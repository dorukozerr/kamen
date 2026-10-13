use cameras::PixelFormat;
use ndarray::Array4;
use ort::environment::Environment;
use ort::ep::{self, coreml};
use ort::{session::Session, value::Tensor};
use std::borrow::Cow;
use std::sync::mpsc;
use std::time::{Duration, Instant};

const MODEL_DIR: &str = "models/face_landmark";
const DETECTOR_PATH: &str = "yolov4_headdetection_480x640.onnx";
const LANDMARK_PATH: &str = "face_landmark_Nx3x160x160.onnx";

const DET_W: usize = 640;
const DET_H: usize = 480;
const DET_INPUT: &str = "input";
const DET_BOXES: &str = "boxes";
const DET_CONFS: &str = "confs";
const CONF_THRESH: f32 = 0.6;
const NMS_THRESH: f32 = 0.5;

const LAND_W: usize = 160;
const LAND_H: usize = 160;
const LAND_INPUT: &str = "images";
const LANDMARK_OUT: &str = "Identity_2";
const POSE_OUT: &str = "Identity_1";
const STATE_OUT: &str = "Identity";

/// How often the detector re-checks the head box while a face is being tracked.
/// Between detections the box follows the landmarks.
const DETECT_INTERVAL: Duration = Duration::from_millis(200);
const HEAD_EXPAND: f32 = 0.25;
/// If the landmark-following box moves further than this fraction of its size
/// from the last detection, the crop has likely slipped off the face: drop it
/// and wait for a fresh detection instead of tracking noise.
const MAX_DRIFT: f32 = 0.5;

pub struct HeadTrack {
    pub rect: [f32; 4],
    pub points: Vec<[f32; 2]>,
    #[allow(dead_code)]
    pub pose: [f32; 3],
    #[allow(dead_code)]
    pub states: [f32; 4],
}

pub struct TrackResult {
    pub frame_w: u32,
    pub frame_h: u32,
    pub head: Option<HeadTrack>,
    pub ms: f32,
}

pub fn spawn_tracker() -> (mpsc::Sender<cameras::Frame>, mpsc::Receiver<TrackResult>) {
    let (frame_tx, frame_rx) = mpsc::channel::<cameras::Frame>();
    let (result_tx, result_rx) = mpsc::channel::<TrackResult>();
    std::thread::Builder::new()
        .name("face-tracker".into())
        .spawn(move || tracker_loop(frame_rx, result_tx))
        .expect("spawn face tracker");
    (frame_tx, result_rx)
}

/// Head box from the detector plus the landmark-centroid offset measured on it,
/// so later frames can re-center the box on the landmarks without drifting.
struct Head {
    rect: [f32; 4],
    detected: [f32; 4],
    centroid_offset: Option<[f32; 2]>,
}

impl Head {
    fn drifted(&self) -> bool {
        let center = |r: [f32; 4]| [(r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0];
        let (a, b) = (center(self.rect), center(self.detected));
        let size = (self.detected[2] - self.detected[0]).max(self.detected[3] - self.detected[1]);
        (a[0] - b[0]).hypot(a[1] - b[1]) > size * MAX_DRIFT
    }
}

fn tracker_loop(rx: mpsc::Receiver<cameras::Frame>, tx: mpsc::Sender<TrackResult>) {
    set_interactive_priority();
    let environment = match ort::init().build() {
        Ok(env) => env,
        Err(error) => {
            eprintln!("face tracker: onnxruntime init failed: {error}");
            return;
        }
    };
    // The landmark model is too small to benefit from more threads, and spinning
    // worker threads would compete with the detector and camera threads.
    let mut landmark = match Session::builder(&environment)
        .and_then(|b| Ok(b.with_intra_threads(1)?))
        .and_then(|b| Ok(b.with_intra_op_spinning(false)?))
        .and_then(|mut b| b.commit_from_file(format!("{MODEL_DIR}/{LANDMARK_PATH}")))
    {
        Ok(s) => s,
        Err(error) => {
            eprintln!("face tracker: loading {LANDMARK_PATH} failed: {error}");
            return;
        }
    };
    let Some(detector) = load_detector(&environment) else {
        return;
    };
    let (det_tx, det_rx) = spawn_detector(detector);

    let mut head: Option<Head> = None;
    let mut detector_busy = false;
    let mut last_detect: Option<Instant> = None;

    while let Ok(mut frame) = rx.recv() {
        while let Ok(newer) = rx.try_recv() {
            frame = newer;
        }
        let start = Instant::now();

        while let Ok(detected) = det_rx.try_recv() {
            detector_busy = false;
            head = detected.map(|rect| Head {
                rect,
                detected: rect,
                centroid_offset: None,
            });
        }
        let detect_due =
            head.is_none() || last_detect.is_none_or(|t| t.elapsed() >= DETECT_INTERVAL);
        if !detector_busy && detect_due {
            if det_tx.send(frame.clone()).is_err() {
                return;
            }
            detector_busy = true;
            last_detect = Some(Instant::now());
        }

        let track = head
            .as_mut()
            .and_then(|head| track_landmarks(&mut landmark, &frame, head));
        if head.as_ref().is_some_and(Head::drifted) {
            head = None;
        }
        let _ = tx.send(TrackResult {
            frame_w: frame.width,
            frame_h: frame.height,
            head: track,
            ms: start.elapsed().as_secs_f32() * 1000.0,
        });
    }
}

/// Runs the head detector on its own thread, since it is far slower than the
/// landmark model. Replies with the best head box in frame coordinates.
fn spawn_detector(mut detector: Session) -> (
    mpsc::Sender<cameras::Frame>,
    mpsc::Receiver<Option<[f32; 4]>>,
) {
    let (frame_tx, frame_rx) = mpsc::channel::<cameras::Frame>();
    let (box_tx, box_rx) = mpsc::channel::<Option<[f32; 4]>>();
    std::thread::Builder::new()
        .name("head-detector".into())
        .spawn(move || {
            while let Ok(mut frame) = frame_rx.recv() {
                while let Ok(newer) = frame_rx.try_recv() {
                    frame = newer;
                }
                if box_tx.send(detect(&mut detector, &frame)).is_err() {
                    return;
                }
            }
        })
        .expect("spawn head detector");
    (frame_tx, box_rx)
}

/// CoreML runs the detector about 8x faster than CPU on Apple Silicon; fall back
/// to CPU where it is unavailable.
fn load_detector(environment: &Environment) -> Option<Session> {
    let path = format!("{MODEL_DIR}/{DETECTOR_PATH}");
    let coreml = Session::builder(environment)
        .and_then(|b| Ok(b.with_intra_op_spinning(false)?))
        .and_then(|b| {
            Ok(b.with_execution_providers([ep::CoreML::default()
                .with_model_format(coreml::ModelFormat::MLProgram)
                .with_compute_units(coreml::ComputeUnits::All)
                .build()
                .error_on_failure()])?)
        })
        .and_then(|mut b| b.commit_from_file(&path));
    match coreml {
        Ok(session) => Some(session),
        Err(error) => {
            eprintln!("face tracker: CoreML unavailable for detector ({error}), using CPU");
            Session::builder(environment)
                .and_then(|mut b| b.commit_from_file(&path))
                .inspect_err(|error| eprintln!("face tracker: loading {DETECTOR_PATH} failed: {error}"))
                .ok()
        }
    }
}

fn detect(detector: &mut Session, frame: &cameras::Frame) -> Option<[f32; 4]> {
    let pixels = Pixels::from_frame(frame)?;
    let (w, h) = (pixels.width as f32, pixels.height as f32);
    let scale = (DET_W as f32 / w).min(DET_H as f32 / h);
    let new_w = (w * scale).round() as usize;
    let new_h = (h * scale).round() as usize;
    let pad_left = (DET_W - new_w) / 2;
    let pad_top = (DET_H - new_h) / 2;

    let mut det_input = Array4::<f32>::zeros((1, 3, DET_H, DET_W));
    pixels.resample_into(
        det_input.as_slice_mut()?,
        DET_W,
        DET_H,
        [0.0, 0.0, w, h],
        [pad_left, pad_top, new_w, new_h],
        1.0 / 255.0,
    );
    let outputs = detector
        .run(ort::inputs![DET_INPUT => Tensor::from_array(det_input).ok()?])
        .ok()?;
    let (_, confs) = outputs[DET_CONFS].try_extract_tensor::<f32>().ok()?;
    let (_, boxes) = outputs[DET_BOXES].try_extract_tensor::<f32>().ok()?;
    let mut candidates: Vec<(f32, [f32; 4])> = (0..confs.len())
        .filter(|&i| confs[i] > CONF_THRESH)
        .map(|i| {
            let b = &boxes[i * 4..i * 4 + 4];
            (
                confs[i],
                [
                    b[0] * DET_W as f32,
                    b[1] * DET_H as f32,
                    b[2] * DET_W as f32,
                    b[3] * DET_H as f32,
                ],
            )
        })
        .collect();
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let kept = nms(&candidates, NMS_THRESH);
    let best = kept.first().map(|(_, b)| *b)?;
    Some([
        (best[0] - pad_left as f32) / scale,
        (best[1] - pad_top as f32) / scale,
        (best[2] - pad_left as f32) / scale,
        (best[3] - pad_top as f32) / scale,
    ])
}

fn track_landmarks(
    landmark: &mut Session,
    frame: &cameras::Frame,
    head: &mut Head,
) -> Option<HeadTrack> {
    let pixels = Pixels::from_frame(frame)?;
    let crop_rect = expand(
        head.rect,
        HEAD_EXPAND,
        pixels.width as f32,
        pixels.height as f32,
    );
    let crop_w = crop_rect[2] - crop_rect[0];
    let crop_h = crop_rect[3] - crop_rect[1];
    if crop_w < 1.0 || crop_h < 1.0 {
        return None;
    }

    let mut input = Array4::<f32>::zeros((1, 3, LAND_H, LAND_W));
    pixels.resample_into(
        input.as_slice_mut()?,
        LAND_W,
        LAND_H,
        crop_rect,
        [0, 0, LAND_W, LAND_H],
        1.0,
    );

    let outputs = landmark
        .run(ort::inputs![LAND_INPUT => Tensor::from_array(input).ok()?])
        .ok()?;
    let (_, points) = outputs[LANDMARK_OUT].try_extract_tensor::<f32>().ok()?;
    let (_, pose) = outputs[POSE_OUT].try_extract_tensor::<f32>().ok()?;
    let (_, states) = outputs[STATE_OUT].try_extract_tensor::<f32>().ok()?;

    let mapped: Vec<[f32; 2]> = points
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| [crop_rect[0] + p[0] * crop_w, crop_rect[1] + p[1] * crop_h])
        .collect();

    // Follow the face between detections: keep the detector's box size, and move
    // its center with the landmark centroid.
    let n = mapped.len().max(1) as f32;
    let centroid = mapped
        .iter()
        .fold([0.0, 0.0], |acc, p| [acc[0] + p[0] / n, acc[1] + p[1] / n]);
    let center = [
        (head.rect[0] + head.rect[2]) / 2.0,
        (head.rect[1] + head.rect[3]) / 2.0,
    ];
    let offset = *head
        .centroid_offset
        .get_or_insert([centroid[0] - center[0], centroid[1] - center[1]]);
    let shift = [
        centroid[0] - offset[0] - center[0],
        centroid[1] - offset[1] - center[1],
    ];
    let rect = head.rect;
    head.rect = [
        rect[0] + shift[0],
        rect[1] + shift[1],
        rect[2] + shift[0],
        rect[3] + shift[1],
    ];

    Some(HeadTrack {
        rect,
        points: mapped,
        pose: [pose[0], pose[1], pose[2]],
        states: [states[0], states[1], states[2], states[3]],
    })
}

/// Read-only view of a frame's pixels as RGB, without converting the whole frame.
struct Pixels<'a> {
    data: Cow<'a, [u8]>,
    width: usize,
    height: usize,
    stride: usize,
    bpp: usize,
    rgb_order: [usize; 3],
}

impl<'a> Pixels<'a> {
    fn from_frame(frame: &'a cameras::Frame) -> Option<Self> {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let packed = |bpp: usize| {
            if frame.stride == 0 {
                width * bpp
            } else {
                frame.stride as usize
            }
        };
        let (data, stride, bpp, rgb_order) = match frame.pixel_format {
            PixelFormat::Bgra8 => (
                Cow::Borrowed(&frame.plane_primary[..]),
                packed(4),
                4,
                [2, 1, 0],
            ),
            PixelFormat::Rgba8 => (
                Cow::Borrowed(&frame.plane_primary[..]),
                packed(4),
                4,
                [0, 1, 2],
            ),
            PixelFormat::Rgb8 => (
                Cow::Borrowed(&frame.plane_primary[..]),
                packed(3),
                3,
                [0, 1, 2],
            ),
            _ => (
                Cow::Owned(cameras::to_rgb8(frame).ok()?),
                width * 3,
                3,
                [0, 1, 2],
            ),
        };
        if width == 0 || height == 0 || data.len() < stride * (height - 1) + width * bpp {
            return None;
        }
        Some(Self {
            data,
            width,
            height,
            stride,
            bpp,
            rgb_order,
        })
    }

    /// Resample the source region `src` (x1, y1, x2, y2) into the `dst` region
    /// (x, y, w, h) of a planar RGB `out` buffer of size `out_w` x `out_h`.
    /// Each output pixel averages a grid of bilinear samples so downscaling
    /// doesn't alias.
    fn resample_into(
        &self,
        out: &mut [f32],
        out_w: usize,
        out_h: usize,
        src: [f32; 4],
        dst: [usize; 4],
        value_scale: f32,
    ) {
        let [dx, dy, dw, dh] = dst;
        let step_x = (src[2] - src[0]) / dw as f32;
        let step_y = (src[3] - src[1]) / dh as f32;
        let taps_x = step_x.ceil().clamp(1.0, 4.0) as usize;
        let taps_y = step_y.ceil().clamp(1.0, 4.0) as usize;
        let weight = value_scale / (taps_x * taps_y) as f32;
        let plane = out_w * out_h;
        for oy in 0..dh {
            for ox in 0..dw {
                let mut sum = [0.0f32; 3];
                for ty in 0..taps_y {
                    let sy =
                        src[1] + (oy as f32 + (ty as f32 + 0.5) / taps_y as f32) * step_y - 0.5;
                    for tx in 0..taps_x {
                        let sx =
                            src[0] + (ox as f32 + (tx as f32 + 0.5) / taps_x as f32) * step_x - 0.5;
                        let rgb = self.bilinear(sx, sy);
                        sum[0] += rgb[0];
                        sum[1] += rgb[1];
                        sum[2] += rgb[2];
                    }
                }
                let index = (dy + oy) * out_w + dx + ox;
                for c in 0..3 {
                    out[c * plane + index] = sum[c] * weight;
                }
            }
        }
    }

    fn bilinear(&self, x: f32, y: f32) -> [f32; 3] {
        let x = x.clamp(0.0, (self.width - 1) as f32);
        let y = y.clamp(0.0, (self.height - 1) as f32);
        let x0 = x as usize;
        let y0 = y as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let mut rgb = [0.0; 3];
        for (c, &channel) in self.rgb_order.iter().enumerate() {
            let at =
                |px: usize, py: usize| self.data[py * self.stride + px * self.bpp + channel] as f32;
            let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
            let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
            rgb[c] = top + (bottom - top) * fy;
        }
        rgb
    }
}

/// Without this macOS tends to schedule the tracker on efficiency cores, which
/// makes landmark inference about 3x slower.
#[cfg(target_os = "macos")]
fn set_interactive_priority() {
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    // SAFETY: only changes the scheduling class of the calling thread.
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn set_interactive_priority() {}

fn expand(rect: [f32; 4], factor: f32, max_w: f32, max_h: f32) -> [f32; 4] {
    let cx = (rect[0] + rect[2]) / 2.0;
    let cy = (rect[1] + rect[3]) / 2.0;
    let w = (rect[2] - rect[0]) * (1.0 + factor);
    let h = (rect[3] - rect[1]) * (1.0 + factor);
    let side = w.max(h);
    [
        (cx - side / 2.0).max(0.0),
        (cy - side / 2.0).max(0.0),
        (cx + side / 2.0).min(max_w),
        (cy + side / 2.0).min(max_h),
    ]
}

fn nms(candidates: &[(f32, [f32; 4])], thresh: f32) -> Vec<(f32, [f32; 4])> {
    let mut kept = Vec::new();
    for (conf, b) in candidates {
        if kept.iter().all(|(_, k)| iou(b, k) <= thresh) {
            kept.push((*conf, *b));
        }
    }
    kept
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let xx1 = a[0].max(b[0]);
    let yy1 = a[1].max(b[1]);
    let xx2 = a[2].min(b[2]);
    let yy2 = a[3].min(b[3]);
    let inter = (xx2 - xx1).max(0.0) * (yy2 - yy1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    inter / (area_a + area_b - inter)
}
