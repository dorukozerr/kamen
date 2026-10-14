//! Stage 2: face tracking with Google's MediaPipe Face Landmarker models
//! (see `models/mediapipe/README.md`).
//!
//! Three small models run on the tracker thread for every camera frame:
//! BlazeFace finds the face (only when it isn't being tracked already), the
//! landmark model places 478 3D points on a rotated crop around it, and the
//! blendshape model turns a subset of those points into 52 expression values.
//! While the face stays in view, the next crop comes from the landmarks.
//!
//! A second thread tracks the body and hands with MediaPipe's Pose and Hand
//! Landmarker models, following MediaPipe Holistic: the body crop comes from
//! the previous frame's pose (or, at first, from the face, since the pose
//! detector model couldn't be converted), and each hand's crop from the pose's
//! wrist and finger points, then from the hand's own landmarks.

use cameras::PixelFormat;
use ndarray::{Array3, Array4};
use ort::{session::Session, value::Tensor};
use std::borrow::Cow;
use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::time::Instant;

const MODEL_DIR: &str = "models/mediapipe";
const DETECTOR_FILE: &str = "face_detector.onnx";
const LANDMARKS_FILE: &str = "face_landmarks_detector.onnx";
const BLENDSHAPES_FILE: &str = "face_blendshapes.onnx";

const DET_SIZE: usize = 128;
const DET_INPUT: &str = "input";
const DET_BOXES: &str = "regressors";
const DET_SCORES: &str = "classificators";
/// Values per detection: box center, size, then 6 key points (x, y).
const DET_VALUES: usize = 16;
const DET_MIN_SCORE: f32 = 0.5;

const LAND_SIZE: usize = 256;
const LAND_INPUT: &str = "input_12";
const LAND_POINTS: &str = "Identity";
const LAND_PRESENCE: &str = "Identity_1";
pub const LANDMARK_COUNT: usize = 478;
const MIN_PRESENCE: f32 = 0.5;

// Pose Landmarker (lite) and Hand Landmarker models. Inputs are RGB 0..1;
// output order and meaning follow MediaPipe's
// modules/pose_landmark/tensors_to_pose_landmarks_and_segmentation.pbtxt and
// modules/hand_landmark/hand_landmark_cpu.pbtxt.
const POSE_FILE: &str = "pose_landmarks_detector.onnx";
const POSE_SIZE: usize = 256;
const POSE_INPUT: &str = "input_1";
/// 39 points x (x, y, z, visibility logit, presence logit), in crop pixels:
/// 33 body landmarks, then auxiliary points used to place the next crop.
const POSE_POINTS: &str = "Identity";
/// Pose presence, already 0..1.
const POSE_FLAG: &str = "Identity_1";
/// 39 x (x, y, z) in meters, origin between the hips, camera-aligned axes.
const POSE_WORLD: &str = "Identity_4";
const POSE_ALL_POINTS: usize = 39;
pub const POSE_POINT_COUNT: usize = 33;
/// Auxiliary points: the body center, and a point that sets the crop's size
/// and rotation (modules/pose_landmark/pose_landmarks_to_roi.pbtxt).
const POSE_ALIGN_CENTER: usize = 33;
const POSE_ALIGN_SCALE: usize = 34;
const POSE_ROI_SCALE: f32 = 1.25;
/// The first body crop, placed from the face: the body's center (between
/// the hips) and a point above the head, in face heights from between the
/// eyes. Rough adult proportions; later crops come from the pose itself.
const BODY_CENTER_BELOW_EYES: f32 = 4.0;
const BODY_TOP_ABOVE_EYES: f32 = 1.1;

const HAND_FILE: &str = "hand_landmarks_detector.onnx";
const HAND_SIZE: usize = 224;
const HAND_INPUT: &str = "input_1";
/// 21 x (x, y, z) in crop pixels.
const HAND_POINTS: &str = "Identity";
/// Hand presence, already 0..1.
const HAND_FLAG: &str = "Identity_1";
/// 21 x (x, y, z) in meters around the hand's center, camera-aligned axes.
const HAND_WORLD: &str = "Identity_3";
pub const HAND_POINT_COUNT: usize = 21;
/// A hand crop from the pose (modules/holistic_landmark/
/// hand_landmarks_from_pose_to_recrop_roi.pbtxt), and from the hand's own
/// landmarks (modules/hand_landmark/hand_landmark_landmarks_to_roi.pbtxt).
const HAND_FROM_POSE_SCALE: f32 = 2.7;
const HAND_FROM_LANDMARKS_SCALE: f32 = 2.0;
const HAND_SHIFT_Y: f32 = -0.1;
/// Hand landmarks MediaPipe uses for the next crop: the wrist and the base
/// two joints of each finger.
const HAND_ROI_POINTS: [usize; 12] = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];
/// Pose points for each of the person's hands: wrist, pinky, index.
const POSE_HAND_POINTS: [[usize; 3]; 2] = [[15, 17, 19], [16, 18, 20]];
/// A wrist must be at least this visible before a hand crop is placed on it.
const MIN_WRIST_VISIBILITY: f32 = 0.5;

const BLEND_INPUT: &str = "serving_default_input_points:0";
const BLEND_OUTPUT: &str = "StatefulPartitionedCall:0";
pub const BLENDSHAPE_COUNT: usize = 52;

/// The crop is this much larger than the face box, as MediaPipe does.
const ROI_SCALE: f32 = 1.5;
/// Landmarks that set the crop's rotation: the outer corners of the eyes.
const EYE_OUTER_IMAGE_LEFT: usize = 33;
const EYE_OUTER_IMAGE_RIGHT: usize = 263;
const FOREHEAD: usize = 10;
const CHIN: usize = 152;
/// The crop holds still unless the face moved by this fraction of its size,
/// grew or shrank by this fraction, or turned by this angle. A crop that
/// follows every pixel of landmark noise feeds that noise back in.
const ROI_MOVE_TOLERANCE: f32 = 0.02;
const ROI_SIZE_TOLERANCE: f32 = 0.05;
const ROI_ANGLE_TOLERANCE: f32 = 0.05;

/// Landmarks the blendshape model reads, in the order it expects them.
const BLENDSHAPE_LANDMARKS: [usize; 146] = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
    81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153,
    154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251,
    263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318,
    321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386,
    387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
];

/// A tracked face. `points` are MediaPipe's 478 landmarks in frame pixels
/// (x right, y down, z toward the camera is negative, in the same scale).
/// `blendshapes` are the 52 ARKit-style expression values, 0..1, in
/// MediaPipe's order (see `face::shape`).
pub struct FaceTrack {
    pub points: Vec<[f32; 3]>,
    pub blendshapes: [f32; BLENDSHAPE_COUNT],
}

/// A tracked body: MediaPipe's 33 pose landmarks.
pub struct PoseTrack {
    /// In frame pixels.
    pub points: Vec<[f32; 2]>,
    /// In meters, origin between the hips, x right / y down / z away from
    /// the camera, as seen in the image.
    pub world: Vec<[f32; 3]>,
    /// How likely each point is visible in the frame, 0..1.
    pub visibility: Vec<f32>,
}

/// A tracked hand: MediaPipe's 21 hand landmarks.
pub struct HandTrack {
    /// In frame pixels.
    pub points: Vec<[f32; 2]>,
    /// In meters around the hand's center, axes as in `PoseTrack::world`.
    pub world: Vec<[f32; 3]>,
}

pub struct BodyResult {
    pub pose: Option<PoseTrack>,
    /// The person's own left hand first, then their right.
    pub hands: [Option<HandTrack>; 2],
}

pub struct TrackResult {
    pub frame_w: u32,
    #[allow(dead_code)]
    pub frame_h: u32,
    pub face: Option<FaceTrack>,
    #[allow(dead_code)]
    pub ms: f32,
}

/// Channels to the two tracking threads. Send every camera frame to both;
/// each thread skips to the newest frame it has when it's ready.
pub struct Trackers {
    pub face_frames: mpsc::Sender<cameras::Frame>,
    pub face_results: mpsc::Receiver<TrackResult>,
    pub body_frames: mpsc::Sender<cameras::Frame>,
    pub body_results: mpsc::Receiver<BodyResult>,
}

/// Where the face is, shared from the face thread so the body thread can
/// place its first crop.
#[derive(Clone, Copy)]
struct FaceBox {
    /// Between the eyes, in frame pixels.
    cx: f32,
    cy: f32,
    /// Forehead to chin, in frame pixels.
    height: f32,
}

type FaceBoxSlot = Arc<Mutex<Option<FaceBox>>>;

pub fn spawn_trackers() -> Trackers {
    let (face_frames, face_frame_rx) = mpsc::channel::<cameras::Frame>();
    let (face_result_tx, face_results) = mpsc::channel::<TrackResult>();
    let (body_frames, body_frame_rx) = mpsc::channel::<cameras::Frame>();
    let (body_result_tx, body_results) = mpsc::channel::<BodyResult>();
    // ONNX Runtime allows one environment per process, so the face thread
    // loads every model and hands the body models to the body thread.
    std::thread::Builder::new()
        .name("face-tracker".into())
        .spawn(move || tracker_loop(face_frame_rx, face_result_tx, body_frame_rx, body_result_tx))
        .expect("spawn face tracker");
    Trackers {
        face_frames,
        face_results,
        body_frames,
        body_results,
    }
}

/// Models live in `Contents/Resources` when running from a macOS .app bundle
/// (where the working directory is `/`), otherwise relative to the working
/// directory.
fn model_path(file: &str) -> std::path::PathBuf {
    let bundled = std::env::current_exe()
        .ok()
        .and_then(|exe| Some(exe.parent()?.parent()?.join("Resources").join(MODEL_DIR).join(file)))
        .filter(|path| path.exists());
    bundled.unwrap_or_else(|| std::path::Path::new(MODEL_DIR).join(file))
}

/// Square crop of the frame, rotated by `angle` around its center.
#[derive(Clone, Copy)]
struct Roi {
    cx: f32,
    cy: f32,
    size: f32,
    angle: f32,
}

impl Roi {
    /// Frame position of a point given in crop coordinates (0..1 across it).
    fn to_frame(&self, u: f32, v: f32) -> [f32; 2] {
        let (sin, cos) = self.angle.sin_cos();
        let (u, v) = ((u - 0.5) * self.size, (v - 0.5) * self.size);
        [self.cx + u * cos - v * sin, self.cy + u * sin + v * cos]
    }

    /// Whether `other` is close enough that keeping this crop is better.
    fn close_to(&self, other: &Roi) -> bool {
        let moved = (self.cx - other.cx).hypot(self.cy - other.cy);
        moved < self.size * ROI_MOVE_TOLERANCE
            && (self.size - other.size).abs() < self.size * ROI_SIZE_TOLERANCE
            && (self.angle - other.angle).abs() < ROI_ANGLE_TOLERANCE
    }
}

struct Models {
    detector: Session,
    landmarks: Session,
    blendshapes: Session,
    anchors: Vec<[f32; 2]>,
}

struct BodyModels {
    pose: Session,
    hand: Session,
}

fn tracker_loop(
    rx: mpsc::Receiver<cameras::Frame>,
    tx: mpsc::Sender<TrackResult>,
    body_rx: mpsc::Receiver<cameras::Frame>,
    body_tx: mpsc::Sender<BodyResult>,
) {
    set_interactive_priority();
    let Some((mut models, body_models)) = load_models() else {
        return;
    };
    let face_box: FaceBoxSlot = Arc::new(Mutex::new(None));
    match body_models {
        Some(body_models) => {
            let face_box = face_box.clone();
            std::thread::Builder::new()
                .name("body-tracker".into())
                .spawn(move || body_loop(body_models, body_rx, body_tx, face_box))
                .expect("spawn body tracker");
        }
        None => eprintln!("body tracker: models missing, tracking the face only"),
    }
    let mut roi: Option<Roi> = None;

    while let Ok(mut frame) = rx.recv() {
        while let Ok(newer) = rx.try_recv() {
            frame = newer;
        }
        let start = Instant::now();
        let face = Pixels::from_frame(&frame).and_then(|pixels| track(&mut models, &pixels, &mut roi));
        if face.is_none() {
            roi = None;
        }
        *face_box.lock().unwrap_or_else(PoisonError::into_inner) = face.as_ref().map(|face| {
            let point = |index: usize| face.points[index];
            let (a, b) = (point(EYE_OUTER_IMAGE_LEFT), point(EYE_OUTER_IMAGE_RIGHT));
            let (top, chin) = (point(FOREHEAD), point(CHIN));
            FaceBox {
                cx: (a[0] + b[0]) / 2.0,
                cy: (a[1] + b[1]) / 2.0,
                height: (top[0] - chin[0]).hypot(top[1] - chin[1]),
            }
        });
        let _ = tx.send(TrackResult {
            frame_w: frame.width,
            frame_h: frame.height,
            face,
            ms: start.elapsed().as_secs_f32() * 1000.0,
        });
    }
}

/// The face models, and the body models if they load too.
fn load_models() -> Option<(Models, Option<BodyModels>)> {
    let environment = ort::init()
        .build()
        .inspect_err(|error| eprintln!("face tracker: onnxruntime init failed: {error}"))
        .ok()?;
    // The models are small; more threads, or threads that spin while idle,
    // only compete with the camera and render threads.
    let load = |file: &str| {
        Session::builder(&environment)
            .and_then(|b| Ok(b.with_intra_threads(2)?))
            .and_then(|b| Ok(b.with_intra_op_spinning(false)?))
            .and_then(|mut b| b.commit_from_file(model_path(file)))
            .inspect_err(|error| eprintln!("face tracker: loading {file} failed: {error}"))
            .ok()
    };
    let models = Models {
        detector: load(DETECTOR_FILE)?,
        landmarks: load(LANDMARKS_FILE)?,
        blendshapes: load(BLENDSHAPES_FILE)?,
        anchors: detector_anchors(),
    };
    let body = (|| {
        Some(BodyModels {
            pose: load(POSE_FILE)?,
            hand: load(HAND_FILE)?,
        })
    })();
    Some((models, body))
}

/// One frame: find the face if it isn't tracked, place the landmarks, read the
/// expression values, and derive the next frame's crop from the landmarks.
fn track(models: &mut Models, pixels: &Pixels, roi: &mut Option<Roi>) -> Option<FaceTrack> {
    let current = match *roi {
        Some(roi) => roi,
        None => detect(models, pixels)?,
    };
    let points = landmarks(&mut models.landmarks, pixels, current)?;
    let next = roi_from_landmarks(&points);
    *roi = Some(if current.close_to(&next) { current } else { next });

    let mut input = Array3::<f32>::zeros((1, BLENDSHAPE_LANDMARKS.len(), 2));
    for (row, &index) in BLENDSHAPE_LANDMARKS.iter().enumerate() {
        input[[0, row, 0]] = points[index][0];
        input[[0, row, 1]] = points[index][1];
    }
    let outputs = models
        .blendshapes
        .run(ort::inputs![BLEND_INPUT => Tensor::from_array(input).ok()?])
        .ok()?;
    let (_, values) = outputs[BLEND_OUTPUT].try_extract_tensor::<f32>().ok()?;
    let mut blendshapes = [0.0; BLENDSHAPE_COUNT];
    if values.len() < BLENDSHAPE_COUNT {
        return None;
    }
    blendshapes.copy_from_slice(&values[..BLENDSHAPE_COUNT]);
    Some(FaceTrack { points, blendshapes })
}

/// BlazeFace's fixed anchor grid: 16x16 cells with 2 anchors each, then 8x8
/// cells with 6. Each entry is the anchor's center, 0..1 across the input.
fn detector_anchors() -> Vec<[f32; 2]> {
    let mut anchors = Vec::with_capacity(896);
    for (cells, per_cell) in [(16, 2), (8, 6)] {
        for y in 0..cells {
            for x in 0..cells {
                let center = [(x as f32 + 0.5) / cells as f32, (y as f32 + 0.5) / cells as f32];
                anchors.extend(std::iter::repeat_n(center, per_cell));
            }
        }
    }
    anchors
}

/// Run the detector on the whole frame (letterboxed into a square) and return
/// the crop for the best face, if any.
fn detect(models: &mut Models, pixels: &Pixels) -> Option<Roi> {
    let (w, h) = (pixels.width as f32, pixels.height as f32);
    let frame = Roi {
        cx: w / 2.0,
        cy: h / 2.0,
        size: w.max(h),
        angle: 0.0,
    };
    let mut input = Array4::<f32>::zeros((1, DET_SIZE, DET_SIZE, 3));
    pixels.sample(input.as_slice_mut()?, DET_SIZE, frame, -1.0, 1.0);
    let outputs = models
        .detector
        .run(ort::inputs![DET_INPUT => Tensor::from_array(input).ok()?])
        .ok()?;
    let (_, scores) = outputs[DET_SCORES].try_extract_tensor::<f32>().ok()?;
    let (_, boxes) = outputs[DET_BOXES].try_extract_tensor::<f32>().ok()?;

    let (best, &logit) = scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
    let score = 1.0 / (1.0 + (-logit.clamp(-100.0, 100.0)).exp());
    let anchor = models.anchors.get(best)?;
    let raw = boxes.get(best * DET_VALUES..(best + 1) * DET_VALUES)?;
    if score < DET_MIN_SCORE {
        return None;
    }
    // Offsets are in input pixels, relative to the anchor.
    let point = |x: f32, y: f32| frame.to_frame(x / DET_SIZE as f32 + anchor[0], y / DET_SIZE as f32 + anchor[1]);
    let center = point(raw[0], raw[1]);
    let size = (raw[2].max(raw[3]) / DET_SIZE as f32) * frame.size;
    // Key points 0 and 1 are the eyes; the crop is rotated to level them.
    let (eye_a, eye_b) = (point(raw[4], raw[5]), point(raw[6], raw[7]));
    Some(Roi {
        cx: center[0],
        cy: center[1],
        size: size * ROI_SCALE,
        angle: (eye_b[1] - eye_a[1]).atan2(eye_b[0] - eye_a[0]),
    })
}

/// Run the landmark model on `roi`. Returns the 478 points in frame pixels,
/// or `None` if the model sees no face there.
fn landmarks(session: &mut Session, pixels: &Pixels, roi: Roi) -> Option<Vec<[f32; 3]>> {
    let mut input = Array4::<f32>::zeros((1, LAND_SIZE, LAND_SIZE, 3));
    pixels.sample(input.as_slice_mut()?, LAND_SIZE, roi, 0.0, 1.0);
    let outputs = session
        .run(ort::inputs![LAND_INPUT => Tensor::from_array(input).ok()?])
        .ok()?;
    let (_, presence) = outputs[LAND_PRESENCE].try_extract_tensor::<f32>().ok()?;
    let presence = 1.0 / (1.0 + (-presence.first()?.clamp(-100.0, 100.0)).exp());
    if presence < MIN_PRESENCE {
        return None;
    }
    let (_, raw) = outputs[LAND_POINTS].try_extract_tensor::<f32>().ok()?;
    if raw.len() < LANDMARK_COUNT * 3 {
        return None;
    }
    // The model answers in crop pixels; z uses the same scale as x.
    let scale = 1.0 / LAND_SIZE as f32;
    Some(
        raw.chunks_exact(3)
            .take(LANDMARK_COUNT)
            .map(|p| {
                let [x, y] = roi.to_frame(p[0] * scale, p[1] * scale);
                [x, y, p[2] * scale * roi.size]
            })
            .collect(),
    )
}

/// The crop for the next frame: the landmarks' bounding box, measured along
/// the face's own axes, enlarged and rotated to keep the eyes level.
fn roi_from_landmarks(points: &[[f32; 3]]) -> Roi {
    let (a, b) = (points[EYE_OUTER_IMAGE_LEFT], points[EYE_OUTER_IMAGE_RIGHT]);
    let angle = (b[1] - a[1]).atan2(b[0] - a[0]);
    let (mut min, mut max) = ([f32::MAX; 2], [f32::MIN; 2]);
    for p in points {
        for axis in 0..2 {
            min[axis] = min[axis].min(p[axis]);
            max[axis] = max[axis].max(p[axis]);
        }
    }
    let center = [(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0];
    let (sin, cos) = (-angle).sin_cos();
    let (mut low, mut high) = ([f32::MAX; 2], [f32::MIN; 2]);
    for p in points {
        let (dx, dy) = (p[0] - center[0], p[1] - center[1]);
        let upright = [dx * cos - dy * sin, dx * sin + dy * cos];
        for axis in 0..2 {
            low[axis] = low[axis].min(upright[axis]);
            high[axis] = high[axis].max(upright[axis]);
        }
    }
    Roi {
        cx: center[0],
        cy: center[1],
        size: (high[0] - low[0]).max(high[1] - low[1]) * ROI_SCALE,
        angle,
    }
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
            PixelFormat::Bgra8 => (Cow::Borrowed(&frame.plane_primary[..]), packed(4), 4, [2, 1, 0]),
            PixelFormat::Rgba8 => (Cow::Borrowed(&frame.plane_primary[..]), packed(4), 4, [0, 1, 2]),
            PixelFormat::Rgb8 => (Cow::Borrowed(&frame.plane_primary[..]), packed(3), 3, [0, 1, 2]),
            _ => (Cow::Owned(cameras::to_rgb8(frame).ok()?), width * 3, 3, [0, 1, 2]),
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

    /// Fill `out` (`size` x `size` pixels, RGB interleaved) with the crop
    /// `roi`, mapping 0..255 to `low..high`. Pixels outside the frame are
    /// black. Each output pixel averages a small grid of bilinear samples, so
    /// strong downscaling doesn't alias.
    fn sample(&self, out: &mut [f32], size: usize, roi: Roi, low: f32, high: f32) {
        let step = roi.size / size as f32;
        let taps = (step / 2.0).ceil().clamp(1.0, 4.0) as usize;
        let scale = (high - low) / 255.0 / (taps * taps) as f32;
        let (sin, cos) = roi.angle.sin_cos();
        let (max_x, max_y) = (self.width as f32, self.height as f32);
        for oy in 0..size {
            for ox in 0..size {
                let mut sum = [0.0f32; 3];
                for ty in 0..taps {
                    let v = (oy as f32 + (ty as f32 + 0.5) / taps as f32) / size as f32 - 0.5;
                    for tx in 0..taps {
                        let u = (ox as f32 + (tx as f32 + 0.5) / taps as f32) / size as f32 - 0.5;
                        let x = roi.cx + (u * cos - v * sin) * roi.size;
                        let y = roi.cy + (u * sin + v * cos) * roi.size;
                        if x >= 0.0 && x < max_x && y >= 0.0 && y < max_y {
                            let rgb = self.bilinear(x - 0.5, y - 0.5);
                            sum[0] += rgb[0];
                            sum[1] += rgb[1];
                            sum[2] += rgb[2];
                        }
                    }
                }
                let index = (oy * size + ox) * 3;
                for channel in 0..3 {
                    out[index + channel] = low + sum[channel] * scale;
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
            let at = |px: usize, py: usize| self.data[py * self.stride + px * self.bpp + channel] as f32;
            let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
            let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
            rgb[c] = top + (bottom - top) * fy;
        }
        rgb
    }
}

/// Without this macOS tends to schedule the tracker on efficiency cores, which
/// makes inference about 3x slower.
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-100.0, 100.0)).exp())
}

/// A crop centered on `center` whose "up" points from `center` toward
/// `toward`: MediaPipe's AlignmentPointsRectsCalculator with a 90 degree
/// target angle, followed by its RectTransformationCalculator.
fn roi_toward(center: [f32; 2], toward: [f32; 2], size: f32, shift_y: f32) -> Roi {
    let (dx, dy) = (toward[0] - center[0], toward[1] - center[1]);
    let angle = dx.atan2(-dy);
    let (sin, cos) = angle.sin_cos();
    // Shift along the crop's own down axis, in units of its size.
    Roi {
        cx: center[0] - sin * shift_y * size,
        cy: center[1] + cos * shift_y * size,
        size,
        angle,
    }
}

fn body_loop(
    mut models: BodyModels,
    rx: mpsc::Receiver<cameras::Frame>,
    tx: mpsc::Sender<BodyResult>,
    face_box: FaceBoxSlot,
) {
    set_interactive_priority();
    let mut pose_roi: Option<Roi> = None;
    let mut hand_rois: [Option<Roi>; 2] = [None; 2];
    while let Ok(mut frame) = rx.recv() {
        while let Ok(newer) = rx.try_recv() {
            frame = newer;
        }
        let Some(pixels) = Pixels::from_frame(&frame) else {
            continue;
        };
        if pose_roi.is_none() {
            pose_roi = face_box
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .map(body_roi_from_face);
        }
        let pose = pose_roi.and_then(|roi| pose_landmarks(&mut models.pose, &pixels, roi));
        pose_roi = match &pose {
            Some((_, next)) => Some(match pose_roi {
                Some(current) if current.close_to(next) => current,
                _ => *next,
            }),
            None => None,
        };
        let pose = pose.map(|(pose, _)| pose);

        let mut hands: [Option<HandTrack>; 2] = [None, None];
        for side in 0..2 {
            let roi = hand_rois[side].or_else(|| pose.as_ref().and_then(|pose| hand_roi_from_pose(pose, side)));
            let hand = roi.and_then(|roi| hand_landmarks(&mut models.hand, &pixels, roi));
            hand_rois[side] = hand.as_ref().map(|(_, next)| match hand_rois[side] {
                Some(current) if current.close_to(next) => current,
                _ => *next,
            });
            hands[side] = hand.map(|(hand, _)| hand);
        }
        if tx.send(BodyResult { pose, hands }).is_err() {
            return;
        }
    }
}

/// The first body crop, from the face: centered between the hips, sized to
/// reach above the head, upright.
fn body_roi_from_face(face: FaceBox) -> Roi {
    let center = [face.cx, face.cy + face.height * BODY_CENTER_BELOW_EYES];
    let top = [face.cx, face.cy - face.height * BODY_TOP_ABOVE_EYES];
    let reach = (top[1] - center[1]).abs();
    roi_toward(center, top, 2.0 * reach * POSE_ROI_SCALE, 0.0)
}

/// Run the pose model on `roi`. Returns the pose and the crop for the next
/// frame, or `None` if no body is there.
fn pose_landmarks(session: &mut Session, pixels: &Pixels, roi: Roi) -> Option<(PoseTrack, Roi)> {
    let mut input = Array4::<f32>::zeros((1, POSE_SIZE, POSE_SIZE, 3));
    pixels.sample(input.as_slice_mut()?, POSE_SIZE, roi, 0.0, 1.0);
    let outputs = session
        .run(ort::inputs![POSE_INPUT => Tensor::from_array(input).ok()?])
        .ok()?;
    let (_, flag) = outputs[POSE_FLAG].try_extract_tensor::<f32>().ok()?;
    if *flag.first()? < MIN_PRESENCE {
        return None;
    }
    let (_, raw) = outputs[POSE_POINTS].try_extract_tensor::<f32>().ok()?;
    let (_, world) = outputs[POSE_WORLD].try_extract_tensor::<f32>().ok()?;
    if raw.len() < POSE_ALL_POINTS * 5 || world.len() < POSE_ALL_POINTS * 3 {
        return None;
    }
    let scale = 1.0 / POSE_SIZE as f32;
    let all: Vec<[f32; 2]> = raw
        .chunks_exact(5)
        .take(POSE_ALL_POINTS)
        .map(|p| roi.to_frame(p[0] * scale, p[1] * scale))
        .collect();
    let next = {
        let (center, toward) = (all[POSE_ALIGN_CENTER], all[POSE_ALIGN_SCALE]);
        let reach = (toward[0] - center[0]).hypot(toward[1] - center[1]);
        roi_toward(center, toward, 2.0 * reach * POSE_ROI_SCALE, 0.0)
    };
    let pose = PoseTrack {
        points: all[..POSE_POINT_COUNT].to_vec(),
        world: rotate_world(&world[..POSE_POINT_COUNT * 3], roi.angle),
        visibility: raw
            .chunks_exact(5)
            .take(POSE_POINT_COUNT)
            .map(|p| sigmoid(p[3]))
            .collect(),
    };
    Some((pose, next))
}

/// World landmarks come out in the crop's orientation; turn them back to the
/// image's (MediaPipe's WorldLandmarkProjectionCalculator).
fn rotate_world(values: &[f32], angle: f32) -> Vec<[f32; 3]> {
    let (sin, cos) = angle.sin_cos();
    values
        .chunks_exact(3)
        .map(|p| [p[0] * cos - p[1] * sin, p[0] * sin + p[1] * cos, p[2]])
        .collect()
}

/// A hand crop from the pose's wrist, pinky and index points, for one of the
/// person's hands (0 = left, 1 = right), if the wrist is visible
/// (MediaPipe's HandDetectionsFromPoseToRectsCalculator).
fn hand_roi_from_pose(pose: &PoseTrack, side: usize) -> Option<Roi> {
    let [wrist, pinky, index] = POSE_HAND_POINTS[side];
    if *pose.visibility.get(wrist)? < MIN_WRIST_VISIBILITY {
        return None;
    }
    let (w, p, i) = (pose.points[wrist], pose.points[pinky], pose.points[index]);
    let middle = [(2.0 * i[0] + p[0]) / 3.0, (2.0 * i[1] + p[1]) / 3.0];
    let size = 2.0 * (middle[0] - w[0]).hypot(middle[1] - w[1]);
    if size < 4.0 {
        return None;
    }
    // Rotation points from the wrist toward the fingers; the crop is
    // centered on the fingers' base.
    let roi = roi_toward(middle, [2.0 * middle[0] - w[0], 2.0 * middle[1] - w[1]], size * HAND_FROM_POSE_SCALE, HAND_SHIFT_Y);
    Some(roi)
}

/// Run the hand model on `roi`. Returns the hand and the crop for the next
/// frame, or `None` if no hand is there.
fn hand_landmarks(session: &mut Session, pixels: &Pixels, roi: Roi) -> Option<(HandTrack, Roi)> {
    let mut input = Array4::<f32>::zeros((1, HAND_SIZE, HAND_SIZE, 3));
    pixels.sample(input.as_slice_mut()?, HAND_SIZE, roi, 0.0, 1.0);
    let outputs = session
        .run(ort::inputs![HAND_INPUT => Tensor::from_array(input).ok()?])
        .ok()?;
    let (_, flag) = outputs[HAND_FLAG].try_extract_tensor::<f32>().ok()?;
    if *flag.first()? < MIN_PRESENCE {
        return None;
    }
    let (_, raw) = outputs[HAND_POINTS].try_extract_tensor::<f32>().ok()?;
    let (_, world) = outputs[HAND_WORLD].try_extract_tensor::<f32>().ok()?;
    if raw.len() < HAND_POINT_COUNT * 3 || world.len() < HAND_POINT_COUNT * 3 {
        return None;
    }
    let scale = 1.0 / HAND_SIZE as f32;
    let points: Vec<[f32; 2]> = raw
        .chunks_exact(3)
        .take(HAND_POINT_COUNT)
        .map(|p| roi.to_frame(p[0] * scale, p[1] * scale))
        .collect();
    let next = hand_roi_from_landmarks(&points);
    let hand = HandTrack {
        world: rotate_world(&world[..HAND_POINT_COUNT * 3], roi.angle),
        points,
    };
    Some((hand, next))
}

/// The next hand crop from the hand's landmarks (MediaPipe's
/// HandLandmarksToRectCalculator): rotated so the fingers point up, around
/// the bounding box of the wrist and finger-base points.
fn hand_roi_from_landmarks(points: &[[f32; 2]]) -> Roi {
    let wrist = points[0];
    // Between the index and ring finger bases, averaged with the middle's.
    let (index, middle, ring) = (points[5], points[9], points[13]);
    let toward = [
        ((index[0] + ring[0]) / 2.0 + middle[0]) / 2.0,
        ((index[1] + ring[1]) / 2.0 + middle[1]) / 2.0,
    ];
    let angle = (toward[0] - wrist[0]).atan2(-(toward[1] - wrist[1]));
    let subset: Vec<[f32; 2]> = HAND_ROI_POINTS.iter().map(|&i| points[i]).collect();
    let (mut min, mut max) = ([f32::MAX; 2], [f32::MIN; 2]);
    for p in &subset {
        for axis in 0..2 {
            min[axis] = min[axis].min(p[axis]);
            max[axis] = max[axis].max(p[axis]);
        }
    }
    let center = [(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0];
    // Bounding box along the hand's own axes.
    let (sin, cos) = (-angle).sin_cos();
    let (mut low, mut high) = ([f32::MAX; 2], [f32::MIN; 2]);
    for p in &subset {
        let (dx, dy) = (p[0] - center[0], p[1] - center[1]);
        let upright = [dx * cos - dy * sin, dx * sin + dy * cos];
        for axis in 0..2 {
            low[axis] = low[axis].min(upright[axis]);
            high[axis] = high[axis].max(upright[axis]);
        }
    }
    let local = [(low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0];
    let (sin, cos) = angle.sin_cos();
    let box_center = [
        center[0] + local[0] * cos - local[1] * sin,
        center[1] + local[0] * sin + local[1] * cos,
    ];
    let size = (high[0] - low[0]).max(high[1] - low[1]) * HAND_FROM_LANDMARKS_SCALE;
    Roi {
        cx: box_center[0] - sin * HAND_SHIFT_Y * size,
        cy: box_center[1] + cos * HAND_SHIFT_Y * size,
        size,
        angle,
    }
}
