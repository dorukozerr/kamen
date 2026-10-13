use image::imageops::FilterType;
use ndarray::Array4;
use ort::{session::Session, value::Tensor};
use std::sync::mpsc;
use std::time::Instant;

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

const DETECT_INTERVAL: u32 = 5;
const HEAD_EXPAND: f32 = 0.25;

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
    #[allow(dead_code)]
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

fn tracker_loop(rx: mpsc::Receiver<cameras::Frame>, tx: mpsc::Sender<TrackResult>) {
    let environment = match ort::init().build() {
        Ok(env) => env,
        Err(_) => return,
    };
    let mut detector = match Session::builder(&environment)
        .and_then(|mut b| b.commit_from_file(format!("{MODEL_DIR}/{DETECTOR_PATH}")))
    {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut landmark = match Session::builder(&environment)
        .and_then(|mut b| b.commit_from_file(format!("{MODEL_DIR}/{LANDMARK_PATH}")))
    {
        Ok(s) => s,
        Err(_) => return,
    };

    let mut frames_since_detect = 0u32;
    let mut last_head: Option<[f32; 4]> = None;

    while let Ok(mut frame) = rx.recv() {
        while let Ok(newer) = rx.try_recv() {
            frame = newer;
        }
        let start = Instant::now();
        let head = track(
            &mut detector,
            &mut landmark,
            &frame,
            frames_since_detect,
            &mut last_head,
        );
        frames_since_detect = (frames_since_detect + 1) % DETECT_INTERVAL;
        let _ = tx.send(TrackResult {
            frame_w: frame.width,
            frame_h: frame.height,
            head,
            ms: start.elapsed().as_secs_f32() * 1000.0,
        });
    }
}

fn track(
    detector: &mut Session,
    landmark: &mut Session,
    frame: &cameras::Frame,
    frames_since_detect: u32,
    last_head: &mut Option<[f32; 4]>,
) -> Option<HeadTrack> {
    let rgba = cameras::to_rgba8(frame).ok()?;
    let rgb: Vec<u8> = rgba.chunks_exact(4).flat_map(|px| [px[0], px[1], px[2]]).collect();

    let crop_rect = if frames_since_detect == 0 || last_head.is_none() {
        let (scale, pad_left, pad_top) = letterbox_params(frame.width as usize, frame.height as usize, DET_W, DET_H);
        let letterboxed = letterbox(&rgb, frame.width as usize, frame.height as usize, DET_W, DET_H, scale, pad_left, pad_top);
        let mut det_input = Array4::<f32>::zeros((1, 3, DET_H, DET_W));
        for y in 0..DET_H {
            for x in 0..DET_W {
                let (src, px) = (y * DET_W + x, 3);
                let base = src * px;
                for (c, channel) in letterboxed[base..base + 3].iter().enumerate() {
                    det_input[[0, c, y, x]] = *channel as f32 / 255.0;
                }
            }
        }
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
                    [b[0] * DET_W as f32, b[1] * DET_H as f32, b[2] * DET_W as f32, b[3] * DET_H as f32],
                )
            })
            .collect();
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        let kept = nms(&candidates, NMS_THRESH);
        let best = kept.first().map(|(_, b)| *b)?;
        let frame_box = [
            (best[0] - pad_left as f32) / scale,
            (best[1] - pad_top as f32) / scale,
            (best[2] - pad_left as f32) / scale,
            (best[3] - pad_top as f32) / scale,
        ];
        *last_head = Some(frame_box);
        frame_box
    } else {
        let Some(head) = *last_head else {
            return None;
        };
        expand(head, HEAD_EXPAND, frame.width as f32, frame.height as f32)
    };

    let crop = crop_rgb(&rgb, frame.width as usize, crop_rect)?;
    let resized = image::imageops::resize(
        &crop,
        LAND_W as u32,
        LAND_H as u32,
        FilterType::Triangle,
    );

    let mut input = Array4::<f32>::zeros((1, 3, LAND_H, LAND_W));
    for y in 0..LAND_H {
        for x in 0..LAND_W {
            for (c, channel) in resized.get_pixel(x as u32, y as u32).0.iter().enumerate() {
                input[[0, c, y, x]] = *channel as f32;
            }
        }
    }

    let outputs = landmark
        .run(ort::inputs![LAND_INPUT => Tensor::from_array(input).ok()?])
        .ok()?;
    let (_, points) = outputs[LANDMARK_OUT].try_extract_tensor::<f32>().ok()?;
    let (_, pose) = outputs[POSE_OUT].try_extract_tensor::<f32>().ok()?;
    let (_, states) = outputs[STATE_OUT].try_extract_tensor::<f32>().ok()?;

    let crop_w = crop_rect[2] - crop_rect[0];
    let crop_h = crop_rect[3] - crop_rect[1];
    let mapped: Vec<[f32; 2]> = points
        .chunks_exact(2)
        .map(|p| {
            [
                crop_rect[0] + p[0] * crop_w,
                crop_rect[1] + p[1] * crop_h,
            ]
        })
        .collect();

    Some(HeadTrack {
        rect: last_head.unwrap_or(crop_rect),
        points: mapped,
        pose: [pose[0], pose[1], pose[2]],
        states: [states[0], states[1], states[2], states[3]],
    })
}

fn letterbox_params(w: usize, h: usize, dw: usize, dh: usize) -> (f32, usize, usize) {
    let scale = (dw as f32 / w as f32).min(dh as f32 / h as f32);
    let new_w = (w as f32 * scale).round() as usize;
    let new_h = (h as f32 * scale).round() as usize;
    (scale, (dw - new_w) / 2, (dh - new_h) / 2)
}

fn letterbox(
    rgb: &[u8],
    w: usize,
    h: usize,
    dw: usize,
    dh: usize,
    scale: f32,
    pad_left: usize,
    pad_top: usize,
) -> Vec<u8> {
    let img = image::RgbImage::from_raw(w as u32, h as u32, rgb.to_vec())
        .expect("frame rgb buffer size mismatch");
    let new_w = (w as f32 * scale).round() as usize;
    let new_h = (h as f32 * scale).round() as usize;
    let resized = image::imageops::resize(
        &img,
        new_w as u32,
        new_h as u32,
        if scale < 1.0 { FilterType::Triangle } else { FilterType::CatmullRom },
    );
    let mut out = image::RgbImage::from_pixel(dw as u32, dh as u32, image::Rgb([0, 0, 0]));
    image::imageops::overlay(&mut out, &resized, pad_left as i64, pad_top as i64);
    out.into_raw()
}

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

fn crop_rgb(rgb: &[u8], w: usize, rect: [f32; 4]) -> Option<image::RgbImage> {
    let img = image::RgbImage::from_raw(w as u32, (rgb.len() / 3 / w) as u32, rgb.to_vec())?;
    let x1 = rect[0].floor().max(0.0) as u32;
    let y1 = rect[1].floor().max(0.0) as u32;
    let x2 = (rect[2].ceil() as u32).min(img.width());
    let y2 = (rect[3].ceil() as u32).min(img.height());
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    Some(
        image::imageops::crop_imm(&img, x1, y1, x2 - x1, y2 - y1).to_image(),
    )
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