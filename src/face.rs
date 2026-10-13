//! Stage 3: turn 68 landmarks into avatar-ready expression values (`FaceState`).
//!
//! Everything is measured relative to a neutral pose captured during
//! calibration, so it works for different faces and camera angles.

use std::time::Instant;

use crate::tracker::TrackResult;

/// Frames averaged to capture the neutral face.
const CALIBRATION_FRAMES: u32 = 20;

// Gains map raw geometric changes to roughly -1..1 at a strong expression.
const YAW_GAIN: f32 = 4.0;
const PITCH_GAIN: f32 = 3.0;
const ROLL_RANGE: f32 = 0.5; // radians (~30°) for a full roll
const MOUTH_OPEN_RANGE: f32 = 0.35; // inner lip gap, in eye distances
const MOUTH_WIDTH_GAIN: f32 = 5.0;
/// An eye counts as fully closed at this fraction of its neutral openness.
const EYE_CLOSED_RATIO: f32 = 0.55;

/// Expression values for the avatar. Head angles and positions are -1..1 with
/// 0 at the calibrated neutral pose; blink and mouth-open are 0..1.
/// Directions are in screen space as the avatar is shown (positive yaw/roll/x
/// = toward screen right); left/right eyes are the avatar's own.
#[derive(Clone, Copy, Default, Debug)]
pub struct FaceState {
    pub head_yaw: f32,
    pub head_pitch: f32,
    pub head_roll: f32,
    pub blink_left: f32,
    pub blink_right: f32,
    pub mouth_open: f32,
    pub mouth_width: f32,
    /// Horizontal face position in the frame.
    pub body_x: f32,
}

/// Raw geometric measurements from one set of landmarks.
#[derive(Clone, Copy, Default)]
struct Features {
    yaw: f32,
    pitch: f32,
    roll: f32,
    ear_left: f32,
    ear_right: f32,
    mouth_gap: f32,
    mouth_width: f32,
    x: f32,
}

impl Features {
    fn from_points(points: &[[f32; 2]], frame_w: f32) -> Option<Self> {
        if points.len() != 68 {
            return None;
        }
        let eye_left = centroid(&points[36..42]);
        let eye_right = centroid(&points[42..48]);
        let eye_dist = dist(eye_left, eye_right);
        if eye_dist < 1.0 {
            return None;
        }
        let roll = (eye_right[1] - eye_left[1]).atan2(eye_right[0] - eye_left[0]);
        let eye_mid = [(eye_left[0] + eye_right[0]) / 2.0, (eye_left[1] + eye_right[1]) / 2.0];

        // Undo roll so yaw/pitch are measured along the face's own axes.
        let (sin, cos) = (-roll).sin_cos();
        let upright = |p: [f32; 2]| {
            let (dx, dy) = (p[0] - eye_mid[0], p[1] - eye_mid[1]);
            [dx * cos - dy * sin, dx * sin + dy * cos]
        };
        let nose = upright(points[30]);
        let jaw_left = upright(points[0]);
        let jaw_right = upright(points[16]);
        let face_w = (jaw_right[0] - jaw_left[0]).max(1.0);

        Some(Self {
            yaw: (nose[0] - jaw_left[0]) / face_w - 0.5,
            pitch: nose[1] / eye_dist,
            roll,
            ear_left: eye_aspect(&points[36..42]),
            ear_right: eye_aspect(&points[42..48]),
            mouth_gap: dist(points[62], points[66]) / eye_dist,
            mouth_width: dist(points[48], points[54]) / eye_dist,
            x: eye_mid[0] / frame_w * 2.0 - 1.0,
        })
    }
}

/// Converts track results into a smoothed, calibrated `FaceState`.
pub struct FaceSolver {
    neutral: Option<Features>,
    calibration_sum: Features,
    calibration_count: u32,
    filters: [OneEuro; 8],
    last_update: Option<Instant>,
    /// Move like a mirror image (VTuber convention) instead of like the camera view.
    pub mirror: bool,
    pub state: FaceState,
}

impl Default for FaceSolver {
    fn default() -> Self {
        let head = OneEuro::new(1.5, 0.8);
        let fast = OneEuro::new(4.0, 1.5);
        Self {
            neutral: None,
            calibration_sum: Features::default(),
            calibration_count: 0,
            filters: [head, head, head, fast, fast, fast, head, head],
            last_update: None,
            mirror: true,
            state: FaceState::default(),
        }
    }
}

impl FaceSolver {
    /// Start a new calibration: hold a neutral face for a moment.
    pub fn calibrate(&mut self) {
        self.neutral = None;
        self.calibration_sum = Features::default();
        self.calibration_count = 0;
    }

    pub fn is_calibrating(&self) -> bool {
        self.neutral.is_none()
    }

    pub fn update(&mut self, result: &TrackResult) {
        let Some(head) = &result.head else {
            return;
        };
        let Some(f) = Features::from_points(&head.points, result.frame_w as f32) else {
            return;
        };
        let Some(n) = self.neutral else {
            let s = &mut self.calibration_sum;
            s.yaw += f.yaw;
            s.pitch += f.pitch;
            s.roll += f.roll;
            s.ear_left += f.ear_left;
            s.ear_right += f.ear_right;
            s.mouth_gap += f.mouth_gap;
            s.mouth_width += f.mouth_width;
            s.x += f.x;
            self.calibration_count += 1;
            if self.calibration_count >= CALIBRATION_FRAMES {
                let k = self.calibration_count as f32;
                self.neutral = Some(Features {
                    yaw: s.yaw / k,
                    pitch: s.pitch / k,
                    roll: s.roll / k,
                    ear_left: s.ear_left / k,
                    ear_right: s.ear_right / k,
                    mouth_gap: s.mouth_gap / k,
                    mouth_width: s.mouth_width / k,
                    x: s.x / k,
                });
            }
            return;
        };

        let blink = |ear: f32, open: f32| {
            let closed = open * EYE_CLOSED_RATIO;
            1.0 - ((ear - closed) / (open - closed).max(1e-3)).clamp(0.0, 1.0)
        };
        let raw = [
            ((f.yaw - n.yaw) * YAW_GAIN).clamp(-1.0, 1.0),
            ((f.pitch - n.pitch) * PITCH_GAIN).clamp(-1.0, 1.0),
            ((f.roll - n.roll) / ROLL_RANGE).clamp(-1.0, 1.0),
            blink(f.ear_left, n.ear_left),
            blink(f.ear_right, n.ear_right),
            ((f.mouth_gap - n.mouth_gap) / MOUTH_OPEN_RANGE).clamp(0.0, 1.0),
            ((f.mouth_width / n.mouth_width - 1.0) * MOUTH_WIDTH_GAIN).clamp(-1.0, 1.0),
            (f.x - n.x).clamp(-1.0, 1.0),
        ];

        let now = Instant::now();
        let dt = self
            .last_update
            .map_or(1.0 / 30.0, |t| now.duration_since(t).as_secs_f32().max(1e-3));
        self.last_update = Some(now);
        let mut out = [0.0; 8];
        for ((o, filter), x) in out.iter_mut().zip(&mut self.filters).zip(raw) {
            *o = filter.filter(x, dt);
        }
        // `out` is in camera-image space, where the eye on the image's left
        // shows on the avatar's right.
        let flip = if self.mirror { -1.0 } else { 1.0 };
        let (image_left_eye, image_right_eye) = (out[3], out[4]);
        let (blink_left, blink_right) = if self.mirror {
            (image_left_eye, image_right_eye)
        } else {
            (image_right_eye, image_left_eye)
        };
        self.state = FaceState {
            head_yaw: out[0] * flip,
            head_pitch: out[1],
            head_roll: out[2] * flip,
            blink_left,
            blink_right,
            mouth_open: out[5],
            mouth_width: out[6],
            body_x: out[7] * flip,
        };
    }
}

/// One Euro filter: smooths jitter when still, follows quickly when moving.
#[derive(Clone, Copy)]
struct OneEuro {
    min_cutoff: f32,
    beta: f32,
    x: Option<f32>,
    dx: f32,
}

impl OneEuro {
    const D_CUTOFF: f32 = 1.0;

    fn new(min_cutoff: f32, beta: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            x: None,
            dx: 0.0,
        }
    }

    fn alpha(cutoff: f32, dt: f32) -> f32 {
        let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
        1.0 / (1.0 + tau / dt)
    }

    fn filter(&mut self, x: f32, dt: f32) -> f32 {
        let Some(prev) = self.x else {
            self.x = Some(x);
            return x;
        };
        let dx = (x - prev) / dt;
        self.dx += Self::alpha(Self::D_CUTOFF, dt) * (dx - self.dx);
        let cutoff = self.min_cutoff + self.beta * self.dx.abs();
        let smoothed = prev + Self::alpha(cutoff, dt) * (x - prev);
        self.x = Some(smoothed);
        smoothed
    }
}

fn centroid(points: &[[f32; 2]]) -> [f32; 2] {
    let n = points.len() as f32;
    points
        .iter()
        .fold([0.0, 0.0], |acc, p| [acc[0] + p[0] / n, acc[1] + p[1] / n])
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Eye aspect ratio: eyelid gap over eye width (6 points, 300-W order).
fn eye_aspect(eye: &[[f32; 2]]) -> f32 {
    (dist(eye[1], eye[5]) + dist(eye[2], eye[4])) / (2.0 * dist(eye[0], eye[3]).max(1e-3))
}
