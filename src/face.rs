//! Stage 3: turn 68 landmarks into avatar-ready expression values (`FaceState`).
//!
//! Everything is measured relative to a neutral pose captured during
//! calibration, so it works for different faces and camera angles.
//!
//! Tracking results arrive at camera rate (20-30 fps) and set a target;
//! `advance` eases the presented state toward it on every drawn frame, so the
//! avatar glides instead of stepping.

use std::time::{Duration, Instant};

use glam::{Mat4, Vec3, Vec4};

use crate::tracker::TrackResult;

/// Frames averaged to capture the neutral face.
const CALIBRATION_FRAMES: u32 = 20;

/// Head angle (radians) that maps to 1.0. These match the avatar's own range
/// (`vrm::MAX_*`), so the avatar turns as far as you do, times `HEAD_GAIN`.
const YAW_RANGE: f32 = 0.70;
const PITCH_RANGE: f32 = 0.50;
const ROLL_RANGE: f32 = 0.50;
const HEAD_GAIN: f32 = 1.2;
const MOUTH_OPEN_RANGE: f32 = 0.35; // inner lip gap, in eye distances
const MOUTH_WIDTH_GAIN: f32 = 5.0;
/// An eye counts as fully closed at this fraction of its neutral openness.
const EYE_CLOSED_RATIO: f32 = 0.55;
/// Blink is on/off with hysteresis: closes above the first, opens below the second.
const BLINK_CLOSE: f32 = 0.65;
const BLINK_OPEN: f32 = 0.40;
/// Eyes this similar blink together; a bigger difference is a wink.
const WINK_DIFFERENCE: f32 = 0.45;
/// Mouth-corner lift / eyebrow raise (in eye distances) for a full expression,
/// after a dead zone so a resting face stays neutral.
const SMILE_RANGE: f32 = 0.12;
const BROW_RANGE: f32 = 0.14;
const EXPRESSION_DEAD_ZONE: f32 = 0.25;
/// Mouth-corner drop / eyebrow lowering (in eye distances) for a full value.
const FROWN_RANGE: f32 = 0.10;
const BROW_LOWER_RANGE: f32 = 0.10;
/// Eye closedness (0 open .. 1 shut) where a held narrowing of the eyes
/// starts and where it counts as full. Blinks are handled separately.
const SQUINT_START: f32 = 0.15;
const SQUINT_FULL: f32 = 0.60;
/// Leaning toward the camera: relative growth of the face that maps to 1.
const LEAN_RANGE: f32 = 0.35;
/// Vertical face travel, as a fraction of the frame width, that maps to 1.
const BODY_Y_RANGE: f32 = 0.20;
/// Small eye darts while idle: how far (of the eye's range), and how often.
const SACCADE_X: f32 = 0.30;
const SACCADE_Y: f32 = 0.15;
const SACCADE_MIN_SECONDS: f32 = 1.2;
const SACCADE_EXTRA_SECONDS: f32 = 2.6;
const GAZE_RATE: f32 = 30.0;

/// How fast the presented state follows its target, per second.
const HEAD_RATE: f32 = 16.0;
const MOUTH_RATE: f32 = 28.0;
const BLINK_RATE: f32 = 45.0;
const EXPRESSION_RATE: f32 = 12.0;
const BODY_RATE: f32 = 8.0;

/// Without a face for this long, the avatar returns to neutral and idles.
const LOST_AFTER: Duration = Duration::from_millis(600);
const AUTO_BLINK_SECONDS: f32 = 0.13;
/// Idle head sway amplitude, in `FaceState` units.
const SWAY: f32 = 0.025;
const BREATH_SECONDS: f32 = 3.6;

/// Expression values for the avatar. Head angles and positions are -1..1 with
/// 0 at the calibrated neutral pose; blink, mouth-open, smile and brow-raise
/// are 0..1. Directions are in screen space as the avatar is shown (positive
/// yaw/roll/x = toward screen right, positive pitch = down); left/right eyes
/// are the avatar's own.
#[derive(Clone, Copy, Default, Debug)]
pub struct FaceState {
    pub head_yaw: f32,
    pub head_pitch: f32,
    pub head_roll: f32,
    pub blink_left: f32,
    pub blink_right: f32,
    pub mouth_open: f32,
    pub mouth_width: f32,
    pub smile: f32,
    pub brow_raise: f32,
    /// Subtle expressions, 0..1: lowered brows (concentrating), mouth
    /// corners down, and eyes held narrowed.
    pub brow_lower: f32,
    pub frown: f32,
    pub eye_squint: f32,
    /// Where the eyes point within their range, -1..1 (right, down).
    pub gaze_x: f32,
    pub gaze_y: f32,
    /// Horizontal face position in the frame.
    pub body_x: f32,
    /// Vertical face position, positive up, and leaning toward the camera.
    pub body_y: f32,
    pub lean: f32,
    /// Breathing cycle, -1..1.
    pub breath: f32,
}

// Indices into `Features`.
const YAW: usize = 0;
const PITCH: usize = 1;
const ROLL: usize = 2;
const EAR_LEFT: usize = 3;
const EAR_RIGHT: usize = 4;
const MOUTH_GAP: usize = 5;
const MOUTH_WIDTH: usize = 6;
const SMILE: usize = 7;
const BROW: usize = 8;
const X: usize = 9;
const Y: usize = 10;
const SCALE: usize = 11;
const FEATURE_COUNT: usize = 12;

/// Raw geometric measurements from one set of landmarks (300-W layout).
/// Left/right are as seen in the image.
type Features = [f32; FEATURE_COUNT];

fn features(points: &[[f32; 2]], frame_w: f32) -> Option<Features> {
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
    let (yaw, pitch) = head_direction(points)?;

    // Height above the eye line along the face's own up axis (undoing roll),
    // in eye distances.
    let (sin, cos) = roll.sin_cos();
    let height = |p: [f32; 2]| -((p[1] - eye_mid[1]) * cos - (p[0] - eye_mid[0]) * sin) / eye_dist;
    let mouth_center = (height(points[51]) + height(points[57])) / 2.0;
    let mouth_corners = (height(points[48]) + height(points[54])) / 2.0;
    let brows = (height(centroid(&points[17..22])) + height(centroid(&points[22..27]))) / 2.0;

    let mut out = [0.0; FEATURE_COUNT];
    out[YAW] = yaw;
    out[PITCH] = pitch;
    out[ROLL] = roll;
    out[EAR_LEFT] = eye_aspect(&points[36..42]);
    out[EAR_RIGHT] = eye_aspect(&points[42..48]);
    out[MOUTH_GAP] = dist(points[62], points[66]) / eye_dist;
    out[MOUTH_WIDTH] = dist(points[48], points[54]) / eye_dist;
    out[SMILE] = mouth_corners - mouth_center;
    out[BROW] = brows;
    out[X] = eye_mid[0] / frame_w * 2.0 - 1.0;
    out[Y] = eye_mid[1] / frame_w;
    out[SCALE] = eye_dist / frame_w;
    Some(out)
}

/// Generic face shape for the pose fit: landmark index and its 3D position
/// (x toward image right, y up, z toward the camera, nose tip at the origin).
const FACE_MODEL: [(usize, [f32; 3]); 6] = [
    (30, [0.0, 0.0, 0.0]),       // nose tip
    (8, [0.0, -63.6, -12.5]),    // chin
    (36, [-43.3, 32.7, -26.0]),  // outer eye corner, image left
    (45, [43.3, 32.7, -26.0]),   // outer eye corner, image right
    (48, [-28.9, -28.9, -24.1]), // mouth corner, image left
    (54, [28.9, -28.9, -24.1]),  // mouth corner, image right
];

/// Head yaw and pitch in radians (yaw toward image right, pitch down), by
/// fitting `FACE_MODEL` to the landmarks under a scaled orthographic camera:
/// least squares for the two image-plane rows of the rotation.
fn head_direction(points: &[[f32; 2]]) -> Option<(f32, f32)> {
    let mut ata = Mat4::ZERO;
    let (mut atb_x, mut atb_y) = (Vec4::ZERO, Vec4::ZERO);
    for (index, model) in FACE_MODEL {
        let row = Vec4::new(model[0], model[1], model[2], 1.0);
        ata += Mat4::from_cols(row * row.x, row * row.y, row * row.z, row * row.w);
        atb_x += row * points[index][0];
        // Image y points down.
        atb_y += row * -points[index][1];
    }
    if ata.determinant().abs() < 1e-3 {
        return None;
    }
    let inverse = ata.inverse();
    let right = (inverse * atb_x).truncate().try_normalize()?;
    let up = (inverse * atb_y).truncate().try_normalize()?;
    let toward_camera = right.cross(up).try_normalize()?;
    // Where the face points, in camera axes.
    let forward = Vec3::new(right.z, up.z, toward_camera.z);
    Some((forward.x.atan2(forward.z), (-forward.y).clamp(-1.0, 1.0).asin()))
}

/// Converts track results into a smoothed, calibrated `FaceState`.
pub struct FaceSolver {
    neutral: Option<Features>,
    calibration_sum: Features,
    calibration_count: u32,
    /// yaw, pitch, roll, mouth open, mouth width, smile, brow raise, x,
    /// brow lower, frown, squint, y, lean.
    filters: [OneEuro; 13],
    last_update: Option<Instant>,
    /// Blink state per eye (image left, image right), with hysteresis.
    eye_closed: [bool; 2],
    /// Latest tracked values, at camera rate.
    target: FaceState,
    /// `target`, eased per drawn frame.
    eased: FaceState,
    last_face: Option<Instant>,
    last_advance: Option<Instant>,
    /// Seconds of animation time, for idle motion.
    clock: f32,
    next_blink: f32,
    /// Current eye-dart target and when the next one is due.
    gaze: [f32; 2],
    next_saccade: f32,
    random: u32,
    /// Move like a mirror image (VTuber convention) instead of like the camera view.
    pub mirror: bool,
    /// What the avatar shows this frame.
    pub state: FaceState,
}

impl Default for FaceSolver {
    fn default() -> Self {
        let head = OneEuro::new(1.2, 0.6);
        let mouth = OneEuro::new(5.0, 2.0);
        let expression = OneEuro::new(1.5, 0.5);
        Self {
            neutral: None,
            calibration_sum: [0.0; FEATURE_COUNT],
            calibration_count: 0,
            filters: [
                head, head, head, mouth, mouth, expression, expression, head, expression, expression, expression,
                head, head,
            ],
            gaze: [0.0; 2],
            next_saccade: 1.5,
            last_update: None,
            eye_closed: [false; 2],
            target: FaceState::default(),
            eased: FaceState::default(),
            last_face: None,
            last_advance: None,
            clock: 0.0,
            next_blink: 2.0,
            random: 0x2545_F491,
            mirror: true,
            state: FaceState::default(),
        }
    }
}

impl FaceSolver {
    /// Start a new calibration: hold a neutral face for a moment.
    pub fn calibrate(&mut self) {
        self.neutral = None;
        self.calibration_sum = [0.0; FEATURE_COUNT];
        self.calibration_count = 0;
    }

    pub fn is_calibrating(&self) -> bool {
        self.neutral.is_none()
    }

    /// Feed one tracking result. Call for every result from the tracker.
    pub fn update(&mut self, result: &TrackResult) {
        let Some(head) = &result.head else {
            return;
        };
        let Some(f) = features(&head.points, result.frame_w as f32) else {
            return;
        };
        self.last_face = Some(Instant::now());
        let Some(n) = self.neutral else {
            for (sum, value) in self.calibration_sum.iter_mut().zip(f) {
                *sum += value;
            }
            self.calibration_count += 1;
            if self.calibration_count >= CALIBRATION_FRAMES {
                let count = self.calibration_count as f32;
                self.neutral = Some(self.calibration_sum.map(|sum| sum / count));
            }
            return;
        };

        // Blinks are not smoothed: they snap shut and open, and both eyes
        // move together unless one is clearly more closed (a wink).
        let closedness = |ear: f32, open: f32| {
            let closed = open * EYE_CLOSED_RATIO;
            1.0 - ((ear - closed) / (open - closed).max(1e-3)).clamp(0.0, 1.0)
        };
        let mut eyes = [
            closedness(f[EAR_LEFT], n[EAR_LEFT]),
            closedness(f[EAR_RIGHT], n[EAR_RIGHT]),
        ];
        if (eyes[0] - eyes[1]).abs() < WINK_DIFFERENCE {
            eyes = [(eyes[0] + eyes[1]) / 2.0; 2];
        }
        for (closed, value) in self.eye_closed.iter_mut().zip(eyes) {
            if value > BLINK_CLOSE {
                *closed = true;
            } else if value < BLINK_OPEN {
                *closed = false;
            }
        }
        // Eyes held partly closed (not a blink) narrow the avatar's eyes. The
        // slow filter on this value keeps quick blinks out of it.
        let squint = if self.eye_closed.contains(&true) {
            0.0
        } else {
            (((eyes[0] + eyes[1]) / 2.0 - SQUINT_START) / (SQUINT_FULL - SQUINT_START)).clamp(0.0, 1.0)
        };

        let expression = |value: f32, range: f32| {
            ((value / range - EXPRESSION_DEAD_ZONE) / (1.0 - EXPRESSION_DEAD_ZONE)).clamp(0.0, 1.0)
        };
        let raw = [
            ((f[YAW] - n[YAW]) / YAW_RANGE * HEAD_GAIN).clamp(-1.0, 1.0),
            ((f[PITCH] - n[PITCH]) / PITCH_RANGE * HEAD_GAIN).clamp(-1.0, 1.0),
            ((f[ROLL] - n[ROLL]) / ROLL_RANGE).clamp(-1.0, 1.0),
            ((f[MOUTH_GAP] - n[MOUTH_GAP]) / MOUTH_OPEN_RANGE).clamp(0.0, 1.0),
            ((f[MOUTH_WIDTH] / n[MOUTH_WIDTH].max(1e-3) - 1.0) * MOUTH_WIDTH_GAIN).clamp(-1.0, 1.0),
            expression(f[SMILE] - n[SMILE], SMILE_RANGE),
            expression(f[BROW] - n[BROW], BROW_RANGE),
            (f[X] - n[X]).clamp(-1.0, 1.0),
            expression(n[BROW] - f[BROW], BROW_LOWER_RANGE),
            expression(n[SMILE] - f[SMILE], FROWN_RANGE),
            squint,
            ((n[Y] - f[Y]) / BODY_Y_RANGE).clamp(-1.0, 1.0),
            ((f[SCALE] / n[SCALE].max(1e-6) - 1.0) / LEAN_RANGE).clamp(-1.0, 1.0),
        ];
        let now = Instant::now();
        let dt = self
            .last_update
            .map_or(1.0 / 30.0, |t| now.duration_since(t).as_secs_f32().max(1e-3));
        self.last_update = Some(now);
        let mut out = [0.0; 13];
        for ((o, filter), x) in out.iter_mut().zip(&mut self.filters).zip(raw) {
            *o = filter.filter(x, dt);
        }
        let [image_left_eye, image_right_eye] = self.eye_closed.map(|closed| if closed { 1.0 } else { 0.0 });

        // Values so far are in camera-image space, where the eye on the
        // image's left shows on the avatar's right.
        let flip = if self.mirror { -1.0 } else { 1.0 };
        let (blink_left, blink_right) = if self.mirror {
            (image_left_eye, image_right_eye)
        } else {
            (image_right_eye, image_left_eye)
        };
        self.target = FaceState {
            head_yaw: out[0] * flip,
            head_pitch: out[1],
            head_roll: out[2] * flip,
            blink_left,
            blink_right,
            mouth_open: out[3],
            mouth_width: out[4],
            smile: out[5],
            brow_raise: out[6],
            brow_lower: out[8],
            frown: out[9],
            eye_squint: out[10],
            gaze_x: 0.0,
            gaze_y: 0.0,
            body_x: out[7] * flip,
            body_y: out[11],
            lean: out[12],
            breath: 0.0,
        };
    }

    /// Step the presented state. Call once per drawn frame.
    pub fn advance(&mut self) {
        let now = Instant::now();
        let dt = self
            .last_advance
            .map_or(0.0, |last| now.duration_since(last).as_secs_f32().min(0.1));
        self.last_advance = Some(now);
        self.clock += dt;

        let lost = self.last_face.is_none_or(|seen| now.duration_since(seen) > LOST_AFTER);
        let mut target = self.target;
        if lost {
            // Nobody in view: relax to neutral and blink now and then.
            target = FaceState::default();
            if self.clock > self.next_blink + AUTO_BLINK_SECONDS {
                self.next_blink = self.clock + 2.5 + self.random_unit() * 3.5;
            }
            let blink = if self.clock >= self.next_blink { 1.0 } else { 0.0 };
            (target.blink_left, target.blink_right) = (blink, blink);
        }

        // Eyes are never perfectly still: every second or three they dart to
        // a new spot near the center and hold there.
        if self.clock >= self.next_saccade {
            self.gaze = [
                (self.random_unit() * 2.0 - 1.0) * SACCADE_X,
                (self.random_unit() * 2.0 - 1.0) * SACCADE_Y,
            ];
            self.next_saccade = self.clock + SACCADE_MIN_SECONDS + self.random_unit() * SACCADE_EXTRA_SECONDS;
        }
        let gaze = self.gaze;

        let ease = |value: &mut f32, target: f32, rate: f32| {
            *value += (target - *value) * (1.0 - (-rate * dt).exp());
        };
        let eased = &mut self.eased;
        ease(&mut eased.head_yaw, target.head_yaw, HEAD_RATE);
        ease(&mut eased.head_pitch, target.head_pitch, HEAD_RATE);
        ease(&mut eased.head_roll, target.head_roll, HEAD_RATE);
        ease(&mut eased.blink_left, target.blink_left, BLINK_RATE);
        ease(&mut eased.blink_right, target.blink_right, BLINK_RATE);
        ease(&mut eased.mouth_open, target.mouth_open, MOUTH_RATE);
        ease(&mut eased.mouth_width, target.mouth_width, MOUTH_RATE);
        ease(&mut eased.smile, target.smile, EXPRESSION_RATE);
        ease(&mut eased.brow_raise, target.brow_raise, EXPRESSION_RATE);
        ease(&mut eased.body_x, target.body_x, BODY_RATE);
        ease(&mut eased.body_y, target.body_y, BODY_RATE);
        ease(&mut eased.lean, target.lean, BODY_RATE);
        ease(&mut eased.brow_lower, target.brow_lower, EXPRESSION_RATE);
        ease(&mut eased.frown, target.frown, EXPRESSION_RATE);
        ease(&mut eased.eye_squint, target.eye_squint, EXPRESSION_RATE);
        ease(&mut eased.gaze_x, gaze[0], GAZE_RATE);
        ease(&mut eased.gaze_y, gaze[1], GAZE_RATE);

        // Idle motion on top, so the avatar is never perfectly still.
        let t = self.clock;
        self.state = FaceState {
            head_yaw: eased.head_yaw + SWAY * (t * 0.43).sin(),
            head_pitch: eased.head_pitch + SWAY * 0.6 * (t * 0.61 + 1.3).sin(),
            head_roll: eased.head_roll + SWAY * 0.5 * (t * 0.37 + 2.1).sin(),
            breath: (t * std::f32::consts::TAU / BREATH_SECONDS).sin(),
            ..*eased
        };
    }

    /// Pseudo-random value in 0..1 (xorshift), for blink timing.
    fn random_unit(&mut self) -> f32 {
        self.random ^= self.random << 13;
        self.random ^= self.random >> 17;
        self.random ^= self.random << 5;
        (self.random >> 8) as f32 / (1u32 << 24) as f32
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
