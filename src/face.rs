//! Stage 3: turn a tracked face into avatar-ready values (`FaceState`).
//!
//! Head angles come from the 3D landmarks, expressions and gaze from the 52
//! blendshape values. Everything is measured relative to a neutral pose
//! captured during calibration, so it works for different faces and camera
//! angles.
//!
//! Tracking results arrive at camera rate (20-30 fps) and set a target;
//! `advance` eases the presented state toward it on every drawn frame, so the
//! avatar glides instead of stepping.

use std::time::{Duration, Instant};

use glam::Vec3;

use crate::tracker::{BLENDSHAPE_COUNT, FaceTrack, TrackResult};

/// Positions of the blendshapes used here in MediaPipe's output. "Left" and
/// "right" are the tracked person's own, so "left" is on the image's right.
mod shape {
    pub const BROW_DOWN_LEFT: usize = 1;
    pub const BROW_DOWN_RIGHT: usize = 2;
    pub const BROW_INNER_UP: usize = 3;
    pub const BROW_OUTER_UP_LEFT: usize = 4;
    pub const BROW_OUTER_UP_RIGHT: usize = 5;
    pub const EYE_BLINK_LEFT: usize = 9;
    pub const EYE_BLINK_RIGHT: usize = 10;
    pub const EYE_LOOK_DOWN_LEFT: usize = 11;
    pub const EYE_LOOK_DOWN_RIGHT: usize = 12;
    pub const EYE_LOOK_IN_LEFT: usize = 13;
    pub const EYE_LOOK_IN_RIGHT: usize = 14;
    pub const EYE_LOOK_OUT_LEFT: usize = 15;
    pub const EYE_LOOK_OUT_RIGHT: usize = 16;
    pub const EYE_LOOK_UP_LEFT: usize = 17;
    pub const EYE_LOOK_UP_RIGHT: usize = 18;
    pub const EYE_SQUINT_LEFT: usize = 19;
    pub const EYE_SQUINT_RIGHT: usize = 20;
    pub const JAW_OPEN: usize = 25;
    pub const MOUTH_FROWN_LEFT: usize = 30;
    pub const MOUTH_FROWN_RIGHT: usize = 31;
    pub const MOUTH_FUNNEL: usize = 32;
    pub const MOUTH_PUCKER: usize = 38;
    pub const MOUTH_SMILE_LEFT: usize = 44;
    pub const MOUTH_SMILE_RIGHT: usize = 45;
    pub const MOUTH_STRETCH_LEFT: usize = 46;
    pub const MOUTH_STRETCH_RIGHT: usize = 47;
}

/// Landmarks that define the head's axes: the outer eye corners (image left
/// and right), the top of the forehead and the chin.
const EYE_OUTER_IMAGE_LEFT: usize = 33;
const EYE_OUTER_IMAGE_RIGHT: usize = 263;
const FOREHEAD: usize = 10;
const CHIN: usize = 152;

/// Frames averaged to capture the neutral face.
const CALIBRATION_FRAMES: u32 = 20;

/// Head angle (radians) that maps to 1.0. These match the avatar's own range
/// (`vrm::MAX_*`), so the avatar turns as far as you do, times `HEAD_GAIN`.
const YAW_RANGE: f32 = 0.70;
const PITCH_RANGE: f32 = 0.50;
const ROLL_RANGE: f32 = 0.50;
const HEAD_GAIN: f32 = 1.2;

// Blendshape change from neutral that counts as the full expression.
const JAW_RANGE: f32 = 0.5;
const MOUTH_WIDTH_RANGE: f32 = 0.5;
const SMILE_RANGE: f32 = 0.6;
const FROWN_RANGE: f32 = 0.4;
const BROW_RAISE_RANGE: f32 = 0.5;
const BROW_LOWER_RANGE: f32 = 0.4;
const SQUINT_RANGE: f32 = 0.5;
const GAZE_RANGE: f32 = 0.6;
/// Blink value at which an eye is fully shut.
const BLINK_FULL: f32 = 0.6;
/// Small changes from neutral are ignored, so a resting face stays neutral.
const EXPRESSION_DEAD_ZONE: f32 = 0.15;
/// Blink is on/off with hysteresis: closes above the first, opens below the second.
const BLINK_CLOSE: f32 = 0.65;
const BLINK_OPEN: f32 = 0.40;
/// Eyes this similar blink together; a bigger difference is a wink.
const WINK_DIFFERENCE: f32 = 0.45;
/// Leaning toward the camera: relative growth of the face that maps to 1.
const LEAN_RANGE: f32 = 0.35;
/// Vertical face travel, as a fraction of the frame width, that maps to 1.
const BODY_Y_RANGE: f32 = 0.20;

/// How fast the presented state follows its target, per second.
const HEAD_RATE: f32 = 16.0;
const MOUTH_RATE: f32 = 28.0;
const BLINK_RATE: f32 = 45.0;
const EXPRESSION_RATE: f32 = 12.0;
const BODY_RATE: f32 = 8.0;
const GAZE_RATE: f32 = 30.0;

/// Without a face for this long, the avatar returns to neutral and idles.
const LOST_AFTER: Duration = Duration::from_millis(600);
const AUTO_BLINK_SECONDS: f32 = 0.13;
/// Idle head sway amplitude, in `FaceState` units.
const SWAY: f32 = 0.025;
const BREATH_SECONDS: f32 = 3.6;
/// Small eye darts while nobody is tracked: how far, and how often.
const SACCADE_X: f32 = 0.30;
const SACCADE_Y: f32 = 0.15;
const SACCADE_MIN_SECONDS: f32 = 1.2;
const SACCADE_EXTRA_SECONDS: f32 = 2.6;

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
const X: usize = 3;
const Y: usize = 4;
const SCALE: usize = 5;
const BLINK_IMAGE_LEFT: usize = 6;
const BLINK_IMAGE_RIGHT: usize = 7;
const JAW: usize = 8;
const MOUTH_WIDE: usize = 9;
const MOUTH_ROUND: usize = 10;
const SMILE: usize = 11;
const FROWN: usize = 12;
const BROW_UP: usize = 13;
const BROW_DOWN: usize = 14;
const SQUINT: usize = 15;
const GAZE_X: usize = 16;
const GAZE_Y: usize = 17;
const FEATURE_COUNT: usize = 18;

/// Raw measurements from one tracked face. Directions are as seen in the
/// camera image (x right, y down); angles are in radians.
type Features = [f32; FEATURE_COUNT];

fn features(face: &FaceTrack, frame_w: f32) -> Option<Features> {
    let point = |index: usize| face.points.get(index).map(|p| Vec3::from(*p));
    let (eye_a, eye_b) = (point(EYE_OUTER_IMAGE_LEFT)?, point(EYE_OUTER_IMAGE_RIGHT)?);
    // The head's own axes. Image y points down and z points away from the
    // camera, so a face looking straight at it has `forward` = (0, 0, -1).
    let right = (eye_b - eye_a).try_normalize()?;
    let up = (point(FOREHEAD)? - point(CHIN)?).try_normalize()?;
    let forward = right.cross(up).try_normalize()?;
    let eye_mid = (eye_a + eye_b) / 2.0;

    let b: &[f32; BLENDSHAPE_COUNT] = &face.blendshapes;
    let pair = |left: usize, right: usize| (b[left] + b[right]) / 2.0;
    let mut out = [0.0; FEATURE_COUNT];
    out[YAW] = forward.x.atan2(-forward.z);
    out[PITCH] = forward.y.clamp(-1.0, 1.0).asin();
    out[ROLL] = right.y.atan2(right.x);
    out[X] = eye_mid.x / frame_w * 2.0 - 1.0;
    out[Y] = eye_mid.y / frame_w;
    // Measured in 3D, so turning the head doesn't read as moving away.
    out[SCALE] = eye_a.distance(eye_b) / frame_w;
    // The person's left eye is the one on the image's right.
    out[BLINK_IMAGE_LEFT] = b[shape::EYE_BLINK_RIGHT];
    out[BLINK_IMAGE_RIGHT] = b[shape::EYE_BLINK_LEFT];
    out[JAW] = b[shape::JAW_OPEN];
    out[MOUTH_WIDE] = pair(shape::MOUTH_STRETCH_LEFT, shape::MOUTH_STRETCH_RIGHT);
    out[MOUTH_ROUND] = b[shape::MOUTH_PUCKER].max(b[shape::MOUTH_FUNNEL]);
    out[SMILE] = pair(shape::MOUTH_SMILE_LEFT, shape::MOUTH_SMILE_RIGHT);
    out[FROWN] = pair(shape::MOUTH_FROWN_LEFT, shape::MOUTH_FROWN_RIGHT);
    out[BROW_UP] =
        (b[shape::BROW_INNER_UP] + pair(shape::BROW_OUTER_UP_LEFT, shape::BROW_OUTER_UP_RIGHT)) / 2.0;
    out[BROW_DOWN] = pair(shape::BROW_DOWN_LEFT, shape::BROW_DOWN_RIGHT);
    out[SQUINT] = pair(shape::EYE_SQUINT_LEFT, shape::EYE_SQUINT_RIGHT);
    // Toward the image's right: the person's left eye looking out, or their
    // right eye looking in.
    out[GAZE_X] = ((b[shape::EYE_LOOK_OUT_LEFT] - b[shape::EYE_LOOK_IN_LEFT])
        + (b[shape::EYE_LOOK_IN_RIGHT] - b[shape::EYE_LOOK_OUT_RIGHT]))
        / 2.0;
    out[GAZE_Y] = pair(shape::EYE_LOOK_DOWN_LEFT, shape::EYE_LOOK_DOWN_RIGHT)
        - pair(shape::EYE_LOOK_UP_LEFT, shape::EYE_LOOK_UP_RIGHT);
    Some(out)
}

/// Converts track results into a smoothed, calibrated `FaceState`.
pub struct FaceSolver {
    neutral: Option<Features>,
    calibration_sum: Features,
    calibration_count: u32,
    /// yaw, pitch, roll, mouth open, mouth width, smile, brow raise, x,
    /// brow lower, frown, squint, y, lean, gaze x, gaze y.
    filters: [OneEuro; 15],
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
    /// Idle eye-dart target and when the next one is due.
    idle_gaze: [f32; 2],
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
        let gaze = OneEuro::new(3.0, 1.5);
        Self {
            neutral: None,
            calibration_sum: [0.0; FEATURE_COUNT],
            calibration_count: 0,
            filters: [
                head, head, head, mouth, mouth, expression, expression, head, expression, expression, expression,
                head, head, gaze, gaze,
            ],
            last_update: None,
            eye_closed: [false; 2],
            target: FaceState::default(),
            eased: FaceState::default(),
            last_face: None,
            last_advance: None,
            clock: 0.0,
            next_blink: 2.0,
            idle_gaze: [0.0; 2],
            next_saccade: 1.5,
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
        let Some(face) = &result.face else {
            return;
        };
        let Some(f) = features(face, result.frame_w as f32) else {
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
        let closedness =
            |index: usize| ((f[index] - n[index]) / (BLINK_FULL - n[index]).max(0.1)).clamp(0.0, 1.0);
        let mut eyes = [closedness(BLINK_IMAGE_LEFT), closedness(BLINK_IMAGE_RIGHT)];
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

        // Change from neutral, as a 0..1 expression with a dead zone.
        let expression = |index: usize, range: f32| {
            (((f[index] - n[index]) / range - EXPRESSION_DEAD_ZONE) / (1.0 - EXPRESSION_DEAD_ZONE)).clamp(0.0, 1.0)
        };
        // Narrowed eyes are a held expression; a blink is not one.
        let squint = if self.eye_closed.contains(&true) {
            0.0
        } else {
            expression(SQUINT, SQUINT_RANGE)
        };
        let mouth_width = ((f[MOUTH_WIDE] - n[MOUTH_WIDE]) - (f[MOUTH_ROUND] - n[MOUTH_ROUND])) / MOUTH_WIDTH_RANGE;
        let raw = [
            ((f[YAW] - n[YAW]) / YAW_RANGE * HEAD_GAIN).clamp(-1.0, 1.0),
            ((f[PITCH] - n[PITCH]) / PITCH_RANGE * HEAD_GAIN).clamp(-1.0, 1.0),
            ((f[ROLL] - n[ROLL]) / ROLL_RANGE).clamp(-1.0, 1.0),
            ((f[JAW] - n[JAW]) / JAW_RANGE).clamp(0.0, 1.0),
            mouth_width.clamp(-1.0, 1.0),
            expression(SMILE, SMILE_RANGE),
            expression(BROW_UP, BROW_RAISE_RANGE),
            (f[X] - n[X]).clamp(-1.0, 1.0),
            expression(BROW_DOWN, BROW_LOWER_RANGE),
            expression(FROWN, FROWN_RANGE),
            squint,
            ((n[Y] - f[Y]) / BODY_Y_RANGE).clamp(-1.0, 1.0),
            ((f[SCALE] / n[SCALE].max(1e-6) - 1.0) / LEAN_RANGE).clamp(-1.0, 1.0),
            ((f[GAZE_X] - n[GAZE_X]) / GAZE_RANGE).clamp(-1.0, 1.0),
            ((f[GAZE_Y] - n[GAZE_Y]) / GAZE_RANGE).clamp(-1.0, 1.0),
        ];
        let now = Instant::now();
        let dt = self
            .last_update
            .map_or(1.0 / 30.0, |t| now.duration_since(t).as_secs_f32().max(1e-3));
        self.last_update = Some(now);
        let mut out = [0.0; 15];
        for ((o, filter), x) in out.iter_mut().zip(&mut self.filters).zip(raw) {
            *o = filter.filter(x, dt);
        }

        // Values so far are in camera-image space, where the eye on the
        // image's left shows on the avatar's right.
        let [image_left_eye, image_right_eye] = self.eye_closed.map(|closed| if closed { 1.0 } else { 0.0 });
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
            gaze_x: out[13] * flip,
            gaze_y: out[14],
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
            // Nobody in view: relax to neutral, blink now and then, and let
            // the eyes dart to a new spot every second or three.
            target = FaceState::default();
            if self.clock > self.next_blink + AUTO_BLINK_SECONDS {
                self.next_blink = self.clock + 2.5 + self.random_unit() * 3.5;
            }
            let blink = if self.clock >= self.next_blink { 1.0 } else { 0.0 };
            (target.blink_left, target.blink_right) = (blink, blink);
            if self.clock >= self.next_saccade {
                self.idle_gaze = [
                    (self.random_unit() * 2.0 - 1.0) * SACCADE_X,
                    (self.random_unit() * 2.0 - 1.0) * SACCADE_Y,
                ];
                self.next_saccade = self.clock + SACCADE_MIN_SECONDS + self.random_unit() * SACCADE_EXTRA_SECONDS;
            }
            (target.gaze_x, target.gaze_y) = (self.idle_gaze[0], self.idle_gaze[1]);
        }

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
        ease(&mut eased.brow_lower, target.brow_lower, EXPRESSION_RATE);
        ease(&mut eased.frown, target.frown, EXPRESSION_RATE);
        ease(&mut eased.eye_squint, target.eye_squint, EXPRESSION_RATE);
        ease(&mut eased.gaze_x, target.gaze_x, GAZE_RATE);
        ease(&mut eased.gaze_y, target.gaze_y, GAZE_RATE);
        ease(&mut eased.body_x, target.body_x, BODY_RATE);
        ease(&mut eased.body_y, target.body_y, BODY_RATE);
        ease(&mut eased.lean, target.lean, BODY_RATE);

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

    /// Pseudo-random value in 0..1 (xorshift), for idle timing.
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
