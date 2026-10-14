//! Turn tracked pose and hands into arm and finger directions for the avatar
//! (`BodyState`), smoothed per drawn frame like `face::FaceSolver`.
//!
//! Directions come from MediaPipe's world landmarks, which are metric 3D
//! points, so the avatar's arms move in depth too.

use std::time::{Duration, Instant};

use glam::Vec3;

use crate::tracker::{BodyResult, HAND_POINT_COUNT, HandTrack, PoseTrack};

/// Pose landmarks for each of the person's arms (left, right): shoulder,
/// elbow, wrist (modules/pose_landmark/pose_landmark_topology.svg).
const ARM_POINTS: [[usize; 3]; 2] = [[11, 13, 15], [12, 14, 16]];
/// Hand landmarks per finger, from the base joint to the tip: thumb, index,
/// middle, ring, little (hand_landmarks.png in MediaPipe's docs).
pub const FINGER_POINTS: [[usize; 4]; 5] = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
];
const WRIST: usize = 0;
const INDEX_BASE: usize = 5;
const MIDDLE_BASE: usize = 9;
const LITTLE_BASE: usize = 17;

/// A pose point's visibility maps to how much tracking drives a bone:
/// nothing below the first value, fully from the second.
const VISIBLE_FROM: f32 = 0.5;
const VISIBLE_FULL: f32 = 0.8;
/// Each new measurement moves the target this far toward it, to calm the
/// frame-to-frame jitter of the landmarks.
const TARGET_SMOOTHING: f32 = 0.5;
/// How fast the shown pose follows the target, per second.
const ARM_RATE: f32 = 14.0;
const FINGER_RATE: f32 = 20.0;
const WEIGHT_RATE: f32 = 6.0;
/// Without a body result for this long, the arms go back to rest.
const LOST_AFTER: Duration = Duration::from_millis(600);

/// One arm and hand of the avatar. Directions are unit vectors in the
/// avatar's world space as shown: +X screen right, +Y up, +Z toward the
/// viewer. A zero vector means "not tracked yet".
#[derive(Clone, Copy, Default, Debug)]
pub struct ArmState {
    /// How much tracking drives each part, 0..1; 0 leaves the resting pose.
    pub upper_weight: f32,
    pub lower_weight: f32,
    pub hand_weight: f32,
    /// Shoulder to elbow, and elbow to wrist.
    pub upper: Vec3,
    pub lower: Vec3,
    /// Wrist to the middle finger's base, and index base to little base.
    pub palm_forward: Vec3,
    pub palm_side: Vec3,
    /// Per finger (thumb, index, middle, ring, little), its three segments.
    pub fingers: [[Vec3; 3]; 5],
}

/// The avatar's own left arm first, then its right.
#[derive(Clone, Copy, Default, Debug)]
pub struct BodyState {
    pub arms: [ArmState; 2],
}

#[derive(Default)]
pub struct BodySolver {
    target: BodyState,
    last_seen: Option<Instant>,
    last_advance: Option<Instant>,
    /// Move like a mirror image instead of like the camera view.
    pub mirror: bool,
    /// What the avatar shows this frame.
    pub state: BodyState,
}


impl BodySolver {
    /// Feed one body result. Call for every result from the body tracker.
    pub fn update(&mut self, result: &BodyResult) {
        self.last_seen = Some(Instant::now());
        for person_side in 0..2 {
            // Facing the viewer, the avatar's left arm is on the screen's
            // right, where the person's left arm appears in the camera image.
            // A mirror swaps sides.
            let avatar_side = if self.mirror { 1 - person_side } else { person_side };
            match &result.pose {
                Some(pose) => self.track_arm(pose, person_side, avatar_side),
                None => {
                    let target = &mut self.target.arms[avatar_side];
                    target.upper_weight = 0.0;
                    target.lower_weight = 0.0;
                }
            }
            match &result.hands[person_side] {
                Some(hand) => self.track_hand(hand, avatar_side),
                None => self.target.arms[avatar_side].hand_weight = 0.0,
            }
        }
    }

    /// Camera-image direction (x right, y down, z away) to avatar space.
    fn to_avatar(&self, from: [f32; 3], to: [f32; 3]) -> Vec3 {
        let x = to[0] - from[0];
        let x = if self.mirror { -x } else { x };
        Vec3::new(x, -(to[1] - from[1]), -(to[2] - from[2])).normalize_or_zero()
    }

    fn track_arm(&mut self, pose: &PoseTrack, person_side: usize, avatar_side: usize) {
        let [shoulder, elbow, wrist] = ARM_POINTS[person_side];
        let weight = |point: usize| {
            ((pose.visibility[point] - VISIBLE_FROM) / (VISIBLE_FULL - VISIBLE_FROM)).clamp(0.0, 1.0)
        };
        let upper = self.to_avatar(pose.world[shoulder], pose.world[elbow]);
        let lower = self.to_avatar(pose.world[elbow], pose.world[wrist]);
        let target = &mut self.target.arms[avatar_side];
        target.upper_weight = weight(elbow).min(weight(shoulder));
        target.lower_weight = weight(wrist).min(target.upper_weight);
        smooth_toward(&mut target.upper, upper);
        smooth_toward(&mut target.lower, lower);
    }

    fn track_hand(&mut self, hand: &HandTrack, avatar_side: usize) {
        if hand.world.len() < HAND_POINT_COUNT {
            return;
        }
        let w = &hand.world;
        let palm_forward = self.to_avatar(w[WRIST], w[MIDDLE_BASE]);
        let palm_side = self.to_avatar(w[INDEX_BASE], w[LITTLE_BASE]);
        let fingers = FINGER_POINTS.map(|chain| {
            [
                self.to_avatar(w[chain[0]], w[chain[1]]),
                self.to_avatar(w[chain[1]], w[chain[2]]),
                self.to_avatar(w[chain[2]], w[chain[3]]),
            ]
        });
        let target = &mut self.target.arms[avatar_side];
        target.hand_weight = 1.0;
        smooth_toward(&mut target.palm_forward, palm_forward);
        smooth_toward(&mut target.palm_side, palm_side);
        for (segments, measured) in target.fingers.iter_mut().zip(fingers) {
            for (segment, measured) in segments.iter_mut().zip(measured) {
                smooth_toward(segment, measured);
            }
        }
    }

    /// Step the shown pose. Call once per drawn frame.
    pub fn advance(&mut self) {
        let now = Instant::now();
        let dt = self
            .last_advance
            .map_or(0.0, |last| now.duration_since(last).as_secs_f32().min(0.1));
        self.last_advance = Some(now);
        if self.last_seen.is_none_or(|seen| now.duration_since(seen) > LOST_AFTER) {
            for arm in &mut self.target.arms {
                arm.upper_weight = 0.0;
                arm.lower_weight = 0.0;
                arm.hand_weight = 0.0;
            }
        }
        let step = |rate: f32| 1.0 - (-rate * dt).exp();
        for (shown, target) in self.state.arms.iter_mut().zip(&self.target.arms) {
            ease_direction(&mut shown.upper, target.upper, step(ARM_RATE));
            ease_direction(&mut shown.lower, target.lower, step(ARM_RATE));
            ease_direction(&mut shown.palm_forward, target.palm_forward, step(FINGER_RATE));
            ease_direction(&mut shown.palm_side, target.palm_side, step(FINGER_RATE));
            for (segments, targets) in shown.fingers.iter_mut().zip(&target.fingers) {
                for (segment, target) in segments.iter_mut().zip(targets) {
                    ease_direction(segment, *target, step(FINGER_RATE));
                }
            }
            shown.upper_weight += (target.upper_weight - shown.upper_weight) * step(WEIGHT_RATE);
            shown.lower_weight += (target.lower_weight - shown.lower_weight) * step(WEIGHT_RATE);
            shown.hand_weight += (target.hand_weight - shown.hand_weight) * step(WEIGHT_RATE);
        }
    }
}

/// Move a target direction part way toward a new measurement.
fn smooth_toward(target: &mut Vec3, measured: Vec3) {
    if measured == Vec3::ZERO {
        return;
    }
    *target = if *target == Vec3::ZERO {
        measured
    } else {
        target.lerp(measured, TARGET_SMOOTHING).normalize_or(measured)
    };
}

/// Ease a shown direction toward its target by `amount` (0..1).
fn ease_direction(shown: &mut Vec3, target: Vec3, amount: f32) {
    if target == Vec3::ZERO {
        return;
    }
    *shown = if *shown == Vec3::ZERO {
        target
    } else {
        shown.lerp(target, amount).normalize_or(target)
    };
}
