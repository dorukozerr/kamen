//! VRM avatars (VRoid Studio exports, VRM 0.x and 1.0) driven by `FaceState`.
//!
//! A VRM file is a binary glTF with extensions for the humanoid bone map,
//! facial expressions and toon materials. Morph targets and skinning are done
//! on the CPU each frame; drawing is a small toon shader into an MSAA
//! framebuffer that is resolved into the avatar render target.

use std::collections::HashMap;
use std::time::Instant;

use glam::{EulerRot, Mat4, Quat, Vec2, Vec3};
use glow::HasContext;
use serde_json::Value;

use crate::body::BodyState;
use crate::face::FaceState;

const MAX_YAW: f32 = 0.70; // radians at head_yaw = 1
const MAX_PITCH: f32 = 0.50;
const MAX_ROLL: f32 = 0.50;
/// How the head rotation is spread along the spine, hips to head.
const BONE_SHARES: [(&str, f32); 4] = [("spine", 0.1), ("chest", 0.1), ("neck", 0.3), ("head", 0.5)];
/// Chest rotation in radians at the peak of a breath.
const BREATH_ANGLE: f32 = 0.012;
/// How strongly a tracked smile / raised brows drive the `happy` and
/// `surprised` expressions. Kept below 1: on VRoid models the full
/// expressions also close or widen the eyes.
const SMILE_WEIGHT: f32 = 0.7;
const BROW_WEIGHT: f32 = 0.6;
/// Subtle expressions: lowered brows borrow a little of `angry`, a frown a
/// little of `sad`, and narrowed eyes part of a blink.
const BROW_LOWER_WEIGHT: f32 = 0.35;
const FROWN_WEIGHT: f32 = 0.4;
const SQUINT_WEIGHT: f32 = 0.45;
/// Eye rotation in radians at full tracked gaze.
const EYE_MAX_YAW: f32 = 0.30;
const EYE_MAX_PITCH: f32 = 0.18;
/// Spine tilt in radians when leaning fully toward the camera.
const LEAN_ANGLE: f32 = 0.10;
/// Outline thickness limit in meters. Models thin their outlines with a
/// width texture we don't read, so the declared width is an upper bound.
const OUTLINE_MAX: f32 = 0.0015;
/// VRM rests in a T-pose; lower the arms by this angle.
const ARM_DOWN: f32 = 1.25;
/// Sideways hip travel in meters at body_x = 1.
const BODY_TRAVEL: f32 = 0.08;
const FOV_Y: f32 = 0.35;
const MSAA_SAMPLES: i32 = 4;
/// Direction toward the light, in world space (the avatar faces +Z).
const LIGHT_DIR: Vec3 = Vec3::new(0.25, 0.45, 0.85);

struct Node {
    parent: Option<usize>,
    rest_translation: Vec3,
    rest_rotation: Quat,
    scale: Vec3,
    translation: Vec3,
    rotation: Quat,
    world: Mat4,
    rest_world_rotation: Quat,
    rest_world_position: Vec3,
}

/// VRM 1.0 node constraint: helper bones (sleeves, twist bones) that follow
/// another bone.
enum ConstraintKind {
    /// Copy the source's rotation around this local axis.
    Roll(Vec3),
    /// Point this local axis at the source.
    Aim(Vec3),
    /// Copy the source's rotation.
    Rotation,
}

struct Constraint {
    node: usize,
    source: usize,
    weight: f32,
    kind: ConstraintKind,
}

/// One simulated bone of a spring chain (hair, clothes, ...). Its tail swings
/// under inertia, gravity and a pull back toward the rest pose.
struct SpringJoint {
    node: usize,
    /// Tail position in the node's local space.
    tail: Vec3,
    stiffness: f32,
    gravity: Vec3,
    drag: f32,
    hit_radius: f32,
    colliders: Vec<usize>,
    /// World-space tail now and one step ago; `None` until first simulated.
    state: Option<(Vec3, Vec3)>,
}

/// Sphere (or capsule, with `tail`) attached to a body bone that spring
/// joints can't enter.
struct Collider {
    node: usize,
    offset: Vec3,
    tail: Option<Vec3>,
    radius: f32,
}

struct Skin {
    joints: Vec<usize>,
    inverse_bind: Vec<Mat4>,
}

#[derive(Clone, Copy, PartialEq)]
enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

struct Material {
    image: Option<usize>,
    color: [f32; 4],
    shade: [f32; 3],
    /// Outline thickness in meters (0 = none) and its color multiplier.
    outline_width: f32,
    outline_color: [f32; 3],
    /// N·L where shading is fully dark / fully lit.
    shade_range: [f32; 2],
    alpha: AlphaMode,
    cutoff: f32,
    double_sided: bool,
    depth_write: bool,
    queue: i32,
}

struct Primitive {
    index_offset: usize,
    index_count: usize,
    material: Option<usize>,
}

/// Primitives of one mesh that share a vertex buffer.
struct Geometry {
    node: usize,
    mesh: usize,
    skin: Option<usize>,
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<[f32; 2]>,
    joints: Vec<[u16; 4]>,
    weights: Vec<[f32; 4]>,
    /// Per morph target: sparse position deltas.
    morphs: Vec<Vec<(u32, Vec3)>>,
    morph_weights: Vec<f32>,
    indices: Vec<u32>,
    primitives: Vec<Primitive>,
    morphed: Vec<Vec3>,
    /// Skinned position + normal per vertex, as uploaded to the GPU.
    vertices: Vec<f32>,
}

struct Expression {
    name: String,
    /// (geometry, morph target, weight)
    binds: Vec<(usize, usize, f32)>,
}

struct DecodedImage {
    size: [u32; 2],
    rgba: Vec<u8>,
}

struct GlGeometry {
    vao: glow::VertexArray,
    vertex_buffer: glow::Buffer,
}

struct Msaa {
    framebuffer: glow::Framebuffer,
    color: glow::Renderbuffer,
    depth: glow::Renderbuffer,
    size: [u32; 2],
}

struct GlState {
    program: glow::Program,
    u_view_proj: Option<glow::UniformLocation>,
    u_color: Option<glow::UniformLocation>,
    u_shade: Option<glow::UniformLocation>,
    u_shade_range: Option<glow::UniformLocation>,
    u_light: Option<glow::UniformLocation>,
    u_cutoff: Option<glow::UniformLocation>,
    u_alpha_mode: Option<glow::UniformLocation>,
    u_outline_width: Option<glow::UniformLocation>,
    u_outline_color: Option<glow::UniformLocation>,
    textures: HashMap<usize, glow::Texture>,
    white: glow::Texture,
    geometries: Vec<GlGeometry>,
    msaa: Msaa,
    background_program: glow::Program,
    u_background_crop: Option<glow::UniformLocation>,
    /// Core-profile OpenGL needs a vertex array bound even to draw without
    /// vertex data.
    background_vao: glow::VertexArray,
    /// The uploaded background image: which `Arc` it came from, the texture
    /// and the image size.
    background: Option<(usize, glow::Texture, [u32; 2])>,
}

pub struct VrmAvatar {
    pub name: Option<String>,
    nodes: Vec<Node>,
    /// Node indices, parents before children.
    order: Vec<usize>,
    root_transform: Mat4,
    skins: Vec<Skin>,
    geometries: Vec<Geometry>,
    materials: Vec<Material>,
    images: HashMap<usize, DecodedImage>,
    /// (geometry, primitive) in draw order.
    draw_order: Vec<(usize, usize)>,
    bones: HashMap<String, usize>,
    constraints: Vec<Constraint>,
    springs: Vec<SpringJoint>,
    colliders: Vec<Collider>,
    last_frame: Option<Instant>,
    expressions: Vec<Expression>,
    /// Feet-to-top height and head joint height in meters, in the rest pose.
    pub height: f32,
    pub head_y: f32,
    gl: Option<GlState>,
}

impl VrmAvatar {
    pub fn load(data: &[u8]) -> Result<Self, String> {
        let json = glb_json(data)?;
        let extensions = &json["extensions"];
        let (vrm0, vrm1) = (&extensions["VRM"], &extensions["VRMC_vrm"]);
        if vrm0.is_null() && vrm1.is_null() {
            return Err("glTF file has no VRM data (not a .vrm avatar)".into());
        }
        let is_vrm0 = vrm1.is_null();

        let gltf = gltf::Gltf::from_slice_without_validation(data).map_err(|e| format!("parse VRM: {e}"))?;
        let blob = gltf.blob.as_deref().ok_or("VRM has no binary chunk")?;
        let document = &gltf.document;
        let buffer = |buffer: gltf::Buffer| (buffer.index() == 0).then_some(blob);

        let mut nodes: Vec<Node> = document
            .nodes()
            .map(|node| {
                let (t, r, s) = node.transform().decomposed();
                let (translation, rotation) = (Vec3::from(t), Quat::from_array(r).normalize());
                Node {
                    parent: None,
                    rest_translation: translation,
                    rest_rotation: rotation,
                    scale: Vec3::from(s),
                    translation,
                    rotation,
                    world: Mat4::IDENTITY,
                    rest_world_rotation: Quat::IDENTITY,
                    rest_world_position: Vec3::ZERO,
                }
            })
            .collect();
        let mut order = Vec::with_capacity(nodes.len());
        let scene = document.default_scene().or_else(|| document.scenes().next());
        let mut stack: Vec<gltf::Node> = scene.map(|s| s.nodes().collect()).unwrap_or_default();
        while let Some(node) = stack.pop() {
            order.push(node.index());
            for child in node.children() {
                nodes[child.index()].parent = Some(node.index());
                stack.push(child);
            }
        }

        let skins = document
            .skins()
            .map(|skin| {
                let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();
                let mut inverse_bind: Vec<Mat4> = skin
                    .reader(buffer)
                    .read_inverse_bind_matrices()
                    .map(|m| m.map(|m| Mat4::from_cols_array_2d(&m)).collect())
                    .unwrap_or_default();
                inverse_bind.resize(joints.len(), Mat4::IDENTITY);
                Skin { joints, inverse_bind }
            })
            .collect();

        let mut geometries: Vec<Geometry> = Vec::new();
        for node in document.nodes() {
            let Some(mesh) = node.mesh() else {
                continue;
            };
            // Primitives usually share one vertex buffer and differ in indices.
            let mut by_accessor: HashMap<usize, usize> = HashMap::new();
            for primitive in mesh.primitives() {
                let Some(position_accessor) = primitive.get(&gltf::Semantic::Positions) else {
                    continue;
                };
                let reader = primitive.reader(buffer);
                let geometry_index = match by_accessor.get(&position_accessor.index()) {
                    Some(&index) => index,
                    None => {
                        let Some(positions) = reader.read_positions() else {
                            continue;
                        };
                        let positions: Vec<Vec3> = positions.map(Vec3::from).collect();
                        let count = positions.len();
                        let mut normals: Vec<Vec3> = reader
                            .read_normals()
                            .map(|n| n.map(Vec3::from).collect())
                            .unwrap_or_default();
                        normals.resize(count, Vec3::Z);
                        let mut uvs: Vec<[f32; 2]> = reader
                            .read_tex_coords(0)
                            .map(|t| t.into_f32().collect())
                            .unwrap_or_default();
                        uvs.resize(count, [0.0; 2]);
                        let mut joints: Vec<[u16; 4]> = reader
                            .read_joints(0)
                            .map(|j| j.into_u16().collect())
                            .unwrap_or_default();
                        joints.resize(count, [0; 4]);
                        let mut weights: Vec<[f32; 4]> = reader
                            .read_weights(0)
                            .map(|w| w.into_f32().collect())
                            .unwrap_or_default();
                        weights.resize(count, [0.0; 4]);
                        let morphs: Vec<Vec<(u32, Vec3)>> = reader
                            .read_morph_targets()
                            .map(|(deltas, _, _)| {
                                deltas
                                    .map(|deltas| {
                                        deltas
                                            .enumerate()
                                            .filter(|(_, d)| d.iter().any(|c| c.abs() > 1e-7))
                                            .map(|(i, d)| (i as u32, Vec3::from(d)))
                                            .collect()
                                    })
                                    .unwrap_or_default()
                            })
                            .collect();
                        geometries.push(Geometry {
                            node: node.index(),
                            mesh: mesh.index(),
                            skin: node.skin().map(|s| s.index()),
                            morph_weights: vec![0.0; morphs.len()],
                            morphed: positions.clone(),
                            vertices: vec![0.0; count * 6],
                            positions,
                            normals,
                            uvs,
                            joints,
                            weights,
                            morphs,
                            indices: Vec::new(),
                            primitives: Vec::new(),
                        });
                        by_accessor.insert(position_accessor.index(), geometries.len() - 1);
                        geometries.len() - 1
                    }
                };
                let geometry = &mut geometries[geometry_index];
                let index_offset = geometry.indices.len();
                match reader.read_indices() {
                    Some(indices) => geometry.indices.extend(indices.into_u32()),
                    None => geometry.indices.extend(0..geometry.positions.len() as u32),
                }
                let vertex_count = geometry.positions.len() as u32;
                if geometry.indices[index_offset..].iter().any(|&i| i >= vertex_count) {
                    return Err("VRM mesh has out-of-range vertex indices".into());
                }
                geometry.primitives.push(Primitive {
                    index_offset,
                    index_count: geometry.indices.len() - index_offset,
                    material: primitive.material().index(),
                });
            }
        }
        if geometries.is_empty() {
            return Err("VRM has no meshes".into());
        }

        let materials: Vec<Material> = document
            .materials()
            .filter_map(|material| Some((material.index()?, material)))
            .map(|(index, material)| read_material(&material, index, &json, is_vrm0))
            .collect();
        let images = decode_images(document, blob, &materials);

        let mut draw_order: Vec<(usize, usize)> = geometries
            .iter()
            .enumerate()
            .flat_map(|(g, geometry)| (0..geometry.primitives.len()).map(move |p| (g, p)))
            .collect();
        draw_order.sort_by_key(|&(g, p)| {
            geometries[g].primitives[p]
                .material
                .and_then(|m| materials.get(m))
                .map_or(2000, |m| m.queue)
        });

        let mut bones = HashMap::new();
        if is_vrm0 {
            for bone in vrm0["humanoid"]["humanBones"].as_array().into_iter().flatten() {
                if let (Some(name), Some(node)) = (bone["bone"].as_str(), bone["node"].as_u64()) {
                    bones.insert(name.to_owned(), node as usize);
                }
            }
        } else if let Some(map) = vrm1["humanoid"]["humanBones"].as_object() {
            for (name, bone) in map {
                if let Some(node) = bone["node"].as_u64() {
                    bones.insert(name.clone(), node as usize);
                }
            }
        }
        bones.retain(|_, node| *node < nodes.len());

        let expressions = read_expressions(vrm0, vrm1, is_vrm0, &geometries);
        let nodes_len = nodes.len();
        let (mut springs, colliders) = if is_vrm0 {
            read_springs_vrm0(&vrm0["secondaryAnimation"], document, &nodes)
        } else {
            read_springs_vrm1(&extensions["VRMC_springBone"], &nodes)
        };
        // Simulate parents before children.
        let mut rank = vec![usize::MAX; nodes_len];
        for (position, &node) in order.iter().enumerate() {
            rank[node] = position;
        }
        springs.sort_by_key(|joint| rank[joint.node]);
        let name = if is_vrm0 {
            vrm0["meta"]["title"].as_str()
        } else {
            vrm1["meta"]["name"].as_str()
        }
        .filter(|name| !name.is_empty())
        .map(str::to_owned);

        let mut avatar = Self {
            name,
            nodes,
            order,
            // VRM 0.x models face -Z, VRM 1.0 models face +Z.
            root_transform: if is_vrm0 {
                Mat4::from_rotation_y(std::f32::consts::PI)
            } else {
                Mat4::IDENTITY
            },
            skins,
            geometries,
            materials,
            images,
            draw_order,
            bones,
            constraints: read_constraints(&json, nodes_len),
            springs,
            colliders,
            last_frame: None,
            expressions,
            height: 1.5,
            head_y: 1.3,
            gl: None,
        };
        avatar.update_world();
        for node in &mut avatar.nodes {
            node.rest_world_rotation = node.world.to_scale_rotation_translation().1;
            node.rest_world_position = node.world.w_axis.truncate();
        }
        avatar.skin();
        let (mut low, mut high) = (f32::MAX, f32::MIN);
        for geometry in &avatar.geometries {
            for vertex in geometry.vertices.chunks_exact(6) {
                low = low.min(vertex[1]);
                high = high.max(vertex[1]);
            }
        }
        if high > low {
            avatar.height = high - low.min(0.0);
        }
        avatar.head_y = match avatar.bones.get("head") {
            Some(&head) => avatar.nodes[head].world.w_axis.y,
            None => avatar.height * 0.85,
        };
        Ok(avatar)
    }

    /// Names shown in the parameter list: driven bones and all expressions.
    pub fn param_names(&self) -> Vec<String> {
        let bones = BONE_SHARES
            .iter()
            .filter(|(bone, _)| self.bones.contains_key(*bone))
            .map(|(bone, _)| format!("Bone: {bone}"));
        let expressions = self.expressions.iter().map(|e| format!("Expression: {}", e.name));
        bones.chain(expressions).collect()
    }

    fn update_world(&mut self) {
        for &index in &self.order {
            let node = &self.nodes[index];
            let local = Mat4::from_scale_rotation_translation(node.scale, node.rotation, node.translation);
            let parent = node.parent.map_or(self.root_transform, |parent| self.nodes[parent].world);
            self.nodes[index].world = parent * local;
        }
    }

    /// Solve node constraints for the current pose. Two passes, since a
    /// constrained node can hang off another constrained node.
    fn apply_constraints(&mut self) {
        if self.constraints.is_empty() {
            return;
        }
        for _ in 0..2 {
            for constraint in &self.constraints {
                let (node, source) = (&self.nodes[constraint.node], &self.nodes[constraint.source]);
                let rest = node.rest_rotation;
                let target = match constraint.kind {
                    ConstraintKind::Rotation => rest * (source.rest_rotation.inverse() * source.rotation),
                    ConstraintKind::Roll(axis) => {
                        let delta_in_parent = source.rotation * source.rest_rotation.inverse();
                        let delta = rest.inverse() * delta_in_parent * rest;
                        let swing = Quat::from_rotation_arc(axis, (delta * axis).normalize());
                        rest * swing.inverse() * delta
                    }
                    ConstraintKind::Aim(axis) => {
                        let parent = match node.parent {
                            Some(parent) => self.nodes[parent].world,
                            None => self.root_transform,
                        };
                        let parent_rotation = parent.to_scale_rotation_translation().1;
                        let position = parent.transform_point3(node.translation);
                        let to = (source.world.w_axis.truncate() - position).normalize_or_zero();
                        if to == Vec3::ZERO {
                            continue;
                        }
                        let from = (parent_rotation * rest * axis).normalize();
                        parent_rotation.inverse() * Quat::from_rotation_arc(from, to) * parent_rotation * rest
                    }
                };
                self.nodes[constraint.node].rotation = rest.slerp(target, constraint.weight).normalize();
            }
            self.update_world();
        }
    }

    /// Step the spring bones (hair and cloth physics) by `dt` seconds.
    fn simulate_springs(&mut self, dt: f32) {
        if self.springs.is_empty() {
            return;
        }
        let colliders: Vec<(Vec3, Option<Vec3>, f32)> = self
            .colliders
            .iter()
            .map(|collider| {
                let world = self.nodes[collider.node].world;
                (
                    world.transform_point3(collider.offset),
                    collider.tail.map(|tail| world.transform_point3(tail)),
                    collider.radius,
                )
            })
            .collect();
        for joint in &mut self.springs {
            let node = &self.nodes[joint.node];
            let parent_world = node.parent.map_or(self.root_transform, |parent| self.nodes[parent].world);
            let rest_world = parent_world
                * Mat4::from_scale_rotation_translation(node.scale, node.rest_rotation, node.rest_translation);
            let head = rest_world.w_axis.truncate();
            let rest_tail = rest_world.transform_point3(joint.tail);
            let length = head.distance(rest_tail);
            if length < 1e-5 {
                continue;
            }
            let (current, previous) = joint.state.unwrap_or((rest_tail, rest_tail));
            let mut next = current
                + (current - previous) * (1.0 - joint.drag)
                + (rest_tail - head) / length * joint.stiffness * dt
                + joint.gravity * dt;
            let keep_length = |tail: Vec3| head + (tail - head).normalize_or(rest_tail - head).normalize() * length;
            next = keep_length(next);
            for &index in &joint.colliders {
                let Some(&(start, end, radius)) = colliders.get(index) else {
                    continue;
                };
                // Closest point on the collider's axis (a point for spheres).
                let closest = match end {
                    Some(end) => {
                        let axis = end - start;
                        let t = ((next - start).dot(axis) / axis.length_squared().max(1e-9)).clamp(0.0, 1.0);
                        start + axis * t
                    }
                    None => start,
                };
                let reach = radius + joint.hit_radius;
                let away = next - closest;
                if away.length_squared() < reach * reach {
                    next = keep_length(closest + away.normalize_or(Vec3::Y) * reach);
                }
            }
            joint.state = Some((next, current));

            let to = rest_world.inverse().transform_point3(next).normalize_or_zero();
            let from = joint.tail.normalize_or_zero();
            if to == Vec3::ZERO || from == Vec3::ZERO {
                continue;
            }
            let node = &mut self.nodes[joint.node];
            node.rotation = (node.rest_rotation * Quat::from_rotation_arc(from, to)).normalize();
            node.world = parent_world * Mat4::from_scale_rotation_translation(node.scale, node.rotation, node.translation);
        }
        self.update_world();
    }

    /// Rotate a bone by `delta` in world space, on top of its rest pose.
    fn rotate_bone(&mut self, bone: &str, delta: Quat) {
        if let Some(&index) = self.bones.get(bone) {
            let node = &mut self.nodes[index];
            let world = node.rest_world_rotation;
            node.rotation = (node.rest_rotation * world.inverse() * delta * world).normalize();
        }
    }

    /// Rest-pose world position of a humanoid bone.
    fn rest_position(&self, bone: &str) -> Option<Vec3> {
        self.bones.get(bone).map(|&index| self.nodes[index].rest_world_position)
    }

    /// Rest-pose direction from `bone` toward `toward`, in world space.
    fn rest_direction(&self, bone: &str, toward: &str) -> Option<Vec3> {
        (self.rest_position(toward)? - self.rest_position(bone)?).try_normalize()
    }

    /// World rotation that takes the bone's rest direction to `target`, blended
    /// from `resting` by `weight`. Falls back to `resting` without tracking.
    fn aimed(rest: Option<Vec3>, target: Vec3, resting: Quat, weight: f32) -> Quat {
        match (rest, target.try_normalize()) {
            (Some(rest), Some(target)) if weight > 0.0 => resting.slerp(Quat::from_rotation_arc(rest, target), weight),
            _ => resting,
        }
    }

    /// Arms and fingers from body tracking. Each bone gets a world rotation
    /// on top of its rest pose; `rotate_bone` takes the part not already
    /// inherited from the parent, `parent⁻¹ * total`.
    fn pose_arms(&mut self, body: &BodyState, values: &mut Vec<(String, Vec2)>) {
        // VRM 1.0 names the thumb's first bone "metacarpal"; VRM 0.x calls the
        // same three bones proximal, intermediate, distal.
        let thumb = if self.bones.contains_key("leftThumbMetacarpal") {
            ["ThumbMetacarpal", "ThumbProximal", "ThumbDistal"]
        } else {
            ["ThumbProximal", "ThumbIntermediate", "ThumbDistal"]
        };
        let fingers = [
            thumb,
            ["IndexProximal", "IndexIntermediate", "IndexDistal"],
            ["MiddleProximal", "MiddleIntermediate", "MiddleDistal"],
            ["RingProximal", "RingIntermediate", "RingDistal"],
            ["LittleProximal", "LittleIntermediate", "LittleDistal"],
        ];
        for (arm, (side, down)) in body.arms.iter().zip([("left", -ARM_DOWN), ("right", ARM_DOWN)]) {
            let name = |bone: &str| format!("{side}{bone}");
            // The arm hangs down at rest; VRM models are stored in a T-pose.
            let resting = Quat::from_rotation_z(down);
            let upper = Self::aimed(
                self.rest_direction(&name("UpperArm"), &name("LowerArm")),
                arm.upper,
                resting,
                arm.upper_weight,
            );
            self.rotate_bone(&name("UpperArm"), upper);
            let lower = Self::aimed(
                self.rest_direction(&name("LowerArm"), &name("Hand")),
                arm.lower,
                upper,
                arm.lower_weight,
            );
            self.rotate_bone(&name("LowerArm"), upper.inverse() * lower);
            values.push((format!("Arm: {side}"), Vec2::new(arm.upper_weight, arm.lower_weight)));

            // The hand turns its rest palm (wrist to middle finger, index to
            // little finger) onto the tracked palm.
            let rest_palm = (|| {
                Some((
                    self.rest_direction(&name("Hand"), &name("MiddleProximal"))?,
                    self.rest_direction(&name("IndexProximal"), &name("LittleProximal"))?,
                ))
            })();
            let hand = match (rest_palm, palm_basis(arm.palm_forward, arm.palm_side)) {
                (Some((forward, side_axis)), Some(tracked)) if arm.hand_weight > 0.0 => match palm_basis(forward, side_axis) {
                    Some(rest) => lower.slerp(Quat::from_mat3(&(tracked * rest.transpose())), arm.hand_weight),
                    None => lower,
                },
                _ => lower,
            };
            self.rotate_bone(&name("Hand"), lower.inverse() * hand);
            values.push((format!("Hand: {side}"), Vec2::new(arm.hand_weight, 0.0)));

            for (segments, targets) in fingers.iter().zip(&arm.fingers) {
                let mut parent = hand;
                for (index, (segment, target)) in segments.iter().zip(targets).enumerate() {
                    let bone = name(segment);
                    // Each segment points at the next; the last continues
                    // the line from the one before it.
                    let rest = match segments.get(index + 1) {
                        Some(next) => self.rest_direction(&bone, &name(next)),
                        None => self.rest_direction(&name(segments[index - 1]), &bone),
                    };
                    let total = Self::aimed(rest, *target, parent, arm.hand_weight);
                    self.rotate_bone(&bone, parent.inverse() * total);
                    parent = total;
                }
            }
        }
    }

    fn pose(&mut self, face: &FaceState, body: &BodyState, values: &mut Vec<(String, Vec2)>) {
        for node in &mut self.nodes {
            node.translation = node.rest_translation;
            node.rotation = node.rest_rotation;
        }
        // Facing the viewer: yaw toward screen right is +Y, looking down is
        // +X, tilting the head toward screen right is -Z.
        let (yaw, pitch, roll) = (
            face.head_yaw * MAX_YAW,
            face.head_pitch * MAX_PITCH,
            -face.head_roll * MAX_ROLL,
        );
        for (bone, share) in BONE_SHARES {
            if self.bones.contains_key(bone) {
                // Breathing tips the chest; the neck cancels it so the head stays level.
                let breath = match bone {
                    "chest" => -face.breath * BREATH_ANGLE,
                    "neck" => face.breath * BREATH_ANGLE,
                    _ => 0.0,
                };
                // Leaning in tilts the spine forward; the neck undoes most of
                // it so the face stays toward the viewer.
                let lean = match bone {
                    "spine" => face.lean * LEAN_ANGLE,
                    "neck" => -face.lean * LEAN_ANGLE * 0.8,
                    _ => 0.0,
                };
                let bone_pitch = pitch * share + breath + lean;
                self.rotate_bone(bone, Quat::from_euler(EulerRot::YXZ, yaw * share, bone_pitch, roll * share));
                values.push((format!("Bone: {bone}"), Vec2::new(yaw * share, pitch * share).map(f32::to_degrees)));
            }
        }
        // Eyes follow the tracked gaze, which is already relative to the head.
        let eye_yaw = face.gaze_x.clamp(-1.0, 1.0) * EYE_MAX_YAW;
        let eye_pitch = face.gaze_y.clamp(-1.0, 1.0) * EYE_MAX_PITCH;
        let eye_rotation = Quat::from_euler(EulerRot::YXZ, eye_yaw, eye_pitch, 0.0);
        self.rotate_bone("leftEye", eye_rotation);
        self.rotate_bone("rightEye", eye_rotation);
        self.pose_arms(body, values);
        if let Some(&hips) = self.bones.get("hips") {
            let parent_rotation = match self.nodes[hips].parent {
                Some(parent) => self.nodes[parent].rest_world_rotation,
                None => self.root_transform.to_scale_rotation_translation().1,
            };
            let travel = Vec3::new(face.body_x, face.body_y * 0.5, face.lean * 0.5) * BODY_TRAVEL;
            self.nodes[hips].translation += parent_rotation.inverse() * travel;
        }
        self.update_world();
        self.apply_constraints();
        let now = Instant::now();
        let dt = self.last_frame.map_or(0.0, |last| now.duration_since(last).as_secs_f32().min(0.05));
        self.last_frame = Some(now);
        self.simulate_springs(dt);

        for geometry in &mut self.geometries {
            geometry.morph_weights.fill(0.0);
        }
        let wide = face.mouth_width.clamp(0.0, 1.0) * (1.0 - face.mouth_open) * 0.6;
        let narrow = (-face.mouth_width).clamp(0.0, 1.0) * (1.0 - face.mouth_open) * 0.6;
        let happy = face.smile * SMILE_WEIGHT;
        // `happy` already narrows the eyes; a full blink on top over-closes them.
        let blink_scale = 1.0 - happy * 0.6;
        for expression in &self.expressions {
            let value = match expression.name.as_str() {
                "blinkLeft" => face.blink_left.max(face.eye_squint * SQUINT_WEIGHT) * blink_scale,
                "blinkRight" => face.blink_right.max(face.eye_squint * SQUINT_WEIGHT) * blink_scale,
                "angry" => face.brow_lower * BROW_LOWER_WEIGHT,
                "sad" => face.frown * FROWN_WEIGHT,
                "aa" => face.mouth_open,
                "ih" => wide,
                "ou" => narrow,
                "happy" => happy,
                "surprised" => face.brow_raise * BROW_WEIGHT,
                _ => continue,
            }
            .clamp(0.0, 1.0);
            values.push((format!("Expression: {}", expression.name), Vec2::new(value, 0.0)));
            for &(geometry, morph, weight) in &expression.binds {
                self.geometries[geometry].morph_weights[morph] += weight * value;
            }
        }
    }

    /// Apply morph targets and skinning on the CPU.
    fn skin(&mut self) {
        let joint_matrices: Vec<Vec<Mat4>> = self
            .skins
            .iter()
            .map(|skin| {
                skin.joints
                    .iter()
                    .zip(&skin.inverse_bind)
                    .map(|(&joint, inverse_bind)| self.nodes[joint].world * *inverse_bind)
                    .collect()
            })
            .collect();
        for geometry in &mut self.geometries {
            geometry.morphed.copy_from_slice(&geometry.positions);
            for (target, &weight) in geometry.morphs.iter().zip(&geometry.morph_weights) {
                if weight != 0.0 {
                    for &(vertex, delta) in target {
                        geometry.morphed[vertex as usize] += delta * weight;
                    }
                }
            }
            let node_world = self.nodes[geometry.node].world;
            let joints = geometry.skin.and_then(|skin| joint_matrices.get(skin));
            for (v, out) in geometry.vertices.chunks_exact_mut(6).enumerate() {
                let matrix = match joints {
                    Some(joints) => {
                        let (indices, weights) = (geometry.joints[v], geometry.weights[v]);
                        let mut blended = Mat4::ZERO;
                        let mut total = 0.0;
                        for (index, weight) in indices.into_iter().zip(weights) {
                            if weight > 0.0
                                && let Some(joint) = joints.get(index as usize)
                            {
                                blended += *joint * weight;
                                total += weight;
                            }
                        }
                        if total > 1e-6 { blended * (1.0 / total) } else { node_world }
                    }
                    None => node_world,
                };
                let position = matrix.transform_point3(geometry.morphed[v]);
                let normal = matrix.transform_vector3(geometry.normals[v]);
                out[..3].copy_from_slice(&position.to_array());
                out[3..].copy_from_slice(&normal.to_array());
            }
        }
    }

    /// Pose from `face` and draw into `target`. `center` is the world point at
    /// the middle of the view and `view_height` the world height it shows.
    /// `clear` is the (premultiplied) color behind the avatar.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        gl: &glow::Context,
        target: glow::Framebuffer,
        size: [u32; 2],
        center: Vec2,
        view_height: f32,
        clear: [f32; 4],
        background: Option<&std::sync::Arc<image::RgbaImage>>,
        face: &FaceState,
        body: &BodyState,
        values: &mut Vec<(String, Vec2)>,
    ) -> Result<(), String> {
        self.pose(face, body, values);
        self.skin();
        if self.gl.is_none() {
            self.gl = Some(self.create_gl(gl, size)?);
        }
        let state = self.gl.as_mut().expect("GL state just created");
        let [w, h] = [size[0] as i32, size[1] as i32];

        let distance = view_height / 2.0 / (FOV_Y / 2.0).tan();
        let projection = Mat4::perspective_rh_gl(FOV_Y, w as f32 / h as f32, distance * 0.1, distance + 20.0);
        let view = Mat4::look_at_rh(center.extend(distance), center.extend(0.0), Vec3::Y);
        let view_proj = projection * view;

        unsafe {
            state.msaa.resize(gl, size);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(state.msaa.framebuffer));
            gl.viewport(0, 0, w, h);
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::STENCIL_TEST);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.depth_mask(true);
            gl.color_mask(true, true, true, true);
            gl.clear_color(clear[0], clear[1], clear[2], clear[3]);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
            if let Some(image) = background {
                state.draw_background(gl, image, size)?;
            }
            gl.front_face(glow::CCW);
            gl.blend_equation(glow::FUNC_ADD);
            // Keeps the framebuffer's alpha premultiplied for egui.
            gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA, glow::ONE, glow::ONE_MINUS_SRC_ALPHA);

            gl.use_program(Some(state.program));
            gl.uniform_matrix_4_f32_slice(state.u_view_proj.as_ref(), false, &view_proj.to_cols_array());
            gl.uniform_3_f32_slice(state.u_light.as_ref(), &LIGHT_DIR.normalize().to_array());
            gl.active_texture(glow::TEXTURE0);

            for (geometry, gl_geometry) in self.geometries.iter().zip(&state.geometries) {
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(gl_geometry.vertex_buffer));
                gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, as_bytes(&geometry.vertices));
            }
            let default_material = Material::default();
            // Solid surfaces, then their outlines, then transparent surfaces.
            // An outline is the same mesh pushed out along its normals and
            // drawn inside-out, so only a rim shows around the solid pass.
            let passes = [(false, false), (true, false), (false, true)];
            for (outline, transparent) in passes {
            for &(g, p) in &self.draw_order {
                let primitive = &self.geometries[g].primitives[p];
                let material = primitive
                    .material
                    .and_then(|m| self.materials.get(m))
                    .unwrap_or(&default_material);
                if (material.alpha == AlphaMode::Blend) != transparent {
                    continue;
                }
                if outline && material.outline_width <= 0.0 {
                    continue;
                }
                let outline_width = if outline { material.outline_width.min(OUTLINE_MAX) } else { 0.0 };
                gl.uniform_1_f32(state.u_outline_width.as_ref(), outline_width);
                gl.uniform_3_f32_slice(state.u_outline_color.as_ref(), &material.outline_color);
                let texture = material.image.and_then(|image| state.textures.get(&image)).unwrap_or(&state.white);
                gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
                gl.uniform_4_f32_slice(state.u_color.as_ref(), &material.color);
                gl.uniform_3_f32_slice(state.u_shade.as_ref(), &material.shade);
                gl.uniform_2_f32_slice(state.u_shade_range.as_ref(), &material.shade_range);
                gl.uniform_1_f32(state.u_cutoff.as_ref(), material.cutoff);
                gl.uniform_1_i32(state.u_alpha_mode.as_ref(), material.alpha as i32);
                if material.alpha == AlphaMode::Blend {
                    gl.enable(glow::BLEND);
                } else {
                    gl.disable(glow::BLEND);
                }
                gl.depth_mask(material.depth_write);
                if outline {
                    gl.enable(glow::CULL_FACE);
                    gl.cull_face(glow::FRONT);
                } else if material.double_sided {
                    gl.disable(glow::CULL_FACE);
                } else {
                    gl.enable(glow::CULL_FACE);
                    gl.cull_face(glow::BACK);
                }
                gl.bind_vertex_array(Some(state.geometries[g].vao));
                gl.draw_elements(
                    glow::TRIANGLES,
                    primitive.index_count as i32,
                    glow::UNSIGNED_INT,
                    (primitive.index_offset * 4) as i32,
                );
            }
            }
            gl.cull_face(glow::BACK);

            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(state.msaa.framebuffer));
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(target));
            gl.blit_framebuffer(0, 0, w, h, 0, 0, w, h, glow::COLOR_BUFFER_BIT, glow::NEAREST);

            // Leave GL the way egui expects it.
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.depth_mask(true);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.enable(glow::BLEND);
        }
        Ok(())
    }

    fn create_gl(&mut self, gl: &glow::Context, size: [u32; 2]) -> Result<GlState, String> {
        unsafe {
            let program = compile_program(gl, VERTEX_SHADER, FRAGMENT_SHADER)?;
            gl.use_program(Some(program));
            gl.uniform_1_i32(gl.get_uniform_location(program, "u_texture").as_ref(), 0);

            let white = upload_texture(gl, [1, 1], &[255; 4])?;
            let mut textures = HashMap::new();
            // Decoded pixels are only needed until they are on the GPU.
            for (index, image) in self.images.drain() {
                textures.insert(index, upload_texture(gl, image.size, &image.rgba)?);
            }

            let mut geometries = Vec::with_capacity(self.geometries.len());
            for geometry in &self.geometries {
                let vao = gl.create_vertex_array()?;
                gl.bind_vertex_array(Some(vao));
                let vertex_buffer = gl.create_buffer()?;
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertex_buffer));
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, as_bytes(&geometry.vertices), glow::DYNAMIC_DRAW);
                gl.enable_vertex_attrib_array(0);
                gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 24, 0);
                gl.enable_vertex_attrib_array(1);
                gl.vertex_attrib_pointer_f32(1, 3, glow::FLOAT, false, 24, 12);
                let uv_buffer = gl.create_buffer()?;
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(uv_buffer));
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, as_bytes(geometry.uvs.as_flattened()), glow::STATIC_DRAW);
                gl.enable_vertex_attrib_array(2);
                gl.vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, 8, 0);
                let index_buffer = gl.create_buffer()?;
                gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(index_buffer));
                gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, as_bytes(&geometry.indices), glow::STATIC_DRAW);
                gl.bind_vertex_array(None);
                geometries.push(GlGeometry { vao, vertex_buffer });
            }

            let background_program = compile_program(gl, BACKGROUND_VERTEX_SHADER, BACKGROUND_FRAGMENT_SHADER)?;
            gl.use_program(Some(background_program));
            gl.uniform_1_i32(gl.get_uniform_location(background_program, "u_texture").as_ref(), 0);

            let mut msaa = Msaa {
                framebuffer: gl.create_framebuffer()?,
                color: gl.create_renderbuffer()?,
                depth: gl.create_renderbuffer()?,
                size: [0, 0],
            };
            msaa.resize(gl, size);

            Ok(GlState {
                program,
                u_view_proj: gl.get_uniform_location(program, "u_view_proj"),
                u_color: gl.get_uniform_location(program, "u_color"),
                u_shade: gl.get_uniform_location(program, "u_shade"),
                u_shade_range: gl.get_uniform_location(program, "u_shade_range"),
                u_light: gl.get_uniform_location(program, "u_light"),
                u_cutoff: gl.get_uniform_location(program, "u_cutoff"),
                u_alpha_mode: gl.get_uniform_location(program, "u_alpha_mode"),
                u_outline_width: gl.get_uniform_location(program, "u_outline_width"),
                u_outline_color: gl.get_uniform_location(program, "u_outline_color"),
                textures,
                white,
                geometries,
                msaa,
                u_background_crop: gl.get_uniform_location(background_program, "u_crop"),
                background_program,
                background_vao: gl.create_vertex_array()?,
                background: None,
            })
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            image: None,
            color: [1.0; 4],
            shade: [0.8; 3],
            outline_width: 0.0,
            outline_color: [0.0; 3],
            shade_range: [-0.1, 0.1],
            alpha: AlphaMode::Opaque,
            cutoff: 0.5,
            double_sided: true,
            depth_write: true,
            queue: 2000,
        }
    }
}

impl Msaa {
    unsafe fn resize(&mut self, gl: &glow::Context, size: [u32; 2]) {
        if size == self.size {
            return;
        }
        self.size = size;
        let [w, h] = [size[0] as i32, size[1] as i32];
        unsafe {
            let samples = MSAA_SAMPLES.min(gl.get_parameter_i32(glow::MAX_SAMPLES)).max(1);
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(self.color));
            gl.renderbuffer_storage_multisample(glow::RENDERBUFFER, samples, glow::RGBA8, w, h);
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(self.depth));
            gl.renderbuffer_storage_multisample(glow::RENDERBUFFER, samples, glow::DEPTH_COMPONENT24, w, h);
            gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.framebuffer));
            gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::RENDERBUFFER, Some(self.color));
            gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::DEPTH_ATTACHMENT, glow::RENDERBUFFER, Some(self.depth));
        }
    }
}

/// Orthonormal basis (forward, side, normal) as matrix columns, from a palm's
/// forward direction and a direction across it.
fn palm_basis(forward: Vec3, across: Vec3) -> Option<glam::Mat3> {
    let forward = forward.try_normalize()?;
    let side = (across - forward * across.dot(forward)).try_normalize()?;
    Some(glam::Mat3::from_cols(forward, side, forward.cross(side)))
}

/// The JSON chunk of a binary glTF: 12-byte header, then length + "JSON" + data.
fn glb_json(data: &[u8]) -> Result<Value, String> {
    if data.len() < 20 || &data[..4] != b"glTF" || &data[16..20] != b"JSON" {
        return Err("not a binary glTF / VRM file".into());
    }
    let length = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let json = data.get(20..20 + length).ok_or("truncated VRM file")?;
    serde_json::from_slice(json).map_err(|e| format!("VRM JSON: {e}"))
}

fn floats<const N: usize>(value: &Value) -> Option<[f32; N]> {
    let array = value.as_array()?;
    let mut out = [0.0; N];
    for (o, v) in out.iter_mut().zip(array) {
        *o = v.as_f64()? as f32;
    }
    (array.len() >= N).then_some(out)
}

fn linear_to_srgb(value: f32) -> f32 {
    value.max(0.0).powf(1.0 / 2.2)
}

/// Core glTF material plus the MToon shading settings from the VRM extension.
/// Colors are kept in sRGB; the shader works in display space.
fn read_material(material: &gltf::Material, index: usize, json: &Value, is_vrm0: bool) -> Material {
    let pbr = material.pbr_metallic_roughness();
    let factor = pbr.base_color_factor();
    let alpha = match material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => AlphaMode::Mask,
        gltf::material::AlphaMode::Blend => AlphaMode::Blend,
    };
    let mut out = Material {
        image: pbr.base_color_texture().map(|info| info.texture().source().index()),
        color: [linear_to_srgb(factor[0]), linear_to_srgb(factor[1]), linear_to_srgb(factor[2]), factor[3]],
        alpha,
        cutoff: material.alpha_cutoff().unwrap_or(0.5),
        double_sided: material.double_sided(),
        depth_write: alpha != AlphaMode::Blend,
        queue: match alpha {
            AlphaMode::Opaque => 2000,
            AlphaMode::Mask => 2450,
            AlphaMode::Blend => 3000,
        },
        ..Material::default()
    };
    if is_vrm0 {
        let properties = &json["extensions"]["VRM"]["materialProperties"][index];
        let (float, vector) = (&properties["floatProperties"], &properties["vectorProperties"]);
        if let Some(color) = floats::<4>(&vector["_Color"]) {
            out.color = color;
        }
        if let Some(shade) = floats::<3>(&vector["_ShadeColor"]) {
            out.shade = shade;
        }
        if let (Some(shift), Some(toony)) = (float["_ShadeShift"].as_f64(), float["_ShadeToony"].as_f64()) {
            let (shift, toony) = (shift as f32, toony as f32);
            out.shade_range = [shift, 1.0 + (shift - 1.0) * toony];
        }
        if let Some(queue) = properties["renderQueue"].as_i64() {
            out.queue = queue as i32;
        }
        if alpha == AlphaMode::Blend {
            out.depth_write = float["_ZWrite"].as_f64().is_some_and(|z| z > 0.5);
        }
        // Width mode 1 is world space, in centimeters; 2 is screen space,
        // approximated with a fixed thin line.
        let width = float["_OutlineWidth"].as_f64().unwrap_or(0.0) as f32;
        out.outline_width = match float["_OutlineWidthMode"].as_f64().unwrap_or(0.0) as i32 {
            1 => width * 0.01,
            2 => 0.001,
            _ => 0.0,
        };
        if let Some(color) = floats::<3>(&vector["_OutlineColor"]) {
            out.outline_color = color;
        }
    } else {
        let mtoon = &json["materials"][index]["extensions"]["VRMC_materials_mtoon"];
        if let Some(shade) = floats::<3>(&mtoon["shadeColorFactor"]) {
            out.shade = shade.map(linear_to_srgb);
        }
        if !mtoon.is_null() {
            let shift = mtoon["shadingShiftFactor"].as_f64().unwrap_or(0.0) as f32;
            let toony = mtoon["shadingToonyFactor"].as_f64().unwrap_or(0.9) as f32;
            out.shade_range = [toony - 1.0 - shift, 1.0 - toony - shift];
        }
        out.queue += mtoon["renderQueueOffsetNumber"].as_i64().unwrap_or(0) as i32;
        if alpha == AlphaMode::Blend {
            out.depth_write = mtoon["transparentWithZWrite"].as_bool().unwrap_or(false);
        }
        out.outline_width = match mtoon["outlineWidthMode"].as_str() {
            Some("worldCoordinates") => mtoon["outlineWidthFactor"].as_f64().unwrap_or(0.0) as f32,
            Some("screenCoordinates") => 0.001,
            _ => 0.0,
        };
        if let Some(color) = floats::<3>(&mtoon["outlineColorFactor"]) {
            out.outline_color = color.map(linear_to_srgb);
        }
    }
    out
}

fn vec3(value: &Value) -> Option<Vec3> {
    match value {
        Value::Array(_) => floats::<3>(value).map(Vec3::from),
        Value::Object(_) => Some(Vec3::new(
            value["x"].as_f64()? as f32,
            value["y"].as_f64()? as f32,
            value["z"].as_f64()? as f32,
        )),
        _ => None,
    }
}

/// VRM 1.0 `VRMC_springBone`: each spring lists its joints root to tip, and
/// each joint's tail is the next joint.
fn read_springs_vrm1(spring_bone: &Value, nodes: &[Node]) -> (Vec<SpringJoint>, Vec<Collider>) {
    let mut colliders = Vec::new();
    // Indices into `colliders` per collider index in the file.
    let mut collider_map = Vec::new();
    for collider in spring_bone["colliders"].as_array().into_iter().flatten() {
        let (sphere, capsule) = (&collider["shape"]["sphere"], &collider["shape"]["capsule"]);
        let shape = if sphere.is_null() { capsule } else { sphere };
        let parsed = collider["node"].as_u64().filter(|&n| (n as usize) < nodes.len()).map(|node| Collider {
            node: node as usize,
            offset: vec3(&shape["offset"]).unwrap_or(Vec3::ZERO),
            tail: vec3(&shape["tail"]),
            radius: shape["radius"].as_f64().unwrap_or(0.0) as f32,
        });
        collider_map.push(parsed.map(|collider| {
            colliders.push(collider);
            colliders.len() - 1
        }));
    }
    let groups: Vec<Vec<usize>> = spring_bone["colliderGroups"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|group| {
            group["colliders"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|index| collider_map.get(index.as_u64()? as usize).copied().flatten())
                .collect()
        })
        .collect();

    let mut joints = Vec::new();
    for spring in spring_bone["springs"].as_array().into_iter().flatten() {
        let spring_colliders: Vec<usize> = spring["colliderGroups"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|group| groups.get(group.as_u64()? as usize))
            .flatten()
            .copied()
            .collect();
        let chain: Vec<&Value> = spring["joints"].as_array().into_iter().flatten().collect();
        for pair in chain.windows(2) {
            let (Some(node), Some(next)) = (pair[0]["node"].as_u64(), pair[1]["node"].as_u64()) else {
                continue;
            };
            let (node, next) = (node as usize, next as usize);
            if node >= nodes.len() || next >= nodes.len() || nodes[next].parent != Some(node) {
                continue;
            }
            let joint = pair[0];
            let gravity_dir = vec3(&joint["gravityDir"]).unwrap_or(Vec3::NEG_Y);
            joints.push(SpringJoint {
                node,
                tail: nodes[next].rest_translation,
                stiffness: joint["stiffness"].as_f64().unwrap_or(1.0) as f32,
                gravity: gravity_dir * joint["gravityPower"].as_f64().unwrap_or(0.0) as f32,
                drag: joint["dragForce"].as_f64().unwrap_or(0.5) as f32,
                hit_radius: joint["hitRadius"].as_f64().unwrap_or(0.0) as f32,
                colliders: spring_colliders.clone(),
                state: None,
            });
        }
    }
    (joints, colliders)
}

/// VRM 0.x `secondaryAnimation`: each group names root bones, and every bone
/// below them is simulated with the group's settings. Vectors are in Unity's
/// left-handed space, so z is flipped.
fn read_springs_vrm0(secondary: &Value, document: &gltf::Document, nodes: &[Node]) -> (Vec<SpringJoint>, Vec<Collider>) {
    let flip_z = |v: Vec3| Vec3::new(v.x, v.y, -v.z);
    let mut colliders = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for group in secondary["colliderGroups"].as_array().into_iter().flatten() {
        let mut indices = Vec::new();
        if let Some(node) = group["node"].as_u64().filter(|&n| (n as usize) < nodes.len()) {
            for collider in group["colliders"].as_array().into_iter().flatten() {
                colliders.push(Collider {
                    node: node as usize,
                    offset: flip_z(vec3(&collider["offset"]).unwrap_or(Vec3::ZERO)),
                    tail: None,
                    radius: collider["radius"].as_f64().unwrap_or(0.0) as f32,
                });
                indices.push(colliders.len() - 1);
            }
        }
        groups.push(indices);
    }

    let children: Vec<Vec<usize>> = document.nodes().map(|n| n.children().map(|c| c.index()).collect()).collect();
    let mut joints = Vec::new();
    for group in secondary["boneGroups"].as_array().into_iter().flatten() {
        // "stiffiness" is the spec's own spelling.
        let stiffness = group["stiffiness"].as_f64().unwrap_or(1.0) as f32;
        let gravity = flip_z(vec3(&group["gravityDir"]).unwrap_or(Vec3::NEG_Y))
            * group["gravityPower"].as_f64().unwrap_or(0.0) as f32;
        let drag = group["dragForce"].as_f64().unwrap_or(0.5) as f32;
        let hit_radius = group["hitRadius"].as_f64().unwrap_or(0.0) as f32;
        let group_colliders: Vec<usize> = group["colliderGroups"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|index| groups.get(index.as_u64()? as usize))
            .flatten()
            .copied()
            .collect();
        let mut stack: Vec<usize> = group["bones"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|bone| bone.as_u64().map(|b| b as usize))
            .filter(|&bone| bone < nodes.len())
            .collect();
        while let Some(node) = stack.pop() {
            // The tail is the first child, or for a leaf a 7 cm extension of
            // the bone (as UniVRM does).
            let tail = match children[node].first() {
                Some(&child) => nodes[child].rest_translation,
                None => nodes[node].rest_translation.normalize_or_zero() * 0.07,
            };
            joints.push(SpringJoint {
                node,
                tail,
                stiffness,
                gravity,
                drag,
                hit_radius,
                colliders: group_colliders.clone(),
                state: None,
            });
            stack.extend(&children[node]);
        }
    }
    (joints, colliders)
}

fn read_constraints(json: &Value, node_count: usize) -> Vec<Constraint> {
    let axis = |name: &str| match name {
        "X" | "PositiveX" => Some(Vec3::X),
        "Y" | "PositiveY" => Some(Vec3::Y),
        "Z" | "PositiveZ" => Some(Vec3::Z),
        "NegativeX" => Some(Vec3::NEG_X),
        "NegativeY" => Some(Vec3::NEG_Y),
        "NegativeZ" => Some(Vec3::NEG_Z),
        _ => None,
    };
    let mut constraints = Vec::new();
    for (node, value) in json["nodes"].as_array().into_iter().flatten().enumerate() {
        let constraint = &value["extensions"]["VRMC_node_constraint"]["constraint"];
        let (roll, aim, rotation) = (&constraint["roll"], &constraint["aim"], &constraint["rotation"]);
        let (spec, kind) = if !roll.is_null() {
            (roll, roll["rollAxis"].as_str().and_then(axis).map(ConstraintKind::Roll))
        } else if !aim.is_null() {
            (aim, aim["aimAxis"].as_str().and_then(axis).map(ConstraintKind::Aim))
        } else if !rotation.is_null() {
            (rotation, Some(ConstraintKind::Rotation))
        } else {
            continue;
        };
        let (Some(kind), Some(source)) = (kind, spec["source"].as_u64()) else {
            continue;
        };
        if node < node_count && (source as usize) < node_count {
            constraints.push(Constraint {
                node,
                source: source as usize,
                weight: spec["weight"].as_f64().unwrap_or(1.0) as f32,
                kind,
            });
        }
    }
    constraints
}

/// Decode the base-color images the materials use, in parallel.
fn decode_images(document: &gltf::Document, blob: &[u8], materials: &[Material]) -> HashMap<usize, DecodedImage> {
    let mut wanted: Vec<usize> = materials.iter().filter_map(|m| m.image).collect();
    wanted.sort_unstable();
    wanted.dedup();
    let sources: Vec<(usize, &[u8])> = wanted
        .into_iter()
        .filter_map(|index| {
            let gltf::image::Source::View { view, .. } = document.images().nth(index)?.source() else {
                return None;
            };
            Some((index, blob.get(view.offset()..view.offset() + view.length())?))
        })
        .collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = sources
            .into_iter()
            .map(|(index, bytes)| {
                scope.spawn(move || {
                    let image = image::load_from_memory(bytes).ok()?.to_rgba8();
                    Some((index, DecodedImage { size: [image.width(), image.height()], rgba: image.into_raw() }))
                })
            })
            .collect();
        handles.into_iter().filter_map(|handle| handle.join().ok().flatten()).collect()
    })
}

/// Expressions under VRM 1.0 preset names, with binds resolved to geometries.
fn read_expressions(vrm0: &Value, vrm1: &Value, is_vrm0: bool, geometries: &[Geometry]) -> Vec<Expression> {
    let resolve = |binds: Option<&Vec<Value>>, key: &str, scale: f32| -> Vec<(usize, usize, f32)> {
        let mut out = Vec::new();
        for bind in binds.into_iter().flatten() {
            let (Some(owner), Some(morph)) = (bind[key].as_u64(), bind["index"].as_u64()) else {
                continue;
            };
            let weight = bind["weight"].as_f64().unwrap_or(0.0) as f32 * scale;
            for (index, geometry) in geometries.iter().enumerate() {
                let geometry_owner = if key == "mesh" { geometry.mesh } else { geometry.node };
                if geometry_owner == owner as usize && (morph as usize) < geometry.morphs.len() {
                    out.push((index, morph as usize, weight));
                }
            }
        }
        out
    };
    let mut expressions = Vec::new();
    if is_vrm0 {
        for group in vrm0["blendShapeMaster"]["blendShapeGroups"].as_array().into_iter().flatten() {
            let preset = group["presetName"].as_str().unwrap_or("unknown");
            let name = match preset {
                "a" => "aa",
                "i" => "ih",
                "u" => "ou",
                "e" => "ee",
                "o" => "oh",
                "blink_l" => "blinkLeft",
                "blink_r" => "blinkRight",
                "joy" => "happy",
                "sorrow" => "sad",
                "fun" => "relaxed",
                "unknown" => group["name"].as_str().unwrap_or("unknown"),
                other => other,
            };
            expressions.push(Expression {
                name: name.to_owned(),
                binds: resolve(group["binds"].as_array(), "mesh", 0.01),
            });
        }
    } else {
        for section in ["preset", "custom"] {
            for (name, expression) in vrm1["expressions"][section].as_object().into_iter().flatten() {
                expressions.push(Expression {
                    name: name.clone(),
                    binds: resolve(expression["morphTargetBinds"].as_array(), "node", 1.0),
                });
            }
        }
    }
    expressions
}

fn as_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    // SAFETY: only used with plain f32/u32 data, which has no padding and no
    // invalid bit patterns; the byte slice covers exactly the same memory.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), std::mem::size_of_val(slice)) }
}

unsafe fn upload_texture(gl: &glow::Context, size: [u32; 2], rgba: &[u8]) -> Result<glow::Texture, String> {
    unsafe {
        let texture = gl.create_texture()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            size[0] as i32,
            size[1] as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(Some(rgba)),
        );
        gl.generate_mipmap(glow::TEXTURE_2D);
        for (parameter, value) in [
            (glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR),
            (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
            (glow::TEXTURE_WRAP_S, glow::REPEAT),
            (glow::TEXTURE_WRAP_T, glow::REPEAT),
        ] {
            gl.tex_parameter_i32(glow::TEXTURE_2D, parameter, value as i32);
        }
        Ok(texture)
    }
}

const VERTEX_SHADER: &str = r#"#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
uniform mat4 u_view_proj;
uniform float u_outline_width; // > 0 in the outline pass: push the surface out
out vec3 v_normal;
out vec2 v_uv;
void main() {
    v_normal = a_normal;
    v_uv = a_uv;
    vec3 position = a_position;
    if (u_outline_width > 0.0 && dot(a_normal, a_normal) > 0.0) {
        position += normalize(a_normal) * u_outline_width;
    }
    gl_Position = u_view_proj * vec4(position, 1.0);
}
"#;

/// Two-tone toon shading: the base color, or the base color tinted by the
/// shade color where the surface faces away from the light.
const FRAGMENT_SHADER: &str = r#"#version 330 core
in vec3 v_normal;
in vec2 v_uv;
uniform sampler2D u_texture;
uniform vec4 u_color;
uniform vec3 u_shade;
uniform vec2 u_shade_range;
uniform vec3 u_light;
uniform float u_cutoff;
uniform int u_alpha_mode; // 0 opaque, 1 mask, 2 blend
uniform float u_outline_width;
uniform vec3 u_outline_color;
out vec4 frag_color;
void main() {
    vec4 base = texture(u_texture, v_uv) * u_color;
    float alpha = 1.0;
    if (u_alpha_mode == 1) {
        if (base.a < u_cutoff) discard;
    } else if (u_alpha_mode == 2) {
        alpha = base.a;
        if (alpha < 0.004) discard;
    }
    if (u_outline_width > 0.0) {
        frag_color = vec4(base.rgb * u_outline_color, 1.0);
        return;
    }
    vec3 normal = normalize(v_normal);
    if (!gl_FrontFacing) normal = -normal;
    float lit = clamp((dot(normal, u_light) - u_shade_range.x) / max(u_shade_range.y - u_shade_range.x, 1e-4), 0.0, 1.0);
    frag_color = vec4(base.rgb * mix(u_shade, vec3(1.0), lit), alpha);
}
"#;

/// Fills the view with the background image, cropped to cover it. Drawn as
/// one triangle larger than the screen, with no vertex data.
const BACKGROUND_VERTEX_SHADER: &str = r#"#version 330 core
uniform vec4 u_crop; // part of the image shown: origin (xy) and size (zw)
out vec2 v_uv;
void main() {
    vec2 corner = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
    vec2 screen = corner * 0.5 + 0.5;
    // Image rows run top to bottom; OpenGL's y runs bottom to top.
    v_uv = u_crop.xy + vec2(screen.x, 1.0 - screen.y) * u_crop.zw;
    gl_Position = vec4(corner, 0.0, 1.0);
}
"#;

const BACKGROUND_FRAGMENT_SHADER: &str = r#"#version 330 core
in vec2 v_uv;
uniform sampler2D u_texture;
out vec4 frag_color;
void main() {
    frag_color = vec4(texture(u_texture, v_uv).rgb, 1.0);
}
"#;

impl GlState {
    /// Draw `image` over the whole framebuffer, cropped to cover `size`.
    /// Uploads it first if it's a different image than last time.
    unsafe fn draw_background(
        &mut self,
        gl: &glow::Context,
        image: &std::sync::Arc<image::RgbaImage>,
        size: [u32; 2],
    ) -> Result<(), String> {
        let key = std::sync::Arc::as_ptr(image) as usize;
        unsafe {
            let dimensions = [image.width(), image.height()];
            match self.background {
                Some((uploaded, _, _)) if uploaded == key => {}
                // A new frame of the same size (a video): reuse the texture.
                Some((_, texture, uploaded_size)) if uploaded_size == dimensions => {
                    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                    gl.tex_sub_image_2d(
                        glow::TEXTURE_2D,
                        0,
                        0,
                        0,
                        dimensions[0] as i32,
                        dimensions[1] as i32,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelUnpackData::Slice(Some(image.as_raw())),
                    );
                    gl.generate_mipmap(glow::TEXTURE_2D);
                    self.background = Some((key, texture, dimensions));
                }
                _ => {
                    if let Some((_, old, _)) = self.background.take() {
                        gl.delete_texture(old);
                    }
                    self.background = Some((key, upload_texture(gl, dimensions, image.as_raw())?, dimensions));
                }
            }
            let Some((_, texture, [image_w, image_h])) = self.background else {
                return Ok(());
            };
            // Cover: show the middle of the image at the view's aspect ratio.
            let view_aspect = size[0] as f32 / size[1].max(1) as f32;
            let image_aspect = image_w as f32 / image_h.max(1) as f32;
            let crop = if image_aspect > view_aspect {
                let width = view_aspect / image_aspect;
                [(1.0 - width) / 2.0, 0.0, width, 1.0]
            } else {
                let height = image_aspect / view_aspect;
                [0.0, (1.0 - height) / 2.0, 1.0, height]
            };
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);
            gl.disable(glow::CULL_FACE);
            gl.use_program(Some(self.background_program));
            gl.uniform_4_f32_slice(self.u_background_crop.as_ref(), &crop);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.bind_vertex_array(Some(self.background_vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);
            gl.enable(glow::DEPTH_TEST);
        }
        Ok(())
    }
}

unsafe fn compile_program(gl: &glow::Context, vertex: &str, fragment: &str) -> Result<glow::Program, String> {
    unsafe {
        let program = gl.create_program()?;
        for (kind, source) in [(glow::VERTEX_SHADER, vertex), (glow::FRAGMENT_SHADER, fragment)] {
            let shader = gl.create_shader(kind)?;
            gl.shader_source(shader, source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                return Err(format!("VRM shader: {}", gl.get_shader_info_log(shader)));
            }
            gl.attach_shader(program, shader);
            gl.delete_shader(shader);
        }
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            return Err(format!("VRM shader link: {}", gl.get_program_info_log(program)));
        }
        Ok(program)
    }
}
