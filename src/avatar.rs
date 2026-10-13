//! Stage 4: Inochi2D avatar (.inp / .inx) driven by `FaceState`.
//!
//! The puppet is rendered offscreen with `inox2d-opengl` into a framebuffer
//! whose color texture is registered with egui, so the UI can draw any
//! background behind it.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use glam::Vec2;
use glow::HasContext;
use inox2d::formats::inp::parse_inp;
use inox2d::model::Model;
use inox2d::render::InoxRendererExt;
use inox2d_opengl::OpenglRenderer;

use crate::face::FaceState;

/// Body follows the head by this fraction, like Inochi Session's defaults.
const BODY_FOLLOW: f32 = 0.4;
/// Framing presets (zoom, offset in puppet units) for an Aka-proportioned
/// puppet. Bust (head and shoulders) is the default.
pub const BUST_VIEW: (f32, Vec2) = (2.2, Vec2::new(0.0, 1230.0));
pub const FACE_VIEW: (f32, Vec2) = (4.5, Vec2::new(0.0, 1620.0));
pub const FULL_BODY_VIEW: (f32, Vec2) = (1.0, Vec2::ZERO);
/// Puppet units that fit the view height at zoom 1.
const UNITS_PER_VIEW_HEIGHT: f32 = 5000.0;

#[derive(Clone, Debug)]
pub struct ParamInfo {
    pub name: String,
    pub min: Vec2,
    pub max: Vec2,
}

pub struct Avatar {
    pub name: String,
    model: Model,
    pub params: Vec<ParamInfo>,
    /// Last value written to each bound param, for display.
    pub values: Vec<(String, Vec2)>,
    renderer: Option<OpenglRenderer>,
    last_frame: Option<Instant>,
    pub zoom: f32,
    /// Camera offset in puppet units.
    pub offset: Vec2,
}

impl Avatar {
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let params = read_params(&data)?;
        let mut model = parse_inp(data.as_slice()).map_err(|e| format!("parse puppet: {e}"))?;
        model.puppet.init_transforms();
        model.puppet.init_rendering();
        model.puppet.init_params();
        model.puppet.init_physics();
        let name = model
            .puppet
            .meta
            .name
            .clone()
            .or_else(|| path.file_stem().map(|s| s.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "avatar".into());
        Ok(Self {
            name,
            model,
            params,
            values: Vec::new(),
            renderer: None,
            last_frame: None,
            zoom: BUST_VIEW.0,
            offset: BUST_VIEW.1,
        })
    }

    pub fn set_view(&mut self, (zoom, offset): (f32, Vec2)) {
        self.zoom = zoom;
        self.offset = offset;
    }

    /// Aka-sized puppets are ~5000 units tall; zoom 1 fits that to the view.
    fn pixels_per_unit(&self, view_height_px: u32) -> f32 {
        self.zoom * view_height_px as f32 / UNITS_PER_VIEW_HEIGHT
    }

    /// Move the view by `delta_px` screen pixels.
    pub fn pan(&mut self, delta_px: Vec2, view_height_px: u32) {
        self.offset += delta_px / self.pixels_per_unit(view_height_px);
    }

    /// Zoom by `factor`, keeping the puppet point under `anchor_px` (pixels
    /// from the view center) in place.
    pub fn zoom_at(&mut self, factor: f32, anchor_px: Vec2, view_height_px: u32) {
        let before = self.pixels_per_unit(view_height_px);
        self.zoom = (self.zoom * factor).clamp(0.1, 20.0);
        let after = self.pixels_per_unit(view_height_px);
        self.offset += anchor_px / after - anchor_px / before;
    }

    /// Pose the puppet from `face` and draw it into `target`.
    pub fn render(
        &mut self,
        gl: &Arc<glow::Context>,
        target: &RenderTarget,
        face: &FaceState,
    ) -> Result<(), String> {
        if self.renderer.is_none() {
            let renderer = OpenglRenderer::new(gl.clone(), &self.model).map_err(|e| e.to_string())?;
            self.renderer = Some(renderer);
        }
        let pixels_per_unit = self.pixels_per_unit(target.size[1]);
        let renderer = self.renderer.as_mut().expect("renderer just created");
        let [w, h] = target.size;
        renderer.target_framebuffer = Some(target.framebuffer);
        if renderer.viewport != glam::uvec2(w, h) {
            renderer.resize(w, h);
        }
        renderer.camera.scale = Vec2::splat(pixels_per_unit);
        renderer.camera.position = self.offset;

        let now = Instant::now();
        let dt = self
            .last_frame
            .map_or(0.0, |t| now.duration_since(t).as_secs_f32().min(0.1));
        self.last_frame = Some(now);

        let puppet = &mut self.model.puppet;
        puppet.begin_frame();
        self.values.clear();
        if let Some(ctx) = puppet.param_ctx.as_mut() {
            for param in &self.params {
                if let Some(value) = bind(&param.name, face) {
                    let value = value.clamp(param.min, param.max);
                    let _ = ctx.set(&param.name, value);
                    self.values.push((param.name.clone(), value));
                }
            }
        }
        puppet.end_frame(dt);

        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(target.framebuffer));
            gl.viewport(0, 0, w as i32, h as i32);
            gl.disable(glow::SCISSOR_TEST);
            gl.color_mask(true, true, true, true);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::STENCIL_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
        }
        renderer.clear();
        renderer.on_begin_draw(puppet);
        renderer.draw(puppet);
        renderer.on_end_draw(puppet);
        // Leave GL the way egui expects it.
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.blend_equation(glow::FUNC_ADD);
            gl.disable(glow::STENCIL_TEST);
            gl.active_texture(glow::TEXTURE0);
            gl.use_program(None);
        }
        Ok(())
    }
}

/// Map a puppet parameter (by Inochi2D's conventional names) to a face value.
/// Returns `None` for params the face doesn't drive (physics, gaze, ...).
fn bind(name: &str, face: &FaceState) -> Option<Vec2> {
    let v = |x: f32, y: f32| Some(Vec2::new(x, y));
    match name {
        "Head:: Yaw-Pitch" => v(face.head_yaw, -face.head_pitch),
        "Head:: Roll" => v(face.head_roll, 0.0),
        "Body:: Yaw-Pitch" => v(face.head_yaw * BODY_FOLLOW, -face.head_pitch * BODY_FOLLOW),
        "Body:: Roll" => v(face.head_roll * BODY_FOLLOW, 0.0),
        "Body:: X:: Move" => v(face.body_x, 0.0),
        "Eye:: Left:: Blink" => v(face.blink_left, 0.0),
        "Eye:: Right:: Blink" => v(face.blink_right, 0.0),
        "Mouth:: Open" => v(face.mouth_open, 0.0),
        "Mouth:: Shape" => v(face.mouth_width * 0.5 + 0.5, face.mouth_open),
        "Mouth:: Width" => v(face.mouth_width, 0.0),
        _ => None,
    }
}

/// Read param names and ranges from the puppet's JSON payload; inox2d keeps
/// its own copy private. Layout: `TRNSRTS\0`, u32 BE length, JSON.
fn read_params(data: &[u8]) -> Result<Vec<ParamInfo>, String> {
    const MAGIC: &[u8] = b"TRNSRTS\0";
    if data.len() < 12 || &data[..8] != MAGIC {
        return Err("not an Inochi2D puppet (.inp/.inx)".into());
    }
    let len = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let json = data.get(12..12 + len).ok_or("truncated puppet file")?;
    let root: serde_json::Value =
        serde_json::from_slice(json).map_err(|e| format!("puppet JSON: {e}"))?;
    let vec2 = |v: &serde_json::Value| {
        Vec2::new(
            v[0].as_f64().unwrap_or(0.0) as f32,
            v[1].as_f64().unwrap_or(0.0) as f32,
        )
    };
    Ok(root["param"]
        .as_array()
        .map(|params| {
            params
                .iter()
                .filter_map(|p| {
                    Some(ParamInfo {
                        name: p["name"].as_str()?.to_owned(),
                        min: vec2(&p["min"]),
                        max: vec2(&p["max"]),
                    })
                })
                .collect()
        })
        .unwrap_or_default())
}

/// Offscreen framebuffer (RGBA color + depth/stencil for masks) whose color
/// texture is shown by egui. Resizing keeps the same texture, so the egui
/// registration stays valid.
pub struct RenderTarget {
    framebuffer: glow::Framebuffer,
    color: glow::Texture,
    depth_stencil: glow::Renderbuffer,
    pub size: [u32; 2],
    pub texture_id: egui::TextureId,
}

impl RenderTarget {
    pub fn new(
        gl: &glow::Context,
        frame: &mut eframe::Frame,
        size: [u32; 2],
    ) -> Result<Self, String> {
        unsafe {
            let framebuffer = gl.create_framebuffer()?;
            let color = gl.create_texture()?;
            let depth_stencil = gl.create_renderbuffer()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(color));
            for (param, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
                (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
                (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
            ] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, param, value as i32);
            }
            gl.bind_texture(glow::TEXTURE_2D, None);
            let mut target = Self {
                framebuffer,
                color,
                depth_stencil,
                size: [0, 0],
                texture_id: frame.register_native_glow_texture(color),
            };
            target.resize(gl, size);
            Ok(target)
        }
    }

    pub fn resize(&mut self, gl: &glow::Context, size: [u32; 2]) {
        let size = [size[0].max(1), size[1].max(1)];
        if size == self.size {
            return;
        }
        self.size = size;
        let [w, h] = [size[0] as i32, size[1] as i32];
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.color));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                w,
                h,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(self.depth_stencil));
            gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH24_STENCIL8, w, h);
            gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.framebuffer));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(self.color),
                0,
            );
            gl.framebuffer_renderbuffer(
                glow::FRAMEBUFFER,
                glow::DEPTH_STENCIL_ATTACHMENT,
                glow::RENDERBUFFER,
                Some(self.depth_stencil),
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }

    /// Read the current frame back as top-down RGBA (premultiplied alpha).
    #[allow(dead_code)]
    pub fn read_rgba(&self, gl: &glow::Context) -> Vec<u8> {
        let [w, h] = [self.size[0] as usize, self.size[1] as usize];
        let mut pixels = vec![0u8; w * h * 4];
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.framebuffer));
            gl.read_pixels(
                0,
                0,
                w as i32,
                h as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        // GL rows are bottom-up.
        pixels
            .chunks_exact(w * 4)
            .rev()
            .flatten()
            .copied()
            .collect()
    }
}
