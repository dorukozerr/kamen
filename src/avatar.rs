//! Stage 4: the avatar, driven by `FaceState`.
//!
//! Avatars are VRM models (.vrm, e.g. from VRoid Studio), loaded and drawn by
//! `crate::vrm`. They render offscreen into a framebuffer whose color texture
//! is registered with egui, so the UI can draw any background behind it.

use std::path::Path;
use std::sync::Arc;

use glam::Vec2;
use glow::HasContext;

use crate::body::BodyState;
use crate::face::FaceState;
use crate::vrm::VrmAvatar;

/// Framing presets.
#[derive(Clone, Copy)]
pub enum View {
    Face,
    Bust,
    FullBody,
}

pub struct Avatar {
    pub name: String,
    /// Everything the model can be driven by, for display.
    pub params: Vec<String>,
    /// Last value written to each driven param, for display.
    pub values: Vec<(String, Vec2)>,
    pub zoom: f32,
    /// How far the model is moved from the view center, in meters (x right,
    /// y down).
    pub offset: Vec2,
    model: VrmAvatar,
}

impl Avatar {
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let model = VrmAvatar::load(&data)?;
        let name = model
            .name
            .clone()
            .or_else(|| path.file_stem().map(|s| s.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "avatar".into());
        let mut avatar = Self {
            name,
            params: model.param_names(),
            values: Vec::new(),
            zoom: 1.0,
            offset: Vec2::ZERO,
            model,
        };
        avatar.set_view(View::Bust);
        Ok(avatar)
    }

    /// Meters that fit the view height at zoom 1 (the full body).
    fn units_per_view_height(&self) -> f32 {
        self.model.height * 1.1
    }

    pub fn set_view(&mut self, view: View) {
        // Head size: head joint (base of the skull) to the top.
        let head = (self.model.height - self.model.head_y).max(0.05);
        let (visible, center_y) = match view {
            View::Face => (head * 1.7, self.model.head_y + head * 0.42),
            View::Bust => (head * 3.4, self.model.head_y + head * 0.05),
            View::FullBody => (self.units_per_view_height(), self.model.height / 2.0),
        };
        self.zoom = self.units_per_view_height() / visible;
        self.offset = Vec2::new(0.0, center_y);
    }

    fn pixels_per_unit(&self, view_height_px: u32) -> f32 {
        self.zoom * view_height_px as f32 / self.units_per_view_height()
    }

    /// Move the view by `delta_px` screen pixels.
    pub fn pan(&mut self, delta_px: Vec2, view_height_px: u32) {
        self.offset += delta_px / self.pixels_per_unit(view_height_px);
    }

    /// Zoom by `factor`, keeping the model point under `anchor_px` (pixels
    /// from the view center) in place.
    pub fn zoom_at(&mut self, factor: f32, anchor_px: Vec2, view_height_px: u32) {
        let before = self.pixels_per_unit(view_height_px);
        self.zoom = (self.zoom * factor).clamp(0.1, 20.0);
        let after = self.pixels_per_unit(view_height_px);
        self.offset += anchor_px / after - anchor_px / before;
    }

    /// Pose the model from `face` and draw it into `target`, over `clear`
    /// (premultiplied; transparent when the UI draws the background itself).
    pub fn render(
        &mut self,
        gl: &Arc<glow::Context>,
        target: &RenderTarget,
        clear: [f32; 4],
        background: Option<&Arc<image::RgbaImage>>,
        face: &FaceState,
        body: &BodyState,
    ) -> Result<(), String> {
        let view_height = self.units_per_view_height() / self.zoom;
        // Moving the model right/down is moving the camera left/up.
        let center = Vec2::new(-self.offset.x, self.offset.y);
        self.values.clear();
        let result = self.model.draw(
            gl,
            target.framebuffer,
            target.size,
            center,
            view_height,
            clear,
            background,
            face,
            body,
            &mut self.values,
        );
        // Leave GL the way egui expects it.
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.active_texture(glow::TEXTURE0);
            gl.use_program(None);
        }
        result
    }
}

/// Offscreen framebuffer (RGBA color + depth/stencil) whose color
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

    /// Send the frame to Syphon clients (OBS). The frame keeps its alpha, so
    /// render it over a transparent clear color.
    pub fn publish_syphon(&self, gl: &glow::Context, server: &crate::platform::SyphonServer) {
        unsafe {
            // Syphon copies the texture with whatever state is current.
            gl.viewport(0, 0, self.size[0] as i32, self.size[1] as i32);
            gl.active_texture(glow::TEXTURE0);
            gl.disable(glow::BLEND);
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.color_mask(true, true, true, true);
            server.publish(self.color.0.get(), self.size);
            // Back to what egui expects.
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.bind_vertex_array(None);
            gl.use_program(None);
            gl.enable(glow::BLEND);
        }
    }

    /// Copy the frame straight into the window's framebuffer, scaled to
    /// `window_size` pixels. Used while egui isn't painting the window.
    pub fn blit_to_window(&self, gl: &glow::Context, window_size: [u32; 2]) {
        let [w, h] = [self.size[0] as i32, self.size[1] as i32];
        let [window_w, window_h] = [window_size[0] as i32, window_size[1] as i32];
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.framebuffer));
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            gl.blit_framebuffer(0, 0, w, h, 0, 0, window_w, window_h, glow::COLOR_BUFFER_BIT, glow::LINEAR);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }
}
