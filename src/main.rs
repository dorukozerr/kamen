mod avatar;
mod face;
mod platform;
mod tracker;
mod vrm;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::time::{Duration, Instant};

use cameras::{Camera, CameraSource, Device, PixelFormat, Resolution, StreamConfig};
use eframe::egui;
use egui_cameras::Pump;

const STREAM_CONFIG: StreamConfig = StreamConfig {
    resolution: Resolution {
        width: 1280,
        height: 720,
    },
    framerate: 30,
    pixel_format: PixelFormat::Bgra8,
};

/// Avatar loaded at startup, relative to the project folder or to
/// `Contents/Resources` in the macOS app bundle.
const DEFAULT_AVATAR: &str = "models/vrm/nihil.vrm";

/// Time between drawn frames. Buffer swaps don't wait for the display
/// (`vsync: false`), so this is what paces the app.
const FRAME_INTERVAL: Duration = Duration::from_millis(16);
const LOW_FPS_FRAME_INTERVAL: Duration = Duration::from_millis(33);
/// Camera frames wider than this are halved for the preview.
const PREVIEW_MAX_WIDTH: usize = 1280;
/// Longest side of the avatar render, in pixels.
const AVATAR_MAX_RENDER_SIDE: f32 = 1600.0;

/// Where the Donate button leads.
const DONATE_URL: &str = "https://doruk.wtf";

const SIDEBAR_WIDTH: f32 = 272.0;
const CANVAS: egui::Color32 = egui::Color32::from_rgb(24, 25, 29);
const SURFACE: egui::Color32 = egui::Color32::from_rgb(32, 33, 38);
const CONTROL: egui::Color32 = egui::Color32::from_rgb(46, 48, 55);
const CONTROL_HOVER: egui::Color32 = egui::Color32::from_rgb(58, 60, 69);
const ACCENT: egui::Color32 = egui::Color32::from_rgb(124, 108, 246);
const TEXT: egui::Color32 = egui::Color32::from_rgb(232, 233, 238);
const TEXT_DIM: egui::Color32 = egui::Color32::from_rgb(140, 143, 155);
const GOOD: egui::Color32 = egui::Color32::from_rgb(84, 200, 130);
const WARN: egui::Color32 = egui::Color32::from_rgb(232, 180, 80);
const BAD: egui::Color32 = egui::Color32::from_rgb(226, 96, 96);
const CHROMA_GREEN: egui::Color32 = egui::Color32::from_rgb(0, 255, 0);

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1080.0, 720.0])
            .with_min_inner_size([640.0, 480.0])
            .with_icon(app_icon()),
        // The avatar renderer draws with OpenGL.
        renderer: eframe::Renderer::Glow,
        // Don't wait for the display when swapping buffers. A window that is
        // off screen (covered, or on another Space) gets no refresh signal, so
        // a waiting swap blocks the whole app until the window is shown
        // again. Frames are paced with `FRAME_INTERVAL` instead.
        vsync: false,
        ..Default::default()
    };
    eframe::run_native(
        "Kamen",
        options,
        Box::new(|cc| {
            platform::disable_app_nap();
            apply_theme(&cc.egui_ctx);
            let mut app = App::new();
            // An avatar given on the command line wins over the default one.
            match std::env::args_os().nth(1) {
                Some(path) => app.load_avatar(std::path::Path::new(&path)),
                None => {
                    if let Some(path) = default_avatar_path() {
                        app.load_avatar(&path);
                    }
                }
            }
            app.connect();
            Ok(Box::new(app))
        }),
    )
}

/// The app icon, for the Dock and the app switcher. Without it eframe swaps
/// in its own default icon at launch, replacing the one from the app bundle.
fn app_icon() -> egui::IconData {
    let Ok(image) = image::load_from_memory(include_bytes!("../assets/AppIcon.png")) else {
        return egui::IconData::default();
    };
    let image = image.to_rgba8();
    egui::IconData {
        width: image.width(),
        height: image.height(),
        rgba: image.into_raw(),
    }
}

/// Dark, quiet look: one accent color, rounded controls, roomy spacing.
fn apply_theme(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = egui::vec2(8.0, 8.0);
        style.spacing.button_padding = egui::vec2(12.0, 6.0);
        style.spacing.interact_size.y = 28.0;
        for (text_style, size) in [
            (egui::TextStyle::Heading, 20.0),
            (egui::TextStyle::Body, 13.5),
            (egui::TextStyle::Button, 13.5),
            (egui::TextStyle::Small, 11.0),
        ] {
            style
                .text_styles
                .insert(text_style, egui::FontId::proportional(size));
        }

        let visuals = &mut style.visuals;
        visuals.panel_fill = SURFACE;
        visuals.window_fill = SURFACE;
        visuals.extreme_bg_color = CANVAS;
        visuals.override_text_color = Some(TEXT);
        visuals.selection.bg_fill = ACCENT;
        visuals.selection.stroke = egui::Stroke::new(1.0_f32, egui::Color32::WHITE);
        visuals.hyperlink_color = ACCENT;
        let radius = egui::CornerRadius::same(8);
        for (widget, fill) in [
            (&mut visuals.widgets.noninteractive, SURFACE),
            (&mut visuals.widgets.inactive, CONTROL),
            (&mut visuals.widgets.hovered, CONTROL_HOVER),
            (&mut visuals.widgets.active, ACCENT),
            (&mut visuals.widgets.open, CONTROL_HOVER),
        ] {
            widget.corner_radius = radius;
            widget.bg_fill = fill;
            widget.weak_bg_fill = fill;
            widget.bg_stroke = egui::Stroke::NONE;
            widget.fg_stroke = egui::Stroke::new(1.0_f32, TEXT);
            widget.expansion = 0.0;
        }
        visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0_f32, CONTROL);
    });
}

struct App {
    devices: Vec<Device>,
    selected_device: usize,
    stream: Option<CameraStream>,
    camera_error: Option<String>,
    /// Show the camera picture in the sidebar. Off by default: converting
    /// and uploading camera frames costs more than the tracking itself.
    show_preview: bool,
    track_flag: Arc<AtomicBool>,
    /// Whether the camera preview is on screen; the camera thread skips
    /// converting frames for it otherwise.
    preview_flag: Arc<AtomicBool>,
    tracker_rx: Option<mpsc::Receiver<tracker::TrackResult>>,
    track_result: Option<tracker::TrackResult>,
    camera_fps: FpsCounter,
    track_fps: FpsCounter,
    face: face::FaceSolver,

    avatar: Option<avatar::Avatar>,
    avatar_target: Option<avatar::RenderTarget>,
    avatar_error: Option<String>,
    /// Sidebar hidden: the window shows only the avatar.
    avatar_only: bool,
    background: Background,
    background_image: Option<egui::TextureHandle>,

    /// Float above other windows. A fully covered window stops being drawn,
    /// which freezes it in OBS's window capture.
    keep_on_top: bool,
    /// Draw at 30 fps instead of 60, for less CPU and GPU use.
    low_fps: bool,
    /// Send avatar frames to OBS through Syphon. The server is created on
    /// the first frame after this is turned on.
    syphon_enabled: bool,
    syphon: Option<platform::SyphonServer>,
    syphon_error: Option<String>,
    hidden_log: HiddenLog,
    /// The window was hidden on the last frame (for logging the change).
    presenting_hidden: bool,
}

impl App {
    fn new() -> Self {
        let mut face = face::FaceSolver::default();
        face.mirror = false;
        Self {
            devices: cameras::devices().unwrap_or_default(),
            selected_device: 0,
            stream: None,
            camera_error: None,
            show_preview: false,
            track_flag: Arc::new(AtomicBool::new(true)),
            preview_flag: Arc::new(AtomicBool::new(false)),
            tracker_rx: None,
            track_result: None,
            camera_fps: FpsCounter::default(),
            track_fps: FpsCounter::default(),
            face,
            avatar: None,
            avatar_target: None,
            avatar_error: None,
            avatar_only: false,
            background: Background::Color(CANVAS),
            background_image: None,
            keep_on_top: false,
            low_fps: true,
            syphon_enabled: true,
            syphon: None,
            syphon_error: None,
            hidden_log: HiddenLog::default(),
            presenting_hidden: false,
        }
    }

    fn frame_interval(&self) -> Duration {
        if self.low_fps {
            LOW_FPS_FRAME_INTERVAL
        } else {
            FRAME_INTERVAL
        }
    }

    /// Open the selected camera and start tracking.
    fn connect(&mut self) {
        self.stream = None;
        self.tracker_rx = None;
        self.track_result = None;
        let Some(device) = self.devices.get(self.selected_device).cloned() else {
            self.camera_error = Some("No camera found".into());
            return;
        };
        match cameras::open_source(CameraSource::Usb(device), STREAM_CONFIG) {
            Ok(camera) => {
                let (tracker_tx, tracker_rx) = tracker::spawn_tracker();
                self.stream = Some(spawn_stream(
                    camera,
                    tracker_tx,
                    self.track_flag.clone(),
                    self.preview_flag.clone(),
                ));
                self.tracker_rx = Some(tracker_rx);
                self.camera_error = None;
            }
            Err(error) => self.camera_error = Some(format!("Camera: {error}")),
        }
    }

    fn refresh_devices(&mut self) {
        self.devices = cameras::devices().unwrap_or_default();
        if self.selected_device >= self.devices.len() {
            self.selected_device = 0;
        }
    }

    fn load_avatar(&mut self, path: &std::path::Path) {
        match avatar::Avatar::load(path) {
            Ok(avatar) => {
                self.avatar = Some(avatar);
                self.avatar_error = None;
            }
            Err(error) => self.avatar_error = Some(error),
        }
    }

    fn pick_avatar(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("VRM avatar", &["vrm"])
            .pick_file()
        {
            self.load_avatar(&path);
        }
    }

    fn pick_background_image(&mut self, ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Image", &["png", "jpg", "jpeg", "webp", "bmp"])
            .pick_file()
        else {
            return;
        };
        match image::open(&path) {
            Ok(image) => {
                let rgba = image.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let color = egui::ColorImage::from_rgba_unmultiplied(size, &rgba);
                self.background_image =
                    Some(ctx.load_texture("background", color, egui::TextureOptions::LINEAR));
                self.background = Background::Image;
                self.avatar_error = None;
            }
            Err(error) => self.avatar_error = Some(format!("Background image: {error}")),
        }
    }

    fn apply_window_level(&self, ctx: &egui::Context) {
        let level = if self.keep_on_top {
            egui::WindowLevel::AlwaysOnTop
        } else {
            egui::WindowLevel::Normal
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(level));
    }

    /// One-line tracking state for the sidebar: a color and a sentence.
    fn status(&self) -> (egui::Color32, String) {
        if let Some(error) = &self.camera_error {
            return (BAD, error.clone());
        }
        if self.stream.is_none() {
            return (TEXT_DIM, "Camera off".into());
        }
        let Some(result) = &self.track_result else {
            return (WARN, "Starting tracking…".into());
        };
        if result.face.is_none() {
            return (WARN, "No face in view".into());
        }
        if self.face.is_calibrating() {
            return (WARN, "Calibrating: look straight ahead".into());
        }
        (GOOD, format!("Tracking at {:.0} fps", self.track_fps.fps))
    }

    fn show_avatar(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, egui::Sense::drag());
        let painter = ui.painter_at(rect);
        match self.background {
            Background::Color(color) => {
                painter.rect_filled(rect, 0.0, color);
            }
            Background::Image => {
                if let Some(texture) = &self.background_image {
                    painter.image(
                        texture.id(),
                        cover_rect(texture.size_vec2(), rect),
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                }
            }
        }
        let Some(avatar) = self.avatar.as_mut() else {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Choose a VRM avatar to begin",
                egui::FontId::proportional(15.0),
                TEXT_DIM,
            );
            return;
        };
        let Some(gl) = frame.gl().cloned() else {
            self.avatar_error = Some("Avatar rendering needs the OpenGL renderer".into());
            return;
        };

        let pixels_per_point = ui.ctx().pixels_per_point();
        let view = rect.size() * pixels_per_point;
        // Render at screen resolution up to a limit; a larger view is scaled
        // up. Every pixel is shaded four times for anti-aliasing.
        let render_scale = (AVATAR_MAX_RENDER_SIDE / view.max_elem()).min(1.0);
        let size = [
            (view.x * render_scale).round() as u32,
            (view.y * render_scale).round() as u32,
        ];
        let view_height = view.y.round() as u32;
        // Scroll (or pinch) to zoom toward the cursor, drag to move.
        if let Some(pointer) = response.hover_pos() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = (scroll * 0.002).exp() * pinch;
            if factor != 1.0 {
                let anchor = (pointer - rect.center()) * pixels_per_point;
                avatar.zoom_at(factor, glam::Vec2::new(anchor.x, anchor.y), view_height);
            }
        }
        if response.dragged() {
            let delta = response.drag_delta() * pixels_per_point;
            avatar.pan(glam::Vec2::new(delta.x, delta.y), view_height);
        }

        let target = match self.avatar_target.as_mut() {
            Some(target) => {
                target.resize(&gl, size);
                target
            }
            None => match avatar::RenderTarget::new(&gl, frame, size) {
                Ok(target) => self.avatar_target.insert(target),
                Err(error) => {
                    self.avatar_error = Some(format!("Avatar render target: {error}"));
                    return;
                }
            },
        };
        if let Err(error) = avatar.render(&gl, target, [0.0; 4], &self.face.state) {
            self.avatar_error = Some(format!("Avatar render: {error}"));
            self.avatar = None;
            return;
        }
        if let Some(server) = &self.syphon {
            target.publish_syphon(&gl, server);
        }
        // GL textures are bottom-up.
        painter.image(
            target.texture_id,
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 1.0), egui::pos2(1.0, 0.0)),
            egui::Color32::WHITE,
        );
    }

    /// Start or stop the Syphon server to match the switch. Runs in `logic`,
    /// where the OpenGL context is current.
    fn update_syphon(&mut self) {
        if !self.syphon_enabled {
            self.syphon = None;
            return;
        }
        if self.syphon.is_none() && self.syphon_error.is_none() {
            match platform::SyphonServer::new(c"Avatar") {
                Ok(server) => self.syphon = Some(server),
                Err(error) => {
                    self.syphon_error = Some(error);
                    self.syphon_enabled = false;
                }
            }
        }
    }

    /// Draw the avatar while the window is hidden. eframe paints nothing
    /// then, which would freeze the picture OBS gets.
    fn present_while_hidden(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let (Some(avatar), Some(target), Some(gl)) = (
            self.avatar.as_mut(),
            self.avatar_target.as_ref(),
            frame.gl().cloned(),
        ) else {
            return;
        };
        // With Syphon on, OBS gets the avatar from Syphon (transparent
        // background), so the window itself doesn't need drawing.
        if let Some(server) = &self.syphon {
            if avatar
                .render(&gl, target, [0.0; 4], &self.face.state)
                .is_ok()
            {
                target.publish_syphon(&gl, server);
            }
            return;
        }
        // The background image is an egui texture, which only egui can draw;
        // it shows as black here.
        let clear = match self.background {
            Background::Color(color) => color.to_normalized_gamma_f32(),
            Background::Image => [0.0, 0.0, 0.0, 1.0],
        };
        if avatar.render(&gl, target, clear, &self.face.state).is_err() {
            return;
        }
        let window = ctx.content_rect().size() * ctx.pixels_per_point();
        target.blit_to_window(&gl, [window.x.round() as u32, window.y.round() as u32]);
        self.hidden_log.before_present();
        let presented = platform::present_gl_frame();
        self.hidden_log.frame(presented);
    }

    fn sidebar(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.heading("Kamen");
        let (color, text) = self.status();
        ui.horizontal(|ui| {
            let (dot, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
            ui.painter().circle_filled(dot.center(), 4.0, color);
            ui.label(egui::RichText::new(text).color(TEXT_DIM));
        });

        section(ui, "Avatar");
        ui.horizontal(|ui| {
            let name = self
                .avatar
                .as_ref()
                .map_or("None", |avatar| avatar.name.as_str());
            ui.label(name);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Change…").clicked() {
                    self.pick_avatar();
                }
            });
        });
        if let Some(error) = &self.avatar_error {
            ui.colored_label(BAD, error);
        }
        if let Some(avatar) = &mut self.avatar {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Framing").color(TEXT_DIM));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    for (label, view) in [
                        ("Full", avatar::View::FullBody),
                        ("Bust", avatar::View::Bust),
                        ("Face", avatar::View::Face),
                    ] {
                        if ui.button(label).clicked() {
                            avatar.set_view(view);
                        }
                    }
                });
            });
            hint(ui, "Scroll to zoom, drag to move.");
        }

        section(ui, "Camera");
        let mut reconnect = false;
        let selected = self
            .devices
            .get(self.selected_device)
            .map_or("No camera", |device| device.name.as_str())
            .to_owned();
        egui::ComboBox::from_id_salt("camera")
            .selected_text(selected)
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for (index, device) in self.devices.iter().enumerate() {
                    if ui
                        .selectable_value(&mut self.selected_device, index, &device.name)
                        .changed()
                    {
                        reconnect = true;
                    }
                }
                if ui
                    .selectable_label(false, "Look for cameras again")
                    .clicked()
                {
                    reconnect = true;
                }
            });
        if reconnect {
            self.refresh_devices();
            self.connect();
        }
        switch_row(ui, "Show camera", &mut self.show_preview);
        if self.show_preview {
            self.camera_preview(ui);
        }

        section(ui, "Tracking");
        switch_row(ui, "Mirror", &mut self.face.mirror);
        let calibrate = if self.face.is_calibrating() {
            "Calibrating…"
        } else {
            "Recalibrate"
        };
        if ui
            .add_sized([ui.available_width(), 28.0], egui::Button::new(calibrate))
            .on_hover_text("Look straight ahead with a relaxed face, then click.")
            .clicked()
        {
            self.face.calibrate();
        }

        section(ui, "Output");
        if switch_row(ui, "Send to OBS", &mut self.syphon_enabled).changed() {
            self.syphon_error = None;
        }
        match &self.syphon_error {
            Some(error) => {
                ui.colored_label(BAD, error);
            }
            None if self.syphon_enabled => {
                hint(ui, "In OBS, add a Syphon Client source and pick Kamen.");
            }
            None => {}
        }
        switch_row(ui, "30 fps", &mut self.low_fps)
            .on_hover_text("Off draws 60 frames a second, for about twice the CPU and GPU use.");

        section(ui, "Background");
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            let is_color = matches!(self.background, Background::Color(_));
            if ui.selectable_label(is_color, "Color").clicked() && !is_color {
                self.background = Background::Color(CANVAS);
            }
            if ui
                .selectable_label(matches!(self.background, Background::Image), "Image…")
                .clicked()
            {
                self.pick_background_image(&ctx);
            }
        });
        if let Background::Color(color) = &mut self.background {
            ui.horizontal(|ui| {
                ui.color_edit_button_srgba(color);
                if ui.button("Dark").clicked() {
                    *color = CANVAS;
                }
                if ui.button("Green").clicked() {
                    *color = CHROMA_GREEN;
                }
            });
        }
        hint(
            ui,
            "Only seen in this window. OBS gets a transparent background.",
        );

        ui.add_space(12.0);
        egui::CollapsingHeader::new(egui::RichText::new("Advanced").color(TEXT_DIM)).show(
            ui,
            |ui| {
                if switch_row(ui, "Keep on top", &mut self.keep_on_top).changed() {
                    self.apply_window_level(ui.ctx());
                }
                if let Some(avatar) = &self.avatar {
                    ui.add_space(4.0);
                    egui::Grid::new("avatar_params")
                        .striped(true)
                        .show(ui, |ui| {
                            for param in &avatar.params {
                                ui.label(egui::RichText::new(param).small());
                                match avatar.values.iter().find(|(name, _)| name == param) {
                                    Some((_, value)) => ui.label(
                                        egui::RichText::new(format!(
                                            "{:+.2} {:+.2}",
                                            value.x, value.y
                                        ))
                                        .small()
                                        .monospace(),
                                    ),
                                    None => ui
                                        .label(egui::RichText::new("idle").small().color(TEXT_DIM)),
                                };
                                ui.end_row();
                            }
                        });
                }
            },
        );

        section(ui, "Support");
        hint(
            ui,
            "Kamen is free. If it's useful to you, a donation helps keep it going.",
        );
        if ui
            .add_sized([ui.available_width(), 28.0], egui::Button::new("Donate"))
            .on_hover_text(DONATE_URL)
            .clicked()
        {
            ui.ctx().open_url(egui::OpenUrl::new_tab(DONATE_URL));
        }

        ui.add_space(12.0);
        if ui
            .add_sized(
                [ui.available_width(), 30.0],
                egui::Button::new("Hide controls"),
            )
            .on_hover_text("Shows only the avatar. Press Esc to bring the controls back.")
            .clicked()
        {
            self.avatar_only = true;
        }
    }

    /// Camera picture with the tracked landmarks, fitted to the sidebar.
    fn camera_preview(&self, ui: &mut egui::Ui) {
        let Some(texture) = self
            .stream
            .as_ref()
            .and_then(|stream| stream.texture.as_ref())
        else {
            hint(ui, "Waiting for the camera…");
            return;
        };
        let width = ui.available_width();
        let size = egui::vec2(width, width / texture.aspect_ratio());
        let image = egui::Image::new((texture.id(), size)).corner_radius(8);
        let rect = ui.add(image).rect;
        let Some(result) = &self.track_result else {
            return;
        };
        let Some(face) = &result.face else {
            return;
        };
        let scale = rect.width() / result.frame_w as f32;
        let painter = ui.painter_at(rect);
        for point in &face.points {
            let position = rect.min + egui::vec2(point[0], point[1]) * scale;
            painter.circle_filled(position, 0.7, ACCENT);
        }
    }
}

impl eframe::App for App {
    /// Runs every frame, also while the window is hidden (covered, or on
    /// another Space), when eframe skips `ui` and painting.
    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if let Some(stream) = &mut self.stream
            && update_texture(stream, ctx)
        {
            self.camera_fps.tick();
        }
        if let Some(rx) = &self.tracker_rx {
            while let Ok(result) = rx.try_recv() {
                self.track_fps.tick();
                self.face.update(&result);
                self.track_result = Some(result);
            }
        }
        self.face.advance();

        let hidden = ctx.input(|i| i.viewport().occluded == Some(true));
        self.preview_flag.store(
            self.show_preview && !self.avatar_only && !hidden,
            Ordering::Relaxed,
        );
        self.update_syphon();
        if hidden && (self.avatar_only || self.syphon.is_some()) {
            if !self.presenting_hidden {
                self.presenting_hidden = true;
                self.hidden_log
                    .note("hidden: drawing the avatar directly (controls hidden, or Syphon)");
            }
            self.present_while_hidden(ctx, frame);
            // `ui` isn't running to ask for the next frame.
            ctx.request_repaint_after(self.frame_interval());
        } else if hidden != self.presenting_hidden {
            self.presenting_hidden = hidden;
            self.hidden_log.note(if hidden {
                "hidden, with controls showing and Syphon off: nothing is drawn"
            } else {
                "visible again"
            });
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if self.avatar_only && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.avatar_only = false;
        }
        if !self.avatar_only {
            egui::Panel::right("controls")
                .resizable(false)
                .exact_size(SIDEBAR_WIDTH)
                .frame(egui::Frame::new().fill(SURFACE).inner_margin(16.0))
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| self.sidebar(ui));
                });
        }
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show_inside(ui, |ui| self.show_avatar(ui, frame));
        ctx.request_repaint_after(self.frame_interval());
    }
}

/// Small dim heading that starts a group of controls.
fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(14.0);
    ui.label(
        egui::RichText::new(title.to_uppercase())
            .small()
            .strong()
            .color(TEXT_DIM),
    );
}

fn hint(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).small().color(TEXT_DIM));
}

/// A label on the left and an on/off switch on the right.
fn switch_row(ui: &mut egui::Ui, label: &str, on: &mut bool) -> egui::Response {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            switch(ui, on)
        })
        .inner
    })
    .inner
}

fn switch(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
    let (rect, mut response) = ui.allocate_exact_size(egui::vec2(36.0, 20.0), egui::Sense::click());
    if response.clicked() {
        *on = !*on;
        response.mark_changed();
    }
    let t = ui.ctx().animate_bool(response.id, *on);
    let mix = |from: u8, to: u8| (from as f32 + (to as f32 - from as f32) * t).round() as u8;
    let track = egui::Color32::from_rgb(
        mix(CONTROL_HOVER.r(), ACCENT.r()),
        mix(CONTROL_HOVER.g(), ACCENT.g()),
        mix(CONTROL_HOVER.b(), ACCENT.b()),
    );
    let radius = rect.height() / 2.0;
    ui.painter().rect_filled(rect, radius, track);
    let knob_x = rect.left() + radius + (rect.width() - 2.0 * radius) * t;
    ui.painter().circle_filled(
        egui::pos2(knob_x, rect.center().y),
        radius - 3.0,
        egui::Color32::WHITE,
    );
    response
}

/// Camera pump plus the egui texture it feeds. Frames are converted to a
/// `ColorImage` on the pump thread, so the UI thread only uploads them.
struct CameraStream {
    /// Keeps the camera running; dropping it stops the stream.
    _pump: Pump,
    latest: Arc<Mutex<Option<egui::ColorImage>>>,
    texture: Option<egui::TextureHandle>,
}

fn spawn_stream(
    camera: Camera,
    tracker_tx: mpsc::Sender<cameras::Frame>,
    track: Arc<AtomicBool>,
    preview: Arc<AtomicBool>,
) -> CameraStream {
    let latest = Arc::new(Mutex::new(None));
    let slot = latest.clone();
    let pump = cameras::pump::spawn(camera, move |frame| {
        if track.load(Ordering::Relaxed) {
            let _ = tracker_tx.send(frame.clone());
        }
        if preview.load(Ordering::Relaxed)
            && let Some(image) = frame_to_color_image(&frame)
        {
            *slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(image);
        }
    });
    CameraStream {
        _pump: pump,
        latest,
        texture: None,
    }
}

fn frame_to_color_image(frame: &cameras::Frame) -> Option<egui::ColorImage> {
    if frame.pixel_format != PixelFormat::Bgra8 {
        return egui_cameras::frame_to_color_image(frame).ok();
    }
    let (width, height) = (frame.width as usize, frame.height as usize);
    let stride = if frame.stride == 0 {
        width * 4
    } else {
        frame.stride as usize
    };
    let data = &frame.plane_primary[..];
    if data.len() < stride * height.saturating_sub(1) + width * 4 {
        return None;
    }
    // The preview is small, so a full-HD frame is halved: a quarter of the
    // pixels to convert and upload.
    let step = if width > PREVIEW_MAX_WIDTH { 2 } else { 1 };
    let (out_width, out_height) = (width / step, height / step);
    let mut pixels = Vec::with_capacity(out_width * out_height);
    for row in 0..out_height {
        let start = row * step * stride;
        pixels.extend(
            data[start..start + width * 4]
                .chunks_exact(4)
                .step_by(step)
                .take(out_width)
                .map(|px| egui::Color32::from_rgb(px[2], px[1], px[0])),
        );
    }
    Some(egui::ColorImage::new([out_width, out_height], pixels))
}

/// Upload the newest converted frame, if any. Returns whether one was uploaded.
fn update_texture(stream: &mut CameraStream, ctx: &egui::Context) -> bool {
    let Some(image) = stream
        .latest
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take()
    else {
        return false;
    };
    match &mut stream.texture {
        Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
        None => {
            stream.texture =
                Some(ctx.load_texture("camera-frame", image, egui::TextureOptions::LINEAR));
        }
    }
    true
}

/// The default avatar's location, if the file exists: inside the app bundle
/// first (an app opened from Finder runs with `/` as working directory), then
/// relative to the working directory.
fn default_avatar_path() -> Option<std::path::PathBuf> {
    let bundled = std::env::current_exe().ok().and_then(|exe| {
        Some(
            exe.parent()?
                .parent()?
                .join("Resources")
                .join(DEFAULT_AVATAR),
        )
    });
    bundled
        .into_iter()
        .chain([std::path::PathBuf::from(DEFAULT_AVATAR)])
        .find(|path| path.exists())
}

/// Diagnostic for the hidden-window path: once a second while the window is
/// hidden, appends how many frames were drawn to `kamen-hidden.log` in the
/// temp directory. An app opened from Finder has no terminal to print to.
#[derive(Default)]
struct HiddenLog {
    window_start: Option<Instant>,
    frames: u32,
    presented: u32,
}

impl HiddenLog {
    fn note(&self, text: &str) {
        use std::io::Write;
        let path = std::env::temp_dir().join("kamen-hidden.log");
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(file, "{} {text}", unix_timestamp());
        }
    }

    /// Call right before presenting. Once a second this leaves a marker, so a
    /// present call that never returns shows up as a marker with no summary.
    fn before_present(&mut self) {
        if self.window_start.is_none() {
            self.window_start = Some(Instant::now());
            self.note("presenting...");
        }
    }

    fn frame(&mut self, presented: bool) {
        let start = *self.window_start.get_or_insert_with(Instant::now);
        self.frames += 1;
        self.presented += presented as u32;
        if start.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.note(&format!(
            "hidden frames drawn={} presented={}",
            self.frames, self.presented
        ));
        *self = Self::default();
    }
}

#[derive(Clone, Copy)]
enum Background {
    Color(egui::Color32),
    Image,
}

/// Rect that covers `area` with an image of `size`, keeping its aspect ratio.
fn cover_rect(size: egui::Vec2, area: egui::Rect) -> egui::Rect {
    let scale = (area.width() / size.x).max(area.height() / size.y);
    egui::Rect::from_center_size(area.center(), size * scale)
}

#[derive(Default)]
struct FpsCounter {
    window_start: Option<Instant>,
    count: u32,
    fps: f32,
}

impl FpsCounter {
    fn tick(&mut self) {
        let now = Instant::now();
        let start = *self.window_start.get_or_insert(now);
        self.count += 1;
        let elapsed = now.duration_since(start).as_secs_f32();
        if elapsed >= 0.5 {
            self.fps = self.count as f32 / elapsed;
            self.count = 0;
            self.window_start = Some(now);
        }
    }
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
