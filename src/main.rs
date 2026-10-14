mod avatar;
mod body;
mod face;
mod platform;
mod tracker;
mod video;
mod vrm;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::time::{Duration, Instant};

use cameras::{Camera, CameraSource, Device, PixelFormat, Resolution, StreamConfig};
use eframe::egui;
use cameras::pump::Pump;

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
// Donations are switched off for now; see the commented-out section in
// `sidebar`.
// const DONATE_URL: &str = "https://doruk.wtf";
/// The project's source code.
const GITHUB_URL: &str = "https://github.com/dorukozerr/kamen";

const SIDEBAR_WIDTH: f32 = 300.0;
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
    /// Whether the camera preview is on screen; the camera thread skips
    /// converting frames for it otherwise.
    preview_flag: Arc<AtomicBool>,
    tracker_rx: Option<mpsc::Receiver<tracker::TrackResult>>,
    track_result: Option<tracker::TrackResult>,
    /// Arms and hands: send frames to the body tracker. Costs about as much
    /// CPU again as face tracking.
    body_flag: Arc<AtomicBool>,
    body_rx: Option<mpsc::Receiver<tracker::BodyResult>>,
    body_result: Option<tracker::BodyResult>,
    body: body::BodySolver,
    track_fps: FpsCounter,
    face: face::FaceSolver,

    avatar: Option<avatar::Avatar>,
    avatar_target: Option<avatar::RenderTarget>,
    avatar_error: Option<String>,
    /// Sidebar hidden: the window shows only the avatar.
    avatar_only: bool,
    /// Drawn into the avatar render itself, so OBS gets the same background
    /// as the window.
    background: Background,
    background_image: Option<Arc<image::RgbaImage>>,
    background_video: Option<video::VideoBackground>,

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
            preview_flag: Arc::new(AtomicBool::new(false)),
            tracker_rx: None,
            body_flag: Arc::new(AtomicBool::new(false)),
            body_rx: None,
            body_result: None,
            body: body::BodySolver::default(),
            track_result: None,
            track_fps: FpsCounter::default(),
            face,
            avatar: None,
            avatar_target: None,
            avatar_error: None,
            avatar_only: false,
            background: Background::Color(CANVAS),
            background_image: None,
            background_video: None,
            keep_on_top: false,
            low_fps: true,
            syphon_enabled: true,
            syphon: None,
            syphon_error: None,
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
        self.body_rx = None;
        self.body_result = None;
        let Some(device) = self.devices.get(self.selected_device).cloned() else {
            self.camera_error = Some("No camera found".into());
            return;
        };
        match cameras::open_source(CameraSource::Usb(device), STREAM_CONFIG) {
            Ok(camera) => {
                let trackers = tracker::spawn_trackers();
                self.stream = Some(spawn_stream(
                    camera,
                    trackers.face_frames,
                    (trackers.body_frames, self.body_flag.clone()),
                    self.preview_flag.clone(),
                ));
                self.tracker_rx = Some(trackers.face_results);
                self.body_rx = Some(trackers.body_results);
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

    fn pick_background(&mut self) {
        const IMAGE_EXTENSIONS: [&str; 6] = ["png", "jpg", "jpeg", "webp", "bmp", "gif"];
        let all: Vec<&str> = IMAGE_EXTENSIONS.iter().chain(&video::VIDEO_EXTENSIONS).copied().collect();
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Image or video", &all)
            .add_filter("Image", &IMAGE_EXTENSIONS)
            .add_filter("Video", &video::VIDEO_EXTENSIONS)
            .pick_file()
        else {
            return;
        };
        let extension = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if video::VIDEO_EXTENSIONS.contains(&extension.as_str()) {
            match video::VideoBackground::open(&path) {
                Ok(video) => {
                    self.background_video = Some(video);
                    self.background = Background::Video;
                    self.avatar_error = None;
                }
                Err(error) => self.avatar_error = Some(format!("Background video: {error}")),
            }
            return;
        }
        match image::open(&path) {
            Ok(image) => {
                self.background_image = Some(Arc::new(image.to_rgba8()));
                self.background = Background::Image;
                self.background_video = None;
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
        // Shows through only with a transparent background; otherwise the
        // avatar render covers it with the chosen background.
        painter.rect_filled(rect, 0.0, CANVAS);
        let (clear, background_image) = self.background_render();
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
        if let Err(error) = avatar.render(
            &gl,
            target,
            clear,
            background_image.as_ref(),
            &self.face.state,
            &self.body.state,
        ) {
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
        let (clear, background_image) = self.background_render();
        let (Some(avatar), Some(target), Some(gl)) = (
            self.avatar.as_mut(),
            self.avatar_target.as_ref(),
            frame.gl().cloned(),
        ) else {
            return;
        };
        let rendered = avatar.render(
            &gl,
            target,
            clear,
            background_image.as_ref(),
            &self.face.state,
            &self.body.state,
        );
        if rendered.is_err() {
            return;
        }
        if let Some(server) = &self.syphon {
            target.publish_syphon(&gl, server);
        }
        // Window capture needs the window itself updated.
        if self.avatar_only {
            let window = ctx.content_rect().size() * ctx.pixels_per_point();
            target.blit_to_window(&gl, [window.x.round() as u32, window.y.round() as u32]);
            platform::present_gl_frame();
        }
    }

    /// The avatar render's clear color (premultiplied alpha) and background
    /// image, for the chosen background.
    fn background_render(&self) -> ([f32; 4], Option<Arc<image::RgbaImage>>) {
        match self.background {
            Background::Color(color) => (color.to_normalized_gamma_f32(), None),
            Background::Image => ([0.0, 0.0, 0.0, 1.0], self.background_image.clone()),
            Background::Video => (
                [0.0, 0.0, 0.0, 1.0],
                self.background_video.as_ref().and_then(|video| video.frame()),
            ),
            Background::Transparent => ([0.0; 4], None),
        }
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
            let views = [avatar::View::Face, avatar::View::Bust, avatar::View::FullBody];
            if let Some(index) = segmented(ui, &["Face", "Bust", "Full body"], None) {
                avatar.set_view(views[index]);
            }
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
        let mut body = self.body_flag.load(Ordering::Relaxed);
        if switch_row(ui, "Arms and hands", &mut body)
            .on_hover_text("Tracks your arms and hands when they're in view. Uses more CPU.")
            .changed()
        {
            self.body_flag.store(body, Ordering::Relaxed);
        }
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
        let selected = match self.background {
            Background::Color(_) => 0,
            Background::Image | Background::Video => 1,
            Background::Transparent => 2,
        };
        match stacked(ui, &["Color", "Image or video…", "Transparent"], selected) {
            Some(0) if selected != 0 => {
                self.background = Background::Color(CANVAS);
                // Stop decoding a video nobody sees.
                self.background_video = None;
            }
            Some(1) => self.pick_background(),
            Some(2) => {
                self.background = Background::Transparent;
                self.background_video = None;
            }
            _ => {}
        }
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
        if selected == 1 {
            hint(ui, "An image, or a looping MP4 / MOV video.");
        }
        hint(
            ui,
            match self.background {
                Background::Transparent => {
                    "OBS gets only the avatar. Turn on Allow Transparency in OBS's Syphon source."
                }
                _ => "OBS gets the same background as this window.",
            },
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

        // section(ui, "Support");
        // hint(
        //     ui,
        //     "Kamen is free. If it's useful to you, a donation helps keep it going.",
        // );
        // if ui
        //     .add_sized([ui.available_width(), 28.0], egui::Button::new("Donate"))
        //     .on_hover_text(DONATE_URL)
        //     .clicked()
        // {
        //     ui.ctx().open_url(egui::OpenUrl::new_tab(DONATE_URL));
        // }

        section(ui, "About");
        hint(ui, "Kamen is free and open source.");
        if ui
            .add_sized([ui.available_width(), 28.0], egui::Button::new("Source code on GitHub"))
            .on_hover_text(GITHUB_URL)
            .clicked()
        {
            ui.ctx().open_url(egui::OpenUrl::new_tab(GITHUB_URL));
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
        let scale = rect.width() / result.frame_w as f32;
        let painter = ui.painter_at(rect);
        let dot = |point: [f32; 2], radius: f32, color: egui::Color32| {
            painter.circle_filled(rect.min + egui::vec2(point[0], point[1]) * scale, radius, color);
        };
        if let Some(face) = &result.face {
            for point in &face.points {
                dot([point[0], point[1]], 0.7, ACCENT);
            }
        }
        let Some(body) = &self.body_result else {
            return;
        };
        if let Some(pose) = &body.pose {
            for (point, visibility) in pose.points.iter().zip(&pose.visibility) {
                if *visibility > 0.5 {
                    dot(*point, 2.0, GOOD);
                }
            }
        }
        for hand in body.hands.iter().flatten() {
            for point in &hand.points {
                dot(*point, 1.5, WARN);
            }
        }
    }
}

impl eframe::App for App {
    /// Runs every frame, also while the window is hidden (covered, or on
    /// another Space), when eframe skips `ui` and painting.
    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if let Some(stream) = &mut self.stream {
            update_texture(stream, ctx);
        }
        if let Some(rx) = &self.tracker_rx {
            while let Ok(result) = rx.try_recv() {
                self.track_fps.tick();
                self.face.update(&result);
                self.track_result = Some(result);
            }
        }
        self.face.advance();
        if let Some(rx) = &self.body_rx {
            while let Ok(result) = rx.try_recv() {
                self.body.update(&result);
                self.body_result = Some(result);
            }
        }
        self.body.mirror = self.face.mirror;
        self.body.advance();

        let hidden = ctx.input(|i| i.viewport().occluded == Some(true));
        self.preview_flag.store(
            self.show_preview && !self.avatar_only && !hidden,
            Ordering::Relaxed,
        );
        self.update_syphon();
        if hidden && (self.avatar_only || self.syphon.is_some()) {
            self.present_while_hidden(ctx, frame);
            // `ui` isn't running to ask for the next frame.
            ctx.request_repaint_after(self.frame_interval());
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
                // Extra room on the right for the scroll bar.
                .frame(egui::Frame::new().fill(SURFACE).inner_margin(egui::Margin {
                    left: 16,
                    right: 22,
                    top: 16,
                    bottom: 16,
                }))
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            // Rows lay out within the panel instead of growing past it.
                            ui.set_max_width(ui.available_width());
                            self.sidebar(ui);
                        });
                });
        }
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show_inside(ui, |ui| self.show_avatar(ui, frame));
        ctx.request_repaint_after(self.frame_interval());
    }
}

/// A row of equal-width buttons filling the available width. `selected`
/// highlights one; returns the index clicked, if any.
fn segmented(ui: &mut egui::Ui, labels: &[&str], selected: Option<usize>) -> Option<usize> {
    let mut clicked = None;
    ui.columns(labels.len(), |columns| {
        for (index, (column, label)) in columns.iter_mut().zip(labels).enumerate() {
            column.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                if ui.selectable_label(selected == Some(index), *label).clicked() {
                    clicked = Some(index);
                }
            });
        }
    });
    clicked
}

/// Full-width options stacked vertically, one selected. Returns the index
/// clicked, if any.
fn stacked(ui: &mut egui::Ui, labels: &[&str], selected: usize) -> Option<usize> {
    let mut clicked = None;
    ui.with_layout(egui::Layout::top_down_justified(egui::Align::Min), |ui| {
        for (index, label) in labels.iter().enumerate() {
            if ui.selectable_label(selected == index, *label).clicked() {
                clicked = Some(index);
            }
        }
    });
    clicked
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
    face_frames: mpsc::Sender<cameras::Frame>,
    body_frames: (mpsc::Sender<cameras::Frame>, Arc<AtomicBool>),
    preview: Arc<AtomicBool>,
) -> CameraStream {
    let latest = Arc::new(Mutex::new(None));
    let slot = latest.clone();
    let pump = cameras::pump::spawn(camera, move |frame| {
        let _ = face_frames.send(frame.clone());
        if body_frames.1.load(Ordering::Relaxed) {
            let _ = body_frames.0.send(frame.clone());
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
        let rgba = cameras::to_rgba8(frame).ok()?;
        return Some(egui::ColorImage::from_rgba_unmultiplied(
            [frame.width as usize, frame.height as usize],
            &rgba,
        ));
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

/// Upload the newest converted frame, if any.
fn update_texture(stream: &mut CameraStream, ctx: &egui::Context) {
    let Some(image) = stream
        .latest
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take()
    else {
        return;
    };
    match &mut stream.texture {
        Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
        None => {
            stream.texture =
                Some(ctx.load_texture("camera-frame", image, egui::TextureOptions::LINEAR));
        }
    }
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

#[derive(Clone, Copy)]
enum Background {
    Color(egui::Color32),
    Image,
    /// A looping video (`background_video`).
    Video,
    /// Nothing behind the avatar: OBS can composite it over the scene.
    Transparent,
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
