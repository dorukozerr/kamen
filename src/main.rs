mod avatar;
mod face;
mod tracker;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::thread::sleep;
use std::time::{Duration, Instant};

#[cfg(any(target_os = "macos", target_os = "windows"))]
use cameras::Credentials;
use cameras::analysis;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use cameras::discover::DiscoverConfig;
use cameras::{
    Camera, CameraSource, ControlCapabilities, ControlKind, ControlRange, Controls, Device,
    PixelFormat, Rect, Resolution, StreamConfig,
};
use eframe::egui;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use egui_cameras::{DiscoverySession, poll_discovery, show_discovery, start_discovery};
use egui_cameras::{Pump, capture_frame, set_active};
use image::{ExtendedColorType, ImageEncoder, codecs::png::PngEncoder};

type ApplyRequest = (Device, Controls);

struct AutofocusState {
    focus_min: f32,
    focus_max: f32,
    step: f32,
    total_samples: u32,
    sample_index: u32,
    best: Option<(f32, f32)>,
    phase: AutofocusPhase,
}

enum AutofocusPhase {
    Apply,
    Settle(Instant),
    Sample,
    Finalize,
}

impl AutofocusState {
    fn focus_at(&self, index: u32) -> f32 {
        (self.focus_min + index as f32 * self.step).min(self.focus_max)
    }
}

const SHARPEST_BURST_SIZE: usize = 16;
const SHARPEST_BURST_INTERVAL: Duration = Duration::from_millis(30);
const AUTOFOCUS_SETTLE: Duration = Duration::from_millis(150);
const AUTOFOCUS_SWEEP_STRIDE_MULTIPLIER: f32 = 4.0;
const AUTOFOCUS_CONTINUOUS_SAMPLES: f32 = 20.0;

const STREAM_CONFIG: StreamConfig = StreamConfig {
    resolution: Resolution {
        width: 1280,
        height: 720,
    },
    framerate: 60,
    pixel_format: PixelFormat::Bgra8,
};

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 820.0])
            .with_transparent(true),
        // The avatar renderer (inox2d-opengl) needs OpenGL.
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    eframe::run_native(
        "cameras egui demo",
        options,
        Box::new(|_cc| {
            let mut app = App::new();
            if let Some(path) = std::env::args_os().nth(1) {
                app.load_avatar(std::path::Path::new(&path));
            }
            Ok(Box::new(app))
        }),
    )
}

#[derive(Clone, Copy, PartialEq)]
enum SourceMode {
    Usb,
    Rtsp,
}

struct App {
    devices: Vec<Device>,
    selected_device: usize,
    source_mode: SourceMode,
    rtsp_url: String,
    rtsp_username: String,
    rtsp_password: String,

    stream: Option<CameraStream>,
    active: bool,
    status: String,
    last_capture: Option<String>,
    capabilities: Option<ControlCapabilities>,
    capabilities_error: Option<String>,
    pending_controls: Controls,
    current_controls: Controls,
    apply_tx: mpsc::Sender<ApplyRequest>,
    live_apply: bool,
    autofocus: Option<AutofocusState>,
    track_enabled: bool,
    track_flag: Arc<AtomicBool>,
    tracker_rx: Option<mpsc::Receiver<tracker::TrackResult>>,
    track_result: Option<tracker::TrackResult>,
    camera_fps: FpsCounter,
    track_fps: FpsCounter,
    face: face::FaceSolver,
    avatar: Option<avatar::Avatar>,
    avatar_target: Option<avatar::RenderTarget>,
    avatar_error: Option<String>,
    avatar_only: bool,
    background: Background,
    background_image: Option<egui::TextureHandle>,
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    discover_open: bool,
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    discover_subnet: String,
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    discover_error: Option<String>,
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    discovery: Option<DiscoverySession>,
}

impl App {
    fn new() -> Self {
        let devices = cameras::devices().unwrap_or_default();
        let apply_tx = spawn_apply_worker();
        Self {
            devices,
            selected_device: 0,
            source_mode: SourceMode::Usb,
            rtsp_url: "rtsp://127.0.0.1:8554/live".into(),
            rtsp_username: String::new(),
            rtsp_password: String::new(),
            stream: None,
            active: true,
            status: "Idle".into(),
            last_capture: None,
            capabilities: None,
            capabilities_error: None,
            pending_controls: Controls::default(),
            current_controls: Controls::default(),
            apply_tx,
            live_apply: false,
            autofocus: None,
            track_enabled: true,
            track_flag: Arc::new(AtomicBool::new(true)),
            tracker_rx: None,
            track_result: None,
            camera_fps: FpsCounter::default(),
            track_fps: FpsCounter::default(),
            face: face::FaceSolver::default(),
            avatar: None,
            avatar_target: None,
            avatar_error: None,
            avatar_only: false,
            background: Background::Color(egui::Color32::from_rgb(0, 255, 0)),
            background_image: None,
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            discover_open: false,
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            discover_subnet: "192.168.1.0/24".into(),
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            discover_error: None,
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            discovery: None,
        }
    }

    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn render_discover(&mut self, ui: &mut egui::Ui) {
        let running = self
            .discovery
            .as_ref()
            .map(|session| !session.done)
            .unwrap_or(false);
        ui.horizontal(|ui| {
            ui.label("Targets");
            ui.add(
                egui::TextEdit::singleline(&mut self.discover_subnet)
                    .hint_text("CIDR or host:port, comma-separated")
                    .desired_width(280.0),
            );
            let start_enabled = !running;
            if ui
                .add_enabled(start_enabled, egui::Button::new("Scan"))
                .clicked()
            {
                self.start_discover();
            }
            if ui
                .add_enabled(running, egui::Button::new("Cancel"))
                .clicked()
            {
                self.discovery = None;
            }
            if let Some(message) = &self.discover_error {
                ui.colored_label(egui::Color32::from_rgb(200, 80, 80), message);
            }
        });
        let clicked = self
            .discovery
            .as_ref()
            .and_then(|session| show_discovery(session, ui));
        if let Some(camera) = clicked {
            self.open_discovered(camera);
        }
    }

    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn start_discover(&mut self) {
        let (subnets, endpoints) = match parse_targets(&self.discover_subnet) {
            Ok(parsed) => parsed,
            Err(error) => {
                self.discover_error = Some(error);
                return;
            }
        };
        let config = DiscoverConfig {
            subnets,
            endpoints,
            ..Default::default()
        };
        match start_discovery(config) {
            Ok(session) => {
                self.discovery = Some(session);
                self.discover_error = None;
            }
            Err(error) => {
                self.discover_error = Some(format!("Start failed: {error}"));
            }
        }
    }

    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn open_discovered(&mut self, camera: cameras::discover::DiscoveredCamera) {
        let CameraSource::Rtsp { url, .. } = &camera.source else {
            return;
        };
        self.source_mode = SourceMode::Rtsp;
        self.rtsp_url = url.clone();
        self.rtsp_username.clear();
        self.rtsp_password.clear();
        self.connect();
    }

    fn queue_apply(&self) {
        let Some(device) = self.devices.get(self.selected_device) else {
            return;
        };
        let _ = self
            .apply_tx
            .send((device.clone(), self.pending_controls.clone()));
    }

    fn refresh_capabilities(&mut self) {
        self.capabilities = None;
        self.capabilities_error = None;
        self.pending_controls = Controls::default();
        self.current_controls = Controls::default();
        if self.source_mode != SourceMode::Usb {
            return;
        }
        let Some(device) = self.devices.get(self.selected_device) else {
            return;
        };
        match cameras::control_capabilities(device) {
            Ok(caps) => {
                self.current_controls = cameras::read_controls(device).unwrap_or_default();
                self.capabilities = Some(caps);
            }
            Err(error) => self.capabilities_error = Some(error.to_string()),
        }
    }

    fn start_autofocus(&mut self) {
        if self.autofocus.is_some() {
            return;
        }
        if self.stream.is_none() {
            return;
        }
        let Some(device) = self.devices.get(self.selected_device).cloned() else {
            return;
        };
        let Some(caps) = self.capabilities.clone() else {
            self.last_capture = Some("No capabilities".into());
            return;
        };
        let Some(focus_range) = caps.focus else {
            self.last_capture = Some("Camera has no controllable focus".into());
            return;
        };
        if let Err(error) = cameras::apply_controls(
            &device,
            &Controls {
                auto_focus: Some(false),
                auto_exposure: Some(false),
                auto_white_balance: Some(false),
                ..Default::default()
            },
        ) {
            self.last_capture = Some(format!("Autofocus setup failed: {error}"));
            return;
        }

        let step = if focus_range.step > 0.0 {
            focus_range.step * AUTOFOCUS_SWEEP_STRIDE_MULTIPLIER
        } else {
            ((focus_range.max - focus_range.min) / AUTOFOCUS_CONTINUOUS_SAMPLES).max(f32::EPSILON)
        };
        let span = (focus_range.max - focus_range.min).max(0.0);
        let total_samples = (span / step).floor() as u32 + 1;

        self.autofocus = Some(AutofocusState {
            focus_min: focus_range.min,
            focus_max: focus_range.max,
            step,
            total_samples,
            sample_index: 0,
            best: None,
            phase: AutofocusPhase::Apply,
        });
        self.last_capture = Some(format!("Autofocus: sweeping {total_samples} samples"));
    }

    fn tick_autofocus(&mut self, ctx: &egui::Context) {
        let Some(mut state) = self.autofocus.take() else {
            return;
        };
        let Some(stream) = self.stream.as_ref() else {
            self.autofocus = None;
            self.last_capture = Some("Autofocus cancelled (stream closed)".into());
            return;
        };
        let Some(device) = self.devices.get(self.selected_device).cloned() else {
            self.autofocus = None;
            return;
        };

        match state.phase {
            AutofocusPhase::Apply => {
                let focus_value = state.focus_at(state.sample_index);
                if cameras::apply_controls(
                    &device,
                    &Controls {
                        focus: Some(focus_value),
                        ..Default::default()
                    },
                )
                .is_err()
                {
                    self.last_capture = Some("Autofocus: apply failed".into());
                    self.autofocus = None;
                    return;
                }
                state.phase = AutofocusPhase::Settle(Instant::now());
                self.autofocus = Some(state);
                ctx.request_repaint_after(AUTOFOCUS_SETTLE);
            }
            AutofocusPhase::Settle(start) => {
                if start.elapsed() >= AUTOFOCUS_SETTLE {
                    state.phase = AutofocusPhase::Sample;
                }
                self.autofocus = Some(state);
                ctx.request_repaint_after(Duration::from_millis(16));
            }
            AutofocusPhase::Sample => {
                let focus_value = state.focus_at(state.sample_index);
                if let Some(frame) = capture_frame(&stream.pump) {
                    let roi = Rect {
                        x: frame.width / 4,
                        y: frame.height / 4,
                        width: frame.width / 2,
                        height: frame.height / 2,
                    };
                    let variance = analysis::blur_variance_in(&frame, roi);
                    if state.best.is_none_or(|(_, score)| variance > score) {
                        state.best = Some((focus_value, variance));
                    }
                }
                state.sample_index += 1;
                if state.sample_index >= state.total_samples {
                    state.phase = AutofocusPhase::Finalize;
                } else {
                    state.phase = AutofocusPhase::Apply;
                }
                self.autofocus = Some(state);
                ctx.request_repaint();
            }
            AutofocusPhase::Finalize => match state.best {
                Some((winner, score)) => {
                    let _ = cameras::apply_controls(
                        &device,
                        &Controls {
                            focus: Some(winner),
                            ..Default::default()
                        },
                    );
                    self.pending_controls.focus = Some(winner);
                    self.pending_controls.auto_focus = Some(false);
                    self.last_capture = Some(format!(
                        "Autofocus locked at {winner:.2} (variance {score:.1})"
                    ));
                    self.autofocus = None;
                }
                None => {
                    self.last_capture = Some("Autofocus: no samples captured".into());
                    self.autofocus = None;
                }
            },
        }
    }

    fn draw_track_overlay(&self, ui: &mut egui::Ui, image_rect: egui::Rect) {
        let Some(result) = &self.track_result else {
            return;
        };
        let Some(head) = &result.head else {
            return;
        };
        let scale_x = image_rect.width() / result.frame_w as f32;
        let scale_y = image_rect.height() / result.frame_h as f32;
        let to_screen = |x: f32, y: f32| {
            egui::pos2(
                image_rect.left() + x * scale_x,
                image_rect.top() + y * scale_y,
            )
        };
        let painter = ui.painter();
        let stroke = egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(80, 200, 120));
        let p1 = to_screen(head.rect[0], head.rect[1]);
        let p2 = to_screen(head.rect[2], head.rect[3]);
        painter.rect_stroke(
            egui::Rect::from_min_max(p1, p2),
            0.0,
            stroke,
            egui::StrokeKind::Outside,
        );
        for point in &head.points {
            painter.circle_filled(
                to_screen(point[0], point[1]),
                3.0,
                egui::Color32::from_rgb(255, 0, 255),
            );
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
            .add_filter("Inochi2D puppet", &["inp", "inx"])
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

    fn show_camera(&self, ui: &mut egui::Ui) {
        if let Some(stream) = &self.stream {
            if let Some(image_rect) = show_stream(stream, ui) {
                self.draw_track_overlay(ui, image_rect);
            }
        } else {
            ui.centered_and_justified(|ui| {
                ui.label("Pick a source, configure it, and press Connect.");
            });
        }
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
            Background::Transparent => {}
        }
        let Some(avatar) = self.avatar.as_mut() else {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Load an Inochi2D avatar (.inp / .inx)",
                egui::FontId::proportional(16.0),
                ui.visuals().text_color(),
            );
            return;
        };
        let Some(gl) = frame.gl().cloned() else {
            self.avatar_error = Some("Avatar rendering needs the OpenGL renderer".into());
            return;
        };

        let pixels_per_point = ui.ctx().pixels_per_point();
        let size = [
            (rect.width() * pixels_per_point).round() as u32,
            (rect.height() * pixels_per_point).round() as u32,
        ];
        // Scroll (or pinch) to zoom toward the cursor, drag to move.
        if let Some(pointer) = response.hover_pos() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = (scroll * 0.002).exp() * pinch;
            if factor != 1.0 {
                let anchor = (pointer - rect.center()) * pixels_per_point;
                avatar.zoom_at(factor, glam::Vec2::new(anchor.x, anchor.y), size[1]);
            }
        }
        if response.dragged() {
            let delta = response.drag_delta() * pixels_per_point;
            avatar.pan(glam::Vec2::new(delta.x, delta.y), size[1]);
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
        if let Err(error) = avatar.render(&gl, target, &self.face.state) {
            self.avatar_error = Some(format!("Avatar render: {error}"));
            self.avatar = None;
            return;
        }
        // GL textures are bottom-up.
        painter.image(
            target.texture_id,
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 1.0), egui::pos2(1.0, 0.0)),
            egui::Color32::WHITE,
        );
    }

    fn avatar_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Avatar");
        ui.horizontal(|ui| {
            if ui.button("Load avatar…").clicked() {
                self.pick_avatar();
            }
            if let Some(avatar) = &self.avatar {
                ui.label(&avatar.name);
            }
        });
        if let Some(error) = &self.avatar_error {
            ui.colored_label(egui::Color32::from_rgb(200, 80, 80), error);
        }
        ui.separator();

        let calibrate_label = if self.face.is_calibrating() {
            "Calibrating… look straight, neutral face"
        } else {
            "Calibrate neutral face"
        };
        if ui.button(calibrate_label).clicked() {
            self.face.calibrate();
        }
        ui.checkbox(&mut self.face.mirror, "Mirror");
        if ui.button("Avatar only (Esc to exit)").clicked() {
            self.avatar_only = true;
        }

        ui.separator();
        ui.label("Background");
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            let is_color = matches!(self.background, Background::Color(_));
            if ui.selectable_label(is_color, "Color").clicked() && !is_color {
                self.background = Background::Color(egui::Color32::from_rgb(0, 255, 0));
            }
            if ui
                .selectable_label(matches!(self.background, Background::Image), "Image…")
                .clicked()
            {
                self.pick_background_image(&ctx);
            }
            if ui
                .selectable_label(matches!(self.background, Background::Transparent), "Transparent")
                .clicked()
            {
                self.background = Background::Transparent;
            }
        });
        if let Background::Color(color) = &mut self.background {
            ui.horizontal(|ui| {
                ui.color_edit_button_srgba(color);
                if ui.small_button("Chroma green").clicked() {
                    *color = egui::Color32::from_rgb(0, 255, 0);
                }
            });
        }

        if let Some(avatar) = &mut self.avatar {
            ui.separator();
            ui.add(egui::Slider::new(&mut avatar.zoom, 0.1..=20.0).logarithmic(true).text("Zoom"));
            ui.horizontal(|ui| {
                if ui.small_button("Face").clicked() {
                    avatar.set_view(avatar::FACE_VIEW);
                }
                if ui.small_button("Bust").clicked() {
                    avatar.set_view(avatar::BUST_VIEW);
                }
                if ui.small_button("Full body").clicked() {
                    avatar.set_view(avatar::FULL_BODY_VIEW);
                }
            });
            ui.weak("Scroll or pinch to zoom toward the cursor, drag to move.");
            ui.collapsing(format!("Parameters ({})", avatar.params.len()), |ui| {
                egui::Grid::new("avatar_params").striped(true).show(ui, |ui| {
                    for param in &avatar.params {
                        ui.label(&param.name);
                        match avatar.values.iter().find(|(name, _)| *name == param.name) {
                            Some((_, value)) => ui.monospace(format!("{:+.2} {:+.2}", value.x, value.y)),
                            None => ui.weak("not tracked"),
                        };
                        ui.end_row();
                    }
                });
            });
        }
    }

    fn build_source(&self) -> Option<CameraSource> {
        match self.source_mode {
            SourceMode::Usb => self
                .devices
                .get(self.selected_device)
                .cloned()
                .map(CameraSource::Usb),
            SourceMode::Rtsp => {
                build_rtsp_source(&self.rtsp_url, &self.rtsp_username, &self.rtsp_password)
            }
        }
    }

    fn connect(&mut self) {
        let Some(source) = self.build_source() else {
            self.status = match self.source_mode {
                SourceMode::Usb => "No camera selected".into(),
                SourceMode::Rtsp => "RTSP URL is empty".into(),
            };
            return;
        };
        self.status = format!("Connecting to {}...", source_label(&source));
        match cameras::open_source(source.clone(), STREAM_CONFIG) {
            Ok(camera) => {
                let (tracker_tx, tracker_rx) = tracker::spawn_tracker();
                self.stream = Some(spawn_stream(camera, tracker_tx, self.track_flag.clone()));
                self.active = true;
                self.tracker_rx = Some(tracker_rx);
                self.track_result = None;
                self.status = format!("Streaming: {}", source_label(&source));
                self.refresh_capabilities();
            }
            Err(error) => {
                self.stream = None;
                self.status = format!("Open failed: {error}");
            }
        }
    }

    fn disconnect(&mut self) {
        self.stream = None;
        self.tracker_rx = None;
        self.track_result = None;
        self.status = "Disconnected".into();
    }

    fn refresh_devices(&mut self) {
        self.devices = cameras::devices().unwrap_or_default();
        if self.selected_device >= self.devices.len() {
            self.selected_device = 0;
        }
    }

    fn snapshot(&mut self) {
        let Some(stream) = &self.stream else {
            return;
        };
        let Some(frame) = capture_frame(&stream.pump) else {
            self.last_capture = Some("Capture failed".into());
            return;
        };
        let variance = analysis::blur_variance(&frame);
        let path = format!("snapshot-{}.png", unix_timestamp());
        match save_png(&frame, &path) {
            Ok(()) => self.last_capture = Some(format!("Wrote {path} (sharpness {variance:.1})")),
            Err(error) => self.last_capture = Some(format!("Save failed: {error}")),
        }
    }

    fn reset_to_defaults(&mut self) {
        let Some(device) = self.devices.get(self.selected_device) else {
            return;
        };
        match cameras::reset_to_defaults(device) {
            Ok(()) => {
                self.current_controls = cameras::read_controls(device).unwrap_or_default();
                self.pending_controls = Controls::default();
                self.last_capture = Some("Controls reset to factory defaults".into());
            }
            Err(error) => {
                self.last_capture = Some(format!("Reset failed: {error}"));
            }
        }
    }

    fn analyze_frame(&mut self) {
        let Some(stream) = &self.stream else {
            return;
        };
        let Some(frame) = capture_frame(&stream.pump) else {
            self.last_capture = Some("Capture failed".into());
            return;
        };
        let variance = analysis::blur_variance(&frame);
        self.last_capture = Some(format!(
            "Current frame sharpness: {variance:.1} (higher = sharper; calibrate per source)"
        ));
    }

    fn pick_sharpest(&mut self) {
        let Some(stream) = &self.stream else {
            return;
        };
        let mut ring = analysis::ring_new(SHARPEST_BURST_SIZE);
        for _ in 0..SHARPEST_BURST_SIZE {
            if let Some(frame) = capture_frame(&stream.pump) {
                analysis::ring_push(&mut ring, frame);
            }
            sleep(SHARPEST_BURST_INTERVAL);
        }
        let Some(sharpest) = analysis::take_sharpest(&ring) else {
            self.last_capture = Some("Burst captured no frames".into());
            return;
        };
        let variance = analysis::blur_variance(&sharpest);
        let path = format!("sharpest-{}.png", unix_timestamp());
        match save_png(&sharpest, &path) {
            Ok(()) => {
                self.last_capture = Some(format!(
                    "Wrote {path} (sharpness {variance:.1}, best of {})",
                    ring.frames.len()
                ))
            }
            Err(error) => self.last_capture = Some(format!("Save failed: {error}")),
        }
    }
}

impl eframe::App for App {
    fn clear_color(&self, visuals: &egui::Visuals) -> [f32; 4] {
        if self.avatar_only && matches!(self.background, Background::Transparent) {
            [0.0; 4]
        } else {
            visuals.panel_fill.to_normalized_gamma_f32()
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if self.autofocus.is_some() {
            self.tick_autofocus(&ctx);
        }
        #[cfg(any(target_os = "macos", target_os = "windows"))]
        if let Some(session) = self.discovery.as_mut() {
            poll_discovery(session);
        }
        if let Some(stream) = &mut self.stream
            && update_texture(stream, &ctx)
        {
            self.camera_fps.tick();
        }
        self.track_flag.store(self.track_enabled, Ordering::Relaxed);
        if let Some(rx) = &self.tracker_rx {
            while let Ok(result) = rx.try_recv() {
                self.track_fps.tick();
                self.face.update(&result);
                self.track_result = Some(result);
            }
        }
        if !self.track_enabled {
            self.track_result = None;
        }

        if self.avatar_only {
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.avatar_only = false;
            }
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show_inside(ui, |ui| self.show_avatar(ui, frame));
            ctx.request_repaint();
            return;
        }

        egui::Panel::top("bar").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("cameras egui demo");
                ui.separator();
                ui.selectable_value(&mut self.source_mode, SourceMode::Usb, "USB");
                ui.selectable_value(&mut self.source_mode, SourceMode::Rtsp, "RTSP");
                ui.separator();
                ui.checkbox(&mut self.track_enabled, "Face tracking");
                #[cfg(any(target_os = "macos", target_os = "windows"))]
                {
                    ui.separator();
                    ui.toggle_value(&mut self.discover_open, "Discover");
                }
            });

            #[cfg(any(target_os = "macos", target_os = "windows"))]
            if self.discover_open {
                self.render_discover(ui);
            }

            let mut selection_changed = false;
            ui.horizontal(|ui| match self.source_mode {
                SourceMode::Usb => {
                    let labels: Vec<String> = if self.devices.is_empty() {
                        vec!["No cameras".into()]
                    } else {
                        self.devices.iter().map(|d| d.name.clone()).collect()
                    };
                    egui::ComboBox::from_label("Camera")
                        .selected_text(
                            labels
                                .get(self.selected_device)
                                .cloned()
                                .unwrap_or_else(|| "No cameras".into()),
                        )
                        .show_ui(ui, |ui| {
                            for (index, label) in labels.iter().enumerate() {
                                let previous = self.selected_device;
                                let response =
                                    ui.selectable_value(&mut self.selected_device, index, label);
                                if response.clicked() && self.selected_device != previous {
                                    selection_changed = true;
                                }
                            }
                        });
                    if ui.button("Refresh").clicked() {
                        self.refresh_devices();
                        selection_changed = true;
                    }
                }
                SourceMode::Rtsp => {
                    let _ = &mut selection_changed;
                    ui.label("URL");
                    ui.text_edit_singleline(&mut self.rtsp_url);
                    ui.label("User");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.rtsp_username).desired_width(120.0),
                    );
                    ui.label("Pass");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.rtsp_password)
                            .password(true)
                            .desired_width(120.0),
                    );
                }
            });

            if selection_changed {
                self.refresh_capabilities();
            }

            ui.horizontal(|ui| {
                if ui.button("Connect").clicked() {
                    self.connect();
                }
                if self.stream.is_some() && ui.button("Disconnect").clicked() {
                    self.disconnect();
                }
                ui.separator();
                ui.label(&self.status);
                if self.stream.is_some() {
                    ui.separator();
                    ui.label(format!("camera {:.0} fps", self.camera_fps.fps));
                    if self.track_enabled {
                        let ms = self.track_result.as_ref().map_or(0.0, |r| r.ms);
                        ui.label(format!(
                            "tracking {:.0} fps ({ms:.1} ms)",
                            self.track_fps.fps
                        ));
                    }
                }
                if let Some(last) = &self.last_capture {
                    ui.separator();
                    ui.label(last);
                }
            });
        });

        if self.stream.is_some() {
            egui::Panel::bottom("actions").show_inside(ui, |ui| {
                ui.horizontal(|ui| {
                    let preview_label = if self.active {
                        "Preview: on"
                    } else {
                        "Preview: off"
                    };
                    if ui.button(preview_label).clicked() {
                        self.active = !self.active;
                        if let Some(stream) = &self.stream {
                            set_active(&stream.pump, self.active);
                        }
                    }
                    if ui.button("Take picture").clicked() {
                        self.snapshot();
                    }
                    if ui.button("Analyze sharpness").clicked() {
                        self.analyze_frame();
                    }
                    if ui
                        .button(format!("Pick sharpest ({SHARPEST_BURST_SIZE}-frame burst)"))
                        .clicked()
                    {
                        self.pick_sharpest();
                    }
                });
                ui.collapsing("Capabilities", |ui| {
                    render_capabilities_panel(
                        ui,
                        self.capabilities.as_ref(),
                        self.capabilities_error.as_deref(),
                    );
                });
                let has_controls_ui = self.capabilities.is_some();
                if has_controls_ui {
                    ui.collapsing("Controls", |ui| {
                        let changed = render_controls_editor(
                            ui,
                            self.capabilities.as_ref().unwrap(),
                            &self.current_controls,
                            &mut self.pending_controls,
                        );
                        if changed && self.live_apply {
                            self.queue_apply();
                        }
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut self.live_apply, "Apply live");
                            if !self.live_apply && ui.button("Apply").clicked() {
                                self.queue_apply();
                            }
                            if ui.button("Reset to defaults").clicked() {
                                self.reset_to_defaults();
                            }
                            let has_focus = self
                                .capabilities
                                .as_ref()
                                .and_then(|caps| caps.focus.as_ref())
                                .is_some();
                            if has_focus {
                                let autofocus_label = match self.autofocus.as_ref() {
                                    Some(state) => format!(
                                        "Autofocusing {}/{}",
                                        state.sample_index.min(state.total_samples),
                                        state.total_samples
                                    ),
                                    None => "Autofocus (sweep)".to_string(),
                                };
                                let autofocus_button = egui::Button::new(autofocus_label);
                                if ui
                                    .add_enabled(self.autofocus.is_none(), autofocus_button)
                                    .clicked()
                                {
                                    self.start_autofocus();
                                }
                            }
                        });
                    });
                }
            });
        }

        egui::Panel::right("avatar")
            .resizable(true)
            .default_size(260.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.avatar_panel(ui));
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let (left, right) = ui.available_rect_before_wrap().split_left_right_at_fraction(0.5);
            ui.scope_builder(egui::UiBuilder::new().max_rect(left), |ui| self.show_camera(ui));
            ui.scope_builder(egui::UiBuilder::new().max_rect(right), |ui| {
                self.show_avatar(ui, frame)
            });
        });

        ctx.request_repaint();
    }
}

/// Camera pump plus the egui texture it feeds. Frames are converted to a
/// `ColorImage` on the pump thread, so the UI thread only uploads them.
struct CameraStream {
    pump: Pump,
    latest: Arc<Mutex<Option<egui::ColorImage>>>,
    texture: Option<egui::TextureHandle>,
}

fn spawn_stream(
    camera: Camera,
    tracker_tx: mpsc::Sender<cameras::Frame>,
    track: Arc<AtomicBool>,
) -> CameraStream {
    let latest = Arc::new(Mutex::new(None));
    let slot = latest.clone();
    let pump = cameras::pump::spawn(camera, move |frame| {
        if track.load(Ordering::Relaxed) {
            let _ = tracker_tx.send(frame.clone());
        }
        if let Some(image) = frame_to_color_image(&frame) {
            *slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(image);
        }
    });
    CameraStream {
        pump,
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
    let mut pixels = Vec::with_capacity(width * height);
    for row in 0..height {
        let start = row * stride;
        pixels.extend(
            data[start..start + width * 4]
                .chunks_exact(4)
                .map(|px| egui::Color32::from_rgb(px[2], px[1], px[0])),
        );
    }
    Some(egui::ColorImage::new([width, height], pixels))
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

/// Draw the stream aspect-fit into the available space; returns the image rect.
fn show_stream(stream: &CameraStream, ui: &mut egui::Ui) -> Option<egui::Rect> {
    let texture = stream.texture.as_ref()?;
    let aspect = texture.aspect_ratio();
    let available = ui.available_size();
    let width = available.x.min(available.y * aspect);
    let size = egui::vec2(width, width / aspect);
    Some(ui.image((texture.id(), size)).rect)
}

#[derive(Clone, Copy)]
enum Background {
    Color(egui::Color32),
    Image,
    Transparent,
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

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn build_rtsp_source(url: &str, username: &str, password: &str) -> Option<CameraSource> {
    let url = url.trim();
    if url.is_empty() {
        return None;
    }
    let username = username.trim();
    let credentials = if username.is_empty() && password.is_empty() {
        None
    } else {
        Some(Credentials {
            username: username.to_string(),
            password: password.to_string(),
        })
    };
    Some(CameraSource::Rtsp {
        url: url.to_string(),
        credentials,
    })
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn build_rtsp_source(_url: &str, _username: &str, _password: &str) -> Option<CameraSource> {
    None
}

fn source_label(source: &CameraSource) -> String {
    match source {
        CameraSource::Usb(device) => device.name.clone(),
        #[cfg(any(target_os = "macos", target_os = "windows"))]
        CameraSource::Rtsp { url, .. } => url.clone(),
    }
}

fn save_png(frame: &cameras::Frame, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let rgba = cameras::to_rgba8(frame)?;
    let file = std::fs::File::create(path)?;
    let encoder = PngEncoder::new(std::io::BufWriter::new(file));
    encoder.write_image(&rgba, frame.width, frame.height, ExtendedColorType::Rgba8)?;
    Ok(())
}

const SUPPORTED_MARK: &str = "✓";
const UNSUPPORTED_MARK: &str = "×";

fn render_capabilities_panel(
    ui: &mut egui::Ui,
    capabilities: Option<&ControlCapabilities>,
    error: Option<&str>,
) {
    if let Some(message) = error {
        ui.colored_label(egui::Color32::from_rgb(200, 80, 80), message);
        return;
    }
    let Some(caps) = capabilities else {
        ui.label("Connect to a USB camera to view its control capabilities.");
        return;
    };
    let rows: [(ControlKind, CapabilityDisplay); 16] = [
        (
            ControlKind::Focus,
            CapabilityDisplay::Range(caps.focus.as_ref()),
        ),
        (
            ControlKind::AutoFocus,
            CapabilityDisplay::Bool(caps.auto_focus),
        ),
        (
            ControlKind::Exposure,
            CapabilityDisplay::Range(caps.exposure.as_ref()),
        ),
        (
            ControlKind::AutoExposure,
            CapabilityDisplay::Bool(caps.auto_exposure),
        ),
        (
            ControlKind::WhiteBalanceTemperature,
            CapabilityDisplay::Range(caps.white_balance_temperature.as_ref()),
        ),
        (
            ControlKind::AutoWhiteBalance,
            CapabilityDisplay::Bool(caps.auto_white_balance),
        ),
        (
            ControlKind::Brightness,
            CapabilityDisplay::Range(caps.brightness.as_ref()),
        ),
        (
            ControlKind::Contrast,
            CapabilityDisplay::Range(caps.contrast.as_ref()),
        ),
        (
            ControlKind::Saturation,
            CapabilityDisplay::Range(caps.saturation.as_ref()),
        ),
        (
            ControlKind::Sharpness,
            CapabilityDisplay::Range(caps.sharpness.as_ref()),
        ),
        (
            ControlKind::Gain,
            CapabilityDisplay::Range(caps.gain.as_ref()),
        ),
        (
            ControlKind::BacklightCompensation,
            CapabilityDisplay::Range(caps.backlight_compensation.as_ref()),
        ),
        (
            ControlKind::PowerLineFrequency,
            CapabilityDisplay::PowerLine(caps.power_line_frequency.is_some()),
        ),
        (
            ControlKind::Pan,
            CapabilityDisplay::Range(caps.pan.as_ref()),
        ),
        (
            ControlKind::Tilt,
            CapabilityDisplay::Range(caps.tilt.as_ref()),
        ),
        (
            ControlKind::Zoom,
            CapabilityDisplay::Range(caps.zoom.as_ref()),
        ),
    ];
    egui::Grid::new("capabilities_grid")
        .num_columns(3)
        .spacing([10.0, 4.0])
        .show(ui, |ui| {
            for (key, display) in rows {
                render_capability_row(ui, key, display);
                ui.end_row();
            }
        });
}

enum CapabilityDisplay<'a> {
    Range(Option<&'a ControlRange>),
    Bool(Option<bool>),
    PowerLine(bool),
}

fn render_capability_row(ui: &mut egui::Ui, key: ControlKind, display: CapabilityDisplay) {
    let green = egui::Color32::from_rgb(80, 200, 120);
    let red = egui::Color32::from_rgb(200, 80, 80);
    let (supported, detail) = match display {
        CapabilityDisplay::Range(Some(range)) => (true, format_range(range)),
        CapabilityDisplay::Range(None) => (false, "not supported".to_string()),
        CapabilityDisplay::Bool(Some(true)) => (true, "auto toggle supported".to_string()),
        CapabilityDisplay::Bool(Some(false)) => (true, "manual only".to_string()),
        CapabilityDisplay::Bool(None) => (false, "not supported".to_string()),
        CapabilityDisplay::PowerLine(true) => (true, "menu supported".to_string()),
        CapabilityDisplay::PowerLine(false) => (false, "not supported".to_string()),
    };
    let mark = if supported {
        SUPPORTED_MARK
    } else {
        UNSUPPORTED_MARK
    };
    let mark_color = if supported { green } else { red };
    let caveat = if supported { None } else { key.caveat() };
    let mark_response = ui.colored_label(mark_color, mark);
    let name_response = ui.label(key.label());
    let detail_response = ui.label(detail);
    if let Some(caveat) = caveat {
        mark_response.on_hover_text(caveat);
        name_response.on_hover_text(caveat);
        detail_response.on_hover_text(caveat);
    }
}

fn auto_lock_hover(paired_label: &str) -> String {
    format!(
        "Disabled because {paired_label} is set to 'auto'. Change {paired_label} to 'leave' or \
         'manual' to enable this slider."
    )
}

fn render_controls_editor(
    ui: &mut egui::Ui,
    capabilities: &ControlCapabilities,
    current: &Controls,
    pending: &mut Controls,
) -> bool {
    let mut changed = false;
    egui::Grid::new("controls_editor_grid")
        .num_columns(2)
        .spacing([12.0, 6.0])
        .show(ui, |ui| {
            changed |= numeric_slider_row(
                ui,
                ControlKind::Focus,
                capabilities.focus.as_ref(),
                current.focus,
                pending.auto_focus == Some(true),
                Some(ControlKind::AutoFocus),
                &mut pending.focus,
            );
            changed |= auto_tri_state_row(
                ui,
                ControlKind::AutoFocus,
                capabilities.auto_focus,
                &mut pending.auto_focus,
            );
            changed |= numeric_slider_row(
                ui,
                ControlKind::Exposure,
                capabilities.exposure.as_ref(),
                current.exposure,
                pending.auto_exposure == Some(true),
                Some(ControlKind::AutoExposure),
                &mut pending.exposure,
            );
            changed |= auto_tri_state_row(
                ui,
                ControlKind::AutoExposure,
                capabilities.auto_exposure,
                &mut pending.auto_exposure,
            );
            changed |= numeric_slider_row(
                ui,
                ControlKind::WhiteBalanceTemperature,
                capabilities.white_balance_temperature.as_ref(),
                current.white_balance_temperature,
                pending.auto_white_balance == Some(true),
                Some(ControlKind::AutoWhiteBalance),
                &mut pending.white_balance_temperature,
            );
            changed |= auto_tri_state_row(
                ui,
                ControlKind::AutoWhiteBalance,
                capabilities.auto_white_balance,
                &mut pending.auto_white_balance,
            );
            changed |= numeric_slider_row(
                ui,
                ControlKind::Brightness,
                capabilities.brightness.as_ref(),
                current.brightness,
                false,
                None,
                &mut pending.brightness,
            );
            changed |= numeric_slider_row(
                ui,
                ControlKind::Contrast,
                capabilities.contrast.as_ref(),
                current.contrast,
                false,
                None,
                &mut pending.contrast,
            );
            changed |= numeric_slider_row(
                ui,
                ControlKind::Saturation,
                capabilities.saturation.as_ref(),
                current.saturation,
                false,
                None,
                &mut pending.saturation,
            );
            changed |= numeric_slider_row(
                ui,
                ControlKind::Sharpness,
                capabilities.sharpness.as_ref(),
                current.sharpness,
                false,
                None,
                &mut pending.sharpness,
            );
            changed |= numeric_slider_row(
                ui,
                ControlKind::Gain,
                capabilities.gain.as_ref(),
                current.gain,
                false,
                None,
                &mut pending.gain,
            );
            changed |= numeric_slider_row(
                ui,
                ControlKind::BacklightCompensation,
                capabilities.backlight_compensation.as_ref(),
                current.backlight_compensation,
                false,
                None,
                &mut pending.backlight_compensation,
            );
            changed |= numeric_slider_row(
                ui,
                ControlKind::Pan,
                capabilities.pan.as_ref(),
                current.pan,
                false,
                None,
                &mut pending.pan,
            );
            changed |= numeric_slider_row(
                ui,
                ControlKind::Tilt,
                capabilities.tilt.as_ref(),
                current.tilt,
                false,
                None,
                &mut pending.tilt,
            );
            changed |= numeric_slider_row(
                ui,
                ControlKind::Zoom,
                capabilities.zoom.as_ref(),
                current.zoom,
                false,
                None,
                &mut pending.zoom,
            );
        });
    changed
}

fn numeric_slider_row(
    ui: &mut egui::Ui,
    key: ControlKind,
    range: Option<&ControlRange>,
    current: Option<f32>,
    locked_by_auto: bool,
    paired_auto: Option<ControlKind>,
    pending: &mut Option<f32>,
) -> bool {
    let mut changed = false;
    ui.label(key.label());
    match range {
        Some(range) => {
            let fallback_value = current.unwrap_or(range.default);
            ui.horizontal(|ui| {
                let lock_hover = if locked_by_auto {
                    paired_auto.map(|auto| auto_lock_hover(auto.label()))
                } else {
                    None
                };
                let mut enabled = pending.is_some();
                let checkbox_enabled = !locked_by_auto;
                let checkbox_response =
                    ui.add_enabled(checkbox_enabled, egui::Checkbox::new(&mut enabled, ""));
                if checkbox_response.changed() {
                    *pending = if enabled { Some(fallback_value) } else { None };
                    changed = true;
                }
                if let Some(hover) = lock_hover.as_deref() {
                    checkbox_response.on_hover_text(hover);
                }
                let slider_enabled = enabled && !locked_by_auto;
                let mut value = pending.unwrap_or(fallback_value);
                let slider_response = ui.add_enabled(
                    slider_enabled,
                    egui::Slider::new(&mut value, range.min..=range.max),
                );
                if slider_response.changed() {
                    *pending = Some(value);
                    changed = true;
                }
                if let Some(hover) = lock_hover.as_deref() {
                    slider_response.on_hover_text(hover);
                }
                if locked_by_auto {
                    ui.weak("auto on");
                } else if let Some(current_value) = current {
                    ui.weak(format!("now {current_value:.2}"));
                }
            });
        }
        None => {
            let response = ui.weak("not supported");
            if let Some(caveat) = key.caveat() {
                response.on_hover_text(caveat);
            }
        }
    }
    ui.end_row();
    changed
}

fn auto_tri_state_row(
    ui: &mut egui::Ui,
    key: ControlKind,
    capability: Option<bool>,
    pending: &mut Option<bool>,
) -> bool {
    ui.label(key.label());
    if capability != Some(true) {
        let response = ui.weak("not supported");
        if let Some(caveat) = key.caveat() {
            response.on_hover_text(caveat);
        }
        ui.end_row();
        return false;
    }
    let mut changed = false;
    ui.horizontal(|ui| {
        changed |= ui.radio_value(pending, None, "leave").changed();
        changed |= ui.radio_value(pending, Some(false), "manual").changed();
        changed |= ui.radio_value(pending, Some(true), "auto").changed();
    });
    ui.end_row();
    changed
}

fn spawn_apply_worker() -> mpsc::Sender<ApplyRequest> {
    let (tx, rx) = mpsc::channel::<ApplyRequest>();
    std::thread::Builder::new()
        .name("egui-demo-controls".into())
        .spawn(move || worker_loop(rx))
        .expect("spawn controls worker");
    tx
}

fn worker_loop(rx: mpsc::Receiver<ApplyRequest>) {
    while let Ok(mut latest) = rx.recv() {
        while let Ok(newer) = rx.try_recv() {
            latest = newer;
        }
        let (device, controls) = latest;
        let _ = cameras::apply_controls(&device, &controls);
    }
}

fn format_range(range: &ControlRange) -> String {
    if range.step > 0.0 {
        format!(
            "{:.0}..{:.0} (step {:.0}, default {:.0})",
            range.min, range.max, range.step, range.default
        )
    } else {
        format!(
            "{:.2}..{:.2} (default {:.2})",
            range.min, range.max, range.default
        )
    }
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn parse_targets(input: &str) -> Result<(Vec<ipnet::IpNet>, Vec<std::net::SocketAddr>), String> {
    let mut subnets = Vec::new();
    let mut endpoints = Vec::new();
    for raw in input.split(',') {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        if let Ok(endpoint) = token.parse::<std::net::SocketAddr>() {
            endpoints.push(endpoint);
        } else if let Ok(net) = token.parse::<ipnet::IpNet>() {
            subnets.push(net);
        } else {
            return Err(format!("could not parse `{token}` as CIDR or host:port"));
        }
    }
    if subnets.is_empty() && endpoints.is_empty() {
        return Err("enter at least one CIDR or host:port".into());
    }
    Ok((subnets, endpoints))
}
