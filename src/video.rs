//! Video backgrounds, decoded with macOS's own AVFoundation: any file
//! QuickTime plays (MP4, MOV, M4V; H.264, HEVC, ProRes) works without extra
//! libraries. A thread decodes frames at the video's own pace, looping, and
//! keeps the newest one for the renderer to pick up.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::time::Duration;

/// File extensions offered for video backgrounds.
pub const VIDEO_EXTENSIONS: [&str; 3] = ["mp4", "mov", "m4v"];
/// Frames wider than this are scaled down while decoding; a background
/// doesn't need more, and 4K frames would cost a lot of CPU per frame.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const MAX_WIDTH: usize = 1920;

type FrameSlot = Arc<Mutex<Option<Arc<image::RgbaImage>>>>;

pub struct VideoBackground {
    latest: FrameSlot,
    stop: Arc<AtomicBool>,
}

impl VideoBackground {
    /// Start decoding `path`. Fails if the file can't be read as a video.
    pub fn open(path: &Path) -> Result<Self, String> {
        let latest: FrameSlot = Arc::new(Mutex::new(None));
        let stop = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = mpsc::channel();
        let (path, slot, stop_flag) = (path.to_path_buf(), latest.clone(), stop.clone());
        std::thread::Builder::new()
            .name("video-background".into())
            .spawn(move || decode(&path, &slot, &stop_flag, ready_tx))
            .map_err(|error| format!("video thread: {error}"))?;
        match ready_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => Ok(Self { latest, stop }),
            Ok(Err(error)) => Err(error),
            Err(_) => Err("the video took too long to open".into()),
        }
    }

    /// The newest decoded frame, if one has arrived.
    pub fn frame(&self) -> Option<Arc<image::RgbaImage>> {
        self.latest.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }
}

impl Drop for VideoBackground {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

#[cfg(not(target_os = "macos"))]
fn decode(_path: &Path, _slot: &FrameSlot, _stop: &AtomicBool, ready: mpsc::Sender<Result<(), String>>) {
    let _ = ready.send(Err("video backgrounds need macOS".into()));
}

#[cfg(target_os = "macos")]
fn decode(path: &Path, slot: &FrameSlot, stop: &AtomicBool, ready: mpsc::Sender<Result<(), String>>) {
    // SAFETY: everything in `avfoundation` follows the documented signatures
    // of the AVFoundation, CoreMedia and CoreVideo calls it makes.
    unsafe { avfoundation::decode(path, slot, stop, ready) }
}

#[cfg(target_os = "macos")]
mod avfoundation {
    use std::ffi::{CStr, CString, c_void};
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, PoisonError, mpsc};
    use std::time::{Duration, Instant};

    use super::{FrameSlot, MAX_WIDTH};
    use crate::platform::objc::{self, Id};

    /// `kCVPixelFormatType_32BGRA`: frames are decoded to 8-bit BGRA.
    const PIXEL_FORMAT_BGRA: u32 = 0x4247_5241;
    /// `kCVPixelBufferLock_ReadOnly`.
    const LOCK_READ_ONLY: u64 = 1;
    /// `AVAssetReaderStatusCompleted`.
    const READER_COMPLETED: i64 = 2;

    #[repr(C)]
    struct CMTime {
        value: i64,
        timescale: i32,
        flags: u32,
        epoch: i64,
    }

    #[link(name = "AVFoundation", kind = "framework")]
    unsafe extern "C" {
        static AVMediaTypeVideo: Id;
    }

    #[link(name = "CoreMedia", kind = "framework")]
    unsafe extern "C" {
        fn CMSampleBufferGetImageBuffer(sample: Id) -> Id;
        fn CMSampleBufferGetPresentationTimeStamp(sample: Id) -> CMTime;
    }

    #[link(name = "CoreVideo", kind = "framework")]
    unsafe extern "C" {
        static kCVPixelBufferPixelFormatTypeKey: Id;
        fn CVPixelBufferLockBaseAddress(buffer: Id, flags: u64) -> i32;
        fn CVPixelBufferUnlockBaseAddress(buffer: Id, flags: u64) -> i32;
        fn CVPixelBufferGetBaseAddress(buffer: Id) -> *const u8;
        fn CVPixelBufferGetBytesPerRow(buffer: Id) -> usize;
        fn CVPixelBufferGetWidth(buffer: Id) -> usize;
        fn CVPixelBufferGetHeight(buffer: Id) -> usize;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFRelease(object: Id);
    }

    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_autoreleasePoolPush() -> *mut c_void;
        fn objc_autoreleasePoolPop(pool: *mut c_void);
    }

    // `objc_msgSend` called with each method's exact signature.
    unsafe fn send0(receiver: Id, selector: &CStr) -> Id {
        unsafe { objc::send_id(receiver, objc::selector(selector)) }
    }

    unsafe fn send1(receiver: Id, selector: &CStr, a: Id) -> Id {
        let send: unsafe extern "C" fn(Id, Id, Id) -> Id = unsafe { std::mem::transmute(objc::msg_send()) };
        unsafe { send(receiver, objc::selector(selector), a) }
    }

    unsafe fn send2(receiver: Id, selector: &CStr, a: Id, b: Id) -> Id {
        let send: unsafe extern "C" fn(Id, Id, Id, Id) -> Id = unsafe { std::mem::transmute(objc::msg_send()) };
        unsafe { send(receiver, objc::selector(selector), a, b) }
    }

    unsafe fn send_u32(receiver: Id, selector: &CStr, a: u32) -> Id {
        let send: unsafe extern "C" fn(Id, Id, u32) -> Id = unsafe { std::mem::transmute(objc::msg_send()) };
        unsafe { send(receiver, objc::selector(selector), a) }
    }

    unsafe fn send_void1(receiver: Id, selector: &CStr, a: Id) {
        let send: unsafe extern "C" fn(Id, Id, Id) = unsafe { std::mem::transmute(objc::msg_send()) };
        unsafe { send(receiver, objc::selector(selector), a) }
    }

    /// Methods returning `BOOL`.
    unsafe fn ask0(receiver: Id, selector: &CStr) -> bool {
        let send: unsafe extern "C" fn(Id, Id) -> i8 = unsafe { std::mem::transmute(objc::msg_send()) };
        unsafe { send(receiver, objc::selector(selector)) != 0 }
    }

    unsafe fn ask1(receiver: Id, selector: &CStr, a: Id) -> bool {
        let send: unsafe extern "C" fn(Id, Id, Id) -> i8 = unsafe { std::mem::transmute(objc::msg_send()) };
        unsafe { send(receiver, objc::selector(selector), a) != 0 }
    }

    /// Methods returning `NSInteger`.
    unsafe fn integer0(receiver: Id, selector: &CStr) -> i64 {
        let send: unsafe extern "C" fn(Id, Id) -> i64 = unsafe { std::mem::transmute(objc::msg_send()) };
        unsafe { send(receiver, objc::selector(selector)) }
    }

    unsafe fn class(name: &CStr) -> Result<Id, String> {
        unsafe { objc::class(name) }.ok_or_else(|| format!("{} is not available", name.to_string_lossy()))
    }

    /// Runs `body` inside an autorelease pool, so temporary Objective-C
    /// objects are freed when it returns.
    unsafe fn with_pool<T>(body: impl FnOnce() -> T) -> T {
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let result = body();
            objc_autoreleasePoolPop(pool);
            result
        }
    }

    /// An `AVAssetReader` reading the video track from the start, and its
    /// output; both retained.
    unsafe fn start_reading(asset: Id, track: Id) -> Result<(Id, Id), String> {
        unsafe {
            let mut error: Id = std::ptr::null_mut();
            let reader = send2(
                class(c"AVAssetReader")?,
                c"assetReaderWithAsset:error:",
                asset,
                (&mut error as *mut Id).cast(),
            );
            if reader.is_null() {
                return Err("the video can't be read".into());
            }
            let format = send_u32(class(c"NSNumber")?, c"numberWithUnsignedInt:", PIXEL_FORMAT_BGRA);
            let settings = send2(
                class(c"NSDictionary")?,
                c"dictionaryWithObject:forKey:",
                format,
                kCVPixelBufferPixelFormatTypeKey,
            );
            let output = send2(
                class(c"AVAssetReaderTrackOutput")?,
                c"assetReaderTrackOutputWithTrack:outputSettings:",
                track,
                settings,
            );
            if output.is_null() || !ask1(reader, c"canAddOutput:", output) {
                return Err("the video's format isn't supported".into());
            }
            send_void1(reader, c"addOutput:", output);
            if !ask0(reader, c"startReading") {
                return Err("the video can't be decoded".into());
            }
            Ok((send0(reader, c"retain"), send0(output, c"retain")))
        }
    }

    unsafe fn release(object: Id) {
        unsafe { objc::send_void(object, objc::selector(c"release")) }
    }

    /// Copy a BGRA pixel buffer into an RGBA image, scaled down to at most
    /// `MAX_WIDTH` wide.
    unsafe fn to_image(buffer: Id) -> Option<image::RgbaImage> {
        unsafe {
            if CVPixelBufferLockBaseAddress(buffer, LOCK_READ_ONLY) != 0 {
                return None;
            }
            let (width, height) = (CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer));
            let stride = CVPixelBufferGetBytesPerRow(buffer);
            let base = CVPixelBufferGetBaseAddress(buffer);
            let image = (!base.is_null() && width > 0 && height > 0).then(|| {
                let data = std::slice::from_raw_parts(base, stride * height);
                let step = width.div_ceil(MAX_WIDTH);
                let (out_w, out_h) = (width / step, height / step);
                let mut rgba = Vec::with_capacity(out_w * out_h * 4);
                for row in 0..out_h {
                    let start = row * step * stride;
                    for px in data[start..start + width * 4].chunks_exact(4).step_by(step).take(out_w) {
                        rgba.extend_from_slice(&[px[2], px[1], px[0], 255]);
                    }
                }
                image::RgbaImage::from_raw(out_w as u32, out_h as u32, rgba)
            });
            CVPixelBufferUnlockBaseAddress(buffer, LOCK_READ_ONLY);
            image.flatten()
        }
    }

    pub unsafe fn decode(path: &Path, slot: &FrameSlot, stop: &AtomicBool, ready: mpsc::Sender<Result<(), String>>) {
        unsafe {
            let opened = with_pool(|| -> Result<(Id, Id, Id, Id), String> {
                let path = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| "bad file path")?;
                let path = objc::send_cstr(class(c"NSString")?, objc::selector(c"stringWithUTF8String:"), path.as_ptr());
                let url = send1(class(c"NSURL")?, c"fileURLWithPath:", path);
                let asset = send2(class(c"AVURLAsset")?, c"URLAssetWithURL:options:", url, std::ptr::null_mut());
                if asset.is_null() {
                    return Err("the file can't be opened".into());
                }
                let tracks = send1(asset, c"tracksWithMediaType:", AVMediaTypeVideo);
                let track = if tracks.is_null() { tracks } else { send0(tracks, c"firstObject") };
                if track.is_null() {
                    return Err("the file has no video in it".into());
                }
                let (reader, output) = start_reading(asset, track)?;
                Ok((send0(asset, c"retain"), send0(track, c"retain"), reader, output))
            });
            let (asset, track, mut reader, mut output) = match opened {
                Ok(objects) => objects,
                Err(error) => {
                    let _ = ready.send(Err(error));
                    return;
                }
            };
            let _ = ready.send(Ok(()));

            // When the first frame was shown, and its timestamp; later frames
            // are shown at the same offsets.
            let mut clock: Option<(Instant, f64)> = None;
            while !stop.load(Ordering::Relaxed) {
                let frame = with_pool(|| {
                    let sample = send0(output, c"copyNextSampleBuffer");
                    if sample.is_null() {
                        return None;
                    }
                    let time = CMSampleBufferGetPresentationTimeStamp(sample);
                    let seconds = if time.timescale > 0 { time.value as f64 / time.timescale as f64 } else { 0.0 };
                    let buffer = CMSampleBufferGetImageBuffer(sample);
                    let image = if buffer.is_null() { None } else { to_image(buffer) };
                    CFRelease(sample);
                    Some((seconds, image))
                });
                let Some((seconds, image)) = frame else {
                    // End of the video: start over. Anything else is an error.
                    let finished = integer0(reader, c"status") == READER_COMPLETED;
                    release(output);
                    release(reader);
                    if !finished {
                        reader = std::ptr::null_mut();
                        break;
                    }
                    match with_pool(|| start_reading(asset, track)) {
                        Ok((new_reader, new_output)) => (reader, output) = (new_reader, new_output),
                        Err(_) => {
                            reader = std::ptr::null_mut();
                            break;
                        }
                    }
                    clock = None;
                    continue;
                };
                let (start, first) = *clock.get_or_insert((Instant::now(), seconds));
                let due = Duration::from_secs_f64((seconds - first).max(0.0));
                let elapsed = start.elapsed();
                if due > elapsed {
                    std::thread::sleep(due - elapsed);
                } else if elapsed - due > Duration::from_millis(500) {
                    // Far behind (the Mac was busy): carry on from here.
                    clock = Some((Instant::now(), seconds));
                }
                if let Some(image) = image {
                    *slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(Arc::new(image));
                }
            }
            if !reader.is_null() {
                send0(reader, c"cancelReading");
                release(output);
                release(reader);
            }
            release(track);
            release(asset);
        }
    }
}
