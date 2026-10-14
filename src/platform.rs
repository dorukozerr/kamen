//! Small OS-specific calls that the windowing crates don't expose.

/// Swap the window's buffers for the current OpenGL context. eframe owns the
/// swap but skips it for hidden windows, and doesn't expose the surface, so
/// this asks AppKit directly.
#[cfg(target_os = "macos")]
pub fn present_gl_frame() {
    // SAFETY: `+[NSOpenGLContext currentContext]` takes no arguments and
    // returns an object or nil; `-flushBuffer` takes none and returns nothing.
    unsafe {
        let Some(class) = objc::class(c"NSOpenGLContext") else {
            return;
        };
        let context = objc::send_id(class, objc::selector(c"currentContext"));
        if !context.is_null() {
            objc::send_void(context, objc::selector(c"flushBuffer"));
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub fn present_gl_frame() {}

/// A Syphon server: publishes frames to other apps on this Mac (OBS's
/// "Syphon Client" source) through shared GPU memory, with transparency and
/// without capturing a window.
///
/// Syphon is loaded at run time from `libSyphon.dylib`, built by
/// `scripts/build.sh syphon`; without that file `new` returns an error and
/// the rest of the app is unaffected.
#[cfg(target_os = "macos")]
pub struct SyphonServer {
    server: objc::Id,
}

#[cfg(target_os = "macos")]
impl SyphonServer {
    /// Create a server named `name` on the current OpenGL context. Call with
    /// the context current, and publish from that same context.
    pub fn new(name: &std::ffi::CStr) -> Result<Self, String> {
        const RTLD_NOW: std::ffi::c_int = 2;
        unsafe extern "C" {
            fn dlopen(path: *const std::ffi::c_char, mode: std::ffi::c_int) -> *mut std::ffi::c_void;
        }
        // In the app bundle, then where the build script leaves it.
        let bundled = std::env::current_exe()
            .ok()
            .and_then(|exe| Some(exe.parent()?.parent()?.join("Frameworks/libSyphon.dylib")));
        let loaded = bundled
            .into_iter()
            .chain([std::path::PathBuf::from("target/syphon/libSyphon.dylib")])
            .filter_map(|path| std::ffi::CString::new(path.into_os_string().into_encoded_bytes()).ok())
            // SAFETY: `dlopen` takes a NUL-terminated path and returns null on failure.
            .any(|path| unsafe { !dlopen(path.as_ptr(), RTLD_NOW).is_null() });
        if !loaded {
            return Err("Syphon library not found. Run scripts/build.sh, which builds it.".into());
        }
        // SAFETY: each message is sent with the signature of the method it
        // names. `-CGLContextObj` returns the context pointer that
        // `-initWithName:context:options:` expects; `options` may be nil.
        unsafe {
            let (Some(server_class), Some(context_class), Some(string_class)) = (
                objc::class(c"SyphonOpenGLServer"),
                objc::class(c"NSOpenGLContext"),
                objc::class(c"NSString"),
            ) else {
                return Err("Syphon library is missing its server class".into());
            };
            let context = objc::send_id(context_class, objc::selector(c"currentContext"));
            if context.is_null() {
                return Err("no OpenGL context is current".into());
            }
            let cgl_context = objc::send_id(context, objc::selector(c"CGLContextObj"));
            let name = objc::send_cstr(string_class, objc::selector(c"stringWithUTF8String:"), name.as_ptr());
            let server = objc::send_init_server(
                objc::send_id(server_class, objc::selector(c"alloc")),
                objc::selector(c"initWithName:context:options:"),
                name,
                cgl_context,
                std::ptr::null_mut(),
            );
            if server.is_null() {
                return Err("Syphon could not start a server".into());
            }
            Ok(Self { server })
        }
    }

    /// Publish an OpenGL 2D texture of `size` pixels (bottom row first, as
    /// OpenGL renders). The caller sets the viewport to `size`, selects
    /// texture unit 0 and disables blending first; Syphon leaves the default
    /// framebuffer bound.
    pub fn publish(&self, texture: u32, size: [u32; 2]) {
        const GL_TEXTURE_2D: u32 = 0x0DE1;
        let dimensions = objc::Size {
            width: size[0] as f64,
            height: size[1] as f64,
        };
        let region = objc::Rect {
            x: 0.0,
            y: 0.0,
            size: dimensions,
        };
        // SAFETY: matches `-publishFrameTexture:textureTarget:imageRegion:
        // textureDimensions:flipped:` (GLuint, GLenum, NSRect, NSSize, BOOL).
        unsafe {
            objc::send_publish(
                self.server,
                objc::selector(c"publishFrameTexture:textureTarget:imageRegion:textureDimensions:flipped:"),
                texture,
                GL_TEXTURE_2D,
                region,
                dimensions,
                false,
            );
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for SyphonServer {
    fn drop(&mut self) {
        // SAFETY: `-stop` and `-release` take no arguments and return nothing;
        // `server` came from alloc/init and is released exactly once.
        unsafe {
            objc::send_void(self.server, objc::selector(c"stop"));
            objc::send_void(self.server, objc::selector(c"release"));
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub struct SyphonServer;

#[cfg(not(target_os = "macos"))]
impl SyphonServer {
    pub fn new(_name: &std::ffi::CStr) -> Result<Self, String> {
        Err("Syphon is only available on macOS".into())
    }

    pub fn publish(&self, _texture: u32, _size: [u32; 2]) {}
}

/// Tell macOS this app is doing user-visible work for its whole lifetime.
/// Otherwise App Nap throttles its timers once its windows are covered or on
/// another Space, which stalls the avatar that OBS is capturing.
/// Call on the main thread, once the app is running.
#[cfg(target_os = "macos")]
pub fn disable_app_nap() {
    // NSActivityUserInitiatedAllowingIdleSystemSleep | NSActivityLatencyCritical
    const OPTIONS: u64 = 0x00EF_FFFF | 0xFF_0000_0000;
    // SAFETY: each message is sent with the signature of the method it names:
    // `+processInfo` and `-retain` return an object, `+stringWithUTF8String:`
    // takes a C string, and `-beginActivityWithOptions:reason:` takes a
    // 64-bit option mask and an NSString.
    unsafe {
        let (Some(process_info), Some(string)) = (objc::class(c"NSProcessInfo"), objc::class(c"NSString")) else {
            return;
        };
        let process_info = objc::send_id(process_info, objc::selector(c"processInfo"));
        let reason = objc::send_cstr(
            string,
            objc::selector(c"stringWithUTF8String:"),
            c"Animating the avatar for capture".as_ptr(),
        );
        if process_info.is_null() || reason.is_null() {
            return;
        }
        let activity = objc::send_activity(
            process_info,
            objc::selector(c"beginActivityWithOptions:reason:"),
            OPTIONS,
            reason,
        );
        if !activity.is_null() {
            // The activity lasts while its token is alive; keep it forever.
            objc::send_id(activity, objc::selector(c"retain"));
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub fn disable_app_nap() {}

/// The few Objective-C runtime calls used above. `objc_msgSend` has to be
/// called through a function pointer with the exact signature of the method.
#[cfg(target_os = "macos")]
pub(crate) mod objc {
    use std::ffi::{CStr, c_char, c_void};

    pub type Id = *mut c_void;

    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> Id;
        fn sel_registerName(name: *const c_char) -> Id;
        fn objc_msgSend();
    }

    pub unsafe fn class(name: &CStr) -> Option<Id> {
        let class = unsafe { objc_getClass(name.as_ptr()) };
        (!class.is_null()).then_some(class)
    }

    /// `objc_msgSend`, to be transmuted to the exact signature of the method
    /// being called.
    pub fn msg_send() -> unsafe extern "C" fn() {
        objc_msgSend
    }

    pub unsafe fn selector(name: &CStr) -> Id {
        unsafe { sel_registerName(name.as_ptr()) }
    }

    pub unsafe fn send_id(receiver: Id, selector: Id) -> Id {
        let send: unsafe extern "C" fn(Id, Id) -> Id =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { send(receiver, selector) }
    }

    pub unsafe fn send_void(receiver: Id, selector: Id) {
        let send: unsafe extern "C" fn(Id, Id) = unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { send(receiver, selector) }
    }

    pub unsafe fn send_cstr(receiver: Id, selector: Id, text: *const c_char) -> Id {
        let send: unsafe extern "C" fn(Id, Id, *const c_char) -> Id =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { send(receiver, selector, text) }
    }

    /// `NSSize` and `NSRect` on 64-bit macOS.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Size {
        pub width: f64,
        pub height: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Rect {
        pub x: f64,
        pub y: f64,
        pub size: Size,
    }

    pub unsafe fn send_init_server(receiver: Id, selector: Id, name: Id, context: Id, options: Id) -> Id {
        let send: unsafe extern "C" fn(Id, Id, Id, Id, Id) -> Id =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { send(receiver, selector, name, context, options) }
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn send_publish(
        receiver: Id,
        selector: Id,
        texture: u32,
        target: u32,
        region: Rect,
        dimensions: Size,
        flipped: bool,
    ) {
        let send: unsafe extern "C" fn(Id, Id, u32, u32, Rect, Size, bool) =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { send(receiver, selector, texture, target, region, dimensions, flipped) }
    }

    pub unsafe fn send_activity(receiver: Id, selector: Id, options: u64, reason: Id) -> Id {
        let send: unsafe extern "C" fn(Id, Id, u64, Id) -> Id =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { send(receiver, selector, options, reason) }
    }
}
