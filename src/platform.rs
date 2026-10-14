//! Small OS-specific calls that the windowing crates don't expose.

/// Swap the window's buffers for the current OpenGL context. eframe owns the
/// swap but skips it for hidden windows, and doesn't expose the surface, so
/// this asks AppKit directly. Returns whether a context was current.
#[cfg(target_os = "macos")]
pub fn present_gl_frame() -> bool {
    // SAFETY: `+[NSOpenGLContext currentContext]` takes no arguments and
    // returns an object or nil; `-flushBuffer` takes none and returns nothing.
    unsafe {
        let Some(class) = objc::class(c"NSOpenGLContext") else {
            return false;
        };
        let context = objc::send_id(class, objc::selector(c"currentContext"));
        if context.is_null() {
            return false;
        }
        objc::send_void(context, objc::selector(c"flushBuffer"));
        true
    }
}

#[cfg(not(target_os = "macos"))]
pub fn present_gl_frame() -> bool {
    false
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
mod objc {
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

    pub unsafe fn send_activity(receiver: Id, selector: Id, options: u64, reason: Id) -> Id {
        let send: unsafe extern "C" fn(Id, Id, u64, Id) -> Id =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { send(receiver, selector, options, reason) }
    }
}
