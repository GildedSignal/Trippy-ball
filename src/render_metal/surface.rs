use winit::window::Window;

#[cfg(target_os = "macos")]
pub struct MetalSurface {
    layer: metal::MetalLayer,
}

#[cfg(target_os = "macos")]
impl MetalSurface {
    pub const PIXEL_FORMAT: metal::MTLPixelFormat = metal::MTLPixelFormat::BGRA8Unorm_sRGB;

    #[allow(unexpected_cfgs)]
    pub fn new(window: &Window, device: &metal::DeviceRef) -> Result<Self, String> {
        use core_graphics_types::geometry::CGSize;
        use metal::objc::runtime::{Object, YES};
        use metal::objc::{msg_send, sel, sel_impl};
        use winit::platform::macos::WindowExtMacOS;

        let layer = metal::MetalLayer::new();
        layer.set_device(device);
        layer.set_pixel_format(Self::PIXEL_FORMAT);
        layer.set_presents_with_transaction(false);
        layer.set_display_sync_enabled(true);
        layer.set_framebuffer_only(false);
        layer.set_opaque(true);
        layer.set_maximum_drawable_count(3);

        let scale = window.scale_factor();
        layer.set_contents_scale(scale);

        let size = window.inner_size();
        layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));

        unsafe {
            let view = window.ns_view() as *mut Object;
            if view.is_null() {
                return Err("window ns_view is null".to_string());
            }

            let () = msg_send![view, setWantsLayer: YES];
            let layer_ref = layer.as_ref() as *const metal::MetalLayerRef as *mut Object;
            let () = msg_send![view, setLayer: layer_ref];
        }

        Ok(Self { layer })
    }

    pub fn resize(&self, width: u32, height: u32, scale_factor: f64) {
        use core_graphics_types::geometry::CGSize;
        self.layer.set_contents_scale(scale_factor);
        self.layer
            .set_drawable_size(CGSize::new(width as f64, height as f64));
    }

    pub fn next_drawable(&self) -> Option<&metal::MetalDrawableRef> {
        self.layer.next_drawable()
    }
}

#[cfg(not(target_os = "macos"))]
pub struct MetalSurface;

#[cfg(not(target_os = "macos"))]
impl MetalSurface {
    pub fn new(_window: &Window, _device: &()) -> Result<Self, String> {
        Err("Metal surface is only available on macOS".to_string())
    }

    pub fn resize(&self, _width: u32, _height: u32, _scale_factor: f64) {}
}
