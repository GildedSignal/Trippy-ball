// Camera Implementation
//
// This file contains the implementation of the camera for the 3D visualization.
// It handles orbital camera movement, zooming, and view matrix calculation.
// Also supports auto-rotation for better visualization experience.

use crate::memory::pod::Pod;
use glam::{Mat4, Quat, Vec3, Vec4};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

// Camera uniform buffer for the GPU
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
}

unsafe impl Pod for CameraUniform {}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 0.0, 10.0, 1.0], // Increased default distance
        }
    }
}

// Frustum planes for culling
#[derive(Debug, Clone)]
pub struct Frustum {
    planes: [Vec4; 6], // Left, Right, Bottom, Top, Near, Far
}

// Camera state
pub struct Camera {
    position: Vec3,
    target: Vec3,
    up: Vec3,
    aspect: f32,
    fov: f32,
    near: f32,
    far: f32,

    // Orbital camera controls
    orbit_distance: f32,
    orbit_pitch: f32,
    orbit_yaw: f32,

    // Auto-rotation
    auto_rotate: bool,
    rotation_speed: f32,

    // Mouse interaction state
    is_dragging: bool,
    last_mouse_pos: (f64, f64),

    // Cached uniform used by native renderer.
    uniform: CameraUniform,
}

impl Camera {
    fn normalize_plane(plane: Vec4) -> Vec4 {
        // Normalize using the xyz normal length only; w must be scaled by the same factor.
        let normal = Vec3::new(plane.x, plane.y, plane.z);
        let len = normal.length();
        if len > 1e-6 {
            plane / len
        } else {
            plane
        }
    }

    // Create a new camera
    pub fn new(aspect: f32) -> Self {
        // Create camera with default orbital position
        let mut camera = Self {
            position: Vec3::new(0.0, 0.0, 10.0), // Increased default distance
            target: Vec3::ZERO,
            up: Vec3::Y,
            aspect,
            fov: 45.0_f32.to_radians(),
            near: 0.1,
            far: 100.0,

            orbit_distance: 10.0, // Increased default distance
            orbit_pitch: 0.0,
            orbit_yaw: 0.0,

            // Auto-rotation settings
            auto_rotate: false,
            rotation_speed: 0.2,

            is_dragging: false,
            last_mouse_pos: (0.0, 0.0),

            uniform: CameraUniform::default(),
        };

        // Update the camera matrices
        camera.update_view_matrix();
        camera.update_projection_matrix();

        camera
    }

    // Update the aspect ratio (when window is resized)
    pub fn update_aspect_ratio(&mut self, aspect: f32) {
        self.aspect = aspect;
        self.update_projection_matrix();
    }

    // Set auto-rotation state
    pub fn set_auto_rotation(&mut self, enabled: bool) {
        self.auto_rotate = enabled;
    }

    // Set rotation speed
    pub fn set_rotation_speed(&mut self, speed: f32) {
        self.rotation_speed = speed;
    }

    // Update camera for auto-rotation
    pub fn update(&mut self, delta_time: f32) {
        if self.auto_rotate && !self.is_dragging {
            // Rotate around the Y axis
            self.orbit_yaw += self.rotation_speed * delta_time;
            self.update_view_matrix();
        }
    }

    // Handle input events
    pub fn handle_input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                // Only start dragging if the left mouse button is pressed
                self.is_dragging = *state == ElementState::Pressed;
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current_pos = (position.x, position.y);

                if self.is_dragging {
                    let dx = (current_pos.0 - self.last_mouse_pos.0) as f32 * 0.005;
                    let dy = (current_pos.1 - self.last_mouse_pos.1) as f32 * 0.005;

                    self.orbit_yaw += dx;
                    self.orbit_pitch += dy;

                    // Clamp pitch to avoid gimbal lock
                    self.orbit_pitch = self.orbit_pitch.clamp(-1.5, 1.5);

                    self.update_view_matrix();
                }

                self.last_mouse_pos = current_pos;
                self.is_dragging
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y * 0.2,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.002,
                };

                // Adjust orbit distance with scroll
                self.orbit_distance -= scroll;
                self.orbit_distance = self.orbit_distance.clamp(2.0, 30.0); // Increased min and max distance

                self.update_view_matrix();
                true
            }
            _ => false,
        }
    }

    // Update the view matrix based on orbital camera parameters
    fn update_view_matrix(&mut self) {
        // Calculate position from orbital parameters
        let pitch_quat = Quat::from_rotation_x(self.orbit_pitch);
        let yaw_quat = Quat::from_rotation_y(self.orbit_yaw);
        let rotation = yaw_quat * pitch_quat;

        let offset = rotation * Vec3::new(0.0, 0.0, self.orbit_distance);
        self.position = self.target + offset;

        // Create view matrix
        let view = Mat4::look_at_rh(self.position, self.target, self.up);
        let proj = self.projection_matrix();

        // Update uniform
        self.uniform.view_proj = (proj * view).to_cols_array_2d();
        self.uniform.camera_pos = [self.position.x, self.position.y, self.position.z, 1.0];
    }

    // Update the projection matrix
    fn update_projection_matrix(&mut self) {
        let proj = self.projection_matrix();
        let view = Mat4::look_at_rh(self.position, self.target, self.up);
        self.uniform.view_proj = (proj * view).to_cols_array_2d();
    }

    // Calculate the projection matrix
    fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
    }

    // Keep call sites stable while camera state is CPU-owned.
    pub fn update_buffer(&self) {}

    // Update with rotation and buffer update
    pub fn update_with_rotation(
        &mut self,
        auto_rotation: bool,
        rotation_speed: f32,
        delta_time: f32,
    ) {
        self.set_auto_rotation(auto_rotation);
        self.set_rotation_speed(rotation_speed);
        self.update(delta_time);
        self.update_buffer();
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn uniform(&self) -> CameraUniform {
        self.uniform
    }

    // Get the current view frustum for culling
    pub fn get_frustum(&self) -> Frustum {
        let view = Mat4::look_at_rh(self.position, self.target, self.up);
        let proj = self.projection_matrix();
        let view_proj = proj * view;

        // Extract frustum planes from the view-projection matrix rows.
        // glam matrices are column-major, so build rows explicitly from columns.
        let c0 = view_proj.col(0);
        let c1 = view_proj.col(1);
        let c2 = view_proj.col(2);
        let c3 = view_proj.col(3);

        let r0 = Vec4::new(c0.x, c1.x, c2.x, c3.x);
        let r1 = Vec4::new(c0.y, c1.y, c2.y, c3.y);
        let r2 = Vec4::new(c0.z, c1.z, c2.z, c3.z);
        let r3 = Vec4::new(c0.w, c1.w, c2.w, c3.w);

        // Each plane is (nx, ny, nz, d) with equation n·p + d >= 0 for inside.
        let mut planes = [Vec4::ZERO; 6];

        // Left plane
        planes[0] = Self::normalize_plane(r3 + r0);

        // Right plane
        planes[1] = Self::normalize_plane(r3 - r0);

        // Bottom plane
        planes[2] = Self::normalize_plane(r3 + r1);

        // Top plane
        planes[3] = Self::normalize_plane(r3 - r1);

        // Near plane
        planes[4] = Self::normalize_plane(r3 + r2);

        // Far plane
        planes[5] = Self::normalize_plane(r3 - r2);

        Frustum { planes }
    }
}

impl Frustum {
    // Check if a point is inside the frustum, with optional margin to reduce edge popping.
    pub fn contains_point_with_margin(&self, point: Vec3, margin: f32) -> bool {
        for plane in &self.planes {
            let normal = Vec3::new(plane.x, plane.y, plane.z);
            let distance = plane.w;

            if normal.dot(point) + distance < -margin {
                return false;
            }
        }

        true
    }

    // Check whether a bounding sphere is fully inside the frustum.
    // If true, per-point culling work is unnecessary.
    pub fn sphere_fully_inside(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let normal = Vec3::new(plane.x, plane.y, plane.z);
            let distance = plane.w;
            if normal.dot(center) + distance < radius {
                return false;
            }
        }
        true
    }
}
