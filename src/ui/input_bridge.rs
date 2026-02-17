use egui::{Event, Key, Modifiers, PointerButton, Pos2, RawInput, Rect, Vec2};
use std::time::Instant;
use winit::event::{
    ElementState, ModifiersState, MouseButton, MouseScrollDelta, VirtualKeyCode, WindowEvent,
};
use winit::window::{CursorIcon, Window};

use crate::{debug, warn};

pub struct InputBridge {
    events: Vec<Event>,
    modifiers: Modifiers,
    pointer_pos: Option<Pos2>,
    focused: bool,
    time_origin: Instant,
}

impl InputBridge {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            modifiers: Modifiers::default(),
            pointer_pos: None,
            focused: true,
            time_origin: Instant::now(),
        }
    }

    pub fn on_window_event(
        &mut self,
        event: &WindowEvent,
        pixels_per_point: f32,
        wants_pointer_input: bool,
        wants_keyboard_input: bool,
    ) -> bool {
        match event {
            WindowEvent::ModifiersChanged(state) => {
                self.modifiers = modifiers_from_winit(*state);
                false
            }
            WindowEvent::Focused(focused) => {
                self.focused = *focused;
                self.events.push(Event::WindowFocused(*focused));
                false
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = Pos2::new(
                    position.x as f32 / pixels_per_point,
                    position.y as f32 / pixels_per_point,
                );
                self.pointer_pos = Some(pos);
                self.events.push(Event::PointerMoved(pos));
                wants_pointer_input
            }
            WindowEvent::CursorLeft { .. } => {
                self.pointer_pos = None;
                self.events.push(Event::PointerGone);
                wants_pointer_input
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let Some(pointer_button) = map_pointer_button(*button) else {
                    return false;
                };
                let pos = self.pointer_pos.unwrap_or_default();
                self.events.push(Event::PointerButton {
                    pos,
                    button: pointer_button,
                    pressed: *state == ElementState::Pressed,
                    modifiers: self.modifiers,
                });
                wants_pointer_input
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let mut delta = match delta {
                    MouseScrollDelta::LineDelta(x, y) => Vec2::new(*x, *y) * 8.0,
                    MouseScrollDelta::PixelDelta(pos) => {
                        Vec2::new(pos.x as f32, pos.y as f32) / pixels_per_point
                    }
                };
                if self.modifiers.shift {
                    delta = Vec2::new(delta.y, 0.0);
                }
                self.events.push(Event::Scroll(delta));
                wants_pointer_input
            }
            WindowEvent::ReceivedCharacter(ch) => {
                if !is_printable_char(*ch) || self.modifiers.ctrl || self.modifiers.command {
                    return false;
                }
                self.events.push(Event::Text(ch.to_string()));
                wants_keyboard_input
            }
            WindowEvent::KeyboardInput { input, .. } => {
                if let Some(key) = input.virtual_keycode.and_then(map_key) {
                    let pressed = input.state == ElementState::Pressed;
                    self.events.push(Event::Key {
                        key,
                        pressed,
                        repeat: false,
                        modifiers: self.modifiers,
                    });
                    return wants_keyboard_input;
                }
                false
            }
            _ => false,
        }
    }

    pub fn take_raw_input(&mut self, window: &Window) -> RawInput {
        let size = window.inner_size();
        let pixels_per_point = window.scale_factor() as f32;
        let screen_size = egui::vec2(
            size.width as f32 / pixels_per_point,
            size.height as f32 / pixels_per_point,
        );

        RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, screen_size)),
            pixels_per_point: Some(pixels_per_point),
            time: Some(self.time_origin.elapsed().as_secs_f64()),
            modifiers: self.modifiers,
            events: std::mem::take(&mut self.events),
            focused: self.focused,
            ..Default::default()
        }
    }

    pub fn apply_platform_output(&self, window: &Window, output: &egui::PlatformOutput) {
        if !output.copied_text.is_empty() {
            debug!("Clipboard output ignored (no clipboard backend)");
        }
        if let Some(open_url) = &output.open_url {
            debug!(
                "Open URL request ignored by local input bridge: {}",
                open_url.url
            );
        }

        if output.cursor_icon == egui::CursorIcon::None {
            window.set_cursor_visible(false);
            return;
        }

        window.set_cursor_visible(true);
        window.set_cursor_icon(map_cursor_icon(output.cursor_icon));
    }
}

fn is_printable_char(ch: char) -> bool {
    !ch.is_ascii_control()
}

fn map_pointer_button(button: MouseButton) -> Option<PointerButton> {
    match button {
        MouseButton::Left => Some(PointerButton::Primary),
        MouseButton::Right => Some(PointerButton::Secondary),
        MouseButton::Middle => Some(PointerButton::Middle),
        MouseButton::Other(1) => Some(PointerButton::Extra1),
        MouseButton::Other(2) => Some(PointerButton::Extra2),
        MouseButton::Other(_) => None,
    }
}

fn map_key(key: VirtualKeyCode) -> Option<Key> {
    Some(match key {
        VirtualKeyCode::Down => Key::ArrowDown,
        VirtualKeyCode::Left => Key::ArrowLeft,
        VirtualKeyCode::Right => Key::ArrowRight,
        VirtualKeyCode::Up => Key::ArrowUp,
        VirtualKeyCode::Escape => Key::Escape,
        VirtualKeyCode::Tab => Key::Tab,
        VirtualKeyCode::Back => Key::Backspace,
        VirtualKeyCode::Return => Key::Enter,
        VirtualKeyCode::Space => Key::Space,
        VirtualKeyCode::Insert => Key::Insert,
        VirtualKeyCode::Delete => Key::Delete,
        VirtualKeyCode::Home => Key::Home,
        VirtualKeyCode::End => Key::End,
        VirtualKeyCode::PageUp => Key::PageUp,
        VirtualKeyCode::PageDown => Key::PageDown,
        VirtualKeyCode::Minus | VirtualKeyCode::NumpadSubtract => Key::Minus,
        VirtualKeyCode::Equals | VirtualKeyCode::Plus | VirtualKeyCode::NumpadAdd => {
            Key::PlusEquals
        }
        VirtualKeyCode::Key0 | VirtualKeyCode::Numpad0 => Key::Num0,
        VirtualKeyCode::Key1 | VirtualKeyCode::Numpad1 => Key::Num1,
        VirtualKeyCode::Key2 | VirtualKeyCode::Numpad2 => Key::Num2,
        VirtualKeyCode::Key3 | VirtualKeyCode::Numpad3 => Key::Num3,
        VirtualKeyCode::Key4 | VirtualKeyCode::Numpad4 => Key::Num4,
        VirtualKeyCode::Key5 | VirtualKeyCode::Numpad5 => Key::Num5,
        VirtualKeyCode::Key6 | VirtualKeyCode::Numpad6 => Key::Num6,
        VirtualKeyCode::Key7 | VirtualKeyCode::Numpad7 => Key::Num7,
        VirtualKeyCode::Key8 | VirtualKeyCode::Numpad8 => Key::Num8,
        VirtualKeyCode::Key9 | VirtualKeyCode::Numpad9 => Key::Num9,
        VirtualKeyCode::A => Key::A,
        VirtualKeyCode::B => Key::B,
        VirtualKeyCode::C => Key::C,
        VirtualKeyCode::D => Key::D,
        VirtualKeyCode::E => Key::E,
        VirtualKeyCode::F => Key::F,
        VirtualKeyCode::G => Key::G,
        VirtualKeyCode::H => Key::H,
        VirtualKeyCode::I => Key::I,
        VirtualKeyCode::J => Key::J,
        VirtualKeyCode::K => Key::K,
        VirtualKeyCode::L => Key::L,
        VirtualKeyCode::M => Key::M,
        VirtualKeyCode::N => Key::N,
        VirtualKeyCode::O => Key::O,
        VirtualKeyCode::P => Key::P,
        VirtualKeyCode::Q => Key::Q,
        VirtualKeyCode::R => Key::R,
        VirtualKeyCode::S => Key::S,
        VirtualKeyCode::T => Key::T,
        VirtualKeyCode::U => Key::U,
        VirtualKeyCode::V => Key::V,
        VirtualKeyCode::W => Key::W,
        VirtualKeyCode::X => Key::X,
        VirtualKeyCode::Y => Key::Y,
        VirtualKeyCode::Z => Key::Z,
        VirtualKeyCode::F1 => Key::F1,
        VirtualKeyCode::F2 => Key::F2,
        VirtualKeyCode::F3 => Key::F3,
        VirtualKeyCode::F4 => Key::F4,
        VirtualKeyCode::F5 => Key::F5,
        VirtualKeyCode::F6 => Key::F6,
        VirtualKeyCode::F7 => Key::F7,
        VirtualKeyCode::F8 => Key::F8,
        VirtualKeyCode::F9 => Key::F9,
        VirtualKeyCode::F10 => Key::F10,
        VirtualKeyCode::F11 => Key::F11,
        VirtualKeyCode::F12 => Key::F12,
        VirtualKeyCode::F13 => Key::F13,
        VirtualKeyCode::F14 => Key::F14,
        VirtualKeyCode::F15 => Key::F15,
        VirtualKeyCode::F16 => Key::F16,
        VirtualKeyCode::F17 => Key::F17,
        VirtualKeyCode::F18 => Key::F18,
        VirtualKeyCode::F19 => Key::F19,
        VirtualKeyCode::F20 => Key::F20,
        _ => return None,
    })
}

fn map_cursor_icon(icon: egui::CursorIcon) -> CursorIcon {
    match icon {
        egui::CursorIcon::Default => CursorIcon::Default,
        egui::CursorIcon::ContextMenu => CursorIcon::ContextMenu,
        egui::CursorIcon::Help => CursorIcon::Help,
        egui::CursorIcon::PointingHand => CursorIcon::Hand,
        egui::CursorIcon::Progress => CursorIcon::Progress,
        egui::CursorIcon::Wait => CursorIcon::Wait,
        egui::CursorIcon::Cell => CursorIcon::Cell,
        egui::CursorIcon::Crosshair => CursorIcon::Crosshair,
        egui::CursorIcon::Text => CursorIcon::Text,
        egui::CursorIcon::VerticalText => CursorIcon::VerticalText,
        egui::CursorIcon::Alias => CursorIcon::Alias,
        egui::CursorIcon::Copy => CursorIcon::Copy,
        egui::CursorIcon::Move => CursorIcon::Move,
        egui::CursorIcon::NoDrop => CursorIcon::NoDrop,
        egui::CursorIcon::NotAllowed => CursorIcon::NotAllowed,
        egui::CursorIcon::Grab => CursorIcon::Grab,
        egui::CursorIcon::Grabbing => CursorIcon::Grabbing,
        egui::CursorIcon::AllScroll => CursorIcon::AllScroll,
        egui::CursorIcon::ResizeHorizontal => CursorIcon::EwResize,
        egui::CursorIcon::ResizeNeSw => CursorIcon::NeswResize,
        egui::CursorIcon::ResizeNwSe => CursorIcon::NwseResize,
        egui::CursorIcon::ResizeVertical => CursorIcon::NsResize,
        egui::CursorIcon::ResizeEast => CursorIcon::EResize,
        egui::CursorIcon::ResizeSouthEast => CursorIcon::SeResize,
        egui::CursorIcon::ResizeSouth => CursorIcon::SResize,
        egui::CursorIcon::ResizeSouthWest => CursorIcon::SwResize,
        egui::CursorIcon::ResizeWest => CursorIcon::WResize,
        egui::CursorIcon::ResizeNorthWest => CursorIcon::NwResize,
        egui::CursorIcon::ResizeNorth => CursorIcon::NResize,
        egui::CursorIcon::ResizeNorthEast => CursorIcon::NeResize,
        egui::CursorIcon::ResizeColumn => CursorIcon::ColResize,
        egui::CursorIcon::ResizeRow => CursorIcon::RowResize,
        egui::CursorIcon::ZoomIn => CursorIcon::ZoomIn,
        egui::CursorIcon::ZoomOut => CursorIcon::ZoomOut,
        egui::CursorIcon::None => CursorIcon::Default,
    }
}

fn modifiers_from_winit(state: ModifiersState) -> Modifiers {
    let alt = state.alt();
    let ctrl = state.ctrl();
    let shift = state.shift();
    let logo = state.logo();
    let mac_cmd = cfg!(target_os = "macos") && logo;
    let command = if cfg!(target_os = "macos") {
        logo
    } else {
        ctrl
    };

    if cfg!(target_os = "macos") && logo && ctrl {
        warn!("Both command and control are active");
    }

    Modifiers {
        alt,
        ctrl,
        shift,
        mac_cmd,
        command,
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use winit::event::{DeviceId, KeyboardInput, TouchPhase};

    fn dummy_device() -> DeviceId {
        unsafe { DeviceId::dummy() }
    }

    fn keyboard_event(key: VirtualKeyCode, state: ElementState) -> WindowEvent<'static> {
        WindowEvent::KeyboardInput {
            device_id: dummy_device(),
            input: KeyboardInput {
                scancode: 0,
                state,
                virtual_keycode: Some(key),
                modifiers: ModifiersState::empty(),
            },
            is_synthetic: false,
        }
    }

    #[test]
    fn key_mapping_covers_expected_keys() {
        assert_eq!(map_key(VirtualKeyCode::Tab), Some(Key::Tab));
        assert_eq!(map_key(VirtualKeyCode::Down), Some(Key::ArrowDown));
        assert_eq!(map_key(VirtualKeyCode::Numpad0), Some(Key::Num0));
        assert_eq!(map_key(VirtualKeyCode::Compose), None);
    }

    #[test]
    fn pointer_button_mapping_covers_supported_buttons() {
        assert_eq!(map_pointer_button(MouseButton::Left), Some(PointerButton::Primary));
        assert_eq!(
            map_pointer_button(MouseButton::Right),
            Some(PointerButton::Secondary)
        );
        assert_eq!(
            map_pointer_button(MouseButton::Middle),
            Some(PointerButton::Middle)
        );
        assert_eq!(
            map_pointer_button(MouseButton::Other(1)),
            Some(PointerButton::Extra1)
        );
        assert_eq!(map_pointer_button(MouseButton::Other(99)), None);
    }

    #[test]
    fn keyboard_events_are_forwarded_when_requested() {
        let mut bridge = InputBridge::new();
        let consumed = bridge.on_window_event(
            &keyboard_event(VirtualKeyCode::Tab, ElementState::Pressed),
            1.0,
            false,
            true,
        );
        assert!(consumed);
        assert!(matches!(
            bridge.events.last(),
            Some(Event::Key {
                key: Key::Tab,
                pressed: true,
                ..
            })
        ));
    }

    #[test]
    fn ctrl_modified_text_input_is_filtered() {
        let mut bridge = InputBridge::new();
        bridge.on_window_event(
            &WindowEvent::ModifiersChanged(ModifiersState::CTRL),
            1.0,
            false,
            false,
        );
        let consumed = bridge.on_window_event(&WindowEvent::ReceivedCharacter('a'), 1.0, false, true);
        assert!(!consumed);
        assert!(!bridge.events.iter().any(|event| matches!(event, Event::Text(_))));
    }

    #[test]
    fn shift_scroll_maps_to_horizontal_scroll_event() {
        let mut bridge = InputBridge::new();
        bridge.on_window_event(
            &WindowEvent::ModifiersChanged(ModifiersState::SHIFT),
            1.0,
            false,
            false,
        );
        let consumed = bridge.on_window_event(
            &WindowEvent::MouseWheel {
                device_id: dummy_device(),
                delta: MouseScrollDelta::LineDelta(1.0, 2.0),
                phase: TouchPhase::Moved,
                modifiers: ModifiersState::SHIFT,
            },
            1.0,
            true,
            false,
        );
        assert!(consumed);
        assert!(matches!(bridge.events.last(), Some(Event::Scroll(v)) if *v == Vec2::new(16.0, 0.0)));
    }

    #[test]
    fn modifiers_mapping_tracks_platform_command_semantics() {
        let mapped = modifiers_from_winit(ModifiersState::CTRL | ModifiersState::SHIFT);
        assert!(mapped.ctrl);
        assert!(mapped.shift);
        if cfg!(target_os = "macos") {
            assert!(!mapped.command);
        } else {
            assert!(mapped.command);
        }
    }
}
