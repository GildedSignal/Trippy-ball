// UI Controls Implementation
//
// This file contains additional UI controls for the wavefunction visualization.
// It includes helper functions and custom widgets.

use egui::{Color32, Ui};

// A helper function to display quantum state information
pub fn quantum_state_info(ui: &mut Ui, n: usize, l: usize, m: isize) {
    ui.horizontal(|ui| {
        ui.label("State:");

        // Display state in spectroscopic notation
        let l_symbol = match l {
            0 => "s",
            1 => "p",
            2 => "d",
            3 => "f",
            4 => "g",
            5 => "h",
            _ => "?",
        };

        let m_symbol = match m.cmp(&0) {
            std::cmp::Ordering::Equal => "".to_string(),
            std::cmp::Ordering::Greater => format!("+{}", m),
            std::cmp::Ordering::Less => format!("{}", m),
        };

        ui.label(format!("{}{}{}", n, l_symbol, m_symbol));
    });

    // Display a brief description of the state
    let description = match (l, m) {
        (0, 0) => "Spherically symmetric state (s orbital)",
        (1, 0) => "p₀ orbital (dumbbell along z-axis)",
        (1, 1) | (1, -1) => "p₁ orbital (dumbbell in xy-plane)",
        (2, 0) => "d₀ orbital (donut with z-axis lobes)",
        (2, _) => "d orbital (complex shape with angular nodes)",
        (3, 0) => "f₀ orbital (complex shape with radial and angular nodes)",
        (3, _) => "f orbital (complex shape with multiple nodes)",
        _ => "Higher angular momentum state with complex nodal structure",
    };

    ui.label(description);
}

// A helper function to display performance tips
pub fn performance_tips(ui: &mut Ui, point_count: usize, fps: f64) {
    if point_count > 100000 && fps < 30.0 {
        ui.label("Performance Tip: Reduce point count for better performance");
    }

    if point_count > 50000 && fps < 15.0 {
        ui.colored_label(Color32::YELLOW, "Warning: Low frame rate detected");
    }
}
