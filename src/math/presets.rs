// Quantum State Presets
//
// This file contains predefined quantum state presets for interesting visualizations.
// Each preset defines a set of quantum numbers and parameters that produce visually
// appealing or physically interesting wavefunction visualizations.

use crate::math::WavefunctionParams;

/// Represents a preset quantum state with a name and parameters
#[derive(Debug, Clone)]
pub struct QuantumStatePreset {
    pub name: &'static str,
    pub params: WavefunctionParams,
}

impl QuantumStatePreset {
    /// Create a new quantum state preset
    pub fn new(name: &'static str, _description: &'static str, params: WavefunctionParams) -> Self {
        Self { name, params }
    }
}

/// Get all available quantum state presets
pub fn get_all_presets() -> Vec<QuantumStatePreset> {
    vec![
        // Ground state (1s)
        QuantumStatePreset::new(
            "Ground State (1s)",
            "The lowest energy state of a hydrogen atom (l=0, m=0)",
            WavefunctionParams {
                n: 1,
                l: 0,
                m: 0,
                n2: 2,
                l2: 1,
                m2: 0,
                mix: 0.20,
                relative_phase: 0.0,
                z: 1.0,
                time_factor: 2.0,
            },
        ),
        // 2p_z state
        QuantumStatePreset::new(
            "2p_z State",
            "The p orbital aligned along the z-axis (l=1, m=0)",
            WavefunctionParams {
                n: 2,
                l: 1,
                m: 0,
                n2: 3,
                l2: 2,
                m2: 0,
                mix: 0.30,
                relative_phase: 0.0,
                z: 1.0,
                time_factor: 2.0,
            },
        ),
        // 2p_x state
        QuantumStatePreset::new(
            "2p_x State",
            "The p orbital aligned along the x-axis (l=1, m=1)",
            WavefunctionParams {
                n: 2,
                l: 1,
                m: 1,
                n2: 3,
                l2: 2,
                m2: 1,
                mix: 0.30,
                relative_phase: 0.0,
                z: 1.0,
                time_factor: 2.0,
            },
        ),
        // 3d_z^2 state
        QuantumStatePreset::new(
            "3d_z^2 State",
            "The d orbital with a donut shape (l=2, m=0)",
            WavefunctionParams {
                n: 3,
                l: 2,
                m: 0,
                n2: 4,
                l2: 3,
                m2: 0,
                mix: 0.35,
                relative_phase: 0.0,
                z: 1.0,
                time_factor: 2.0,
            },
        ),
        // 3d_xy state
        QuantumStatePreset::new(
            "3d_xy State",
            "The d orbital with four lobes in the xy plane (l=2, m=2)",
            WavefunctionParams {
                n: 3,
                l: 2,
                m: 2,
                n2: 4,
                l2: 3,
                m2: 2,
                mix: 0.35,
                relative_phase: 0.0,
                z: 1.0,
                time_factor: 2.0,
            },
        ),
        // 4f state
        QuantumStatePreset::new(
            "4f State",
            "A complex f orbital with intricate structure (l=3, m=2)",
            WavefunctionParams {
                n: 4,
                l: 3,
                m: 2,
                n2: 5,
                l2: 4,
                m2: 2,
                mix: 0.40,
                relative_phase: 0.0,
                z: 1.0,
                time_factor: 2.0,
            },
        ),
        // 5g state
        QuantumStatePreset::new(
            "5g State",
            "A highly complex g orbital (l=4, m=0)",
            WavefunctionParams {
                n: 5,
                l: 4,
                m: 0,
                n2: 6,
                l2: 5,
                m2: 0,
                mix: 0.45,
                relative_phase: 0.0,
                z: 1.0,
                time_factor: 2.0,
            },
        ),
        // Superposition state
        // Note: This is just a placeholder - actual superposition would require
        // more complex implementation in the wavefunction evaluation
        QuantumStatePreset::new(
            "Superposition-like",
            "Parameters that visually resemble a superposition state",
            WavefunctionParams {
                n: 4,
                l: 3,
                m: 3,
                n2: 5,
                l2: 2,
                m2: 1,
                mix: 0.50,
                relative_phase: 1.2,
                z: 1.0,
                time_factor: 2.0,
            },
        ),
    ]
}

/// Get a preset by name
pub fn get_preset_by_name(name: &str) -> Option<QuantumStatePreset> {
    get_all_presets().into_iter().find(|p| p.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_presets_exist() {
        let presets = get_all_presets();
        assert!(!presets.is_empty());
    }

    #[test]
    fn test_get_preset_by_name() {
        let preset = get_preset_by_name("Ground State (1s)");
        assert!(preset.is_some());

        let preset = preset.unwrap();
        assert_eq!(preset.params.l, 0);
        assert_eq!(preset.params.m, 0);
    }
}
