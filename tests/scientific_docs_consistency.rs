use std::fs;

const CONTRACT_PATH: &str = "docs/scientific_contract.md";
const README_PATH: &str = "README.md";
const DOCUMENTATION_PATH: &str = "DOCUMENTATION.md";
const TECHNICAL_DESIGN_PATH: &str = "TECHNICAL_DESIGN.md";

fn read_text(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|err| panic!("failed to read {}: {}", path, err))
}

#[test]
fn scientific_contract_exists_with_core_claims() {
    let contract = read_text(CONTRACT_PATH);
    let required = [
        "hydrogenic radial",
        "spherical harmonics",
        "coherent two-state superposition",
        "hydrogen-like energy levels",
        "relative phase",
        "time scale (fs/s)",
        "nuclear charge z",
    ];

    let contract_lc = contract.to_ascii_lowercase();
    for phrase in required {
        assert!(
            contract_lc.contains(phrase),
            "contract is missing required phrase '{}'",
            phrase
        );
    }
}

#[test]
fn designated_docs_include_required_scientific_terms() {
    let merged = format!(
        "{}\n{}\n{}",
        read_text(README_PATH),
        read_text(DOCUMENTATION_PATH),
        read_text(TECHNICAL_DESIGN_PATH)
    );
    let merged_lc = merged.to_ascii_lowercase();

    let required = [
        "hydrogenic radial",
        "spherical harmonics",
        "coherent two-state superposition",
        "hydrogen-like energy levels",
        "relative phase",
    ];

    for phrase in required {
        assert!(
            merged_lc.contains(phrase),
            "designated docs are missing required phrase '{}'",
            phrase
        );
    }
}

#[test]
fn designated_docs_reject_stale_scientific_claims() {
    let merged = format!(
        "{}\n{}\n{}",
        read_text(README_PATH),
        read_text(DOCUMENTATION_PATH),
        read_text(TECHNICAL_DESIGN_PATH)
    );
    let merged_lc = merged.to_ascii_lowercase();

    let banned = [
        "radial term: gaussian radial function",
        "time modulation: sinusoid-derived amplitude",
        "radial parameter: `alpha`",
        "wavefunction parameter ui (`l`, `m`, `alpha`, `time_factor`)",
    ];

    for phrase in banned {
        assert!(
            !merged_lc.contains(phrase),
            "stale scientific phrase still present: '{}'",
            phrase
        );
    }
}
