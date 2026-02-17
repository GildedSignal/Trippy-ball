use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
}

static LOG_LEVEL: OnceLock<Level> = OnceLock::new();

fn parse_level(raw: Option<String>) -> Level {
    match raw
        .unwrap_or_else(|| "warn".to_string())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "error" => Level::Error,
        "warn" | "warning" => Level::Warn,
        "info" => Level::Info,
        "debug" | "trace" => Level::Debug,
        _ => Level::Warn,
    }
}

fn current_level() -> Level {
    *LOG_LEVEL.get_or_init(|| {
        parse_level(
            std::env::var("TRIPPY_BALL_LOG")
                .ok()
                .or_else(|| std::env::var("RUST_LOG").ok()),
        )
    })
}

pub fn init() {
    let _ = current_level();
}

pub fn enabled_debug() -> bool {
    current_level() >= Level::Debug
}

pub fn enabled_info() -> bool {
    current_level() >= Level::Info
}

pub fn enabled_warn() -> bool {
    current_level() >= Level::Warn
}

pub fn enabled_error() -> bool {
    current_level() >= Level::Error
}

#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {{
        if $crate::telemetry::enabled_debug() {
            eprintln!("[DEBUG] {}", format_args!($($arg)*));
        }
    }};
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {{
        if $crate::telemetry::enabled_info() {
            eprintln!("[INFO] {}", format_args!($($arg)*));
        }
    }};
}

#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {{
        if $crate::telemetry::enabled_warn() {
            eprintln!("[WARN] {}", format_args!($($arg)*));
        }
    }};
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {{
        if $crate::telemetry::enabled_error() {
            eprintln!("[ERROR] {}", format_args!($($arg)*));
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_level_maps_expected_values() {
        assert_eq!(parse_level(Some("error".to_string())), Level::Error);
        assert_eq!(parse_level(Some("warn".to_string())), Level::Warn);
        assert_eq!(parse_level(Some("warning".to_string())), Level::Warn);
        assert_eq!(parse_level(Some("info".to_string())), Level::Info);
        assert_eq!(parse_level(Some("debug".to_string())), Level::Debug);
        assert_eq!(parse_level(Some("trace".to_string())), Level::Debug);
        assert_eq!(parse_level(Some("unknown".to_string())), Level::Warn);
        assert_eq!(parse_level(None), Level::Warn);
    }

    #[test]
    fn parse_level_is_trimmed_and_case_insensitive() {
        assert_eq!(parse_level(Some("  DeBuG  ".to_string())), Level::Debug);
        assert_eq!(parse_level(Some(" INFO ".to_string())), Level::Info);
    }
}
