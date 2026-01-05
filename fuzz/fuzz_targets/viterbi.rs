#![no_main]

use libfuzzer_sys::fuzz_target;
use mecrab::MeCrab;

fuzz_target!(|data: &str| {
    // Skip empty or very long inputs
    if data.is_empty() || data.len() > 10000 {
        return;
    }

    // Try to parse - should not panic
    if let Ok(mecrab) = MeCrab::new() {
        let _ = mecrab.parse(data);
    }
});
