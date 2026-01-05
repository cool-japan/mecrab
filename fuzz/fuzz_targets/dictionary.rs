#![no_main]

use libfuzzer_sys::fuzz_target;
use mecrab::dict::Dictionary;

fuzz_target!(|data: &str| {
    // Skip empty inputs
    if data.is_empty() || data.len() > 1000 {
        return;
    }

    // Try to lookup in dictionary - should not panic
    if let Ok(dict) = Dictionary::default_dictionary() {
        let _ = dict.lookup(data);

        // Test character info
        for c in data.chars().take(100) {
            let _ = dict.char_info(c);
            let _ = dict.char_category(c);
        }
    }
});
