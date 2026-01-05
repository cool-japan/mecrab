#![no_main]

use libfuzzer_sys::fuzz_target;
use mecrab::dict::Dictionary;
use mecrab::lattice::Lattice;

fuzz_target!(|data: &str| {
    // Skip empty or very long inputs
    if data.is_empty() || data.len() > 5000 {
        return;
    }

    // Build lattice - should not panic
    if let Ok(dict) = Dictionary::default_dictionary() {
        let _ = Lattice::build(data, &dict);
    }
});
