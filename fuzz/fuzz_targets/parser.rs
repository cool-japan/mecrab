#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use mecrab::MeCrab;

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    text: String,
    add_words: Vec<(String, String, i16)>,
}

fuzz_target!(|input: FuzzInput| {
    // Skip very long inputs
    if input.text.len() > 5000 {
        return;
    }

    if let Ok(mecrab) = MeCrab::new() {
        // Add some words
        for (surface, reading, cost) in input.add_words.iter().take(10) {
            if surface.len() < 100 && reading.len() < 100 {
                mecrab.add_word(surface, reading, reading, *cost);
            }
        }

        // Parse
        let _ = mecrab.parse(&input.text);

        // Wakati
        let _ = mecrab.wakati(&input.text);
    }
});
