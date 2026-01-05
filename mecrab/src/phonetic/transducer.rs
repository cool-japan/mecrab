//! Phonetic transducer implementations

use std::collections::HashMap;

/// Convert katakana to hiragana
pub fn to_hiragana(text: &str) -> String {
    text.chars()
        .map(|c| {
            if ('\u{30A1}'..='\u{30F6}').contains(&c) {
                // Katakana to Hiragana: subtract 0x60
                char::from_u32(c as u32 - 0x60).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

/// Convert hiragana to katakana
pub fn to_katakana(text: &str) -> String {
    text.chars()
        .map(|c| {
            if ('\u{3041}'..='\u{3096}').contains(&c) {
                // Hiragana to Katakana: add 0x60
                char::from_u32(c as u32 + 0x60).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

/// Convert kana to romaji (Hepburn romanization)
pub fn to_romaji(text: &str) -> String {
    let hiragana = to_hiragana(text);
    let mut result = String::with_capacity(hiragana.len() * 2);

    let map = romaji_map();
    let mut chars = hiragana.chars().peekable();

    while let Some(c) = chars.next() {
        // Check for small tsu (っ) - gemination
        if c == 'っ' {
            if let Some(&next) = chars.peek() {
                if let Some(romaji) = map.get(&next.to_string()) {
                    if let Some(first) = romaji.chars().next() {
                        result.push(first);
                        continue;
                    }
                }
            }
            result.push_str("tsu");
            continue;
        }

        // Check for two-character combinations (with small kana)
        if let Some(&next) = chars.peek() {
            let combo = format!("{}{}", c, next);
            if let Some(romaji) = map.get(&combo) {
                result.push_str(romaji);
                chars.next(); // consume the peeked character
                continue;
            }
        }

        // Single character
        if let Some(romaji) = map.get(&c.to_string()) {
            result.push_str(romaji);
        } else {
            result.push(c);
        }
    }

    result
}

/// Convert kana to IPA (International Phonetic Alphabet)
pub fn to_ipa(text: &str) -> String {
    let hiragana = to_hiragana(text);
    let mut result = String::with_capacity(hiragana.len() * 3);

    let map = ipa_map();
    let mut chars = hiragana.chars().peekable();

    while let Some(c) = chars.next() {
        // Check for small tsu (っ) - gemination
        if c == 'っ' {
            // Duplicate the next consonant
            if let Some(&next) = chars.peek() {
                if let Some(ipa) = map.get(&next.to_string()) {
                    if let Some(first) = ipa.chars().next() {
                        if first != 'a'
                            && first != 'e'
                            && first != 'i'
                            && first != 'o'
                            && first != 'ɯ'
                        {
                            result.push(first);
                        }
                    }
                }
            }
            continue;
        }

        // Check for long vowel marker (ー)
        if c == 'ー' {
            result.push('ː');
            continue;
        }

        // Check for two-character combinations
        if let Some(&next) = chars.peek() {
            let combo = format!("{}{}", c, next);
            if let Some(ipa) = map.get(&combo) {
                result.push_str(ipa);
                chars.next(); // consume the peeked character
                continue;
            }
        }

        // Single character
        if let Some(ipa) = map.get(&c.to_string()) {
            result.push_str(ipa);
        } else {
            result.push(c);
        }
    }

    result
}

/// Convert kana to X-SAMPA (phonetic transcription)
pub fn to_xsampa(text: &str) -> String {
    let hiragana = to_hiragana(text);
    let mut result = String::with_capacity(hiragana.len() * 3);

    let map = xsampa_map();
    let mut chars = hiragana.chars().peekable();

    while let Some(c) = chars.next() {
        // Check for small tsu (っ) - gemination
        if c == 'っ' {
            result.push(':');
            continue;
        }

        // Check for long vowel marker (ー)
        if c == 'ー' {
            result.push(':');
            continue;
        }

        // Check for two-character combinations
        if let Some(&next) = chars.peek() {
            let combo = format!("{}{}", c, next);
            if let Some(xsampa) = map.get(&combo) {
                result.push_str(xsampa);
                chars.next();
                continue;
            }
        }

        // Single character
        if let Some(xsampa) = map.get(&c.to_string()) {
            result.push_str(xsampa);
        } else {
            result.push(c);
        }
    }

    result
}

#[allow(clippy::too_many_lines)]
fn romaji_map() -> HashMap<String, &'static str> {
    let mut map = HashMap::new();

    // Basic hiragana
    let pairs = [
        ("あ", "a"),
        ("い", "i"),
        ("う", "u"),
        ("え", "e"),
        ("お", "o"),
        ("か", "ka"),
        ("き", "ki"),
        ("く", "ku"),
        ("け", "ke"),
        ("こ", "ko"),
        ("さ", "sa"),
        ("し", "shi"),
        ("す", "su"),
        ("せ", "se"),
        ("そ", "so"),
        ("た", "ta"),
        ("ち", "chi"),
        ("つ", "tsu"),
        ("て", "te"),
        ("と", "to"),
        ("な", "na"),
        ("に", "ni"),
        ("ぬ", "nu"),
        ("ね", "ne"),
        ("の", "no"),
        ("は", "ha"),
        ("ひ", "hi"),
        ("ふ", "fu"),
        ("へ", "he"),
        ("ほ", "ho"),
        ("ま", "ma"),
        ("み", "mi"),
        ("む", "mu"),
        ("め", "me"),
        ("も", "mo"),
        ("や", "ya"),
        ("ゆ", "yu"),
        ("よ", "yo"),
        ("ら", "ra"),
        ("り", "ri"),
        ("る", "ru"),
        ("れ", "re"),
        ("ろ", "ro"),
        ("わ", "wa"),
        ("を", "wo"),
        ("ん", "n"),
        // Voiced
        ("が", "ga"),
        ("ぎ", "gi"),
        ("ぐ", "gu"),
        ("げ", "ge"),
        ("ご", "go"),
        ("ざ", "za"),
        ("じ", "ji"),
        ("ず", "zu"),
        ("ぜ", "ze"),
        ("ぞ", "zo"),
        ("だ", "da"),
        ("ぢ", "ji"),
        ("づ", "zu"),
        ("で", "de"),
        ("ど", "do"),
        ("ば", "ba"),
        ("び", "bi"),
        ("ぶ", "bu"),
        ("べ", "be"),
        ("ぼ", "bo"),
        ("ぱ", "pa"),
        ("ぴ", "pi"),
        ("ぷ", "pu"),
        ("ぺ", "pe"),
        ("ぽ", "po"),
        // Combinations with small ya/yu/yo
        ("きゃ", "kya"),
        ("きゅ", "kyu"),
        ("きょ", "kyo"),
        ("しゃ", "sha"),
        ("しゅ", "shu"),
        ("しょ", "sho"),
        ("ちゃ", "cha"),
        ("ちゅ", "chu"),
        ("ちょ", "cho"),
        ("にゃ", "nya"),
        ("にゅ", "nyu"),
        ("にょ", "nyo"),
        ("ひゃ", "hya"),
        ("ひゅ", "hyu"),
        ("ひょ", "hyo"),
        ("みゃ", "mya"),
        ("みゅ", "myu"),
        ("みょ", "myo"),
        ("りゃ", "rya"),
        ("りゅ", "ryu"),
        ("りょ", "ryo"),
        ("ぎゃ", "gya"),
        ("ぎゅ", "gyu"),
        ("ぎょ", "gyo"),
        ("じゃ", "ja"),
        ("じゅ", "ju"),
        ("じょ", "jo"),
        ("びゃ", "bya"),
        ("びゅ", "byu"),
        ("びょ", "byo"),
        ("ぴゃ", "pya"),
        ("ぴゅ", "pyu"),
        ("ぴょ", "pyo"),
    ];

    for (kana, romaji) in pairs {
        map.insert(kana.to_string(), romaji);
    }

    map
}

#[allow(clippy::too_many_lines)]
fn xsampa_map() -> HashMap<String, &'static str> {
    let mut map = HashMap::new();

    // X-SAMPA transcription for Japanese
    let pairs = [
        ("あ", "a"),
        ("い", "i"),
        ("う", "M"),
        ("え", "e"),
        ("お", "o"),
        ("か", "ka"),
        ("き", "ki"),
        ("く", "kM"),
        ("け", "ke"),
        ("こ", "ko"),
        ("さ", "sa"),
        ("し", "Ci"),
        ("す", "sM"),
        ("せ", "se"),
        ("そ", "so"),
        ("た", "ta"),
        ("ち", "tCi"),
        ("つ", "tsM"),
        ("て", "te"),
        ("と", "to"),
        ("な", "na"),
        ("に", "Ji"),
        ("ぬ", "nM"),
        ("ね", "ne"),
        ("の", "no"),
        ("は", "ha"),
        ("ひ", "Ci"),
        ("ふ", "p\\M"),
        ("へ", "he"),
        ("ほ", "ho"),
        ("ま", "ma"),
        ("み", "mi"),
        ("む", "mM"),
        ("め", "me"),
        ("も", "mo"),
        ("や", "ja"),
        ("ゆ", "jM"),
        ("よ", "jo"),
        ("ら", "4a"),
        ("り", "4i"),
        ("る", "4M"),
        ("れ", "4e"),
        ("ろ", "4o"),
        ("わ", "wa"),
        ("を", "o"),
        ("ん", "N"),
        // Voiced
        ("が", "ga"),
        ("ぎ", "gi"),
        ("ぐ", "gM"),
        ("げ", "ge"),
        ("ご", "go"),
        ("ざ", "za"),
        ("じ", "dZi"),
        ("ず", "zM"),
        ("ぜ", "ze"),
        ("ぞ", "zo"),
        ("だ", "da"),
        ("ぢ", "dZi"),
        ("づ", "zM"),
        ("で", "de"),
        ("ど", "do"),
        ("ば", "ba"),
        ("び", "bi"),
        ("ぶ", "bM"),
        ("べ", "be"),
        ("ぼ", "bo"),
        ("ぱ", "pa"),
        ("ぴ", "pi"),
        ("ぷ", "pM"),
        ("ぺ", "pe"),
        ("ぽ", "po"),
    ];

    for (kana, xsampa) in pairs {
        map.insert(kana.to_string(), xsampa);
    }

    map
}

/// IPA (International Phonetic Alphabet) mapping for Japanese
#[allow(clippy::too_many_lines)]
fn ipa_map() -> HashMap<String, &'static str> {
    let mut map = HashMap::new();

    // IPA transcription for Japanese
    let pairs = [
        // Basic vowels
        ("あ", "a"),
        ("い", "i"),
        ("う", "ɯ"), // Japanese /u/ is unrounded
        ("え", "e"),
        ("お", "o"),
        // K-series
        ("か", "ka"),
        ("き", "ki"),
        ("く", "kɯ"),
        ("け", "ke"),
        ("こ", "ko"),
        // S-series
        ("さ", "sa"),
        ("し", "ɕi"), // Voiceless alveolo-palatal fricative
        ("す", "sɯ"),
        ("せ", "se"),
        ("そ", "so"),
        // T-series
        ("た", "ta"),
        ("ち", "tɕi"), // Voiceless alveolo-palatal affricate
        ("つ", "tsɯ"), // Voiceless alveolar affricate
        ("て", "te"),
        ("と", "to"),
        // N-series
        ("な", "na"),
        ("に", "ɲi"), // Palatal nasal
        ("ぬ", "nɯ"),
        ("ね", "ne"),
        ("の", "no"),
        // H-series
        ("は", "ha"),
        ("ひ", "çi"), // Voiceless palatal fricative
        ("ふ", "ɸɯ"), // Voiceless bilabial fricative
        ("へ", "he"),
        ("ほ", "ho"),
        // M-series
        ("ま", "ma"),
        ("み", "mi"),
        ("む", "mɯ"),
        ("め", "me"),
        ("も", "mo"),
        // Y-series
        ("や", "ja"),
        ("ゆ", "jɯ"),
        ("よ", "jo"),
        // R-series (tap/flap)
        ("ら", "ɾa"),
        ("り", "ɾi"),
        ("る", "ɾɯ"),
        ("れ", "ɾe"),
        ("ろ", "ɾo"),
        // W-series
        ("わ", "wa"),
        ("を", "o"), // Modern Japanese: same as お
        ("ん", "ɴ"), // Uvular nasal
        // Voiced K-series (G)
        ("が", "ɡa"),
        ("ぎ", "ɡi"),
        ("ぐ", "ɡɯ"),
        ("げ", "ɡe"),
        ("ご", "ɡo"),
        // Voiced S-series (Z)
        ("ざ", "dza"), // Voiced alveolar affricate
        ("じ", "dʑi"), // Voiced alveolo-palatal affricate
        ("ず", "dzɯ"),
        ("ぜ", "dze"),
        ("ぞ", "dzo"),
        // Voiced T-series (D)
        ("だ", "da"),
        ("ぢ", "dʑi"), // Same as じ in modern Japanese
        ("づ", "dzɯ"), // Same as ず in modern Japanese
        ("で", "de"),
        ("ど", "do"),
        // Voiced H-series (B)
        ("ば", "ba"),
        ("び", "bi"),
        ("ぶ", "bɯ"),
        ("べ", "be"),
        ("ぼ", "bo"),
        // P-series
        ("ぱ", "pa"),
        ("ぴ", "pi"),
        ("ぷ", "pɯ"),
        ("ぺ", "pe"),
        ("ぽ", "po"),
        // Combinations with small ya/yu/yo (palatalization)
        ("きゃ", "kʲa"),
        ("きゅ", "kʲɯ"),
        ("きょ", "kʲo"),
        ("しゃ", "ɕa"),
        ("しゅ", "ɕɯ"),
        ("しょ", "ɕo"),
        ("ちゃ", "tɕa"),
        ("ちゅ", "tɕɯ"),
        ("ちょ", "tɕo"),
        ("にゃ", "ɲa"),
        ("にゅ", "ɲɯ"),
        ("にょ", "ɲo"),
        ("ひゃ", "ça"),
        ("ひゅ", "çɯ"),
        ("ひょ", "ço"),
        ("みゃ", "mʲa"),
        ("みゅ", "mʲɯ"),
        ("みょ", "mʲo"),
        ("りゃ", "ɾʲa"),
        ("りゅ", "ɾʲɯ"),
        ("りょ", "ɾʲo"),
        ("ぎゃ", "ɡʲa"),
        ("ぎゅ", "ɡʲɯ"),
        ("ぎょ", "ɡʲo"),
        ("じゃ", "dʑa"),
        ("じゅ", "dʑɯ"),
        ("じょ", "dʑo"),
        ("びゃ", "bʲa"),
        ("びゅ", "bʲɯ"),
        ("びょ", "bʲo"),
        ("ぴゃ", "pʲa"),
        ("ぴゅ", "pʲɯ"),
        ("ぴょ", "pʲo"),
    ];

    for (kana, ipa) in pairs {
        map.insert(kana.to_string(), ipa);
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_hiragana() {
        assert_eq!(to_hiragana("トウキョウ"), "とうきょう");
        assert_eq!(to_hiragana("カタカナ"), "かたかな");
    }

    #[test]
    fn test_to_katakana() {
        assert_eq!(to_katakana("ひらがな"), "ヒラガナ");
        assert_eq!(to_katakana("とうきょう"), "トウキョウ");
    }

    #[test]
    fn test_to_romaji() {
        assert_eq!(to_romaji("とうきょう"), "toukyou");
        assert_eq!(to_romaji("にほん"), "nihon");
        assert_eq!(to_romaji("しんぶん"), "shinbun");
    }

    #[test]
    fn test_gemination() {
        assert_eq!(to_romaji("がっこう"), "gakkou");
        assert_eq!(to_romaji("きって"), "kitte");
    }

    #[test]
    fn test_xsampa() {
        let result = to_xsampa("とうきょう");
        assert!(result.contains("to"));
    }

    #[test]
    fn test_ipa_basic() {
        // Test basic conversion
        assert_eq!(to_ipa("あいうえお"), "aiɯeo");
        assert_eq!(to_ipa("かきくけこ"), "kakikɯkeko");
    }

    #[test]
    fn test_ipa_tokyo() {
        // トウキョウ → とうきょう → toːkʲoː
        let ipa = to_ipa("とうきょう");
        assert!(ipa.contains("to"));
        assert!(ipa.contains("kʲo")); // Palatalized k
    }

    #[test]
    fn test_ipa_palatalization() {
        // きゃ should be kʲa (palatalized)
        assert_eq!(to_ipa("きゃ"), "kʲa");
        assert_eq!(to_ipa("しゃ"), "ɕa");
        assert_eq!(to_ipa("ちゃ"), "tɕa");
    }

    #[test]
    fn test_ipa_gemination() {
        // っ should double the consonant
        let ipa = to_ipa("がっこう");
        assert!(ipa.contains("kk")); // doubled k
    }

    #[test]
    fn test_ipa_long_vowel() {
        // ー should become ː (length mark)
        let ipa = to_ipa("ラーメン");
        assert!(ipa.contains("ː"));
    }
}
