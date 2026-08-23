//! Sentence splitting for document sweeps — ours, not UAX #29.
//!
//! `unicode-segmentation`'s sentence boundaries behaved opaquely on the
//! corpus this exists for (Война и мир: dialogue dashes, «quotes»,
//! initials): the fragments it emitted classified erratically where the
//! same text split at plain linguistic boundaries classified cleanly.
//! This splitter implements the small rule set a language-identification
//! sweep actually needs, with no dependencies and visible behavior:
//!
//! - a boundary sits after a run of terminators (`. ! ? …`, so `?!`,
//!   `...`, `?..` are one run) plus any closing quotes/brackets that
//!   follow (`.»`, `!)`), when the run ends the line or meets whitespace;
//! - a dot after a single letter is an initial (`Л. Н. Толстой`) — no
//!   boundary, twice;
//! - a dot between digits is a decimal (`3.5`) — no boundary;
//! - `, —` attribution never splits; `. — ` (terminator before a dash)
//!   does, which is exactly the dialogue-turn rule;
//! - a line with no terminator is one sentence.
//!
//! The rules trade linguistic generality for predictability — the caller
//! is a language detector, not a translator.
//!
//! `glue_short` lives here too: merging sub-threshold fragments into
//! neighbors is the same sweep policy (attribution tails rejoin their
//! sentence), not a detection concern. Known cost: two-plus-letter
//! abbreviations (`См.`, `стр.`) split when followed by a space.
//!
//! Unit tests live below (`cargo test --example detect_md`); examples
//! are outside plain `cargo test`'s target set, which is deliberate —
//! the splitter is sweep policy, not part of the crate's detection
//! surface.

/// Sentence terminators; a maximal run of these plus trailing closers
/// forms a boundary when followed by whitespace or end of line.
const TERMINATORS: [char; 4] = ['.', '!', '?', '…'];

/// Quotes/brackets glued to a terminator on its right (`.»`, `?!)`) —
/// the boundary lands after them.
const CLOSERS: &str = "»\"”')]";

fn is_terminator(c: char) -> bool {
    TERMINATORS.contains(&c)
}

fn is_closer(c: char) -> bool {
    CLOSERS.contains(c)
}

/// True when the dot at `chars[dot]` terminates a single-letter initial
/// (`Л. Н. Толстой`): the char before it is a lone letter.
fn is_initial(chars: &[(usize, char)], dot: usize) -> bool {
    match dot.checked_sub(1).map(|k| chars[k].1) {
        Some(p) if p.is_alphabetic() => !matches!(
            dot.checked_sub(2).map(|k| chars[k].1),
            Some(b) if b.is_alphabetic()
        ),
        _ => false,
    }
}

/// Split one line into trimmed, non-empty sentence slices.
///
/// Splitting the caller's text into lines first (and dropping markdown
/// headings) stays the caller's policy; this function is purely the
/// within-line segmentation.
pub fn split_line(line: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    let mut out = Vec::new();
    let mut start = 0;

    let mut i = 0;
    while i < chars.len() {
        if !is_terminator(chars[i].1) {
            i += 1;
            continue;
        }
        // consume the terminator run, then any closers glued to it
        let run = i;
        let mut j = i;
        while j < chars.len() && is_terminator(chars[j].1) {
            j += 1;
        }
        while j < chars.len() && is_closer(chars[j].1) {
            j += 1;
        }
        let end = if j < chars.len() {
            chars[j].0
        } else {
            line.len()
        };

        // digit.digit = decimal (3.5, 2026.08): the dot binds tighter
        // than the sentence
        let decimal = match run.checked_sub(1).map(|k| chars[k].1) {
            Some(p) if p.is_numeric() => j < chars.len() && chars[j].1.is_numeric(),
            _ => false,
        };

        let boundary = (j >= chars.len() || chars[j].1.is_whitespace())
            && !is_initial(&chars, run)
            && !decimal;

        if boundary {
            let piece = line[start..end].trim();
            if !piece.is_empty() {
                out.push(piece);
            }
            start = end;
        }
        i = j;
    }

    let tail = line[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

/// Glue attribution tails back onto their sentence: a sub-`min_chars`
/// fragment starting with an em-dash whose predecessor ends with a
/// closing quote/bracket («Да!» / "— сказал он.") is one linguistic
/// unit and is merged. Nothing else merges — MEASURED: broadly gluing
/// every short fragment resurrected the rapid-fire dialogue the length
/// filter used to drop ("— Нет. — Да. — Отвяжись." are three speakers,
/// not one sentence), and those glued units classified so badly that
/// tyv went from 77 to 1,904 on the voyna sweep. Sub-threshold
/// fragments with no quote to attach to are returned as-is; the
/// caller's length filter drops them, which is the right call.
pub fn glue_short(pieces: Vec<String>, min_chars: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for piece in pieces {
        let is_attribution_tail = piece.starts_with('—')
            && piece.chars().count() < min_chars
            && out
                .last()
                .is_some_and(|prev| prev.chars().last().is_some_and(is_closer));
        if is_attribution_tail && let Some(last) = out.last_mut() {
            last.push(' ');
            last.push_str(&piece);
            continue;
        }
        out.push(piece);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{glue_short, split_line};

    fn parts(line: &str) -> Vec<String> {
        split_line(line).iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn plain_sentences() {
        assert_eq!(
            parts("Однажды в студёную зимнюю пору. Я из лесу вышел."),
            ["Однажды в студёную зимнюю пору.", "Я из лесу вышел."]
        );
    }

    #[test]
    fn dialogue_attribution_stays_whole() {
        // comma + dash mid-sentence is attribution, not a turn
        assert_eq!(
            parts("— Моя жена, — продолжал князь Андрей, — прекрасная женщина."),
            ["— Моя жена, — продолжал князь Андрей, — прекрасная женщина."]
        );
        // terminator before a dash = new dialogue turn
        assert_eq!(
            parts("— Нет, — сказал он. — Да."),
            ["— Нет, — сказал он.", "— Да."]
        );
        // question + closing quote + dash attribution: two pieces
        assert_eq!(
            parts("«Эй, кто там?» — крикнул вахтёр"),
            ["«Эй, кто там?»", "— крикнул вахтёр"]
        );
    }

    #[test]
    fn initials_never_split() {
        assert_eq!(
            parts("Л. Н. Толстой родился в 1828 г. и умер в 1910 г."),
            ["Л. Н. Толстой родился в 1828 г. и умер в 1910 г."]
        );
        assert_eq!(parts("Роман написан Толстым."), ["Роман написан Толстым."]);
    }

    #[test]
    fn decimals_and_hosts_stay_whole() {
        assert_eq!(
            parts("Цена 3.5 рубля. Дорого!"),
            ["Цена 3.5 рубля.", "Дорого!"]
        );
        assert_eq!(
            parts("Сайт example.com/abc открыт. Позже."),
            ["Сайт example.com/abc открыт.", "Позже."]
        );
    }

    #[test]
    fn terminator_runs_and_closers() {
        assert_eq!(parts("Что?.. О!"), ["Что?..", "О!"]);
        assert_eq!(parts("«Да!» — сказал он."), ["«Да!»", "— сказал он."]);
        assert_eq!(parts("(Приказ № 5.) Всё."), ["(Приказ № 5.)", "Всё."]);
        assert_eq!(parts("Правда?! Как же!"), ["Правда?!", "Как же!"]);
    }

    #[test]
    fn no_terminator_is_one_piece() {
        assert_eq!(
            parts("Просто строка без конца"),
            ["Просто строка без конца"]
        );
        assert!(split_line("   ").is_empty());
        assert!(split_line("").is_empty());
    }

    #[test]
    fn glue_joins_attribution_tails() {
        // the classic: quote + tail recombine into one evidential unit
        let pieces = parts("«Да!» — сказал он.");
        assert_eq!(glue_short(pieces, 20), ["«Да!» — сказал он."]);
        // question + closer + attribution works the same
        let pieces = parts("«Эй, кто там?» — крикнул вахтёр");
        assert_eq!(
            glue_short(pieces, 20),
            ["«Эй, кто там?» — крикнул вахтёр"]
        );
    }

    #[test]
    fn glue_leaves_rapid_dialogue_alone() {
        // three speakers, no closers: NOT one sentence; the length filter
        // drops these (measured: gluing them resurrected 1.8k tyv picks)
        let pieces = vec![
            "— Нет.".to_string(),
            "— Да.".to_string(),
            "— Отвяжись.".to_string(),
        ];
        assert_eq!(glue_short(pieces.clone(), 20), pieces);
    }

    #[test]
    fn glue_ignores_short_without_closer_before_it() {
        // dash fragment whose predecessor ends in a plain terminator: a new
        // dialogue turn, not an attribution
        let pieces = parts("Он замолчал. — Ну и что.");
        assert_eq!(glue_short(pieces.clone(), 20), pieces);
    }

    #[test]
    fn glue_isolated_short_is_kept_for_the_filter() {
        // no neighbor: nothing to glue to, the caller's min-chars filter drops it
        assert_eq!(glue_short(vec!["— Нет.".to_string()], 20), ["— Нет."]);
    }
}
