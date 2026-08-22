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
        if is_attribution_tail
            && let Some(last) = out.last_mut() {
                last.push(' ');
                last.push_str(&piece);
                continue;
            }
        out.push(piece);
    }
    out
}
