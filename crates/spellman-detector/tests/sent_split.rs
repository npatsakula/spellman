//! Tests for the example-local sentence splitter (`detect_md/sent.rs`)
//! — examples don't run unit tests under `cargo test`, so the module is
//! pulled in by path. The splitter is sweep policy, deliberately NOT
//! part of the crate's public detection surface.

#[path = "../examples/detect_md/sent.rs"]
mod sent;

fn parts(line: &str) -> Vec<String> {
    sent::split_line(line)
        .iter()
        .map(|s| s.to_string())
        .collect()
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
    assert!(sent::split_line("   ").is_empty());
    assert!(sent::split_line("").is_empty());
}

#[test]
fn glue_joins_attribution_tails() {
    // the classic: quote + tail recombine into one evidential unit
    let pieces = parts("«Да!» — сказал он.");
    assert_eq!(sent::glue_short(pieces, 20), ["«Да!» — сказал он."]);
    // question + closer + attribution works the same
    let pieces = parts("«Эй, кто там?» — крикнул вахтёр");
    assert_eq!(
        sent::glue_short(pieces, 20),
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
    assert_eq!(sent::glue_short(pieces.clone(), 20), pieces);
}

#[test]
fn glue_ignores_short_without_closer_before_it() {
    // dash fragment whose predecessor ends in a plain terminator: a new
    // dialogue turn, not an attribution
    let pieces = parts("Он замолчал. — Ну и что.");
    assert_eq!(sent::glue_short(pieces.clone(), 20), pieces);
}

#[test]
fn glue_isolated_short_is_kept_for_the_filter() {
    // no neighbor: nothing to glue to, the caller's min-chars filter drops it
    assert_eq!(sent::glue_short(vec!["— Нет.".to_string()], 20), ["— Нет."]);
}
