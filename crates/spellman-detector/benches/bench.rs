//! Benchmarks: feature extraction + CPU detection throughput.
//!
//! Model-dependent benchmarks run only when `SPELLMAN_MODEL` points at an
//! exported model directory; the feature pipeline is benchmarked standalone.

use std::path::PathBuf;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use spellman_detector::features::{bucket_tokens, token_keys, FeatureConfig};
use spellman_detector::hash::FeatureHasher;

const RUS_TEXT: &str = "Съешь ещё этих мягких французских булок да выпей же чаю. \
    Съешь ещё этих мягких французских булок да выпей же чаю. \
    Быстрая бурая лиса прыгает через ленивую собаку.";

const ENG_TEXT: &str = "The quick brown fox jumps over the lazy dog. \
    Pack my box with five dozen liquor jugs. \
    How vexingly quick daft zebras jump when the sun rises.";

fn bench_token_keys(c: &mut Criterion) {
    let cfg = FeatureConfig::default();
    let mut group = c.benchmark_group("token_keys");
    for (name, text) in [("rus", RUS_TEXT), ("eng", ENG_TEXT)] {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::new("pangrams", name), &text, |b, text| {
            b.iter(|| black_box(token_keys(black_box(text), black_box(&cfg)).len()))
        });
    }
    group.finish();
}

fn bench_bucket_tokens(c: &mut Criterion) {
    let cfg = FeatureConfig::default();
    let hasher = FeatureHasher::default();
    let mut group = c.benchmark_group("bucket_tokens");
    group.throughput(Throughput::Bytes(RUS_TEXT.len() as u64));
    group.bench_function("rus fmix32 d17", |b| {
        b.iter(|| black_box(bucket_tokens(black_box(RUS_TEXT), black_box(&cfg), black_box(&hasher), 17)).len())
    });
    group.finish();
}

/// Bulk batched detection through the svod JIT plan
/// (`SPELLMAN_MODEL=... SPELLMAN_K=128 SPELLMAN_MAX_BATCH=512 cargo bench`).
fn bench_bulk(c: &mut Criterion) {
    let Ok(model_dir) = std::env::var("SPELLMAN_MODEL") else {
        return;
    };
    let k: usize = std::env::var("SPELLMAN_K").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
    let max_batch: usize =
        std::env::var("SPELLMAN_MAX_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let mut bulk = spellman_detector::BulkDetector::load(&PathBuf::from(&model_dir), k, max_batch)
        .expect("prepare bulk plan");

    let texts: Vec<&str> = std::iter::repeat(RUS_TEXT).take(max_batch / 2).chain(std::iter::repeat(ENG_TEXT).take(max_batch / 2)).collect();
    let bytes: u64 = texts.iter().map(|t| t.len() as u64).sum();

    let mut group = c.benchmark_group("detect_bulk");
    group.throughput(Throughput::Bytes(bytes));
    group.sample_size(20);
    group.bench_function(format!("svit k={k} b={max_batch}"), |b| {
        b.iter(|| black_box(bulk.detect_batch(black_box(&texts)).expect("bulk detect")))
    });
    group.finish();
}

criterion_group!(benches, bench_token_keys, bench_bucket_tokens, bench_bulk);
criterion_main!(benches);
