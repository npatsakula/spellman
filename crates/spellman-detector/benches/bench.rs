//! Benchmarks: feature extraction + CPU detection throughput.
//!
//! Model-dependent benchmarks run only when `SPELLMAN_MODEL` points at an
//! exported model directory; the feature pipeline is benchmarked standalone.

use std::path::PathBuf;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use spellman_detector::features::{
    WordClass, bucket_tokens, classify_word, fill_signed_indices, token_keys, FeatureConfig,
};
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

/// Word classification in isolation (the per-word pass `for_each_key` runs
/// before packing): splits and classifies the same words the extraction
/// benches see.
fn bench_classify(c: &mut Criterion) {
    let mut group = c.benchmark_group("classify");
    group.throughput(Throughput::Bytes(RUS_TEXT.len() as u64));
    group.bench_function("rus words", |b| {
        b.iter(|| {
            let mut real = 0usize;
            for w in black_box(RUS_TEXT).split_whitespace() {
                let w = w.strip_prefix('#').unwrap_or(w);
                if !w.is_empty() && classify_word(w) == WordClass::Word {
                    real += 1;
                }
            }
            real
        })
    });
    group.finish();
}

/// The zero-copy streaming path the JIT input fill uses (signed indices
/// written straight into the plan's row buffer, k-capped).
fn bench_fill_indices(c: &mut Criterion) {
    let cfg = FeatureConfig::default();
    let hasher = FeatureHasher::default();
    let mut dst = vec![0i32; 1024];
    let mut group = c.benchmark_group("fill_indices");
    group.throughput(Throughput::Bytes(RUS_TEXT.len() as u64));
    group.bench_function("rus fmix32 d17 k1024", |b| {
        b.iter(|| black_box(fill_signed_indices(black_box(RUS_TEXT), black_box(&cfg), black_box(&hasher), 17, 1024, &mut dst)))
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

    let texts: Vec<&str> = std::iter::repeat_n(RUS_TEXT, max_batch / 2)
        .chain(std::iter::repeat_n(ENG_TEXT, max_batch / 2))
        .collect();
    let bytes: u64 = texts.iter().map(|t| t.len() as u64).sum();

    let mut group = c.benchmark_group("detect_bulk");
    group.throughput(Throughput::Bytes(bytes));
    group.sample_size(20);
    group.bench_function(format!("svit k={k} b={max_batch}"), |b| {
        b.iter(|| black_box(bulk.detect_batch(black_box(&texts)).expect("bulk detect")))
    });
    group.finish();
}

/// Hash stage in isolation: the same key stream through the per-key scalar
/// path and the 8-key block path, benched in one run (shared machine state).
fn bench_hash_stage(c: &mut Criterion) {
    let hasher = FeatureHasher::default();
    let keys: Vec<u64> = token_keys(RUS_TEXT, &FeatureConfig::default());
    let mut dst = vec![0i32; 1024];
    let mut group = c.benchmark_group("hash_stage");
    group.throughput(Throughput::Elements(keys.len() as u64));
    group.bench_function("scalar per-key", |b| {
        b.iter(|| {
            for (d, &k) in dst.iter_mut().zip(keys.iter().cycle()).take(1024) {
                *d = black_box(hasher.signed_index(black_box(k), 17));
            }
        })
    });
    group.bench_function("block8", |b| {
        b.iter(|| {
            for (out, chunk) in dst
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(keys.as_chunks::<8>().0.iter().cycle())
                .take(128)
            {
                hasher.signed_index_block(black_box(chunk), 17, out);
            }
        })
    });
    group.finish();
}

criterion_group!(benches, bench_token_keys, bench_bucket_tokens, bench_classify, bench_fill_indices, bench_hash_stage, bench_bulk);
criterion_main!(benches);
