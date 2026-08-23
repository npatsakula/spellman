# spellman-cli

`spellman` — the command-line interface of the
[spellman](https://github.com/npatsakula/spellman) Cyrillic-optimized
language detector: `detect` (stdin → ISO 639-3), `eval` (accuracy +
throughput), `bench` (probes + timings).

```bash
echo "Съешь ещё этих мягких французских булок" | spellman detect
# rus

# The model comes from the Hugging Face Hub through the standard HF cache
# (a plain path also works; default ./model):
spellman detect --model hf:vpermilp/spellman
spellman eval   --model hf:vpermilp/spellman eval.tsv
spellman bench  --single
```

See the [repository README](https://github.com/npatsakula/spellman) for the
detector crate, benchmarks and the training pipeline.

> **Publishing status:** this crate is not yet on crates.io — it depends on
> `spellman-detector`, which depends on
> [svod](https://github.com/npatsakula/svod) via git (crates.io rejects git
> dependencies, and the svod release on crates.io predates APIs spellman
> uses). It will be published together with the detector once a matching
> svod release lands; until then build it from the repository:
> `cargo build --release -p spellman-cli`.
