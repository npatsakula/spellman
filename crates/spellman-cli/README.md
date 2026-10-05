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

Install with `cargo install spellman-cli` (the binary is named `spellman`).
