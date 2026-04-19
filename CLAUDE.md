# Claude Rules

## Communication Style

Respond terse like smart caveman. All technical substance stay. Only fluff die.

Rules:
- Drop: articles (a/an/the), filler (just/really/basically), pleasantries, hedging
- Fragments OK. Short synonyms. Technical terms exact. Code unchanged.
- Pattern: [thing] [action] [reason]. [next step].
- Not: "Sure! I'd be happy to help you with that."
- Yes: "Bug in auth middleware. Fix:"

Switch level: /caveman lite|full|ultra
Stop: "stop caveman" or "normal mode"

Auto-Clarity: drop caveman for security warnings, irreversible actions, user confused. Resume after.

Boundaries: code/commits/PRs written normal.

---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                        # install / update dependencies
uv run main.py single          # run single-pass experiment
uv run main.py single --eb_n0_db 5 --threshold 0.2   # with CLI overrides
uv run main.py sweep           # stub — NotImplementedError
uv run main.py minimize        # stub — NotImplementedError
```

All scalar params from `config.yaml` can be overridden on any subcommand. `single`-section params (`eb_n0_db`, `bandpass_bw`, `corr_len`, `threshold`) are passed after the `single` subcommand.

## Architecture

Pipeline: **bits → TX pulse → AWGN → bandpass filter → sliding correlator → decisions**

### Signal conventions
- `bits`: sparse `float64` array (0/1) with guaranteed `min_gap_bits` spacing between ones.
- TX signal: strictly non-negative (unipolar laser pulse), built via `np.convolve(impulse_train, tx_pulse)`.
- Correlation template: zero-mean bipolar (integral ≈ 0), shaped like `−1…−1 | +1…+1 | −1…−1`. Intentionally different from TX pulse — `make_tx_pulse` vs `make_pulse_template` in `src/pulse_gen.py`.
- `sps` (samples per symbol) is the upsample rate. All sample-rate arrays have length `n_bits * sps`.

### Data flow
`src/runner.py` is the shared core. `run_trial(...)` returns `TrialResult(ber, snr_out, stages, corr_threshold, decisions)`. All experiment modes call this. `stages` is an ordered dict of named waveforms passed directly to `plots.plot_waveforms`.

### Run modes (`experiments/`)
- `single.py`: one pass, fixed params from `config.yaml single:` section → saves `output/single/templates.png` and `output/single/waveforms.png`.
- `sweep.py`, `minimize.py`: stubs ready to implement. Sweep receives full `p` dict with list params (`bandpass_bw_sweep`, `correlator_len_sweep`, `eb_n0_range_db`). Minimize should wrap `run_trial` in `scipy.optimize` with `p["single"]` as starting point.

### Plotting (`src/plots.py`)
All figures saved headless (Agg backend). `plot_waveforms` auto-detects bit-rate signals (`len(sig) * sps == total_samples`) and renders them as bars; sample-rate signals use line plots. Threshold line drawn on any stage whose key contains `"Correlator"`.

### Config loading (`main.py: load_params`)
YAML loaded, `eb_n0_range_db` expanded via `np.arange(*...)`. CLI overrides split into top-level params and `single:`-section params (`eb_n0_db`, `bandpass_bw`, `corr_len`, `threshold`).
