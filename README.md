# Experimental Artifact

**"From Perfect to Approximate Hints: Efficient LWE Secret Recovery Leveraging Low Hamming Weight"** (S&P 2026)

Reproduces Figures 1–7 and Table 2. Single self-contained SageMath file; no external dependencies beyond SageMath + matplotlib.

## Requirements

- SageMath ≥ 9.6 (tested on 10.2)
- matplotlib (bundled with SageMath)

## Quick Start

```bash
# List all available figures and their parameters
sage paper_experiments.sage --list

# Reproduce a single figure (ternary mode, paper default)
sage paper_experiments.sage --figure 6

# Reproduce all figures + Table 2
sage paper_experiments.sage --figure all --table2

# Quick smoke test (fewer trials)
sage paper_experiments.sage --figure 1 --trials 10

# Deterministic run (fully reproducible)
sage paper_experiments.sage --figure 1 --seed 42
```

## Secret-Key Modes

The code supports two secret distributions via `--mode`:

| Mode | Secret distribution | Default |
|------|-------------------|---------|
| `ternary` | s ∈ {−1, 0, 1}ⁿ, Hamming weight h | **yes** (paper setting) |
| `binary` | s ∈ {0, 1}ⁿ, Hamming weight h | no |

```bash
sage paper_experiments.sage --mode binary  --figure 1
sage paper_experiments.sage --mode ternary --figure 1   # same as default
```

Algorithmic differences between modes (sign handling in argmax, variance comparison, threshold filtering) are documented inline in the source code at each branch point.

## Figures and Parameters

| Fig | x-axis | Fixed parameters | m rule | Algorithms | Trials |
|-----|--------|-----------------|--------|------------|--------|
| 1 | n (÷100) | h=8, q≈2¹⁰⁰, σ=3.19 | ⌊2h log₂h⌉ = 48 | 1, 2, 3 | 100 |
| 2 | n (÷100) | h=8, q≈2¹⁰⁰ | ⌊2h log₂h⌉ = 48 | 3 only (σ=q/8 vs σ=3.19) | 100 |
| 3 | t (σ = q/2ᵗ) | n=512, h=8, m=48 | fixed | 1, 2, 3 | 100 |
| 4 | t (q ≈ 2¹⁰⁰ᵗ) | n=512, h=8, m=48, σ=3.19 | fixed | 1, 2, 3 | 100 |
| 5 | t (m = ⌊t·h·log₂h⌉) | n=512, h=8, q≈2¹⁰⁰, σ=3.19 | factor of t | 1, 2, 3 | 100 |
| 6 | h ∈ {4,...,14} | n=512, **m=48 fixed**, q≈2¹⁰⁰, σ=3.19 | fixed | 1, 2, 3 | 1100 |
| 7 | h ∈ {4,...,14} | n=1024, q≈2¹⁰⁰, σ=3.19 | **⌊2h log₂h⌉** | 1, 2, 3 | 100 |

**Figs 6 vs 7:** Fig 6 fixes m=48, so success decreases as h grows. Fig 7 scales m=⌊2h log₂h⌉ with h, so success increases.

σ = 3.19 = 8/√(2π) is the standard discrete Gaussian width used by OpenFHE, SEAL, and Lattigo.

### Table 2

Concrete FHE parameter sets with m = ⌊2h log₂h⌉:

| Scheme | Hints | n | log₂q | h |
|--------|-------|---|-------|---|
| [7],[28] | approximate | 2¹⁵ | 161 | 32, 64 |
| [12] | approximate | 2¹⁴, 2¹⁶ | 66, 438 | 32 |
| [34] | perfect | 2¹⁴, 2¹⁵ | 40 | 128, 192 |

```bash
sage paper_experiments.sage --table2
sage paper_experiments.sage --table2 --trials 10   # fast smoke test
```

**Note:** The last two rows (n=2¹⁴–2¹⁵, h=128–192) involve large matrices and may require significant memory/time. Use `--trials 10` for a quick check.

## Customizing Parameters

### Modifying an existing figure

Edit the `PAPER_FIGURES` dictionary in the source. Each entry has the structure:

```python
'6': {
    'desc'       : 'Success rate vs h  (n=512, m=48 fixed)',
    'sweep'      : 'h',                              # variable on x-axis
    'sweep_vals' : [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # x-axis values
    'fixed'      : {'n': 512, 'm': 48, 'q': Q100, 'sigma': SIGMA_DEFAULT},
    'm_rule'     : 'fixed',       # how m is determined (see below)
    'algos'      : ['nmw', 'ccp', 'ours'],
    'trials'     : 1100,
    ...
}
```

For example, to run Figure 6 with n=1024 instead of 512:

```python
'fixed': {'n': 1024, 'm': 48, 'q': Q100, 'sigma': SIGMA_DEFAULT},
```

### Adding a new figure

Add a new entry to `PAPER_FIGURES`:

```python
'8': {
    'desc'       : 'My custom experiment',
    'sweep'      : 'h',
    'sweep_vals' : [4, 8, 12, 16, 20, 24, 32],
    'fixed'      : {'n': 2048, 'm': 96, 'q': int(next_prime(2**200)), 'sigma': SIGMA_DEFAULT},
    'm_rule'     : 'fixed',
    'algos'      : ['ours', 'ccp'],
    'trials'     : 50,
    'x_transform': lambda v: v,
    'x_label'    : 'Hamming weight $h$',
    'x_ticks'    : [4, 8, 12, 16, 20, 24, 32],
    'y_ticks'    : [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'legend_loc' : 'upper right',
},
```

Then run: `sage paper_experiments.sage --figure 8`

### Sweep variable types

The `sweep` field determines what the x-axis varies. The `m_rule` field determines how m is set.

| `sweep` value | Meaning | Example |
|--------------|---------|---------|
| `'n'` | Dimension n | Fig 1, 2 |
| `'h'` | Hamming weight h | Fig 6, 7 |
| `'m'` | Number of hints m (directly) | — |
| `'sigma_t'` | σ = q / 2ᵗ, sweep over t | Fig 3 |
| `'q_t'` | q = next_prime(2¹⁰⁰ᵗ), sweep over t | Fig 4 |
| `'m_factor'` | m = ⌊t · h · log₂h⌉, sweep over t | Fig 5 |

| `m_rule` value | How m is determined |
|----------------|-------------------|
| `'fixed'` | Use `m` from the `fixed` dict |
| `'scaling'` | m = round(2h · log₂h) |
| `'m_factor'` | m = round(t · h · log₂h), where t is the sweep value |

### Adding a new Table 2 row

Append to the `TABLE2_ROWS` list:

```python
TABLE2_ROWS = [
    ...
    # (scheme_label, hint_type, n, log2q, h, default_trials)
    ('MyScheme', 'approximate', 2**14, 100, 64, 100),
]
```

### Algorithm selection

The `algos` list controls which algorithms run. Any subset of:

| Key | Algorithm | Paper reference |
|-----|-----------|----------------|
| `'ours'` | Greedy inner-product with residual update | Algorithm 3 (this paper) |
| `'nmw'` | Per-column variance comparison | Algorithm 1, [26] (Nolte et al.) |
| `'ccp'` | Threshold filtering + column-mean ranking | Algorithm 2, [27] (Cheon et al.) |

To run only Algorithm 3: `'algos': ['ours']`

### Normalized score variant

The `--normalize-score` flag divides `⟨A_i, c⟩` by `‖A_i‖²` before argmax, giving a least-squares estimate of s_i. This is a variance-reducing heuristic **not in the paper**; the default uses raw inner products (paper-faithful).

```bash
sage paper_experiments.sage --figure 1 --normalize-score
```

## Outputs

```
results/
  figure_1_ternary.json      # raw data + 95% Wilson CIs
  figure_6_ternary.json
  table2_ternary.json
figures/
  figure_1_ternary.pdf       # publication-quality plot
  figure_1_ternary.png
  figure_6_ternary.pdf
  table2_ternary.pdf         # table rendered as image
```

### JSON structure (figures)

```json
{
  "figure": "6",
  "mode": "ternary",
  "trials": 1100,
  "base_seed": 42,
  "data": {
    "ours": [
      {
        "sv": 8, "x": 8.0,
        "n": 512, "h": 8, "m": 48, "sigma": 3.19,
        "successes": 523, "trials": 1100,
        "p": 0.4754, "ci_lo": 0.447, "ci_hi": 0.504
      }
    ]
  }
}
```

## Reproducibility

Use `--seed` for deterministic runs. Each (sweep point, trial) gets seed = `BASE + point_index * 10⁶ + trial_index`, so results are identical across runs with the same seed.

```bash
# These two runs produce identical results:
sage paper_experiments.sage --figure 1 --seed 42
sage paper_experiments.sage --figure 1 --seed 42
```

Re-plot from saved data without re-running experiments:

```bash
sage paper_experiments.sage --plot-only 6
```

## Bug Fixes

Two bugs in the original `Integer_LWE_solver` code were fixed:

1. **Shared instance across trials.** A, s, e, b were generated once outside the trial loop, so all 100 trials tested the same instance (biasing success rate to either 0% or 100%). Fix: fresh instance per trial.

2. **Variable name collision.** In the CCP block, `b = a*s + D()` overwrote the shared vector `b` with a scalar, corrupting subsequent Algorithm 3 calls in the same trial. Fix: separate variable `b_ccp`.
