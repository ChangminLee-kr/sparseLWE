#!/usr/bin/env sage
"""
paper_experiments.sage
======================
"From Perfect to Approximate Hints: Efficient LWE Secret Recovery
 Leveraging Low Hamming Weight" (S&P 2026)

Reproduces Figures 1-7 and Table 2 in two secret-key distribution modes:
  --mode binary   : s in {0,1}^n,    Hamming weight h
  --mode ternary  : s in {-1,0,1}^n, Hamming weight h  (paper setting)

Bug fixes applied relative to the original Integer_LWE_solver code:
  Bug 1) A, s, e, b were generated only once outside the trial loop,
         so all 100 trials shared the same instance (biasing success to 0 or 1).
         Fix: generate a fresh instance inside each trial.
  Bug 2) Inside the CCP block, "b = a*s + D()" overwrote the shared vector b
         with a scalar, corrupting subsequent Algorithm 3 calls.
         Fix: use a separate variable b_ccp for the CCP scalar.

Algorithm differences between binary and ternary modes are documented
at each site where the logic changes.

Usage:
    sage paper_experiments.sage --list
    sage paper_experiments.sage --mode ternary --figure 1
    sage paper_experiments.sage --mode ternary --figure 1 --seed 42
    sage paper_experiments.sage --mode ternary --figure 1 --normalize-score
    sage paper_experiments.sage --mode binary  --figure all --trials 30 --seed 42
    sage paper_experiments.sage --mode ternary --table2 --trials 10   # fast smoke-test
    sage paper_experiments.sage --mode ternary --figure all --table2
    sage paper_experiments.sage --mode ternary --plot-only 7
    sage paper_experiments.sage --calibration
"""

import numpy as np
import random, time, json, os, argparse
from datetime import datetime
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler


# ==============================================================================
#  Exact integer variance helper
#
#  Returns m * sum(v_i^2) - (sum(v_i))^2, which equals m^2 * Var(v) and is
#  always an integer when v is an integer vector.  Using this avoids float64
#  catastrophic cancellation when entries are O(q) with q up to ~2^1100
#  (Figure 4, t=11).  The comparison Var(b - A_i) < Var(b) is equivalent to
#  var_scaled(b - A_i) < var_scaled(b) because the m^2 factor cancels.
# ==============================================================================

def _var_scaled(v):
    """Return m*sum(v^2) - (sum(v))^2 using exact Python integers."""
    m  = len(v)
    s1 = sum(v)
    s2 = sum(x * x for x in v)
    return m * s2 - s1 * s1


# ==============================================================================
#  §0. Shared utility functions
# ==============================================================================

def secret_vector(n, h, mode):
    """
    Sample a sparse secret vector s of dimension n with Hamming weight h.

    mode='binary'  : s_i in {0, 1},    nonzero entries all +1
    mode='ternary' : s_i in {-1, 0, 1}, nonzero entries chosen uniformly from {-1, +1}
    """
    if h > n:
        raise ValueError("h cannot exceed n")
    if mode == 'binary':
        data = [1] * h + [0] * (n - h)
    elif mode == 'ternary':
        signs = [random.choice([-1, 1]) for _ in range(h)]
        data  = signs + [0] * (n - h)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'binary' or 'ternary'.")
    random.shuffle(data)
    return vector(ZZ, data)

def rand_vector(n, q):
    """Return an integer vector with entries drawn uniformly from [-q, q]."""
    return vector(ZZ, [randint(-q, q) for _ in range(n)])

def nonzero_positions(v):
    """Return the sorted list of indices where v is nonzero."""
    return sorted([i for i, val in enumerate(v) if val != 0])

def find_top_indices(lst, k):
    """Return indices of the k largest values in lst (descending order)."""
    return sorted(range(len(lst)), key=lambda x: lst[x], reverse=True)[:k]

def m_from_h(h):
    """Compute m = round(2h * log2(h)), the adaptive sample count used in the paper."""
    if h <= 1:
        return 2
    return int(round(2 * h * log(h, 2)))


# ==============================================================================
#  §1. Single-trial runner
#
#  Binary vs. ternary differences per algorithm:
#
#  Algorithm 3 (Ours) — greedy inner-product with residual update
#    binary : argmax  <A_i, c>       (signal is always positive for s_i=+1)
#             update  c <- c - A_j   (fixed sign s_j=+1)
#    ternary: argmax |<A_i, c>|      (signal is negative when s_i=-1)
#             update  c <- c - sign(<A_j,c>) * A_j  (estimate s_j from sign)
#
#  Algorithm 1 (NMW [26]) — variance comparison
#    binary : flag i if std(b - A_i) < std(b)             (s_i=+1 branch only)
#    ternary: flag i if std(b - A_i) < std(b)  [s_i=+1]
#                   OR std(b + A_i) < std(b)   [s_i=-1, new branch]
#
#  Algorithm 2 (CCP [27]) — threshold filtering + column mean
#    binary : accept row if b_ccp > K*q          (one-sided threshold)
#             rank by  mean(M_j)                 (positive signal for s_j=+1)
#    ternary: accept row if |b_ccp| > K*q        (two-sided threshold)
#             store  sign(b_ccp) * a             (sign-normalize before storing)
#             rank by |mean(M_j)|               (|signal| for s_j=±1)
#
#             Reason for sign-normalization in ternary CCP:
#               For s_j=+1: E[a_j | b>Kq]=+delta, E[a_j | b<-Kq]=-delta
#               Pooling both tails without sign flip cancels the signal exactly.
#               Storing sign(b)*a makes both tails contribute +delta for s_j=+1
#               and -delta for s_j=-1, so |mean| correctly identifies nonzero coords.
# ==============================================================================

def run_one_trial(n, Hw, q, m, sigma, mode, algorithms=('ours', 'nmw', 'ccp'),
                  trial_seed=None, normalize_score=False):
    """
    Generate one random Integer-LWE instance (A, s, e) and test each algorithm.

    Parameters
    ----------
    n               : secret dimension
    Hw              : Hamming weight of s
    q               : modulus (used as range bound for A entries)
    m               : number of hints (rows of A)
    sigma           : noise standard deviation
    mode            : 'binary' or 'ternary'
    algorithms      : subset of ('ours', 'nmw', 'ccp')
    trial_seed      : integer RNG seed for full reproducibility; None = random
    normalize_score : if True, divide <A_i,c> by ||A_i||^2 before argmax
                      (variance-reducing heuristic, not in paper Algorithm 3)

    Returns
    -------
    dict mapping algorithm name -> bool (True = correct recovery)
    """
    if trial_seed is not None:
        random.seed(trial_seed)
        # Sage's randint uses its own RNG; re-seed via set_random_seed
        set_random_seed(trial_seed)
    D        = DiscreteGaussianDistributionIntegerSampler(sigma=sigma)
    s        = secret_vector(n, Hw, mode)
    solindex = nonzero_positions(s)

    # Shared instance for Algorithm 1 and Algorithm 3
    A = matrix(ZZ, m, n)
    for j in range(m):
        A[j] = rand_vector(n, (q - 1) // 2)
    e = vector(ZZ, [D() for _ in range(m)])
    b = A * s + e

    results = {}

    # ------------------------------------------------------------------
    # Algorithm 3 (Ours): greedy argmax with residual update
    # ------------------------------------------------------------------
    if 'ours' in algorithms:
        Sol = []
        At_work = copy(A.transpose())
        b_work  = copy(b)
        for _ in range(Hw):
            C      = At_work * b_work
            C_list = list(C)

            # Score for argmax: raw inner product (paper Algorithm 3) or
            # column-normalized variant (--normalize-score flag).
            #
            # Raw:        score_i = <A_i, c>          (paper-faithful)
            # Normalized: score_i = <A_i, c>/||A_i||^2 (reduces bias from
            #             column-length variation; not in paper Algorithm 3)
            if normalize_score:
                norms_sq = [float(At_work[i].dot_product(At_work[i]))
                            for i in range(At_work.nrows())]
                scores   = [C_list[i] / norms_sq[i] if norms_sq[i] > 0 else 0.0
                            for i in range(len(C_list))]
            else:
                scores = C_list  # paper-original: raw inner product

            if mode == 'binary':
                idx = scores.index(max(scores))
                b_work -= At_work[idx]
            else:
                idx = max(range(len(scores)), key=lambda i: abs(scores[i]))
                sj  = 1 if scores[idx] >= 0 else -1
                b_work -= sj * At_work[idx]
            Sol.append(idx)
            At_work[idx] = zero_vector(m)
        Sol.sort()
        results['ours'] = (Sol == solindex)

    # ------------------------------------------------------------------
    # Algorithm 1 (NMW [26]): variance-based column selection
    #
    # Criterion: select column i if Var(b - A_i) < Var(b)  [s_i = +1]
    #                          or  Var(b + A_i) < Var(b)   [s_i = -1, ternary]
    #
    # Implementation uses exact integer arithmetic via _var_scaled() to avoid
    # float64 catastrophic cancellation at large q (e.g. q ~ 2^1100 in Fig 4).
    # _var_scaled(v) = m*sum(v^2) - (sum(v))^2  ∝  Var(v), all integers.
    # ------------------------------------------------------------------
    if 'nmw' in algorithms:
        Sol_NMW = []
        At      = A.transpose()
        b_list  = list(b)
        vb      = _var_scaled(b_list)
        for i in range(n):
            ai_list = list(At[i])
            if _var_scaled([b_list[j] - ai_list[j] for j in range(m)]) < vb:
                # s_i = +1: removing A_i from b reduces (scaled) variance
                Sol_NMW.append(i)
            elif mode == 'ternary' and                  _var_scaled([b_list[j] + ai_list[j] for j in range(m)]) < vb:
                # s_i = -1: removing -A_i from b reduces variance  [ternary only]
                Sol_NMW.append(i)
        Sol_NMW.sort()
        results['nmw'] = (Sol_NMW == solindex)

    # ------------------------------------------------------------------
    # Algorithm 2 (CCP [27]): threshold filtering + column-mean ranking
    # ------------------------------------------------------------------
    if 'ccp' in algorithms:
        Threshold = round(sqrt(Hw) / sqrt(12))   # K = sqrt(h/12), same as original
        M   = matrix(ZZ, m, n)
        pos = 0
        while pos < m:
            a     = rand_vector(n, (q - 1) // 2)
            b_ccp = a * s + D()    # scalar — separate variable (Bug 2 fix)
            if mode == 'binary':
                # Binary: one-sided filter; positive tail signals s_j=+1.
                if b_ccp > Threshold * q:
                    M[pos] = a
                    pos += 1
            else:
                # Ternary: two-sided filter with sign-normalization.
                if abs(b_ccp) > Threshold * q:
                    b_sign = 1 if b_ccp > 0 else -1
                    M[pos] = b_sign * a   # sign-normalized row
                    pos += 1
        Mt      = M.transpose()
        # Use exact integer column sums instead of float mean to avoid precision
        # loss when matrix entries are O(q) with large q.
        # Ranking by sum is equivalent to ranking by mean (positive scaling).
        if mode == 'binary':
            # Binary: larger positive sum -> more likely s_j=+1.
            Avg = [sum(list(Mt[j])) for j in range(n)]
        else:
            # Ternary: |sum| identifies nonzero columns regardless of sign(s_j).
            Avg = [abs(sum(list(Mt[j]))) for j in range(n)]
        Sol_CCP = find_top_indices(Avg, Hw)
        Sol_CCP.sort()
        results['ccp'] = (Sol_CCP == solindex)

    return results


# ==============================================================================
#  §2. Wilson score confidence interval
# ==============================================================================

def wilson_ci(successes, trials, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    if trials == 0:
        return (0.0, 0.0, 0.0)
    p   = successes / trials
    d   = 1 + z**2 / trials
    ctr = (p + z**2 / (2 * trials)) / d
    hw  = (z / d) * float(sqrt(RR(p * (1-p) / trials + z**2 / (4 * trials**2))))
    return (float(p), max(0.0, ctr - hw), min(1.0, ctr + hw))


# ==============================================================================
#  §3. Figure configurations (Figures 1–7)
#
#  Each entry specifies:
#    sweep       : parameter to vary; special tokens: 'sigma_t', 'q_t', 'm_factor'
#    sweep_vals  : list of values for the sweep variable
#    fixed       : dict of fixed parameters
#    m_rule      : how to determine m:
#                    'fixed'   – use m from 'fixed' dict
#                    'scaling' – m = round(2h log2 h)
#                    'm_factor'– m = round(sv * h * log2 h)  [sv = sweep value]
#    algos       : algorithms to run
#    trials      : number of trials per point (paper original)
#    x_transform : function mapping sweep value to plot x-coordinate
#    x_label, x_ticks, y_ticks : axis configuration
# ==============================================================================

Q100          = int(next_prime(2**100))
SIGMA_DEFAULT = float(8 / sqrt(2 * pi))   # approx 3.19, CKKS default (OpenFHE/SEAL/Lattigo)

PAPER_FIGURES = {

    # Figure 1 ----------------------------------------------------------------
    # Sweep: n (x-axis = n/100).  Three algorithms compared.
    # 21 data points: n = 50, 100, 200, ..., 2000  (matches tikzpicture coords)
    '1': {
        'desc'       : 'Success rate vs n  (h=8, m=2h*log2(h)=48, sigma=3.19)',
        'sweep'      : 'n',
        'sweep_vals' : [50, 100, 200, 300, 400, 500, 600, 700, 800,
                        900, 1000, 1100, 1200, 1300, 1400, 1500,
                        1600, 1700, 1800, 1900, 2000],
        'fixed'      : {'h': 8, 'q': Q100, 'sigma': SIGMA_DEFAULT},
        'm_rule'     : 'scaling',       # m = 2*8*log2(8) = 48
        'algos'      : ['nmw', 'ccp', 'ours'],
        'trials'     : 100,
        'x_transform': lambda v: v / 100,
        'x_label'    : '$n / 100$',
        'x_ticks'    : [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'y_ticks'    : [0.1, 0.3, 0.5, 0.7, 0.9],
        'legend_loc' : 'upper right',
    },

    # Figure 2 ----------------------------------------------------------------
    # Sweep: n.  Algorithm 3 only.  Two curves: sigma=q/8 vs sigma=3.19.
    # 21 data points, same n range as Figure 1.
    '2': {
        'desc'       : 'Success rate vs n  (Alg3: sigma=q/8 vs sigma=3.19)',
        'sweep'      : 'n',
        'sweep_vals' : [50, 100, 200, 300, 400, 500, 600, 700, 800,
                        900, 1000, 1100, 1200, 1300, 1400, 1500,
                        1600, 1700, 1800, 1900, 2000],
        'fixed'      : {'h': 8, 'q': Q100},
        'm_rule'     : 'scaling',
        'algos'      : ['ours'],
        'trials'     : 100,
        'multi_curve': {
            'var'   : 'sigma',
            'values': [Q100 / 8, SIGMA_DEFAULT],
            'labels': [r'$\sigma = q/8$', r'$\sigma = 3.19$'],
        },
        'x_transform': lambda v: v / 100,
        'x_label'    : '$n / 100$',
        'x_ticks'    : [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'y_ticks'    : [0.1, 0.3, 0.5, 0.7, 0.9],
        'legend_loc' : 'upper right',
    },

    # Figure 3 ----------------------------------------------------------------
    # Sweep: t where sigma = q / 2^t.
    # t=0 (sigma=q) omitted: noise equals signal magnitude, trivially fails.
    # 7 data points: t = 1, 2, ..., 7  (matches tikzpicture x-coordinates)
    '3': {
        'desc'       : 'Success rate vs sigma  (n=512, h=8, m=48, sigma=q/2^t)',
        'sweep'      : 'sigma_t',
        'sweep_vals' : [1, 2, 3, 4, 5, 6, 7],
        'fixed'      : {'n': 512, 'h': 8, 'q': Q100, 'm': 48},
        'm_rule'     : 'fixed',
        'algos'      : ['nmw', 'ccp', 'ours'],
        'trials'     : 100,
        'x_transform': lambda v: v,
        'x_label'    : r'$t$   (where $\sigma = q / 2^t$)',
        'x_ticks'    : [0, 2, 4, 6, 7],
        'y_ticks'    : [0.1, 0.3, 0.5, 0.6],
        'legend_loc' : 'lower right',
    },

    # Figure 4 ----------------------------------------------------------------
    # Sweep: t where q = next_prime(2^(100t)).
    # 11 data points: t = 1, 2, ..., 11
    '4': {
        'desc'       : 'Success rate vs q  (n=512, h=8, m=48, sigma=3.19)',
        'sweep'      : 'q_t',
        'sweep_vals' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'fixed'      : {'n': 512, 'h': 8, 'm': 48, 'sigma': SIGMA_DEFAULT},
        'm_rule'     : 'fixed',
        'algos'      : ['nmw', 'ccp', 'ours'],
        'trials'     : 100,
        'x_transform': lambda v: v,
        'x_label'    : r'$t$   (where $q \approx 2^{100t}$)',
        'x_ticks'    : [0, 2, 4, 6, 8, 10, 11],
        'y_ticks'    : [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'legend_loc' : 'lower right',
    },

    # Figure 5 ----------------------------------------------------------------
    # Sweep: t where m = floor(t * h * log2(h)).
    # 12 data points: t = 1, 2, ..., 12
    '5': {
        'desc'       : 'Success rate vs m  (n=512, h=8, m=floor(t*h*log2(h)))',
        'sweep'      : 'm_factor',
        'sweep_vals' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'fixed'      : {'n': 512, 'h': 8, 'q': Q100, 'sigma': SIGMA_DEFAULT},
        'm_rule'     : 'm_factor',
        'algos'      : ['nmw', 'ccp', 'ours'],
        'trials'     : 100,
        'x_transform': lambda v: v,
        'x_label'    : r'$t$   (where $m = \lfloor t \cdot h \log_2 h \rceil$)',
        'x_ticks'    : [1, 2, 3, 4, 6, 8, 10, 12],
        'y_ticks'    : [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'legend_loc' : 'lower right',
    },

    # Figure 6 ----------------------------------------------------------------
    # Sweep: h.  m = 48 fixed.
    # 11 data points: h = 4, 5, 6, ..., 14  (every integer, not just even)
    # Paper uses 1100 trials for this figure.
    '6': {
        'desc'       : 'Success rate vs h  (n=512, m=48 fixed)',
        'sweep'      : 'h',
        'sweep_vals' : [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'fixed'      : {'n': 512, 'm': 48, 'q': Q100, 'sigma': SIGMA_DEFAULT},
        'm_rule'     : 'fixed',
        'algos'      : ['nmw', 'ccp', 'ours'],
        'trials'     : 1100,
        'x_transform': lambda v: v,
        'x_label'    : 'Hamming weight $h$',
        'x_ticks'    : [4, 6, 8, 10, 12, 14],
        'y_ticks'    : [0.1, 0.3, 0.5, 0.7, 0.9],
        'legend_loc' : 'upper right',
    },

    # Figure 7 ----------------------------------------------------------------
    # Sweep: h.  m = floor(2h * log2(h)) adaptive.  n=1024.
    # 11 data points: h = 4, 5, 6, ..., 14
    '7': {
        'desc'       : 'Success rate vs h  (n=1024, m=floor(2h*log2(h)) adaptive)',
        'sweep'      : 'h',
        'sweep_vals' : [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'fixed'      : {'n': 1024, 'q': Q100, 'sigma': SIGMA_DEFAULT},
        'm_rule'     : 'scaling',
        'algos'      : ['nmw', 'ccp', 'ours'],
        'trials'     : 100,
        'x_transform': lambda v: v,
        'x_label'    : 'Hamming weight $h$',
        'x_ticks'    : [4, 6, 8, 10, 12, 14],
        'y_ticks'    : [0.1, 0.3, 0.5, 0.7, 0.9],
        'legend_loc' : 'upper right',
    },
}

# Table 2 parameter rows -------------------------------------------------------
# (scheme_label, hint_type, n, log2q, h, default_trials)
TABLE2_ROWS = [
    ('[7],[28]',  'approximate', 2**15, 161,  32, 100),
    ('[7],[28]',  'approximate', 2**15, 161,  64, 100),
    ('[12]',      'approximate', 2**14,  66,  32, 100),
    ('[12]',      'approximate', 2**16, 438,  32, 100),
    ('[34]',      'perfect',     2**14,  40, 128, 100),
    ('[34]',      'perfect',     2**15,  40, 192, 100),
]


# ==============================================================================
#  §4. Parameter resolution helper
# ==============================================================================

def _resolve(cfg, sv, extra):
    """
    Resolve (n, h, q, m, sigma) from a figure config, sweep value, and
    multi-curve extra parameters.
    """
    p     = dict(cfg['fixed'])
    p.update(extra)
    sweep = cfg['sweep']

    if   sweep == 'n':        p['n'] = sv
    elif sweep == 'h':        p['h'] = sv
    elif sweep == 'm':        p['m'] = sv
    elif sweep == 'sigma_t':  p['sigma'] = int(p['q']) / (2 ** sv)
    elif sweep == 'q_t':      p['q'] = int(next_prime(2 ** (sv * 100)))
    elif sweep == 'm_factor':
        h    = int(p['h'])
        p['m'] = max(1, int(round(sv * h * log(h, 2))))

    rule = cfg['m_rule']
    if   rule == 'scaling':   p['m'] = m_from_h(int(p['h']))
    elif rule == 'fixed':     pass
    elif rule == 'm_factor':  pass   # already computed above

    return int(p['n']), int(p['h']), int(p['q']), int(p['m']), float(p['sigma'])


# ==============================================================================
#  §5. Figure experiment loop
# ==============================================================================

def run_figure(fig_name, mode, override_trials=None, outdir='results',
               base_seed=None, normalize_score=False):
    """Run all parameter points for one figure and save JSON results."""
    cfg        = PAPER_FIGURES[fig_name]
    num_trials = override_trials or cfg['trials']
    algos      = cfg['algos']
    is_multi   = 'multi_curve' in cfg

    print(f"\n{'='*68}")
    print(f"  Figure {fig_name} [{mode}]: {cfg['desc']}")
    print(f"  Trials per point: {num_trials}")
    print(f"{'='*68}\n")

    curves = ([(lbl, {cfg['multi_curve']['var']: v})
               for v, lbl in zip(cfg['multi_curve']['values'],
                                 cfg['multi_curve']['labels'])]
              if is_multi else [(None, {})])

    all_results = {}
    total = len(cfg['sweep_vals']) * len(curves)
    idx   = 0
    t0    = time.time()

    for sv in cfg['sweep_vals']:
        for curve_label, extra in curves:
            idx += 1
            n, h, q, m, sigma = _resolve(cfg, sv, extra)

            eta = ''
            if idx > 1:
                elapsed   = time.time() - t0
                remaining = elapsed / (idx - 1) * (total - idx + 1)
                eta       = f'  ETA {remaining:.0f}s'

            lbl = f'  [{curve_label}]' if curve_label else ''
            print(f'  [{idx}/{total}] {cfg["sweep"]}={sv}  '
                  f'n={n} h={h} m={m} sigma={sigma:.4f}{lbl}{eta}')

            counts = {a: 0 for a in algos}
            for t_idx in range(num_trials):
                # Deterministic seed per (point, trial) when base_seed is set.
                # Formula: seed = base_seed + point_idx * 10^6 + trial_idx
                # This ensures different points use different random instances
                # while remaining fully reproducible from the same base_seed.
                tseed = (base_seed + idx * 1_000_000 + t_idx
                         if base_seed is not None else None)
                res = run_one_trial(n, h, q, m, sigma, mode, algorithms=algos,
                                    trial_seed=tseed,
                                    normalize_score=normalize_score)
                for a, ok in res.items():
                    if ok:
                        counts[a] += 1

            for a in algos:
                p_hat, ci_lo, ci_hi = wilson_ci(counts[a], num_trials)
                key = f'{a}|{curve_label}' if curve_label else a
                if key not in all_results:
                    all_results[key] = []
                all_results[key].append({
                    'sv'   : sv,
                    'x'    : float(cfg['x_transform'](sv)),
                    'n': n, 'h': h, 'm': m, 'sigma': sigma,
                    'successes': counts[a], 'trials': num_trials,
                    'p'    : p_hat, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                })
                bar = '\u2588' * int(p_hat * 20)
                print(f'      {key:30s}: {p_hat*100:5.1f}%  '
                      f'[{ci_lo*100:.1f}, {ci_hi*100:.1f}]  {bar}')

    elapsed = time.time() - t0
    print(f'\n  Done in {elapsed:.1f}s\n')

    record = {
        'figure': fig_name, 'mode': mode, 'desc': cfg['desc'],
        'sweep' : cfg['sweep'], 'trials': num_trials,
        'elapsed': elapsed, 'timestamp': datetime.now().isoformat(),
        'base_seed': base_seed,          # None = non-reproducible run
        'normalize_score': normalize_score,
        'data'  : all_results,
    }
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f'figure_{fig_name}_{mode}.json')
    with open(fname, 'w') as f:
        json.dump(record, f, indent=2, default=str)
    print(f'  JSON saved: {fname}')
    return record


# ==============================================================================
#  §6. Table 2 experiment
# ==============================================================================

def run_table2(mode, override_trials=None, outdir='results',
               base_seed=None, normalize_score=False):
    """Run Table 2 experiments on concrete FHE parameter sets.

    WARNING: Table 2 parameters include n=2^15, h=192, m~2913.
    The matrix A has ~9.5*10^7 entries per trial.  With default trials=100
    this can require significant memory and runtime.
    Use --trials 10 for a quick smoke-test, or provide precomputed JSON.
    """
    print(f"\n{'='*68}")
    print(f'  Table 2 [{mode}]: Experimental results on FHE target parameters')
    print(f'  m = floor(2h * log2(h)) for each row')
    print(f"{'='*68}\n")

    hdr = (f"{'Scheme':<10} {'Hints':^11} {'n':>6} {'log2q':>6} "
           f"{'h':>4} {'m':>5} | "
           f"{'Alg1 [26]':>10} {'Alg2 [27]':>10} {'Alg3 Ours':>10}")
    print(hdr)
    print('-' * len(hdr))

    rows = []
    for row_idx, (scheme, hint_type, n, log2q, h, default_trials) in             enumerate(TABLE2_ROWS):
        num_trials = override_trials or default_trials
        q     = int(next_prime(2 ** log2q))
        m     = m_from_h(h)
        sigma = 0.001 if hint_type == 'perfect' else SIGMA_DEFAULT

        counts = {'nmw': 0, 'ccp': 0, 'ours': 0}
        t0 = time.time()
        for t_idx in range(num_trials):
            tseed = (base_seed + row_idx * 1_000_000 + t_idx
                     if base_seed is not None else None)
            res = run_one_trial(n, h, q, m, sigma, mode,
                                algorithms=['nmw', 'ccp', 'ours'],
                                trial_seed=tseed,
                                normalize_score=normalize_score)
            for a, ok in res.items():
                if ok:
                    counts[a] += 1
        elapsed = time.time() - t0

        p_nmw  = counts['nmw']  / num_trials
        p_ccp  = counts['ccp']  / num_trials
        p_ours = counts['ours'] / num_trials
        print(f"{scheme:<10} {hint_type:^11} {n:>6} {log2q:>6} "
              f"{h:>4} {m:>5} | "
              f"{p_nmw:>10.2f} {p_ccp:>10.2f} {p_ours:>10.2f}"
              f"  [{elapsed:.0f}s]")
        rows.append({
            'scheme': scheme, 'hint_type': hint_type, 'mode': mode,
            'n': n, 'log2q': log2q, 'h': h, 'm': m,
            'trials': num_trials, 'elapsed': elapsed,
            'p_nmw': p_nmw, 'p_ccp': p_ccp, 'p_ours': p_ours,
        })

    print('-' * len(hdr))
    print()

    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f'table2_{mode}.json')
    with open(fname, 'w') as f:
        json.dump({'mode': mode, 'timestamp': datetime.now().isoformat(),
                   'rows': rows}, f, indent=2)
    print(f'  JSON saved: {fname}\n')
    return rows


# ==============================================================================
#  §7. Plotting  (reproduces tikzpicture style)
#
#  Color scheme (matching paper): NMW [26] = black, CCP [27] = blue, Ours = red
#  Note: all numeric values passed to matplotlib are cast to Python float/int
#  to avoid TypeError from Sage's RealLiteral in the PDF backend.
# ==============================================================================

_STYLES = {
    'nmw' : ('black', 's', '-',  '[26] (Nolte et al.)'),
    'ccp' : ('blue',  's', '-',  '[27] (Cheon et al.)'),
    'ours': ('red',   's', '-',  r'Ours $\sigma=3.19$'),
}

def _style(key):
    """Return (color, marker, linestyle, label) for a result key."""
    base = key.split('|')[0] if '|' in key else key
    lbl  = key.split('|')[1].strip() if '|' in key else None
    col, mk, ls, default_lbl = _STYLES.get(base, ('gray', 'o', '-', base))
    display = lbl if lbl else default_lbl
    # Figure 2 multi-curve color override
    if lbl and 'q/8' in lbl:
        col = 'red'
    elif lbl and '3.19' in lbl:
        col = 'blue'
    return col, mk, ls, display

def _f(lst):
    """Cast a list to pure Python floats (avoids Sage RealLiteral in PDF backend)."""
    return [float(v) for v in lst]


def plot_figure(record, figdir='figures'):
    """Save PDF and PNG plots for one figure record."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator

    fname = record['figure']
    mode  = record.get('mode', '')
    cfg   = PAPER_FIGURES[fname]
    os.makedirs(figdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for key, entries in record['data'].items():
        xs  = _f([e['x']     for e in entries])
        ys  = _f([e['p']     for e in entries])
        lo  = _f([e['ci_lo'] for e in entries])
        hi  = _f([e['ci_hi'] for e in entries])

        col, mk, ls, disp = _style(key)
        ax.plot(xs, ys, color=col, marker=mk, ms=float(4.5), lw=float(1.4),
                linestyle=ls, label=disp)
        ax.fill_between(xs, lo, hi, color=col, alpha=float(0.12))

    ax.set_xlabel(cfg['x_label'], fontsize=11)
    ax.set_ylabel('Success probability', fontsize=11)
    ax.xaxis.set_major_locator(FixedLocator(_f(cfg['x_ticks'])))
    ax.yaxis.set_major_locator(FixedLocator(_f(cfg['y_ticks'])))
    ax.set_ylim(float(-0.02), float(1.04))
    ax.grid(True, linestyle='--', alpha=float(0.5))
    ax.legend(loc=cfg.get('legend_loc', 'best'), fontsize=9, framealpha=float(0.88))
    title_mode = f' [{mode}]' if mode else ''
    ax.set_title(f"Figure {fname}{title_mode}: {cfg['desc']}", fontsize=9)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        path = os.path.join(figdir, f'figure_{fname}_{mode}.{ext}')
        fig.savefig(path, dpi=int(200), bbox_inches='tight')
    plt.close(fig)
    print(f'  Plots: {figdir}/figure_{fname}_{mode}.{{pdf,png}}')


def plot_table2(rows, mode, figdir='figures'):
    """Render Table 2 as a PDF/PNG image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, max(2.5, len(rows) * 0.6 + 1.6)))
    ax.axis('off')
    col_labels = ['Scheme', 'Hints', 'n', 'log2q', 'h', 'm',
                  'Alg1 [26]', 'Alg2 [27]', 'Alg3 Ours']
    data = [[r['scheme'], r['hint_type'], str(r['n']), str(r['log2q']),
             str(r['h']), str(r['m']),
             f"{r['p_nmw']:.2f}", f"{r['p_ccp']:.2f}", f"{r['p_ours']:.2f}"]
            for r in rows]
    tbl = ax.table(cellText=data, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.7)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#ccd8ee')
    for i in range(1, len(rows) + 1):
        tbl[(i, 8)].set_facecolor('#dff5e3')
    ax.set_title(f"Table 2 [{mode}]: Experimental results on concrete FHE parameters",
                 fontsize=11, pad=12)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        path = os.path.join(figdir, f'table2_{mode}.{ext}')
        fig.savefig(path, dpi=int(200), bbox_inches='tight')
    plt.close(fig)
    print(f'  Table image: {figdir}/table2_{mode}.{{pdf,png}}')


# ==============================================================================
#  §8. GAA theoretical lower bound (Lemma 3.1)
# ==============================================================================

def gaa_lower_bound(n, h, q, m, sigma):
    """
    Compute the GAA lower bound on success probability (Lemma 3.1 of the paper).
    Uses a union-bound argument over noise coordinates at each greedy step.
    """
    from sage.all import erf as sage_erf
    def Phi(x):
        return float(0.5 * (1 + sage_erf(RR(x) / sqrt(RR(2)))))
    q   = RR(q)
    Ea2 = (q**2 - 1) / 12
    Ea4 = (q**2 - 1) * (3*q**2 - 7) / 240
    se2 = RR(sigma)**2
    p   = 1.0
    for j in range(h):
        h_r = h - j
        n_n = n - h + j
        mu  = float(m * Ea2)
        vs  = float(m * (Ea4 - Ea2**2) + m * max(0, h_r-1) * Ea2**2 + m * se2 * Ea2)
        vn  = float(m * h_r * Ea2**2 + m * se2 * Ea2)
        ss  = float(sqrt(RR(vs)))
        sn  = float(sqrt(RR(vn)))
        em  = sn * float(sqrt(2 * log(max(2, n_n)))) if n_n > 1 else 0.0
        z   = (mu - em) / ss if ss > 0 else 0.0
        p  *= Phi(z)
    return p


def print_calibration():
    """Print GAA lower bounds and noise moment calibration."""
    print(f"\n{'='*60}")
    print('  GAA Lower Bounds (Lemma 3.1 of the paper)')
    print(f"{'='*60}")
    q100 = int(next_prime(2**100))
    cases = [
        (1024, 8, 48,           'n=1024 h=8  m=48  (paper Sec.3.1.1 example)'),
        (1024, 8, m_from_h(8),  'n=1024 h=8  m=2h*log2(h)=48'),
        (512,  8, 48,           'n=512  h=8  m=48'),
    ]
    for n, h, m, desc in cases:
        p = gaa_lower_bound(n, h, q100, m, SIGMA_DEFAULT)
        print(f'  {desc}: P >= {p:.6f}')
    print()


# ==============================================================================
#  §9. JSON load / re-plot helpers
# ==============================================================================

def load_record(fig_name, mode, outdir='results'):
    fname = os.path.join(outdir, f'figure_{fig_name}_{mode}.json')
    with open(fname) as f:
        return json.load(f)

def load_table2(mode, outdir='results'):
    with open(os.path.join(outdir, f'table2_{mode}.json')) as f:
        return json.load(f)['rows']


# ==============================================================================
#  §10. Command-line interface
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Reproduce Figures 1-7 and Table 2 (binary or ternary mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  sage paper_experiments.sage --list
  sage paper_experiments.sage --mode ternary --figure 1
  sage paper_experiments.sage --mode ternary --figure 1 --trials 20
  sage paper_experiments.sage --mode binary  --figure all
  sage paper_experiments.sage --mode ternary --figure all --table2
  sage paper_experiments.sage --mode ternary --plot-only 6
  sage paper_experiments.sage --calibration
        """)

    parser.add_argument('--mode',       type=str, default='ternary',
                        choices=['binary', 'ternary'],
                        help='Secret key distribution: binary {0,1} or ternary {-1,0,1}')
    parser.add_argument('--figure',     type=str,
                        help='Figure number to run (1-7 or all)')
    parser.add_argument('--table2',     action='store_true',
                        help='Run Table 2 experiments')
    parser.add_argument('--trials',     type=int,
                        help='Override number of trials per point')
    parser.add_argument('--list',       action='store_true',
                        help='List available figures and exit')
    parser.add_argument('--plot-only',  type=str, metavar='N',
                        help='Re-plot figure N from saved JSON (no experiments)')
    parser.add_argument('--calibration', action='store_true',
                        help='Print GAA theoretical bounds and exit')
    parser.add_argument('--seed',       type=int, default=None,
                        help='Base RNG seed for reproducibility. '
                             'Each (point, trial) gets seed = BASE + point*10^6 + trial. '
                             'Omit for non-deterministic runs.')
    parser.add_argument('--normalize-score', action='store_true',
                        help='Divide <A_i,c> by ||A_i||^2 before argmax in Algorithm 3. '
                             'This is a variance-reducing heuristic NOT in the paper. '
                             'Default is paper-faithful (raw inner product).')
    parser.add_argument('--outdir',     type=str, default='results',
                        help='Directory for JSON output (default: results/)')
    parser.add_argument('--figdir',     type=str, default='figures',
                        help='Directory for plot output (default: figures/)')

    args, _ = parser.parse_known_args()

    if args.list:
        fmt = '  {:>4s}  {:>10s}  {:>6s}  {:>24s}  {}'
        print('\nAvailable figures:\n')
        print(fmt.format('Fig', 'Sweep', 'Trials', 'Algorithms', 'Description'))
        print('  ' + '-' * 78)
        for name, cfg in sorted(PAPER_FIGURES.items()):
            print(fmt.format(name, cfg['sweep'], str(cfg['trials']),
                             str(cfg['algos']), cfg['desc'][:55]))
        print(f'\nCurrent mode: {args.mode}  (change with --mode binary|ternary)\n')
        return

    if args.calibration:
        print_calibration()
        return

    if args.plot_only:
        record = load_record(args.plot_only, args.mode, args.outdir)
        plot_figure(record, args.figdir)
        return

    did_something = False

    if args.figure:
        names = (sorted(PAPER_FIGURES.keys())
                 if args.figure.lower() == 'all' else [args.figure])
        for name in names:
            if name not in PAPER_FIGURES:
                print(f"[Error] Figure '{name}' not found. Use --list.")
                continue
            record = run_figure(name, args.mode,
                                override_trials=args.trials, outdir=args.outdir,
                                base_seed=args.seed,
                                normalize_score=args.normalize_score)
            plot_figure(record, args.figdir)
            did_something = True

    if args.table2:
        rows = run_table2(args.mode, override_trials=args.trials, outdir=args.outdir,
                          base_seed=args.seed, normalize_score=args.normalize_score)
        plot_table2(rows, args.mode, args.figdir)
        did_something = True

    if not did_something:
        parser.print_help()


if __name__ == '__main__':
    main()
