"""Every constant of launch record 135, and every reading of an ambiguity as a named constant.

Registered values carry their section. Constants under "Readings of ambiguities" were the literal
reading of text that does not settle a choice; amendment 1 registers them as the rules (its item
13, with IMPLEMENTATION_NOTES.md), and changes some (items 1, 2, 5, 6, 7, 11, 12, marked below).
Each changes only by committed amendment.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]

# ---- Files (section 2) ------------------------------------------------------------------------
REGISTRATION = REPO / "docs/experiments/logbooks/supporting/135-neural-readout-ceiling/launch.md"
READS_LOG = REGISTRATION.parent / "READS.md"
STAMS_PATH = REPO / "investigation/spike/session0/exports/measured_stams.npz"
STAMS_SHA256 = "1c98ae81f5422ac5f9cc35bbd375ac7ea7815c59e3cf0d5545d08e3898237b28"
TRAIN_MEMBER = "measured_stams.train.npy"
TEST_MEMBER = "measured_stams.test.npy"  # read only by ceiling.scorer
NAMES_PATH = REPO / "investigation/spike/session0/creamer_154_names.json"
HINGE4_PATH = REPO / "docs/experiments/logbooks/supporting/131-c2-clean-start/probes/hinge4.json"
PROBE_DIR = REPO / "scripts/analysis"

# ---- Sizes ------------------------------------------------------------------------------------
N_MEASURED = 154
N_CONNECTOME = 302
N_ENCODER = 137
BOX_CENTRE = 0.5

# ---- Splits (section 4) -----------------------------------------------------------------------
N_FOLDS = 5
SPLITS = ("stimulations", "neurons", "worms")
HELD_OUT_SPLITS = ("stimulations", "neurons")

# ---- Seeds (section 10) -----------------------------------------------------------------------
SEED_WORMS_INNER = 9770
SEED_FOLDS = 9771
SEED_BOOTSTRAP = 9772
SEEDS_P = (9773, 9775, 9776)
SEEDS_N_NOISE = (9777, 9778, 9779)
SEED_TIED_INIT = 9774
SEEDS_NULL_R = tuple(range(9781, 9800))
SEEDS_ER = tuple(range(9801, 9806))
SEEDS_NULL_D = tuple(range(9741, 9760))
SEED_PARITY = 9732  # M-a's and H0's parity inputs

# ---- Arms (section 5) -------------------------------------------------------------------------
SERIES_TERMS = 4
TIED_SWEEP_RANKS = (2, 5, 10, 20, 40)
UNTIED_RANK = 17
INIT_SD = 0.01
NULL_R_SWAPS_PER_ENTRY = 10
NULL_R_MAX_SWAPS_PER_ENTRY = 30
NULL_R_MIN_MOVED = 0.8

# ---- Fitting and selection (section 6) --------------------------------------------------------
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
N_INNER = 3
MAX_ITER = 500
MAX_ITER_K = 2000
# Amendment 1, item 2: the preflight fits masked on held-out-stimulation fold 0 at lambda 1e-2 for
# 500 and for 2,000 iterations; if the visible loss at 500 exceeds the loss at 2,000 by more than 1%
# (relative), every sparse-support and tied fit (real and oracle data alike) gets a cap of 2,000.
CONVERGENCE_LAMBDA = 1e-2
CONVERGENCE_SHORT = MAX_ITER
CONVERGENCE_LONG = 2000
CONVERGENCE_RELATIVE = 0.01
LONG_CAP_FAMILIES = ("masked", "null-D", "null-R", "sparse-ER", "masked-prior", "tied")
LBFGS_HISTORY = 20
LOSS_CHANGE_TOL = 1e-10
SELECTION_NULL_D = (9741, 9742, 9743, 9744)
SELECTION_NULL_R = (9781, 9782, 9783, 9784)
NULL_READABLE_MIN = 17

# ---- Oracles (section 7) ----------------------------------------------------------------------
K_BAR = 0.8
P_MIN_DRAWS = 2
N_TRUTH_SEEDS = (9741, 9742, 9743)
N_FAIL_MIN_TRUTHS = 2
COUNT_MODEL_K = 80 / 30

# ---- Readings (section 8) ---------------------------------------------------------------------
N_BOOT = 2000
M_FRACTION = 0.25
M0 = 0.0109
CI_LEVEL = 0.95

# ---- Budget (section 10) ----------------------------------------------------------------------
WORKERS = 8
BUDGET_HOURS = 6.0
BUSY_PATTERN = r"c2_fit|c2_encoder_probe|run_campaign|benchmark_grade"

# ---- The guarded scorer (amendment 1, items 11 and 12) -----------------------------------------
# The scorer refuses while this spike's code has uncommitted changes.
CODE_DIR = HERE / "ceiling"
# A named amendment must change the registration or add a file whose name starts with this, in
# the registration's folder.
AMENDMENT_FILE_PREFIX = "amendment"
# READS.md is committed after every read, alone, with this message.
READS_COMMIT_MESSAGE = ("Launch record 135: log the {quantity} read of S in READS.md\n\n"
                        "Committed by the guarded scorer after the read.\n\n"
                        "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
# Products of a read in a run directory; a read is refused if READS.md is missing while any exists.
READ_PRODUCTS = ("entry_sets.npz", "entry_sets.json", "section3.json", "results.json")

# ---- Readings of ambiguities (each a named constant; see the build report) --------------------
# Section 4: classes are listed in sorted name order before default_rng(9771).permutation.
CLASS_ORDER_BEFORE_PERMUTATION = "sorted"
# Section 6: the inner class folds are the visible classes, in the outer permutation's order,
# dealt round-robin into 3 (no seed is registered for them).
INNER_CLASS_FOLDS = "round_robin_in_outer_order"
# Section 6: held-out-worm inner folds: default_rng(9770).permutation of T's finite off-diagonal
# entries (row-major order), entry at permuted position p in inner fold p % 3.
WORMS_INNER_FOLDS = "permutation_mod_3"
# Amendment 1, item 9 (registered as built): floor-T does not exist on held-out worms, so there
# floor-C is a + b*floor-ME and the calibrated arms drop their floor-T term.
FLOOR_T_ON_HELD_OUT_WORMS = False
# Amendment 1, item 5: the held-out-neuron entry set requires every reference defined, as the
# primary split does: floor-T (T[j, i] visible and finite) and floor-ME (column j has a finite entry
# in a visible row; hidden rows do not count). Every arm is then scored on the same entries, and
# each arm's count of defined entries is asserted equal to the set's size on every split.
NEURON_SPLIT_REQUIRES_TRANSPOSE = True
# Section 6: the ridge is lambda times the mean of the squared penalised parameters, taken jointly
# over every penalised tensor of the arm.
RIDGE_MEAN = "joint"
# Section 6: the refit rule (non-finite loss -> once at the next larger lambda) applies to every
# fit, the inner-CV fits included; a still-unreadable inner fit scores +inf.
REFIT_IN_SELECTION = True
# Section 6: summed inner-CV loss is the sum of each fit's validation MSE (in the outer unit's
# standardised units), over inner folds and the selection set's supports.
SELECTION_LOSS = "sum_of_fold_mse"
# Section 6: dense starts at W = 0 (the sparse arms' rule); untied draws U then V from N(0, 0.01^2)
# with default_rng(9774), as tied draws U then R.
DENSE_INIT = "zeros"
UNTIED_INIT_SEED = SEED_TIED_INIT
# Section 6: torch's L-BFGS also stops on a gradient tolerance and on a per-step parameter change.
# The gradient tolerance is 0 so it never stops a fit (tied starts with a tiny gradient); the
# parameter-change tolerance is torch's, set to the registered 1e-10. Function evaluations are
# capped generously so that the 500-iteration cap governs.
LBFGS_TOLERANCE_GRAD = 0.0
LBFGS_MAX_EVAL_PER_ITER = 25
# Section 5: each fit's per-hop share is Cov(W^k, M) / Var(M) over the 154's off-diagonal entries
# (shares sum to 1); Var(W^k) / Var(M) is reported beside it.
VARIANCE_SHARE_RULE = "covariance"
# Section 5: null-R runs its undirected swaps first, then its directed swaps; "10 swaps per entry"
# is per directed entry of each part; a draw under 80% moved continues in rounds of one attempt per
# entry (both parts) until it reaches 80% or 30 attempts per entry.
NULL_R_ORDER = "undirected_then_directed"
NULL_R_EXTENSION_ROUND = 1
# Section 5: sparse-ER draws its entries by rng.choice over the 302*301 off-diagonal positions
# (row-major), without replacement.
ER_DRAW = "choice_without_replacement"
# Amendment 1, item 8 (registered as built): K fits masked at a fixed lambda of 1e-4 with no
# selection, and calls a visible loss near zero below this value, in standardised units.
K_LAMBDA = LAMBDA_GRID[0]
K_NEAR_ZERO_LOSS = 1e-3
# Section 7: the noise and calibration patterns: K's T_syn and S_syn are M* on T's and S's
# finite-entry patterns, as P's are.
K_ON_PATTERNS = True
# Amendment 1, item 1: every oracle data set (K, P, N), in every fold unit and inner fold, is
# standardised with its truth fit's own (mean, SD), saved with the data set; the truth fits
# themselves are standardised as the real run is (by their visible entries).
ORACLE_STANDARDISATION = "truth_constants"
# Section 7: a column block for target column j is one source column drawn uniformly from the
# columns with a finite residual; entries the source has are copied by responder, and the rest are
# drawn with replacement from the source column's finite residuals.
RESIDUAL_BLOCK_FILL = "aligned_then_pool"
# Section 7: the noise scale solves Var(M*)/Var(M* + a*e) = rel exactly on the pattern (the
# covariance of M* and e kept), taking the positive root.
NOISE_SCALE_RULE = "exact_quadratic"
# Section 3/7: an estimate outside (0, 1) is replaced by its own count-model value; the other is
# kept.
CALIBRATION_FALLBACK = "per_estimate"
# Amendment 1, item 10 (registered as built): N's lambda selection keeps the fixed selection set,
# its generating draw included; the generating draw is left out of the rank only.
N_SELECTION_INCLUDES_GENERATING = True
# Amendment 1, item 6: the gates of each reading. "P-worms" is P's held-out-worm line. R-0 is
# reported on every split whatever the gates say; a split whose masked fit is unreadable is
# reported unreadable, not refused. K failing on held-out stimulations also leaves the
# held-out-neuron split unread (section 9: "only the held-out-worm split is read").
GATES = {"R-A": ("K",), "R-B": ("K", "P", "N"), "R-D": ("K", "P"),
         "R-C": ("P-worms",), "denoising": ("P-worms",)}
K_STIM_FAIL_BLOCKS_NEURONS = True
# Amendment 1, item 7: R-C's sparse-ER contrast needs at least 4 of the 5 draws readable, and its
# tied contrast a readable tied; otherwise that contrast is unreadable (never "fails").
ER_READABLE_MIN = 4
# Section 8: the 2,000 resamples are drawn once from default_rng(9772) as integers(0, 74, (2000,
# 74)) over the classes in sorted order and reused for every pair, split and the ceiling interval;
# intervals are the 2.5th and 97.5th percentiles.
BOOTSTRAP_DRAW = "integers_once"
BOOT_INTERVAL = "percentile"
# Section 8: r_ref's arm is chosen once on the split's point r; each resample recomputes that arm.
R_REF_CHOSEN_ON_POINT = True
# Section 9: "first of 20" is a strict maximum among masked and the family's readable draws.
RANK_STRICT = True
# Section 6: an arm (or a null draw) is readable on a split only if every one of its fold fits is.
ARM_READABLE_RULE = "all_folds"
# Section 10: timed preflight fits use the middle of the grid.
PREFLIGHT_LAMBDA = 1e-2
# Section 10: the cuts are applied one at a time, stopping at the first that brings the projection
# under the budget.
CUTS = ("description_arms", "neuron_split_oracles", "n_two_truths")


def single_thread() -> None:
    """Pin BLAS, OpenMP and torch to one thread (call before heavy imports where possible)."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set in this process
