import os
import glob
from pathlib import Path

# Project root is the parent of the Non_Sentiment_Train folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

# Set RUN_ZERO_SHOT_COMPARISON = True to print a zero-shot leaderboard for
# each date range before fine-tuning begins.
RUN_ZERO_SHOT_COMPARISON = False

# Set AUTO_SELECT_BEST_MODEL = True to have the zero-shot winner chosen
# automatically (only relevant when RUN_ZERO_SHOT_COMPARISON = True).
AUTO_SELECT_BEST_MODEL = False

CANDIDATE_MODELS      = ["Heston", "Bates", "Bergomi", "rBergomi"]
CANDIDATE_DATE_RANGES = ["2010-2012", "2013-2015", "2016-2019", "2020-2022"]

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------

OFFLINE_WEIGHT = 0.70

# rBergomi's offline surfaces are far more volatile than the calm regimes
# (offline mean/std 0.319/0.461 vs real 0.187/0.070 on 2013-2015), so a
# 0.70 offline weight pushes the scaler off-centre and the network sees a
# nearly constant input -> collapse at ~epoch 3. Lower weight for rBergomi
# only; every other family keeps the original value so their numbers stay
# comparable with earlier runs.
# V2: the rBergomi override was justified by V1 offline moments
# (mean/std 0.319/0.461) produced by the corrupted sentiment
# channel. The V2 surfaces measure -0.3156/0.2496, so the scaler
# collapse that motivated it should no longer occur. Restored to
# a uniform weight; watch rBergomi's first ~20 epochs.
OFFLINE_WEIGHT_BY_MODEL = {}

def offline_weight_for(model_name):
    return OFFLINE_WEIGHT_BY_MODEL.get(model_name, OFFLINE_WEIGHT)

EPOCHS         = 750
LR             = 5e-6
BATCH_SIZE     = 128
VAL_FRACTION   = 0.2
SPLIT_SEED     = 10479694

# ---------------------------------------------------------------------------
# Output & data paths
# ---------------------------------------------------------------------------

# Output root depends on the arm so the sentiment run and the baseline run
# do NOT land in the same folder (they are otherwise indistinguishable).
_ARM_DIR = "sentiment_output" if os.environ.get("SDP_USE_SENTIMENT", "1") == "1" else "base_output"
MODEL_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "fine_tuning_output", _ARM_DIR, "models")
PLOT_OUTPUT_ROOT  = os.path.join(PROJECT_ROOT, "fine_tuning_output", _ARM_DIR, "plots")
DATA_FILES_DIR    = os.path.join(PROJECT_ROOT, "Fine_Tuning_Files", "SPX_Data")

# ---------------------------------------------------------------------------
# SPX file resolver
# ---------------------------------------------------------------------------

def get_spx_files_for_range(date_range: str):
    """Return a sorted list of Path objects for SPX CSVs that fall within date_range."""
    start_yr, end_yr = [int(p) for p in date_range.split("-")]
    files = sorted(
        [
            Path(f)
            for f in glob.glob(
                os.path.join(DATA_FILES_DIR, "cleanerSPXData_*_merged.csv")
            )
            if start_yr <= int(Path(f).stem.split("_")[1]) <= end_yr
        ],
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not files:
        raise FileNotFoundError(
            f"No CSV files found for {date_range} in: {DATA_FILES_DIR}"
        )
    return files
