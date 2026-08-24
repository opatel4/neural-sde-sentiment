# Neural-SDE Option Pricing with Financial News Sentiment

Reproducibility package for "Does Financial News Sentiment Improve Neural-SDE
Option Pricing? Evidence from a Corrected Two-Stage Calibration Pipeline."

This repository reproduces every number, table, and figure in the paper from
raw inputs. It has two external entry points that require credentialed data or
one-time model inference (marked **[EXTERNAL]** below); everything downstream of
them is fully scripted.

The pipeline has two stages, mirroring the paper: an **offline** stage
(`src/generate_offline_data.py`, `src/pretrain_offline.py`) that generates
synthetic surfaces and pre-trains the CNN, and an **online** stage
(`src/fine_tuning/`) that fine-tunes against real SPX surfaces and evaluates.

---

## Repository layout

```
.
├── README.md                      <- this file
├── requirements.txt               <- pinned Python dependencies
├── environment.md                 <- exact versions/hardware for published results
├── audit_repo.sh                  <- verifies package completeness + imports
│
├── data/
│   ├── README.md                  <- schemas for the [EXTERNAL] licensed inputs
│   ├── spx_returns_sentiment_merged.csv   <- daily returns + FinBERT sentiment
│   ├── treasury_curve.csv         <- FRED DGS1MO..DGS3Y (public)
│   └── dividend_yield.csv         <- constant 1.9%
│
├── src/
│   ├── generate_offline_data.py   <- OFFLINE: synthetic generator (all corrections)
│   ├── pretrain_offline.py        <- OFFLINE: CNN pre-training
│   │
│   ├── fine_tuning/               <- ONLINE: calibration + evaluation
│   │   ├── config.py              <- regimes, model registry, paths
│   │   ├── model.py               <- baseline CNN (In1)
│   │   ├── model_in2.py           <- two-channel CNN (In2, + sentiment)
│   │   ├── pricers.py             <- differentiable pricers (all four families)
│   │   ├── data_utils.py          <- forward reconstruction, band filter, grids
│   │   ├── in2_adapter.py         <- sentiment channel + per-regime standardisation
│   │   ├── utils.py               <- surface interpolation helpers
│   │   ├── trainer.py             <- fine-tuning loop, early-stopping rule
│   │   ├── train.py               <- fine-tune driver
│   │   ├── launch.py              <- multi-GPU/worker launcher
│   │   ├── evaluate.py            <- in-sample evaluation
│   │   ├── export_surfaces.py     <- dump per-day predicted surfaces to npz
│   │   ├── walkforward.py         <- expanding-window OOS evaluation
│   │   ├── compare_variants.py    <- in-sample DM / benchmarks (Tables 3, 5)
│   │   ├── compare_variants_wf.py <- out-of-sample DM (Table 4)
│   │   ├── hedging.py             <- delta-hedge engine
│   │   ├── run_hedging.py         <- hedging driver (Sec 6.4)
│   │   ├── param_identification.py<- identification ratios (Sec 6.6)
│   │   ├── calendar_diagnostic.py <- no-arbitrage-is-noise diagnostic (Sec 6.5)
│   │   └── eval_upgrades.py       <- shared test utilities (DM, GW, CW)
│   │
│   └── sentiment/
│       ├── finbert_scoring.py     <- [EXTERNAL] FinBERT inference over news text
│       └── regression_sentiment.py<- sentiment regression (Table 1)
│
├── scripts/                       <- one script per stage, env vars baked in
│   ├── 00_generate_offline.sh
│   ├── 01_pretrain.sh
│   ├── 02_finetune.sh
│   ├── 03_export.sh
│   ├── 04_evaluate.sh
│   ├── 05_walkforward.sh
│   └── 06_hedging.sh
│
├── results/
│   ├── in_sample/analysis/        <- 16-cell results_master.csv + per-cell JSON
│   ├── walkforward/analysis/      <- 16-cell OOS results_master.csv + JSON
│   ├── hedging_all.log            <- delta-hedged P&L, 16 cells
│   └── tanh_sentiment_regression_fitted_values.csv
│
├── figures/
│   ├── make_figures.py            <- regenerates paper figures from results/
│   ├── hedge_*.png                <- 16 hedging P&L figures
│   └── paper/                     <- convergence, smile, parity, error heatmaps
│
└── paper/
    └── neural_sde_sentiment_paper.md
```

Trained weights and exported surfaces are too large for git and are published
under [Releases](https://github.com/opatel4/neural-sde-sentiment/releases).

---

## Environment

- Python 3.10, PyTorch 2.x + CUDA
- One CUDA GPU with >= 16 GB (results produced on an NVIDIA H100 slice)
- ~96 CPU cores used for 4-worker parallelism; fewer works, just slower
- `pip install -r requirements.txt`

See `environment.md` for the exact versions the published numbers were produced on.

---

## Pre-trained artifacts (reproduce without retraining)

Download from [Releases](https://github.com/opatel4/neural-sde-sentiment/releases)
and extract into `results/`:

```bash
unzip checkpoints.zip          -d results/   # 32 fine-tuned cells
unzip surfaces_in_sample.zip   -d results/   # 16 npz per arm
unzip surfaces_walkforward.zip -d results/   # 16 npz per arm
```

With these in place you can run Steps 0, 4-7 below (all CPU-only) and reproduce
every table and figure without repeating the ~13 h of GPU training in Steps 1-3.

---

## The two external entry points

Everything except these two steps is scripted and deterministic.

1. **SPX option data and LSEG news** are licensed and cannot be redistributed.
   `data/README.md` documents the expected schema. With equivalent data in that
   schema, the pipeline runs unchanged.

2. **FinBERT scoring** (`src/sentiment/finbert_scoring.py`) runs the pretrained
   FinBERT model over article text to produce the daily sentiment index. This is
   one-time inference; `data/spx_returns_sentiment_merged.csv` contains the
   scored output so reviewers without the raw news can reproduce everything
   downstream of sentiment. Note this script reads from a Postgres database;
   adapt the source for flat files if needed.

---

## Reproduction order

Each step lists its required environment variables. **All online steps require
`SDP_NORM=forward`** (forward-measure normalisation; without it a legacy price
cap re-fires and surfaces will not match the paper). The `scripts/` set these
correctly — prefer them over invoking the Python entry points by hand.

### Step 0 — sentiment regression (Table 1)
```bash
python src/sentiment/regression_sentiment.py
# R^2 = 0.508, n = 3774; reproduces Table 1
```

### Step 1 — generate synthetic surfaces
```bash
bash scripts/00_generate_offline.sh
# 50k surfaces x 4 families; applies martingale correction, full truncation,
# log-moneyness grid, and the Table-1 sentiment coefficients (see Appendix A)
```

### Step 2 — offline pre-training (Table 2)
```bash
bash scripts/01_pretrain.sh
# 8 CNNs; In2 beats In1 by 4-25% (Table 2)
```

### Step 3 — online fine-tuning (32 cells)
```bash
bash scripts/02_finetune.sh
# 4 models x 4 regimes x 2 arms; per-regime SDP_R set inside the launcher
# ~4-5 h on 4 workers. Produces 32 checkpoints.
```

### Step 4 — in-sample evaluation + export (Tables 3, 5)
```bash
bash scripts/03_export.sh      # per-day predicted surfaces -> npz
bash scripts/04_evaluate.sh    # Table 3 + persistence benchmark (Table 5)
```

### Step 5 — walk-forward (Table 4)
```bash
bash scripts/05_walkforward.sh
# expanding-window OOS, both arms x 4 models x 4 regimes, then compare_variants_wf
```

### Step 6 — hedging, identification, no-arbitrage (Sec 6.4-6.6)
```bash
bash scripts/06_hedging.sh
export PYTHONPATH="$PWD/src/fine_tuning"
python src/fine_tuning/param_identification.py
python src/fine_tuning/calendar_diagnostic.py
```

### Step 7 — figures
```bash
python figures/make_figures.py
```

---

## Verifying the package

```bash
bash audit_repo.sh
```

Checks structure, that the corrected constants and coefficients are present (not
a stale copy), that results have the expected shape, and that every local import
resolves. Expect `FAIL 0`.

---

## Reproducing without a GPU

With the Release artifacts extracted into `results/`, a reviewer without a GPU
can reproduce every **table and figure** by running Steps 0 and 4-7 (the
`compare_*` and diagnostic scripts are CPU-only). Only Steps 1-3 (data
generation and training) require a GPU.

---

## Notes on the corrected pipeline

This pipeline supersedes an earlier version whose offline data contained several
defects (documented in Appendix A of the paper): a martingale drift error, naive
variance clamping, catastrophic cancellation in the quadrature weights, a
sentiment channel written along the wrong array axis, a mismatched strike grid,
and superseded regression coefficients. The corrections are baked into
`src/generate_offline_data.py`.

Do **not** regenerate synthetic data from any older notebook; those lack the
fixes and reproduce the earlier, misleading result. Several of these defects are
invisible in aggregate training-loss curves — they were found only by inspecting
the generated data directly. The scripts in this repository are the canonical
source.
