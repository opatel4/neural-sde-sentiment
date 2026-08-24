# Neural-SDE Option Pricing with Financial News Sentiment

Reproducibility package for "Does Financial News Sentiment Improve Neural-SDE
Option Pricing? Evidence from a Corrected Two-Stage Calibration Pipeline."

This repository reproduces every number, table, and figure in the paper from
raw inputs. It has two external entry points that require credentialed data or
one-time model inference (marked **[EXTERNAL]** below); everything downstream of
them is fully scripted.

---

## Repository layout

```
.
├── README.md                     <- this file
├── requirements.txt              <- pinned Python dependencies
├── environment.md                <- hardware/OS the results were produced on
│
├── data/
│   ├── raw/
│   │   ├── spx_options/           <- [EXTERNAL] daily SPX option quotes by regime
│   │   ├── fred_curves/           <- DGS1MO..DGS3Y treasury yields (FRED, public)
│   │   └── lseg_news/             <- [EXTERNAL] Refinitiv/LSEG news (licensed)
│   ├── sentiment/
│   │   ├── finbert_scores.py      <- [EXTERNAL] FinBERT inference over news text
│   │   └── daily_sentiment.csv    <- output: date -> daily sentiment index
│   └── regression/
│       ├── regression_sentiment_tanh.py
│       ├── spx_returns_sentiment_merged.csv
│       └── tanh_sentiment_regression_fitted_values.csv
│
├── offline/
│   ├── generate_offline_data.py   <- synthetic surface generator (all corrections)
│   ├── pretrain_offline.py        <- offline CNN pre-training
│   ├── models/                    <- 8 pretrained CNNs (In1/In2 x 4 families)
│   └── surfaces/                  <- 200k synthetic surfaces (or regenerate)
│
├── online/
│   ├── config.py                  <- regimes, model registry, weights
│   ├── data_utils.py              <- forward reconstruction, band filter, grids
│   ├── in2_adapter.py             <- sentiment channel + per-regime standardisation
│   ├── trainer.py                 <- fine-tuning loop, early-stopping rule
│   ├── train.py                   <- fine-tune driver
│   ├── run_parallel.sh            <- 4-worker fine-tuning launcher
│   ├── evaluate.py                <- in-sample evaluation
│   ├── export_surfaces.py         <- dump per-day predicted surfaces to npz
│   ├── walkforward.py             <- expanding-window OOS evaluation
│   ├── compare_variants.py        <- in-sample DM / CW / benchmarks
│   ├── compare_variants_wf.py     <- out-of-sample DM
│   ├── run_hedging.py             <- delta-hedged P&L
│   ├── param_identification.py    <- identification ratios
│   ├── calendar_diagnostic.py     <- no-arbitrage-is-noise diagnostic
│   └── eval_upgrades.py           <- shared test utilities (DM, GW, CW)
│
├── results/
│   ├── checkpoints/               <- 32 fine-tuned models
│   ├── in_sample/                 <- 16-cell results_master.csv + per-cell JSON
│   ├── walkforward/               <- 16-cell OOS results_master.csv + JSON
│   ├── hedging/                   <- hedging_all.log
│   └── logs/                      <- run logs (methods-appendix evidence)
│
├── figures/
│   ├── make_figures.py            <- regenerates all paper figures from results/
│   └── paper/                     <- generated figures
│
└── paper/
    └── neural_sde_sentiment_paper.md
```

---

## Environment

- Python 3.10, PyTorch 2.x + CUDA
- One CUDA GPU with >= 16 GB (results produced on an NVIDIA H100 slice)
- ~96 CPU cores used for 4-worker parallelism; fewer works, just slower
- `pip install -r requirements.txt`

See `environment.md` for the exact versions the published numbers were produced on.

---

## The two external entry points

Everything except these two steps is scripted and deterministic.

1. **SPX option data and LSEG news** are licensed and cannot be redistributed.
   `data/raw/` documents the expected schema. With equivalent data in that
   schema, the pipeline runs unchanged.

2. **FinBERT scoring** (`data/sentiment/finbert_scores.py`) runs the pretrained
   FinBERT model over article text to produce `daily_sentiment.csv`. This is
   one-time inference; the output CSV is included so reviewers without the raw
   news can still reproduce everything downstream of sentiment.

---

## Reproduction order

Each step lists its required environment variables. **All online steps require
`SDP_NORM=forward`** (forward-measure normalisation; without it a legacy price
cap re-fires and surfaces will not match the paper).

### Step 0 - sentiment regression  (Table 1)
```bash
cd data/regression
python regression_sentiment_tanh.py
# writes tanh_sentiment_regression_coefficients.csv
# R^2 = 0.508, n = 3774; reproduces Table 1
```

### Step 1 - generate synthetic surfaces
```bash
cd offline
SDP_MODEL=all python generate_offline_data.py
# 50k surfaces x 4 families; applies martingale correction, full truncation,
# log-moneyness grid, and the Table-1 sentiment coefficients (see Appendix A)
```

### Step 2 - offline pre-training  (Table 2)
```bash
SDP_DATA_DIR=./surfaces SDP_WEIGHTS_DIR=./models SDP_SENTIMENT=both \
  python pretrain_offline.py
# 8 CNNs; In2 beats In1 by 4-25% (Table 2)
```

### Step 3 - online fine-tuning  (32 cells)
```bash
cd online
SDP_NORM=forward SDP_USE_IN2=1 SDP_Q=0.015 ./run_parallel.sh
# 4 models x 4 regimes x 2 arms; per-regime SDP_R set inside the launcher
# ~4-5 h on 4 workers. Produces 32 checkpoints.
```

### Step 4 - in-sample evaluation + export  (Table 3)
```bash
export SDP_NORM=forward SDP_USE_IN2=1 SDP_Q=0.015
SDP_USE_SENTIMENT=1 python evaluate.py
SDP_USE_SENTIMENT=0 python evaluate.py
SDP_USE_SENTIMENT=1 python export_surfaces.py
SDP_USE_SENTIMENT=0 python export_surfaces.py
python compare_variants.py     # Table 3 + persistence benchmark (Table 5)
```

### Step 5 - walk-forward  (Table 4)
```bash
# both arms x 4 models x 4 regimes, expanding-window OOS
# (see online/run_walkforward.sh for the 4-worker launcher)
python compare_variants_wf.py  # Table 4
```

### Step 6 - hedging, identification, no-arbitrage  (Sec 6.4-6.6)
```bash
python run_hedging.py --model <M> --regime <R>   # per cell; SDP_R per regime
python param_identification.py
python calendar_diagnostic.py
```

### Step 7 - figures
```bash
cd figures && python make_figures.py
```

---

## Reproducing without a GPU

`results/` ships with all 32 checkpoints and the exported npz surfaces, so a
reviewer without a GPU can reproduce every **table and figure** by running only
Steps 0, 4 (the `compare_*` scripts, which are CPU-only), 5, 6, and 7 against
the shipped artifacts. Only Steps 1-3 (data generation and training) require a
GPU.

---

## Notes on the corrected pipeline

This pipeline supersedes an earlier version whose offline data contained several
defects (documented in Appendix A of the paper). The corrections are baked into
`offline/generate_offline_data.py`. Do **not** regenerate synthetic data from any
older notebook; those lack the fixes and reproduce the earlier, misleading
result. The scripts in this repository are the canonical source.
```
```
