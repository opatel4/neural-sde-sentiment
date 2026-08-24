# Data directory

## Committed (redistributable)

| File | Source | Notes |
|---|---|---|
| `treasury_curve.csv` | FRED series DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS3 | Daily constant-maturity Treasury yields; interpolated to each option's maturity for the forward reconstruction. |
| `dividend_yield.csv` | constant 1.9% | Flat S&P 500 dividend yield used throughout. |
| `spx_returns_sentiment_merged.csv` | derived | Daily SPX returns joined to FinBERT daily sentiment. Input to the sentiment regression. See LSEG terms before redistributing; if unable to share, regenerate via the two external steps below. |

### `spx_returns_sentiment_merged.csv` schema
```
date               ISO date
daily.returns      SPX daily log return
daily_sentiment    mean article-level FinBERT score (P_pos - P_neg) for the date
articles_per_day   article count (metadata; not used as a regression predictor)
```
The regression predictors (lagged sentiment, rolling means/vols, drawdown, etc.)
are constructed inside `src/sentiment/regression_sentiment.py`; they are not
columns in this file.

## NOT committed (licensed — acquire separately)

### SPX option quotes
Daily S&P 500 index option quotes, 2010–2022. The pipeline expects, per trade date,
enough quotes to build an 8x11 (maturity x log-moneyness) grid.

Required columns consumed by `src/fine_tuning/data_utils.py`:
```
Trade Date    trade date
Tau           time to maturity (years)
Forward       forward price
Strike        option strike
<price/IV>    option price or implied vol (see data_utils for the exact field)
r, q          optional per-row rate / dividend (falls back to SDP_R / SDP_Q)
```
Grid: maturities [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], log-moneyness 11 nodes
on [-0.5, 0.5]. Days with < 6 quotes, < 2 distinct maturities, or < 3 distinct
log-moneyness levels are dropped.

Place the per-year cleaned files under `data/SPX_Data/` following the naming the
loader expects (see `get_spx_files_for_range` in `data_utils.py`).

### LSEG Machine Readable News
Refinitiv/LSEG news corpus, filtered to U.S. equity-market relevance. Consumed
only by `src/sentiment/finbert_scoring.py` to produce `daily_sentiment`. Licensed;
cannot be redistributed. A reviewer without access starts from the committed
`spx_returns_sentiment_merged.csv`.

## Regenerating the non-committed inputs

1. Score the LSEG corpus:  `python3 src/sentiment/finbert_scoring.py`  -> daily_sentiment
2. Merge with SPX returns  -> `spx_returns_sentiment_merged.csv`
3. Fit the regression:     `python3 src/sentiment/regression_sentiment.py`
   -> coefficients compiled into `src/generate_offline_data.py`

Steps 1–3 are upstream of the automated pipeline (README §5) and only need
re-running if you are reproducing from raw news rather than from the committed
merged file.
