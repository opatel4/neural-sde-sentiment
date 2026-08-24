# Environment (exact versions the published numbers were produced on)

- Python 3.10.12
- torch 2.10.0 (CUDA 12.8 build)
- numpy 2.2.6
- pandas 2.3.3
- scipy 1.15.3
- matplotlib 3.10.9
- NVIDIA H100 (80 GB slice), driver 580.126.09

The sentiment regression uses only numpy least-squares (no statsmodels or
scikit-learn). FinBERT scoring (src/sentiment/finbert_scoring.py) additionally
requires the `transformers` library and a BERT checkpoint; the pipeline can be
run from the pre-scored `data/spx_returns_sentiment_merged.csv` without it.

See requirements.txt for a pip-installable subset.
