# Does Financial News Sentiment Improve Neural-SDE Option Pricing? Evidence from a Corrected Two-Stage Calibration Pipeline

**Ethan Itkis, Matthew Lascoe, Stephen Linehan-Reckford, Om Patel, Samuel Shlyam**

*Stevens Institute of Technology*

---

## Abstract

We test whether daily financial-news sentiment, extracted from Refinitiv/LSEG news via FinBERT, improves the pricing accuracy of neural stochastic-volatility models calibrated to S&P 500 index options. Following the two-stage deep-calibration framework of Horvath, Muguruza and Tomas (2021), we pre-train a convolutional network offline on synthetic implied-volatility surfaces generated from four stochastic-volatility families — Heston, Bates, one-factor Bergomi and rough Bergomi — and then fine-tune it online against historical SPX surfaces through a fully differentiable pricer. Two arms are compared under identical architecture and hyperparameters: a baseline that sees only the implied-volatility surface, and a sentiment arm that additionally receives a daily sentiment channel. We evaluate 32 (model, regime, arm) cells across four market regimes spanning 2010–2022.

Offline, the sentiment channel is genuinely informative: the sentiment arm attains 4–25% lower validation loss than the baseline across all four model families. Online, the picture is far more equivocal. In-sample, Diebold-Mariano tests find statistically significant differences in 11 of 16 cells, but split 7 favourable and 4 unfavourable with effect sizes below 5%. Under expanding-window walk-forward evaluation — the more demanding test — 7 cells favour sentiment and 6 favour the baseline, and the in-sample and out-of-sample verdicts disagree in 6 of 16 cells. The largest in-sample sentiment advantage (Heston, 2010–2012, DM *t* = 12.46) vanishes out-of-sample (*t* = −0.44), while Bates reverses from in-sample unfavourable to out-of-sample favourable in three regimes. Robust, sign-consistent sentiment gains survive in only a minority of cells, concentrated in Heston during 2016–2019 and 2020–2022.

Two further findings bound any interpretation. First, a naive previous-day persistence forecast attains lower surface error than every calibrated model variant in all 16 cells, by a mean factor of 2.0. Second, an identification analysis shows that most structural parameters are only weakly recovered — mean-reversion speed in particular is statistically indistinguishable from random draws within its bounds — and the sentiment channel does not improve identification. We therefore report a carefully qualified result: news sentiment produces small, heterogeneous and only partially transferable improvements in neural-SDE option pricing, and in-sample model comparison materially misrepresents which of those improvements are real.

**Keywords:** deep calibration, stochastic volatility, implied volatility surface, FinBERT, sentiment analysis, walk-forward evaluation

---

## 1. Introduction

### 1.1 Motivation

Deep calibration has made parametric stochastic-volatility models tractable at speeds classical optimisation cannot approach. By learning the map from implied-volatility surfaces to model parameters offline, a neural network replaces a costly inner optimisation loop with a single forward pass, reducing calibration from minutes to milliseconds. This opens a question that was previously impractical to ask: if the calibration network can absorb an arbitrary input channel at negligible cost, can it exploit information beyond the surface itself?

Financial news sentiment is a natural candidate. A substantial literature links investor sentiment to returns and to the cross-section of asset prices. If sentiment carries information about the risk-neutral density that the current surface has not yet fully impounded, a calibration network given both should outperform one given only the surface.

### 1.2 Research question

What is the effect of supplying a daily financial-news sentiment signal to the training and calibration of a neural stochastic-differential-equation option-pricing model, and does that effect survive out-of-sample evaluation?

### 1.3 Contributions

1. We construct a two-arm experiment in which the sentiment and baseline models are identical in architecture, hyperparameters, data, and random seeds, differing only in the presence of a sentiment input channel, enabling a clean attribution of any performance difference.
2. We evaluate the arms not only on in-sample surface fit but under expanding-window walk-forward calibration, and show that the two disagree in 6 of 16 cells — including cases where the strongest in-sample effect disappears entirely.
3. We benchmark both arms against a previous-day persistence forecast and find that neither beats it in any cell, a comparison we argue should be standard in this literature.
4. We document and correct a set of data-generation defects in the offline stage that, left in place, produce qualitatively different and misleading conclusions (Appendix A).

### 1.4 Hypotheses

The study was designed around two pre-registered hypotheses:

- **H1 (pricing).** The sentiment arm attains lower surface pricing error than the baseline, with the largest improvements in high-volatility regimes.
- **H2 (hedging).** The sentiment arm attains lower delta-hedged profit-and-loss variance than the baseline.

All other analyses reported below — persistence benchmarking, no-arbitrage diagnostics, parameter identification, and the conditional Giacomini-White tests — are exploratory and are labelled as such.

---

## 2. Related work

Our calibration architecture follows the two-stage "deep calibration" approach, in which a network is trained offline on synthetic surfaces with known ground-truth parameters and then applied to market data. The rough Bergomi specification we include as a fourth family reflects the now-substantial evidence that volatility is rough, with Hurst exponents well below 1/2.

On the sentiment side, we use FinBERT, a BERT variant fine-tuned for financial text, to produce per-article sentiment probabilities. The broader question of whether sentiment carries pricing-relevant information beyond price history is long-standing; our contribution is narrower and methodological — whether a sentiment channel improves a specific, well-defined calibration task under evaluation protocols strong enough to distinguish transferable gains from in-sample artifacts.

---

## 3. Data

### 3.1 Option data

We use daily S&P 500 index option quotes partitioned into four regimes chosen to span distinct volatility environments:

| Regime | Character | Trading days used |
|---|---|---|
| 2010–2012 | Post-crisis recovery | 744 |
| 2013–2015 | Low-volatility expansion | 727 |
| 2016–2019 | Extended calm, four years | 995 |
| 2020–2022 | COVID shock and aftermath | 749 |

For each trade date we build an 8 × 11 grid of 8 maturity slices by 11 log-moneyness levels spanning [−0.5, +0.5]. Days with insufficient coverage (fewer than 6 quotes, 2 distinct maturities, or 3 distinct log-moneyness levels) are dropped.

### 3.2 Forward reconstruction and normalisation

Option prices are normalised by the forward rather than by a spot proxy. Discount factors are built from FRED Treasury yields (DGS1MO through DGS3Y) interpolated to each option's maturity, with a constant 1.9% dividend yield. Working in forward units removes the dividend yield from the pricing map entirely, since the forward is by construction the martingale numeraire.

We replace the percentile-based global price cap used in earlier iterations with a no-arbitrage band filter that drops quotes outside the interval [intrinsic value, forward]. This is both more principled and far less destructive: the band filter removes 0.35% of rows, and after filtering, 100% of the retained prices are invertible to implied volatility, with a median shift of −0.0031 volatility points relative to the vendor's own implied volatilities.

### 3.3 News sentiment

Article-level sentiment is extracted with FinBERT from Refinitiv/LSEG Machine Readable News filtered to U.S. equity-market relevance. Long articles are split into 400-character chunks; FinBERT returns a probability distribution over positive, negative and neutral for each chunk; chunk probabilities are averaged within an article; and the continuous article score is defined as *P*(positive) − *P*(negative). Article scores are averaged by publication date to yield a daily sentiment index aligned to the option data.

---

## 4. Synthetic data generation

### 4.1 Parameter sampling and surface construction

For each of the four model families we sample 50,000 parameter vectors uniformly from economically plausible bounds and price the corresponding 8 × 11 surface, using Fourier methods where a characteristic function is available and Monte Carlo otherwise. Prices are inverted to implied volatilities by a robust root-finder.

Three properties of the generator are worth stating explicitly, because each was a defect in an earlier iteration of this work and each materially changes the resulting data (see Appendix A):

- **Martingale correction.** Simulated terminal prices are rescaled so that the sample mean of *S<sub>T</sub>* matches the forward. Without this correction the discretised variance process induces a drift error reaching −19% at *T* = 2.
- **Full truncation.** The variance process uses the full-truncation scheme rather than naive clamping, which otherwise biases the put wing severely.
- **Log-moneyness grid.** Synthetic surfaces are built on the same log-moneyness grid as the market surfaces, so that offline and online inputs are geometrically comparable.

After these corrections, the fraction of synthetic surface points that fail to invert (implied volatility at the solver floor) falls from 16.8–37.6% to 0.22–0.70% across families. The residual is confined to the shortest maturity and deepest in-the-money strikes, where vega is numerically negligible.

### 4.2 Synthetic sentiment

The offline network must be shown a sentiment channel during pre-training, but simulated paths carry no news. We therefore estimate an empirical map from return-based path statistics to sentiment on real data, and apply it to simulated paths.

Because sentiment is bounded in (−1, 1), we model it in unbounded space: the response is *y* = arctanh(*s*), and predictions are mapped back through tanh. The predictors are computable from any price path — contemporaneous, squared and absolute returns; 5- and 21-day rolling means and volatilities; and 21-day downside deviation, drawdown and realised variance — together with the lagged latent sentiment.

**Table 1. Regression of arctanh-transformed daily FinBERT sentiment on return-based path statistics.** Residual standard error 0.1122 on 3762 degrees of freedom. *R*² = 0.5080, adjusted *R*² = 0.5065, *n* = 3774. In sentiment space (after the tanh transform) *R*² = 0.4979.

| Variable | Estimate | Std. Error | *t* | Pr(>&#124;*t*&#124;) | |
|---|---:|---:|---:|---:|---|
| (Intercept) | 0.06556 | 0.00608 | 10.783 | < 2 × 10⁻¹⁶ | \*\*\* |
| y_lag1 | 0.68555 | 0.01170 | 58.599 | < 2 × 10⁻¹⁶ | \*\*\* |
| ret | −3.22856 | 0.19308 | −16.721 | < 2 × 10⁻¹⁶ | \*\*\* |
| ret_sq | 13.05354 | 7.07931 | 1.844 | 0.0652 | . |
| abs_ret | −2.27957 | 0.41895 | −5.441 | 5.29 × 10⁻⁸ | \*\*\* |
| roll_mean_5 | 1.04134 | 0.60250 | 1.728 | 0.0839 | . |
| roll_mean_21 | −6.59953 | 2.54445 | −2.594 | 0.0095 | \*\* |
| roll_vol_5 | 1.03807 | 0.55849 | 1.859 | 0.0631 | . |
| roll_vol_21 | 4.60709 | 2.40796 | 1.913 | 0.0557 | . |
| downside_21 | −5.99732 | 3.11629 | −1.925 | 0.0543 | . |
| drawdown_21 | 0.12228 | 0.17800 | 0.687 | 0.4920 | |
| realized_var_21 | −10.62001 | 15.99044 | −0.664 | 0.5070 | |

Signif. codes: 0 '\*\*\*' 0.001 '\*\*' 0.01 '\*' 0.05 '.' 0.1

The fit explains roughly half the variation in daily sentiment. It is dominated by persistence: lagged sentiment enters with coefficient 0.686 and *t* = 58.6, consistent with news tone being strongly autocorrelated day to day. The clearest market linkages are contemporaneous returns (*t* = −16.7) and absolute returns (*t* = −5.4) — sentiment falls with negative returns and with volatility — with an additional medium-horizon effect from the 21-day rolling mean (*t* = −2.6). The longer-horizon drawdown and realised-variance terms are individually insignificant but are retained because they are path-computable.

**A limitation of this construction.** The dominant predictor, lagged sentiment, has no analogue on a simulated path with no news history. The generator substitutes a path-derived proxy (a *z*-scored lagged return) in that slot. The synthetic channel therefore reproduces the return-driven component of sentiment but not its autoregressive persistence, and the offline network is trained on a structurally simpler signal than the one it meets online. We return to this in Section 8.

### 4.3 Channel representation

Each surface is presented to the network as an image. The baseline arm receives a single-channel 8 × 11 implied-volatility grid. The sentiment arm receives two channels: the surface, and a sentiment channel broadcast across strikes within each maturity slice. The two arms are otherwise identical.

Because the synthetic and market sentiment distributions differ in location and scale, the sentiment channel is standardised per regime before entering the network, mapping the market distribution onto the offline distribution the network was pre-trained against. The offline channel has pooled mean −0.0296 and standard deviation 0.0927; the resulting per-regime scale factors are 0.647 (2010–2012), 0.917 (2013–2015), 0.790 (2016–2019) and 0.863 (2020–2022). That all four are below unity indicates the corrected synthetic channel and the empirical channel occupy comparable ranges, requiring only mild rescaling.

---

## 5. Method

### 5.1 Offline pre-training

A convolutional network maps the input surface to the model's parameter vector. The architecture is a stack of 2-D convolutions with batch normalisation and ELU activations, followed by a dense head with dropout, a sigmoid output, and a scaling layer mapping to each parameter's admissible range. Training minimises error against the known generating parameters using Adam with a ReduceLROnPlateau schedule.

Convolution is appropriate here because the surface is genuinely spatial: a dense network on a flattened vector discards the adjacency between neighbouring strikes and maturities that encodes smile convexity and term-structure slope.

**Table 2. Offline pre-training, best validation loss.**

| Model | Baseline (In1) | Sentiment (In2) | Improvement |
|---|---:|---:|---:|
| Heston | 0.00349 | 0.00261 | 25% |
| Bates | 0.01213 | 0.01123 | 7% |
| Bergomi | 0.00336 | 0.00322 | 4% |
| rBergomi | 0.00347 | 0.00321 | 7% |

The sentiment arm attains lower validation loss in every family. This establishes that the channel is learnable: given a sentiment signal generated from the same paths that generated the surface, the network extracts information from it. Whether that capability transfers to real sentiment is the question the online stage answers.

### 5.2 Online fine-tuning

The pre-trained weights initialise an unsupervised refinement loop against market surfaces. The network predicts parameters; a differentiable pricer built in PyTorch prices the 8 × 11 grid from those parameters; and the reconstruction error between predicted and observed surfaces is backpropagated through the pricer into the network. Because every pricer operation — characteristic-function evaluation, Monte Carlo simulation, Black-Scholes inversion — is expressed in differentiable tensor operations, gradients flow cleanly end to end.

The objective is Huber loss on the surface, chosen over squared error for insensitivity to the small number of large deviations that occur while the network is still adapting from the synthetic to the empirical regime. Each of the 32 cells is fine-tuned independently from the same offline checkpoint for 750 epochs.

**A note on early stopping.** Convergence speed varies sharply across regimes. In the calmest regime (2013–2015), three cells initially appeared to collapse at epoch 1 under a stopping rule that terminated training after 10 consecutive non-improving epochs. Inspection showed this was a false positive: the learning-rate scheduler does not engage until epoch 15, every cell in that regime exhibits a rising validation loss over the first 10–20 epochs, and a surviving cell in the same regime showed an identical early trajectory before descending to the best loss in its regime. Relaxing the rule to 30 consecutive increases, armed only after epoch 20, recovered all three cells, which then trained to epochs 717–744 with 12–21% improvements over their initialisation. Naive early stopping can misclassify slow-but-healthy convergence as failure; the diagnostic that distinguishes them is whether a comparable cell under the same configuration recovers.

### 5.3 Evaluation protocols

**In-sample.** All fine-tuned cells are evaluated on the full regime, reporting vega-weighted implied-volatility RMSE. Because the two arms are separately estimated rather than nested in the estimation sense, we use two-sided Diebold-Mariano tests as the primary comparison; Clark-West statistics are computed but interpreted with caution.

**Walk-forward.** Within each regime the sample is split into quarterly blocks. The model trains on all quarters up to *Q* and is evaluated on *Q* + 1, rolling forward, with each window warm-started from the previous window's weights. This yields 8 out-of-sample quarters per three-year regime and 12 for 2016–2019 — 36 windows per model per arm.

**Benchmarks.** Both arms are compared against a previous-day persistence forecast (yesterday's surface as today's prediction) and an at-the-money-flat surface.

**Hedging.** For each cell we run a rolling at-the-money delta hedge over every trading day, using 20,000 Monte Carlo paths and the regime's risk-free rate, and compare the variance of the hedging error between arms.

---

## 6. Results

### 6.1 In-sample pricing

**Table 3. In-sample vega-weighted IV RMSE and Diebold-Mariano tests.** NS = baseline, S = sentiment. Positive *t* favours sentiment.

| Model | Regime | *n* | RMSE (NS) | RMSE (S) | DM *t* | |
|---|---|---:|---:|---:|---:|---|
| Bates | 2010–2012 | 744 | 0.04758 | 0.04774 | −1.87 | \* |
| Bates | 2013–2015 | 727 | 0.06571 | 0.06554 | 1.64 | |
| Bates | 2016–2019 | 995 | 0.06033 | 0.06111 | −6.89 | \*\*\* |
| Bates | 2020–2022 | 749 | 0.06799 | 0.06773 | 1.72 | \* |
| Bergomi | 2010–2012 | 744 | 0.05123 | 0.05141 | −0.96 | |
| Bergomi | 2013–2015 | 727 | 0.05430 | 0.05416 | 0.34 | |
| Bergomi | 2016–2019 | 995 | 0.05331 | 0.05216 | 4.01 | \*\*\* |
| Bergomi | 2020–2022 | 749 | 0.07303 | 0.07889 | −0.90 | |
| Heston | 2010–2012 | 744 | 0.05242 | 0.05004 | 12.46 | \*\*\* |
| Heston | 2013–2015 | 727 | 0.04786 | 0.04850 | −4.94 | \*\*\* |
| Heston | 2016–2019 | 995 | 0.04746 | 0.04611 | 8.01 | \*\*\* |
| Heston | 2020–2022 | 749 | 0.06957 | 0.06706 | 7.57 | \*\*\* |
| rBergomi | 2010–2012 | 744 | 0.06064 | 0.06002 | 0.81 | |
| rBergomi | 2013–2015 | 727 | 0.05141 | 0.05044 | 2.16 | \*\* |
| rBergomi | 2016–2019 | 995 | 0.04836 | 0.04635 | 7.37 | \*\*\* |
| rBergomi | 2020–2022 | 749 | 0.08111 | 0.08464 | −3.57 | \*\*\* |

Eleven of sixteen cells show a statistically significant difference, split seven favourable to sentiment and four unfavourable. Effect sizes are uniformly small: the largest, Heston 2010–2012, is a 4.5% RMSE reduction. Large *t*-statistics arise from consistent small daily differences across hundreds of observations rather than from large effects.

Two patterns are visible. Heston benefits most consistently, in three of four regimes. The Bergomi family — which already imposes flexible forward-variance dynamics — shows little consistent gain, which is what one would expect if the surface already encodes most of what sentiment could contribute for those specifications.

### 6.2 Out-of-sample walk-forward

**Table 4. Walk-forward out-of-sample results, with in-sample comparison.**

| Model | Regime | *n* | RMSE (NS) | RMSE (S) | OOS DM *t* | | In-sample DM *t* | Agree? |
|---|---|---:|---:|---:|---:|---|---:|---|
| Bates | 2010–2012 | 494 | 0.05735 | 0.05606 | 10.13 | \*\*\* | −1.87 | no |
| Bates | 2013–2015 | 478 | 0.10152 | 0.10130 | 1.14 | | 1.64 | yes |
| Bates | 2016–2019 | 745 | 0.09470 | 0.09371 | 9.23 | \*\*\* | −6.89 | no |
| Bates | 2020–2022 | 496 | 0.06217 | 0.06110 | 9.39 | \*\*\* | 1.72 | yes |
| Bergomi | 2010–2012 | 494 | 0.05606 | 0.05633 | −2.15 | \*\* | −0.96 | yes |
| Bergomi | 2013–2015 | 478 | 0.11608 | 0.11657 | −2.40 | \*\* | 0.34 | no |
| Bergomi | 2016–2019 | 745 | 0.10329 | 0.10400 | −2.92 | \*\*\* | 4.01 | no |
| Bergomi | 2020–2022 | 496 | 0.04357 | 0.04349 | 0.92 | | −0.90 | no |
| Heston | 2010–2012 | 494 | 0.04629 | 0.04637 | −0.44 | | 12.46 | no |
| Heston | 2013–2015 | 478 | 0.08369 | 0.08382 | −0.95 | | −4.94 | yes |
| Heston | 2016–2019 | 745 | 0.07014 | 0.06943 | 8.99 | \*\*\* | 8.01 | yes |
| Heston | 2020–2022 | 496 | 0.04215 | 0.04150 | 6.40 | \*\*\* | 7.57 | yes |
| rBergomi | 2010–2012 | 494 | 0.08136 | 0.08197 | −3.23 | \*\*\* | 0.81 | no |
| rBergomi | 2013–2015 | 478 | 0.05837 | 0.05802 | 2.68 | \*\*\* | 2.16 | yes |
| rBergomi | 2016–2019 | 745 | 0.05123 | 0.05146 | −2.30 | \*\* | 7.37 | no |
| rBergomi | 2020–2022 | 496 | 0.04417 | 0.04470 | −4.96 | \*\*\* | −3.57 | yes |

Out-of-sample, seven cells favour sentiment significantly and six favour the baseline. The aggregate verdict is therefore as equivocal as in-sample — but the cell-level agreement is not. **The in-sample and out-of-sample signs disagree in six of sixteen cells**, and the disagreements are not confined to marginal cases:

- **Heston 2010–2012**, the largest in-sample sentiment advantage in the entire study (*t* = 12.46), is statistically indistinguishable from zero out-of-sample (*t* = −0.44). The in-sample advantage did not transfer.
- **Bates reverses direction entirely.** In-sample it is the family sentiment appears to harm most; out-of-sample sentiment wins its three significant cells decisively (*t* = 10.13, 9.23, 9.39).
- **Bergomi and rBergomi 2016–2019** flip from significantly favourable in-sample to significantly unfavourable out-of-sample.

What survives with the same sign and significance in both protocols is a minority: Heston in 2016–2019 and 2020–2022, rBergomi in 2013–2015, and Bates in 2020–2022. These are the only cells for which we would claim a robust sentiment effect.

We regard this as the paper's central methodological finding. In-sample surface-fit comparison, which is the standard reporting practice in this literature, is not a reliable guide to whether an additional input channel carries transferable information.

### 6.3 Benchmark comparison

**Table 5. Previous-day persistence versus the better of the two calibrated arms (in-sample RMSE).**

| Regime | Persistence | Best model RMSE range | Ratio range |
|---|---:|---|---:|
| 2010–2012 | 0.02768 | 0.04798–0.06066 | 1.73–2.19 |
| 2013–2015 | 0.02310 | 0.04821–0.06563 | 2.09–2.84 |
| 2016–2019 | 0.02077 | 0.04636–0.06044 | 2.23–2.91 |
| 2020–2022 | 0.05889 | 0.06820–0.08180 | 1.16–1.39 |

**Persistence attains lower surface error than every model variant in all sixteen cells, by a mean factor of 2.0.** This is not an artifact of the sentiment channel or of any correction we made; it holds for both arms and every family.

The interpretation is straightforward and should be stated plainly. Implied-volatility surfaces are highly persistent day to day. A parametric model calibrated to fit today's surface is not thereby a good forecast of tomorrow's, and against a random-walk benchmark it underperforms. The gap narrows sharply in 2020–2022 (ratio 1.16–1.39), precisely the regime in which surfaces move most and persistence is weakest — which is what one would predict if the mechanism is surface persistence rather than model inadequacy per se.

We report this because it bounds the practical significance of everything else in the paper. A 4% RMSE difference between arms is a second-order effect inside a framework a trivial baseline outperforms by a factor of two.

### 6.4 Delta-hedged profit and loss (H2)

Across the sixteen cells, the sentiment arm attains lower hedging-error variance in nine and higher in seven — again balanced. More informative is the relationship to the pricing result: **the hedging and in-sample pricing verdicts agree in twelve of sixteen cells.** Hedging does not reveal a sentiment benefit invisible to pricing; it largely confirms the pricing verdict.

The largest hedging effects appear in the COVID regime: Heston 2020–2022 shows a 12.5% variance reduction with sentiment, and rBergomi 2020–2022 a 15.8% reduction. Heston is fully consistent across metrics — all four regimes agree in direction between pricing and hedging, and three of four favour sentiment. The four disagreements cluster in rBergomi, which is also the family with the weakest parameter identification and the highest calendar-violation rates.

H2 is therefore not supported in general, with the qualified exception of specific high-volatility cells.

### 6.5 No-arbitrage diagnostics

Butterfly violations are essentially absent (rates of 0.0000–0.0016 across all cells). Calendar-spread violations range from 1.4% to 19%, which at face value would be concerning for models that are martingale by construction.

A targeted diagnostic shows these are Monte Carlo noise rather than model pathology. Violations concentrate overwhelmingly at the long end of the maturity axis — for a representative cell, the violation rate rises monotonically from 0.000 in the 0.1 → 0.3 gap to 0.153 in the 1.8 → 2.0 gap — where the true increment in total variance between adjacent maturities is smallest and therefore most easily flipped in sign by simulation error. The median violation magnitude is 0.00311 against a median positive increment of 0.00812, a ratio of 0.38. Violation rates decline steadily as tolerance is relaxed past the Monte Carlo error scale. Real pathology would concentrate at the short end and survive a generous tolerance; neither holds.

We therefore do not report calendar violations as a model defect. We note this explicitly because an earlier iteration of this work did report them as one.

### 6.6 Parameter identification

We assess whether calibrated parameters are genuinely pinned down by the surface using an identification ratio: the standard deviation of observed daily parameter changes divided by the standard deviation that would arise if the parameter were redrawn uniformly from its bounds each day. A ratio near 1 indicates the network is effectively guessing.

The results are sobering. Mean-reversion speed (κ in Heston and Bates) has ratios of 1.0–1.2 in nearly every cell — statistically indistinguishable from random draws. Long-run variance (θ) and initial variance (*v*₀) frequently exceed 0.9. In the Bergomi family, β and the rough-Bergomi Hurst parameter *H* sit at 0.9–1.1. The best-identified parameters are vol-of-vol, correlation, and the jump parameters, at ratios of 0.4–0.6 — "weak" to "moderate" on any reasonable reading. Identification is uniformly best in 2020–2022, where the surface carries the most structure to constrain against.

Critically for this paper's question, **the sentiment and baseline arms have nearly identical identification ratios in every row**, differing only in the third decimal. The sentiment channel does not help the network pin down structural parameters.

This means calibrated parameter values from this pipeline should not be interpreted as estimates of physical quantities. The network finds parameter vectors that reproduce the surface; it does not uniquely determine them.

---

## 7. Discussion

### 7.1 On H1

H1 is not supported in the form stated. Sentiment does not produce a general reduction in pricing error, and the regime prediction is wrong in the specific sense that gains are not concentrated in high-volatility periods. In-sample, the calmest regime and the most turbulent one both contain favourable and unfavourable cells.

The more defensible statement the data supports is conditional: sentiment produces small, model-dependent effects that are robust to out-of-sample validation in a minority of cells, concentrated in the Heston family during 2016–2019 and 2020–2022.

### 7.2 Why offline gains do not transfer

The gap between the offline result (Table 2: sentiment helps in all four families, by 4–25%) and the online result requires explanation. The network demonstrably can extract information from a sentiment channel. What differs online is the channel itself.

Two mechanisms are plausible and mutually compatible. First, the synthetic sentiment used offline is generated *from the same simulated paths* that generated the surface, making it partially redundant with the surface but perfectly consistent with it; real sentiment is neither. Second, and more specifically, the synthetic channel omits the dominant term of the empirical sentiment process. As Table 1 shows, real sentiment is overwhelmingly autoregressive (lagged sentiment, *t* = 58.6), and that term cannot be reproduced on a simulated path. The offline network therefore learns to exploit a return-driven sentiment signal and then meets an online signal whose principal component is persistence it has never seen.

This is a testable prediction rather than a demonstrated mechanism: a generator that carried synthetic sentiment forward along the maturity axis, so that a genuine lagged-sentiment term were available, should narrow the gap. We did not implement this and flag it as the most direct extension of this work.

### 7.3 On the persistence benchmark

The persistence result deserves emphasis rather than burial. It does not invalidate deep calibration, whose purpose is fast and accurate fitting of the current surface for pricing and risk applications, not forecasting. But it does mean that claims about surface accuracy in this literature should be accompanied by a random-walk comparison. We would encourage this as standard practice; it costs nothing to compute and materially changes how a reported RMSE should be read.

---

## 8. Limitations

**Sentiment representation.** The signal is a single daily scalar. Topic-conditioned scores, intraday timing, article-volume weighting, or source diversification (retail-forum sentiment, for instance) might carry information this aggregation destroys.

**The synthetic sentiment gap.** As discussed, the generator substitutes a return-based proxy for the lagged-sentiment term that dominates the empirical fit. The offline network is therefore pre-trained on a structurally simpler signal than it encounters online.

**Parameter identification.** Most structural parameters are weakly identified. Conclusions about surface fit are unaffected, but the calibrated parameters themselves carry little interpretive weight.

**Residual synthetic-data floor.** Approximately 0.22–0.70% of synthetic surface points remain at the inversion floor, confined to the shortest maturity and deepest in-the-money strikes where vega is negligible. We judged the cost of eliminating these to exceed the benefit.

**Scope.** Four model families, four regimes, one index, one sentiment source, 750 fine-tuning epochs, 20 refit epochs per walk-forward window. The walk-forward protocol in particular uses a reduced epoch budget for tractability, and it is possible that longer refits would change some marginal cells.

**Multiple comparisons.** Sixteen cells are tested under each of two protocols. We have reported all of them, including the unfavourable ones, and we have not selected a subset post hoc. Readers should nonetheless interpret individual marginal cells (*p* between 0.01 and 0.10) accordingly; our substantive conclusions rest on the pattern across cells and on the in-sample/out-of-sample comparison rather than on any single test.

---

## 9. Conclusion

We set out to test whether financial-news sentiment improves neural-SDE option pricing, expecting the clearest benefits in volatile markets. After rebuilding the pipeline to correct several data-generation defects, training 32 model-regime-arm cells, and evaluating under both in-sample and expanding-window walk-forward protocols, we find that it does not, in the general form hypothesised.

What we find instead is more specific and, we think, more useful. The sentiment channel is learnable in principle — it improves offline validation loss in all four model families. Online, it produces small and heterogeneous effects that survive out-of-sample validation in only a minority of cells. In-sample and out-of-sample evaluation disagree in six of sixteen cells, including a complete reversal for one model family and the disappearance of the study's largest in-sample effect. And every model variant, in every cell, is outperformed by a previous-day persistence forecast by a mean factor of two.

The methodological conclusions are the ones we would most want carried forward: that in-sample surface comparison is insufficient to establish that an auxiliary input carries transferable information, that a random-walk benchmark belongs alongside any reported surface RMSE, and that calibrated parameters from a deep-calibration pipeline should not be read as identified estimates without an explicit identification check.

---

## Appendix A. Corrections to the offline data-generation stage

An earlier iteration of this study produced qualitatively different conclusions — most notably that sentiment helped most in the calmest regime. That finding did not survive correction of the following defects, all of which affected the offline training data.

**A.1 Martingale defect.** Terminal simulated prices did not satisfy *E*[*S<sub>T</sub>*] = *F*. The discretised variance process under naive clamping produced a drift error reaching −19% at *T* = 2, biasing all Monte Carlo-priced surfaces. Corrected by rescaling *S<sub>T</sub>* by its sample mean at every pricing site.

**A.2 Variance discretisation.** Naive clamping of the variance process was replaced with the full-truncation scheme, which is the standard remedy for the negative-variance problem in Euler discretisation of square-root processes.

**A.3 Quadrature weights.** Gauss-Laguerre weights were constructed by a manual recursion subject to catastrophic cancellation, producing a maximum relative weight error of 6.6%. Replaced with a standard library implementation.

**A.4 Sentiment channel axis error.** The synthetic sentiment channel was written by indexing the Monte Carlo *path* axis rather than broadcasting a single value across strikes. The resulting channel contained eleven unrelated path draws presented as a strike profile — that is, noise structured to look like a signal. After correction the channel is constant across strikes within a maturity slice, as intended, with adjacent-maturity correlations of +0.74 to +0.89.

**A.5 Strike grid.** Synthetic surfaces were built on a linear strike grid while market surfaces used log-moneyness, so offline and online inputs were not geometrically comparable. Unified to log-moneyness on [−0.5, +0.5].

**A.6 Regression coefficients.** The generator applied a superseded coefficient set whose return and volatility loadings were 5.6–7.1 times those of the current fit. Replaced with the coefficients in Table 1.

**Effect of the corrections.** The synthetic-surface inversion-failure rate fell from 16.8–37.6% to 0.22–0.70%. The offline sentiment-arm advantage fell from 6–42% to 4–25%, indicating that part of the apparent offline benefit had come from the inflated variance of the corrupted channel. The earlier finding that sentiment helps most in 2013–2015 disappeared entirely; that regime now shows the smallest in-sample effects of any.

We document these in full because the corrected and uncorrected pipelines support opposite conclusions, and because several of the defects (A.1, A.2, A.4) would be invisible in aggregate loss curves. Training-loss diagnostics alone were not sufficient to detect any of them.
