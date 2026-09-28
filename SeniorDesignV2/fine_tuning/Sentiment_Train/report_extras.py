"""
report_extras.py
================
Reads the per-cell JSONs that compare_variants.py ALREADY wrote and surfaces the
two result sets that never made it into results_master.csv:

  1. BENCHMARKS      - CNN variants vs previous-day persistence and ATM-flat.
                       (If persistence wins, that reframes the whole paper.)
  2. CONDITIONAL     - direction of the sentiment effect, bucketed by lagged
     BUCKETS           sentiment magnitude and news volume, with t-stats.
                       (This is what makes the GW rejections interpretable.)

No recomputation - pure reading. Run from the REPO ROOT:

    python report_extras.py

Writes:
    full_project_output/eval_surfaces/analysis/benchmarks_table.csv
    full_project_output/eval_surfaces/analysis/buckets_table.csv
"""
import os, glob, json, csv

ANA = "full_project_output/eval_surfaces/analysis"


def main():
    files = sorted(glob.glob(os.path.join(ANA, "*.json")))
    if not files:
        print(f"No JSONs found in {ANA}. Run compare_variants.py first.")
        return

    bench_rows, bucket_rows = [], []

    print("=" * 96)
    print("BENCHMARKS - vega-weighted IV RMSE (lower is better)")
    print("=" * 96)
    print(f"{'cell':<24}{'prev_day':>10}{'atm_flat':>10}{'cnn_NS':>10}{'cnn_S':>10}"
          f"{'  winner':<16}{'CNN beats persistence?':>22}")
    print("-" * 96)

    for f in files:
        cell = os.path.basename(f)[:-5]
        d = json.load(open(f))
        b = d.get("benchmarks", {})
        if not b:
            continue
        pd_ = b.get("previous_day", float("nan"))
        af = b.get("atm_flat", float("nan"))
        ns = b.get("cnn_no_sent", float("nan"))
        s = b.get("cnn_sentiment", float("nan"))
        winner = min(b, key=b.get)
        beats = "YES" if min(ns, s) < pd_ else "NO  <-- problem"
        print(f"{cell:<24}{pd_:>10.5f}{af:>10.5f}{ns:>10.5f}{s:>10.5f}"
              f"  {winner:<14}{beats:>22}")
        row = {"cell": cell, **{k: round(v, 6) for k, v in b.items()},
               "winner": winner, "cnn_beats_persistence": beats.split()[0]}
        bench_rows.append(row)

    print()
    print("=" * 96)
    print("CONDITIONAL BUCKETS - mean CW differential (>0 favours SENTIMENT), with t-stat")
    print("=" * 96)

    for cond_key, label in [("by_sentiment_magnitude", "SENTIMENT MAGNITUDE (lagged)"),
                            ("by_news_volume", "NEWS VOLUME (lagged)")]:
        print(f"\n--- conditioned on {label} ---")
        print(f"{'cell':<24}{'low':>18}{'mid':>18}{'high':>18}   {'pattern':<22}")
        print("-" * 96)
        for f in files:
            cell = os.path.basename(f)[:-5]
            d = json.load(open(f))
            cb = d.get("conditional_buckets", {}).get(cond_key, {})
            if not cb:
                continue
            cells_txt, vals = [], {}
            for bkt in ("low", "mid", "high"):
                r = cb.get(bkt, {})
                m, t = r.get("mean_diff", float("nan")), r.get("t", float("nan"))
                vals[bkt] = m
                star = "*" if abs(t) > 1.96 else " "
                cells_txt.append(f"{m:>+11.2e}(t{t:>5.1f}){star}")
            # is the effect monotone in the conditioner?
            lo, mi, hi = vals["low"], vals["mid"], vals["high"]
            if lo < mi < hi:
                pattern = "rising with cond."
            elif lo > mi > hi:
                pattern = "falling with cond."
            else:
                pattern = "non-monotone"
            print(f"{cell:<24}" + "".join(f"{c:>18}" for c in cells_txt) + f"   {pattern:<22}")
            bucket_rows.append({"cell": cell, "conditioner": cond_key,
                                "low": lo, "mid": mi, "high": hi, "pattern": pattern})

    os.makedirs(ANA, exist_ok=True)
    if bench_rows:
        keys = sorted({k for r in bench_rows for k in r})
        with open(os.path.join(ANA, "benchmarks_table.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(bench_rows)
    if bucket_rows:
        with open(os.path.join(ANA, "buckets_table.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(bucket_rows[0])); w.writeheader(); w.writerows(bucket_rows)
    print(f"\nWrote benchmarks_table.csv and buckets_table.csv to {ANA}/")


if __name__ == "__main__":
    main()
