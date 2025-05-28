import pandas as pd
import numpy as np

# 1. Load real data
orig = pd.read_csv("Updated_Final_merged_dataset_with_mediation.csv")

# 2. Precompute per-column stats
col_stats = {}
for col in orig.columns:
    ser = orig[col].dropna()
    if pd.api.types.is_numeric_dtype(ser):
        # determine if all values are integer‐like
        is_int = ser.apply(lambda x: float(x).is_integer()).all()
        col_stats[col] = {
            "mean": ser.mean(),
            "std": ser.std(),
            "min": ser.min(),
            "max": ser.max(),
            "is_int": is_int
        }
    else:
        col_stats[col] = {
            "categories": ser.unique().tolist()
        }

# For Start/End sample, compute typical segment length distribution
seg_lens = orig["End_Sample"] - orig["Start_Sample"]
seg_mean, seg_std = seg_lens.mean(), seg_lens.std()

def generate_synthetic(n_rows=1000):
    df = pd.DataFrame()
    for col, stats in col_stats.items():
        if "mean" in stats:
            # numeric
            vals = np.random.normal(loc=stats["mean"], scale=stats["std"], size=n_rows)
            # clip to original range
            vals = np.clip(vals, stats["min"], stats["max"])
            # round if integer
            if stats["is_int"]:
                vals = np.round(vals).astype(int)
            df[col] = vals
        else:
            # categorical/text
            df[col] = np.random.choice(stats["categories"], size=n_rows)
    # override Start_Sample/End_Sample to keep realistic segments
    starts = np.random.randint(orig["Start_Sample"].min(),
                               orig["Start_Sample"].max(), size=n_rows)
    lens = np.random.normal(seg_mean, seg_std, size=n_rows).astype(int)
    ends = starts + lens
    ends = np.minimum(ends, orig["End_Sample"].max())
    df["Start_Sample"] = starts
    df["End_Sample"] = ends
    return df

if __name__ == "__main__":
    synth = generate_synthetic(n_rows=500)  # change as needed
    synth.to_csv("synthetic_dataset.csv", index=False)
    print("Wrote synthetic_dataset.csv with", len(synth), "rows.")
