import pandas as pd
import numpy as np
df=pd.read_csv("delhi_aqi.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
numeric_cols = df.select_dtypes(include="number").columns.tolist()
print("Shape before anomaly fixes:", df.shape)

if df.isnull().sum().sum()==0:
    print("\nFix-1 \n\tNo missing value")
else:
    print("Missing value: ", df.isnull().sum().sum())

# checking duplication
date_dups = df.duplicated(subset="date").sum()
if df.duplicated().sum()==0 and date_dups==0:
    print("\nFix-2\n\tNo duplicated row")
else:
    print(f"duplicated row: {df.duplicated().sum()}")
    print(f"duplicate timestamps: {date_dups}")

neg_found = False
for col in numeric_cols:
    neg = (df[col] < 0).sum()
    if neg > 0:
        neg_found = True
        print(f"{col}:{neg} negative values")
        print(df[df[col] < 0][["date", col]].head(5).to_string())
if not neg_found:
    print("\nFix-3\n\tNo negative values found.")

full_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="1h")
df = df.set_index("date").reindex(full_range)   # inserts NaN rows for gaps
df.index.name = "date"
df[numeric_cols] = df[numeric_cols].interpolate(method="time")

df = df.reset_index()
# print(f"Shape after gap fill : {df.shape}  (+{df.shape[0] - 18776} rows filled)")
MIN_FREEZE_RUN = 5
def fix_sensor_freeze(df, col, min_run=5):
    same_as_prev = df[col] == df[col].shift(1)
    cumsum_series = (same_as_prev != same_as_prev.shift()).cumsum()
    groups = same_as_prev.groupby(cumsum_series)
    run_lengths = groups.sum()                       # length of each run
    frozen_group_ids = run_lengths[run_lengths >= min_run].index

    n_replaced = 0
    for gid in frozen_group_ids:
        mask = cumsum_series == gid
        df.loc[mask, col] = np.nan                  # mark as NaN
        n_replaced += mask.sum()


    df = df.set_index("date")
    df[col] = df[col].interpolate(method="time")
    df = df.reset_index()
    print(f"  {col}: {n_replaced} frozen values replaced via interpolation")
    return df

print("\nFix 4 — Outlier capping (IQR):")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper_fence = Q3 + 1.5 * IQR

    n_outliers = (df[col] > upper_fence).sum()
    if n_outliers > 0:
        df[col] = df[col].clip(upper=upper_fence)
        print(f"  {col}: {n_outliers} values capped at {upper_fence:.2f}")
    else:
        print(f"  {col}: no outliers above IQR fence")

print("\nFix 5 — Sensor freeze (pass 1 — before capping):")
for col in ["no", "o3"]:
    df = fix_sensor_freeze(df, col, MIN_FREEZE_RUN)

IQR_MULTIPLIER = 2.5
caps_25 = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    caps_25[col] = Q3 + IQR_MULTIPLIER * IQR

for col, new_cap in caps_25.items():
    old_cap_rows = (df[col] >= df[col].max() - 0.1).sum()  # rows at old ceiling
    print(f"  {col}: old cap={df[col].max():.2f} | new cap={new_cap:.2f} "
          f"| rows released from old ceiling={old_cap_rows}")

print("\nFix 6 — Nighttime O3 correction:")

NIGHT_O3_THRESHOLD = 40
NIGHT_START_HOUR   = 20
NIGHT_END_HOUR     = 6

df = df.set_index("date")

df["hour"] = df.index.hour
night_mask = (
    (df["hour"] >= NIGHT_START_HOUR) | (df["hour"] < NIGHT_END_HOUR)
) & (df["o3"] > NIGHT_O3_THRESHOLD)

n_fixed = night_mask.sum()
df.loc[night_mask, "o3"] = np.nan
df["o3"] = df["o3"].interpolate(method="time")
night_hours = (df["hour"] >= NIGHT_START_HOUR) | (df["hour"] < NIGHT_END_HOUR)
still_high = (night_hours & (df["o3"] > NIGHT_O3_THRESHOLD)).sum()
df.loc[night_hours & (df["o3"] > NIGHT_O3_THRESHOLD), "o3"] = NIGHT_O3_THRESHOLD
df = df.drop(columns="hour").reset_index()
print(f"  {n_fixed} nighttime O3 rows corrected")
print(f"  {still_high} rows hard-capped at {NIGHT_O3_THRESHOLD} µg/m³ after interpolation")

print("\nFix 7 — Smooth cap plateaus + apply 2.5x ceiling:")
df = df.set_index("date")

for col in numeric_cols:
    old_ceiling = df[col].max()
    new_ceiling = caps_25[col]
    at_ceiling = (df[col] >= old_ceiling - 0.01)
    run_id = (at_ceiling != at_ceiling.shift()).cumsum()
    plateau_count = 0

    for gid, group in df[at_ceiling].groupby(run_id[at_ceiling]):
        if len(group) >= 2:
            interior_idx = group.index[1:-1]
            if len(interior_idx) > 0:
                df.loc[interior_idx, col] = np.nan
                plateau_count += 1
    df[col] = df[col].interpolate(method="time")
    df[col] = df[col].clip(upper=new_ceiling)

    remaining_plateaus = (df[col] >= new_ceiling - 0.01).sum()
    print(f"  {col}: {plateau_count} plateaus smoothed | "
          f"new ceiling={new_ceiling:.2f} | rows at new ceiling={remaining_plateaus}")
print("\n  Post-smooth freeze cleanup:")
for col in numeric_cols:
    s = df[col] == df[col].shift(1)
    cumsum_series = (s != s.shift()).cumsum()
    runs = s.groupby(cumsum_series).sum()
    long_groups = runs[runs >= 5].index.tolist()
    if not long_groups:
        print(f"  {col}: OK")
        continue
    n_replaced = 0
    for gid in long_groups:
        mask = cumsum_series == gid
        df.loc[mask, col] = np.nan
        n_replaced += mask.sum()
    df[col] = df[col].interpolate(method="time")
    df[col] = df[col].clip(upper=caps_25[col])
    if col == "o3":
        night_mask2 = (df.index.hour >= NIGHT_START_HOUR) | (df.index.hour < NIGHT_END_HOUR)
        df.loc[night_mask2 & (df[col] > NIGHT_O3_THRESHOLD), col] = NIGHT_O3_THRESHOLD
    print(f"  {col}: {n_replaced} post-smooth frozen values re-interpolated")

df = df.reset_index()
print("VERIFICATION")
gaps = (df["date"].diff().dropna() > pd.Timedelta("1h")).sum()
print(f"Time gaps             : {gaps}")
df_tmp = df.set_index("date")
max_freezes = {}
for col in numeric_cols:
    series = df_tmp[col].copy()
    if col == "o3":
        night_hrs = (df_tmp.index.hour >= NIGHT_START_HOUR) | (df_tmp.index.hour < NIGHT_END_HOUR)
        series[night_hrs & (series == NIGHT_O3_THRESHOLD)] = np.nan
    s = series == series.shift(1)
    runs = s.groupby((s != s.shift()).cumsum()).sum()
    max_freezes[col] = int(runs.max())
freeze_ok = all(v < 5 for v in max_freezes.values())
print(f"Max freeze runs       : {max(max_freezes.values())} "
      f"({'good all < 5' if freeze_ok else 'bad'})")
df["hour"] = df["date"].dt.hour
night_o3 = df[df["hour"].isin([0,1,2,3])]["o3"]
print(f"Night O3 (0–3am) max  : {night_o3.max():.2f}  "
      f"({'good' if night_o3.max() <= NIGHT_O3_THRESHOLD else 'bad'})")
print(f"Night O3 (0–3am) mean : {night_o3.mean():.2f}")

print("Cliff edges remaining :")
for col in numeric_cols:
    cap = caps_25[col]
    at_cap = df[col] >= cap * 0.98
    next_val = df[col].shift(-1)
    big_drop = (df[col] - next_val) / df[col] > 0.40
    cliff = (at_cap & big_drop).sum()
    tag = "good" if cliff == 0 else "bad"
    print(f"  {col}: {cliff} {tag}")

print(f"pm2_5 ~ pm10 corr     : {df['pm2_5'].corr(df['pm10']):.3f}  (expect > 0.85)")
print(f"no ~ no2 corr         : {df['no'].corr(df['no2']):.3f}  (expect > 0.5)")

df = df.drop(columns="hour")
print(f"\nFinal shape           : {df.shape}")
df[numeric_cols] = df[numeric_cols].round(2)

df.to_csv("delhi_aqi_final.csv", index=False)
print("Saved → delhi_aqi_final.csv")
