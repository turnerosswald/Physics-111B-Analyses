import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# RFS_per_degree_overlay.py
#
# Quick helper plot:
#   overplot the 3 per-degree runs + the combined run on ONE graph.
#
# Reuses the same conventions as the data-only plotting script:
#   - 4 V threshold
#   - nominal 2° = physical 0°
#   - run 1 has a 63 s leading offset
#   - run 3 nominal 12° point was measured for 240 s
#   - same total angle uncertainty used as x-error bars
# ============================================================

BASE_DIR = Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS")
OUTPUT_DIR = BASE_DIR / "requested_data_only_plots_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_V = 4.0
TRUE_ZERO_OFFSET_DEG = 2.0

PROTRACTOR_SIGMA_DEG = 1.0
COLLIMATOR_SEPARATION_CM = 1.99
COLLIMATOR_SEPARATION_SIGMA_CM = 0.005
APERTURE_DIAMETER_CM = (1.0 / 16.0) * 2.54

alpha_sigma_rad = (
    APERTURE_DIAMETER_CM
    / (COLLIMATOR_SEPARATION_CM**2 + APERTURE_DIAMETER_CM**2)
    * COLLIMATOR_SEPARATION_SIGMA_CM
)
COLLIMATOR_ANGLE_SIGMA_DEG = np.degrees(alpha_sigma_rad)
TOTAL_ANGLE_SIGMA_DEG = np.hypot(
    PROTRACTOR_SIGMA_DEG,
    COLLIMATOR_ANGLE_SIGMA_DEG,
)

DETECTOR_DISTANCE_CM = 4.5
DETECTOR_DISTANCE_SIGMA_CM = 0.05
DETECTOR_GEOM_REL_SIGMA = 2.0 * DETECTOR_DISTANCE_SIGMA_CM / DETECTOR_DISTANCE_CM
COLLIMATOR_GEOM_REL_SIGMA = 2.0 * COLLIMATOR_SEPARATION_SIGMA_CM / COLLIMATOR_SEPARATION_CM
GEOMETRY_REL_SIGMA = np.hypot(DETECTOR_GEOM_REL_SIGMA, COLLIMATOR_GEOM_REL_SIGMA)

FILES = {
    "inc_run1": BASE_DIR / "PER_DEGREE_RUN_ONE_30MINUTES_RUN_ARIYANTURNER_RFS.csv",
    "inc_run2": BASE_DIR / "PER_DEGREE_RUN_ONE_HOUR_RUN_ARIYANTURNER_RFS_RUN2.csv",
    "inc_run3": BASE_DIR / "PER_DEGREE_RUN_ONE_HOUR_RUN_ARIYANTURNER_RFS_RUN3.csv",
}


def load_run(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["pulse_height_V"] = pd.to_numeric(df["pulse_height_V"], errors="coerce")
    return (
        df.dropna(subset=["timestamp", "pulse_height_V"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def incremental_summary(path, label, leading_offset_s, extended_angle12=False):
    df = load_run(path)
    raw_start = df["timestamp"].min()
    raw_end = df["timestamp"].max()
    zero_start = raw_start + pd.to_timedelta(leading_offset_s, unit="s")

    elapsed = (df["timestamp"] - zero_start).dt.total_seconds().to_numpy()
    accepted_v = df["pulse_height_V"].to_numpy() >= THRESHOLD_V
    available_end_s = (raw_end - zero_start).total_seconds()

    rows = []
    t = 0.0
    for angle in range(16):
        measurement_s = 240.0 if (extended_angle12 and angle == 12) else 120.0
        start_s = t
        scheduled_end_s = t + measurement_s
        actual_end_s = min(scheduled_end_s, available_end_s)
        live_s = max(0.0, actual_end_s - start_s)

        if live_s > 0:
            mask = (
                (elapsed >= start_s)
                & (elapsed < actual_end_s)
                & accepted_v
            )
            N = int(mask.sum())
            live_h = live_s / 3600.0
            rate_h = N / live_h
            poisson_h = math.sqrt(N) / live_h
            geom_h = GEOMETRY_REL_SIGMA * rate_h
            total_h = math.hypot(poisson_h, geom_h)
        else:
            N = 0
            rate_h = poisson_h = geom_h = total_h = np.nan

        rows.append({
            "run": label,
            "nominal_angle_deg": float(angle),
            "calibrated_angle_deg": float(angle) - TRUE_ZERO_OFFSET_DEG,
            "angle_sigma_deg": TOTAL_ANGLE_SIGMA_DEG,
            "actual_live_time_s": live_s,
            "accepted_counts": N,
            "rate_counts_per_hour": rate_h,
            "total_sigma_counts_per_hour": total_h,
            "rate_counts_per_minute": rate_h / 60.0 if np.isfinite(rate_h) else np.nan,
            "total_sigma_counts_per_minute": total_h / 60.0 if np.isfinite(total_h) else np.nan,
        })
        t = scheduled_end_s + 5.0

    return pd.DataFrame(rows)


inc1 = incremental_summary(FILES["inc_run1"], "Incremental run 1", 63.0, False)
inc2 = incremental_summary(FILES["inc_run2"], "Incremental run 2", 0.0, False)
inc3 = incremental_summary(FILES["inc_run3"], "Incremental run 3", 0.0, True)

all_inc = pd.concat([inc1, inc2, inc3], ignore_index=True)

combined_rows = []
for angle in range(16):
    chunk = all_inc[
        (all_inc["nominal_angle_deg"] == angle)
        & (all_inc["actual_live_time_s"] > 0)
    ].copy()

    if len(chunk) == 0:
        continue

    N = int(chunk["accepted_counts"].sum())
    live_s = float(chunk["actual_live_time_s"].sum())
    live_h = live_s / 3600.0
    rate_h = N / live_h
    poisson_h = math.sqrt(N) / live_h
    geom_h = GEOMETRY_REL_SIGMA * rate_h
    total_h = math.hypot(poisson_h, geom_h)

    combined_rows.append({
        "run": "Combined live-time weighted",
        "nominal_angle_deg": float(angle),
        "calibrated_angle_deg": float(angle) - TRUE_ZERO_OFFSET_DEG,
        "angle_sigma_deg": TOTAL_ANGLE_SIGMA_DEG,
        "actual_live_time_s": live_s,
        "accepted_counts": N,
        "rate_counts_per_hour": rate_h,
        "total_sigma_counts_per_hour": total_h,
        "rate_counts_per_minute": rate_h / 60.0,
        "total_sigma_counts_per_minute": total_h / 60.0,
    })

combined = pd.DataFrame(combined_rows)

# ------------------------------------------------------------
# Single overlay plot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13.8, 7.8))

series = [
    (inc1[inc1["actual_live_time_s"] > 0].copy(), "Incremental run 1", "o-", 0.80, 1.1, 4.5, 3),
    (inc2[inc2["actual_live_time_s"] > 0].copy(), "Incremental run 2", "s-", 0.80, 1.1, 4.5, 3),
    (inc3[inc3["actual_live_time_s"] > 0].copy(), "Incremental run 3", "^-", 0.80, 1.1, 4.8, 3),
    (combined[combined["actual_live_time_s"] > 0].copy(), "Combined", "D-", 1.00, 2.0, 5.6, 5),
]

for table, label, fmt, alpha, lw, ms, zorder in series:
    ax.errorbar(
        table["calibrated_angle_deg"],
        table["rate_counts_per_minute"],
        xerr=table["angle_sigma_deg"],
        yerr=table["total_sigma_counts_per_minute"],
        fmt=fmt,
        linewidth=lw,
        markersize=ms,
        capsize=3,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )

# Label the combined points only, to keep the graph readable.
for i, (_, row) in enumerate(combined[combined["actual_live_time_s"] > 0].iterrows()):
    dy = 8 if i % 2 == 0 else -12
    ax.annotate(
        f"{row['rate_counts_per_minute']:.1f}",
        (row["calibrated_angle_deg"], row["rate_counts_per_minute"]),
        xytext=(4, dy),
        textcoords="offset points",
        fontsize=7,
        rotation=30,
        ha="left",
        va="center",
    )

# Highlight the extended point in run 3.
special = inc3[(inc3["actual_live_time_s"] > 0) & (inc3["nominal_angle_deg"] == 12)]
if len(special) == 1:
    s = special.iloc[0]
    ax.scatter(
        [s["calibrated_angle_deg"]],
        [s["rate_counts_per_minute"]],
        marker="s",
        s=95,
        facecolors="none",
        edgecolors="black",
        linewidths=1.4,
        zorder=7,
        label="Run 3 nominal 12° point measured for 4 min",
    )

ax.axvline(
    0.0,
    linestyle="--",
    linewidth=1.0,
    alpha=0.70,
    label="Physical 0°",
)

ax.set_xlabel("Calibrated scattering angle (degrees)")
ax.set_ylabel("Count rate (counts/minute)")
ax.set_title(
    "Per-degree foil scans overplotted on one graph\n"
    "Runs 1–3 plus combined live-time-weighted result"
)
ax.grid(alpha=0.25)
ax.legend(fontsize=9, ncols=2)

fig.tight_layout()
plot_path = OUTPUT_DIR / "per_degree_runs_overlay_with_combined.png"
fig.savefig(plot_path, dpi=240, bbox_inches="tight")
plt.show()

# Export the data used in the overlay plot too.
export = pd.concat([inc1, inc2, inc3, combined], ignore_index=True)
export = export[export["actual_live_time_s"] > 0].copy()
export.to_csv(
    OUTPUT_DIR / "per_degree_runs_overlay_with_combined.csv",
    index=False,
)

print("Saved overlay plot to:")
print(plot_path)
print("Saved overlay data CSV to:")
print(OUTPUT_DIR / "per_degree_runs_overlay_with_combined.csv")
