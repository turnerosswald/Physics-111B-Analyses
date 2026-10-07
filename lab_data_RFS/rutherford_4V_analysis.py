import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math

# ============================================================
# RUTHERFORD DATA-ONLY PLOTTING V2
#
# - 4 V software pulse-height cut on ALL datasets
# - no hard 5-minute startup cut for fixed-angle runs
# - each fixed-angle run gets its own baseline mean/std
# - iterative 3-sigma clipping estimates that baseline
# - a point is ACTUALLY removed only if:
#       (a) it is in the first/last 15% of that run's time bins, and
#       (b) it is > 3 sigma from that run's baseline
# - interior excursions are kept
# - gradient maps are made for zero offsets 0, 1, 2, and 3 degrees
# - no Rutherford fit / no Z extraction / no residual analysis
# - foil areal density is tagged by run date (old through 9/1, new after 9/1)
# ============================================================

BASE_DIR = Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS")
OUTPUT_DIR = BASE_DIR / "requested_data_only_plots_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_V = 4.0
SIGMA_LIMIT = 3.0
EDGE_FRACTION = 0.15
ZERO_OFFSETS_DEG = [0, 1, 2, 3]

# Nominal protractor 2° corresponds to the physical 0° beam direction.
TRUE_ZERO_OFFSET_DEG = 2.0

# Foil areal number density:
# - original/heavier foil used through 2026-09-01
# - replacement/lighter foil used after 2026-09-01
N_AREAL_CM2 = 8.26e18
N_AREAL_CM2_NEW = 4.72e18
FOIL_REPLACEMENT_DATE = pd.Timestamp("2026-09-01")


def areal_density_for_timestamp(timestamp):
    """Return the foil areal density that applies to a run."""
    ts = pd.Timestamp(timestamp)

    if pd.isna(ts):
        raise ValueError("Cannot choose foil areal density from a missing timestamp.")

    # Keep the comparison date-only in case timestamps contain clock times.
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    run_date = ts.normalize()

    if run_date <= FOIL_REPLACEMENT_DATE:
        return N_AREAL_CM2
    elif run_date > FOIL_REPLACEMENT_DATE:
        return N_AREAL_CM2_NEW

    raise ValueError(f"Could not determine foil areal density for {timestamp!r}.")


PROTRACTOR_SIGMA_DEG = 1.0
COLLIMATOR_SEPARATION_CM = 1.99
COLLIMATOR_SEPARATION_SIGMA_CM = 0.005
DETECTOR_DISTANCE_CM = 4.5
DETECTOR_DISTANCE_SIGMA_CM = 0.05
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

DETECTOR_GEOM_REL_SIGMA = 2.0 * DETECTOR_DISTANCE_SIGMA_CM / DETECTOR_DISTANCE_CM
COLLIMATOR_GEOM_REL_SIGMA = 2.0 * COLLIMATOR_SEPARATION_SIGMA_CM / COLLIMATOR_SEPARATION_CM
GEOMETRY_REL_SIGMA = np.hypot(DETECTOR_GEOM_REL_SIGMA, COLLIMATOR_GEOM_REL_SIGMA)

FILES = {
    "5_run1": BASE_DIR / "5DEGREE_RUN_ONE_HOUR_ARIYANTURNER_RFS.csv",
    "5_run2": BASE_DIR / "5DEGREES_HOUR_RUN_ARIYANTURNER_RFS.csv",
    "10_run1": BASE_DIR / "10DEGREES_OVERNIGHT_RUN_ARIYANTURNER_RFS.csv",
    "10_run2": BASE_DIR / "10DEGREE_OVERNIGHT_RUN_ARIYAN_TURNER_RFS_RUN2.csv",
    "15_run1": BASE_DIR / "15DEGREES_OVERNIGHT_ARIYANTURNER_RFS.csv",
    "15_run2": BASE_DIR / "15DEGREES_WEEKEND_RUN_ARIYANTURNER_RFS.csv",
    "15_run3": BASE_DIR / "15DEGREES_OVERNIGHT_ARIYANTURNER_RUN3.csv",
    "no_foil": BASE_DIR / "NO_FOIL_TEN_MINUTE_RUN_ARIYANTURNER_RFS.csv",
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

def make_rate_bins(df, bin_width_s, rate_unit_s):
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    total_s = (end - start).total_seconds()
    elapsed = (df["timestamp"] - start).dt.total_seconds().to_numpy()
    accepted = df["pulse_height_V"].to_numpy() >= THRESHOLD_V

    edges = np.arange(0.0, total_s + bin_width_s, bin_width_s)
    if edges[-1] > total_s:
        edges[-1] = total_s
    if edges[-1] < total_s:
        edges = np.append(edges, total_s)
    edges = np.unique(edges)

    rows = []
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        live_s = right - left
        mask = (elapsed >= left) & (elapsed < right) & accepted
        N = int(mask.sum())
        live_units = live_s / rate_unit_s
        rate = N / live_units
        poisson = math.sqrt(N) / live_units
        geom = GEOMETRY_REL_SIGMA * rate
        total = math.hypot(poisson, geom)

        rows.append({
            "bin_index": i,
            "elapsed_center_s": 0.5 * (left + right),
            "live_time_s": live_s,
            "accepted_counts": N,
            "rate": rate,
            "poisson_sigma": poisson,
            "geometry_sigma": geom,
            "total_sigma": total,
        })
    return pd.DataFrame(rows)

def iterative_sigma_baseline(values):
    values = np.asarray(values, float)
    mask = np.isfinite(values)

    for _ in range(20):
        v = values[mask]
        mean = v.mean()
        std = v.std(ddof=1)
        if std <= 0:
            break
        new_mask = np.isfinite(values) & (np.abs(values - mean) <= 3.0 * std)
        if np.array_equal(new_mask, mask):
            mask = new_mask
            break
        mask = new_mask

    v = values[mask]
    return v.mean(), v.std(ddof=1)

def mark_edge_outliers(table):
    table = table.copy()
    mean, std = iterative_sigma_baseline(table["rate"].to_numpy())

    n = len(table)
    edge_n = max(1, int(np.ceil(EDGE_FRACTION * n)))
    near_edge = np.zeros(n, dtype=bool)
    near_edge[:edge_n] = True
    near_edge[-edge_n:] = True

    z = np.abs(table["rate"].to_numpy() - mean) / std
    table["baseline_mean_rate"] = mean
    table["baseline_std_rate"] = std
    table["z_from_run_baseline"] = z
    table["near_start_or_end"] = near_edge
    table["excluded_edge_outlier"] = near_edge & (z > 3.0)
    table["kept_for_run_average"] = ~table["excluded_edge_outlier"]
    return table

fixed_groups = {
    5: (60.0, 60.0, 60.0, "minutes", [
        ("5° run 1", FILES["5_run1"]),
        ("5° run 2", FILES["5_run2"]),
    ]),
    10: (3600.0, 3600.0, 3600.0, "hours", [
        ("10° run 1", FILES["10_run1"]),
        ("10° run 2", FILES["10_run2"]),
    ]),
    15: (3600.0, 3600.0, 3600.0, "hours", [
        ("15° overnight run 1", FILES["15_run1"]),
        ("15° weekend run", FILES["15_run2"]),
        ("15° overnight run 3", FILES["15_run3"]),
    ]),
}

# Save the run-average values too, since these are useful later in the analysis.
fixed_average_rows = []

for angle, (bin_s, unit_s, divisor, unit_name, runs) in fixed_groups.items():
    # Wider plots leave room for the quantitative point labels.
    if angle == 15:
        plt.figure(figsize=(17.0, 8.5))
    else:
        plt.figure(figsize=(14.5, 7.5))

    for run_index, (label, path) in enumerate(runs):
        run_df = load_run(path)
        run_start = run_df["timestamp"].min()
        run_N_areal = areal_density_for_timestamp(run_start)

        table = mark_edge_outliers(
            make_rate_bins(
                run_df,
                bin_s,
                unit_s,
            )
        )

        kept = table[
            table["kept_for_run_average"]
        ].copy()

        removed = table[
            table["excluded_edge_outlier"]
        ].copy()

        # --------------------------------------------------------
        # Average count rate for THIS RUN
        # --------------------------------------------------------
        # Use total retained counts / total retained live time.
        # This is preferable to a plain arithmetic mean when the
        # final time bin is only partially filled.
        total_kept_counts = kept[
            "accepted_counts"
        ].sum()

        total_kept_live_units = (
            kept["live_time_s"].sum()
            / unit_s
        )

        average_rate = (
            total_kept_counts
            / total_kept_live_units
        )

        average_poisson_sigma = (
            math.sqrt(total_kept_counts)
            / total_kept_live_units
        )
        average_geometry_sigma = (
            GEOMETRY_REL_SIGMA * average_rate
        )
        average_rate_sigma = math.hypot(
            average_poisson_sigma,
            average_geometry_sigma,
        )

        rate_unit_text = (
            "counts/min"
            if angle == 5
            else "counts/hour"
        )

        fixed_average_rows.append({
            "run": label,
            "nominal_angle_deg": angle,
            "calibrated_angle_deg": angle - TRUE_ZERO_OFFSET_DEG,
            "average_rate": average_rate,
            "sigma_average_rate": average_rate_sigma,
            "rate_unit": rate_unit_text,
            "retained_counts": total_kept_counts,
            "retained_live_time_s": kept["live_time_s"].sum(),
            "excluded_edge_bins": len(removed),
            "run_start_timestamp": run_start,
            "N_areal_cm2": run_N_areal,
        })

        # --------------------------------------------------------
        # Main count-rate curve + uncertainty bars
        # --------------------------------------------------------
        x_kept = (
            kept["elapsed_center_s"]
            / divisor
        )

        plt.errorbar(
            x_kept,
            kept["rate"],
            yerr=kept["total_sigma"],
            fmt="o-",
            markersize=3.5,
            linewidth=1.0,
            capsize=2,
            alpha=0.80,
            label=(
                f"{label} "
                f"(calibrated {angle - TRUE_ZERO_OFFSET_DEG:g}°, "
                f"avg = {average_rate:.1f} ± {average_rate_sigma:.1f} {rate_unit_text})"
            ),
        )

        # Dashed horizontal line = average for this run.
        plt.axhline(
            average_rate,
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
        )

        # --------------------------------------------------------
        # Label EACH retained count-rate point quantitatively.
        # The labels are just the rate value; the y-axis supplies
        # the unit. Different runs are offset in different directions
        # to reduce overlap.
        # --------------------------------------------------------
        run_vertical_offsets = [8, -12, 17]
        base_dy = run_vertical_offsets[
            run_index % len(run_vertical_offsets)
        ]

        for point_index, (x_value, rate_value, sigma_value) in enumerate(
            zip(
                x_kept.to_numpy(),
                kept["rate"].to_numpy(),
                kept["total_sigma"].to_numpy(),
            )
        ):
            # Small alternating nudge helps nearby labels separate.
            extra_dy = 3 if point_index % 2 else 0

            plt.annotate(
                f"{rate_value:.1f} ± {sigma_value:.1f}",
                (x_value, rate_value),
                xytext=(3, base_dy + extra_dy),
                textcoords="offset points",
                fontsize=6.2,
                rotation=45,
                ha="left",
                va="center",
            )

        # Keep rejected edge points visible and label their numerical
        # rates too, so the data cleaning remains transparent.
        if len(removed):
            x_removed = (
                removed["elapsed_center_s"]
                / divisor
            )

            plt.errorbar(
                x_removed,
                removed["rate"],
                yerr=removed["total_sigma"],
                fmt="x",
                markersize=7,
                capsize=2,
                linewidth=1.0,
                label=f"{label} — excluded edge >3σ",
            )

            for x_value, rate_value, sigma_value in zip(
                x_removed.to_numpy(),
                removed["rate"].to_numpy(),
                removed["total_sigma"].to_numpy(),
            ):
                plt.annotate(
                    f"{rate_value:.1f} ± {sigma_value:.1f}",
                    (x_value, rate_value),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=6.2,
                    rotation=45,
                    ha="left",
                    va="bottom",
                )

    plt.xlabel(
        f"Elapsed time from raw run start ({unit_name})"
    )

    plt.ylabel(
        "Count rate (counts/minute)"
        if angle == 5
        else "Count rate (counts/hour)"
    )

    plt.title(
        (
            f"{angle - TRUE_ZERO_OFFSET_DEG:g}° fixed-angle count rate "
        )
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{angle}deg_count_rate_edge_3sigma_cleaned_labeled.png",
        dpi=240,
        bbox_inches="tight",
    )

    plt.show()

# Save the quantitative average count rates for later analysis.
pd.DataFrame(
    fixed_average_rows
).to_csv(
    OUTPUT_DIR / "fixed_angle_average_count_rates.csv",
    index=False,
)

# ---------------- Incremental scans ----------------

RUN1_OFFSET_S = 63.0
ANGLE_MEASUREMENT_S = 120.0
ANGLE_CHANGE_S = 5.0

def incremental_summary(path, label, leading_offset_s, extended_angle12=False):
    df = load_run(path)
    raw_start = df["timestamp"].min()
    run_N_areal = areal_density_for_timestamp(raw_start)
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
            rate = N / live_h
            poisson = math.sqrt(N) / live_h
            geom = GEOMETRY_REL_SIGMA * rate
            total = math.hypot(poisson, geom)
        else:
            N = 0
            rate = poisson = geom = total = np.nan

        rows.append({
            "run": label,
            "nominal_angle_deg": float(angle),
            "angle_sigma_deg": TOTAL_ANGLE_SIGMA_DEG,
            "actual_live_time_s": live_s,
            "accepted_counts": N,
            "rate_counts_per_hour": rate,
            "total_sigma_counts_per_hour": total,
            "run_start_timestamp": raw_start,
            "N_areal_cm2": run_N_areal,
            "areal_exposure_cm2_s": run_N_areal * live_s,
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
    ]
    N = int(chunk["accepted_counts"].sum())
    live_s = float(chunk["actual_live_time_s"].sum())
    live_h = live_s / 3600.0
    rate = N / live_h
    areal_exposure = float(chunk["areal_exposure_cm2_s"].sum())
    effective_N_areal = areal_exposure / live_s

    combined_rows.append({
        "run": "Combined live-time weighted",
        "nominal_angle_deg": float(angle),
        "angle_sigma_deg": TOTAL_ANGLE_SIGMA_DEG,
        "actual_live_time_s": live_s,
        "accepted_counts": N,
        "rate_counts_per_hour": rate,
        "N_areal_cm2": effective_N_areal,
        "areal_exposure_cm2_s": areal_exposure,
    })
combined = pd.DataFrame(combined_rows)

def calibrated_radial_profile(table, zero_offset_deg):
    w = table[table["actual_live_time_s"] > 0].copy()
    w["calibrated_angle_deg"] = w["nominal_angle_deg"] - zero_offset_deg
    w["radial_angle_deg"] = np.abs(w["calibrated_angle_deg"])

    grouped = (
        w.groupby("radial_angle_deg", as_index=False)
        .agg(
            accepted_counts=("accepted_counts", "sum"),
            live_time_s=("actual_live_time_s", "sum"),
        )
        .sort_values("radial_angle_deg")
    )
    grouped["rate_counts_per_hour"] = (
        grouped["accepted_counts"]
        / (grouped["live_time_s"] / 3600.0)
    )
    return grouped

datasets = {
    "incremental_run1": ("Incremental run 1", inc1),
    "incremental_run2": ("Incremental run 2", inc2),
    "incremental_run3": ("Incremental run 3", inc3),
}

# ============================================================
# CIRCULAR GRADIENT MAPS — VISUAL ONLY
#
# One 2x2 sheet per incremental run:
#     0° offset | 1° offset
#     2° offset | 3° offset
#
# These maps are intentionally kept clean. The quantitative rate
# labels belong on the fixed-angle count-rate plots above.
# ============================================================

profiles = {}
all_rates = []

for key, (label, table) in datasets.items():
    for z0 in ZERO_OFFSETS_DEG:
        p = calibrated_radial_profile(
            table,
            z0,
        )

        profiles[
            (key, z0)
        ] = p

        all_rates.extend(
            p[
                "rate_counts_per_hour"
            ].to_numpy(
                dtype=float
            )
        )

# One shared color scale so the visual intensity means the same thing
# across every run and every zero-offset choice.
vmin = np.nanmin(
    all_rates
)

vmax = np.nanmax(
    all_rates
)

MAP_EXTENT_DEG = 15.0
GRID_SIZE = 700

for key, (label, table) in datasets.items():

    # ONE window per run rather than four separate windows.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.5, 13.0),
    )

    axes = axes.flatten()
    last_im = None

    for ax, z0 in zip(
        axes,
        ZERO_OFFSETS_DEG,
    ):
        p = profiles[
            (key, z0)
        ]

        r_data = p[
            "radial_angle_deg"
        ].to_numpy(
            dtype=float
        )

        rate_data = p[
            "rate_counts_per_hour"
        ].to_numpy(
            dtype=float
        )

        axis = np.linspace(
            -MAP_EXTENT_DEG,
            MAP_EXTENT_DEG,
            GRID_SIZE,
        )

        X, Y = np.meshgrid(
            axis,
            axis,
        )

        R = np.sqrt(
            X**2 + Y**2
        )

        field = np.full_like(
            R,
            np.nan,
            dtype=float,
        )

        max_measured_radius = np.nanmax(
            r_data
        )

        measured_region = (
            R <= max_measured_radius
        )

        field[
            measured_region
        ] = np.interp(
            R[measured_region],
            r_data,
            rate_data,
        )

        last_im = ax.imshow(
            field,
            extent=[
                -MAP_EXTENT_DEG,
                MAP_EXTENT_DEG,
                -MAP_EXTENT_DEG,
                MAP_EXTENT_DEG,
            ],
            origin="lower",
            interpolation="bilinear",
            vmin=vmin,
            vmax=vmax,
        )

        # Visual guide circles only.
        for rr in [
            5,
            10,
            15,
        ]:
            ax.add_patch(
                plt.Circle(
                    (0, 0),
                    rr,
                    fill=False,
                    linewidth=0.8,
                    alpha=0.42,
                )
            )

            ax.text(
                0.45,
                rr - 0.8,
                f"{rr}°",
                fontsize=9,
                alpha=0.75,
            )

        # Center = calibrated zero angle.
        ax.scatter(
            [0],
            [0],
            marker="+",
            s=48,
            linewidths=1.4,
        )

        ax.set_aspect(
            "equal"
        )

        ax.set_xlim(
            -MAP_EXTENT_DEG,
            MAP_EXTENT_DEG,
        )

        ax.set_ylim(
            -MAP_EXTENT_DEG,
            MAP_EXTENT_DEG,
        )

        ax.set_xlabel(
            "Calibrated scattering-angle coordinate (degrees)"
        )

        ax.set_ylabel(
            "Calibrated scattering-angle coordinate (degrees)"
        )

        ax.set_title(
            f"Zero offset = {z0}°",
            fontsize=13,
            pad=10,
        )

        ax.grid(
            alpha=0.10
        )

    fig.suptitle(
        (
            f"{label}: circular count-rate gradients\n"
            f"zero-angle offsets 0°, 1°, 2°, and 3°"
        ),
        fontsize=16,
        y=0.98,
    )

    # Shared colorbar for the whole 2x2 sheet.
    fig.subplots_adjust(
        left=0.06,
        right=0.90,
        bottom=0.06,
        top=0.91,
        wspace=0.24,
        hspace=0.23,
    )

    colorbar_ax = fig.add_axes(
        [
            0.925,
            0.16,
            0.018,
            0.68,
        ]
    )

    colorbar = fig.colorbar(
        last_im,
        cax=colorbar_ax,
    )

    colorbar.set_label(
        "Count rate after 4 V cut (counts/hour)",
        fontsize=11,
    )

    plt.savefig(
        OUTPUT_DIR
        / f"gradient_{key}_offsets_0_1_2_3_one_sheet.png",
        dpi=240,
        bbox_inches="tight",
    )

    # Only ONE window opens for this run.
    plt.show()


# ============================================================
# ADDITIONAL DATA-ONLY PLOTS
# Known calibration: nominal 2° = physical 0°.
# NO Rutherford fit / Z extraction / residual analysis here.
# ============================================================

# ============================================================
# PER-DEGREE FOIL RUNS — COUNT RATE VS CALIBRATED ANGLE
#
# Reuses the timing logic above unchanged:
#   120 s measurement + 5 s angle-change allowance
#   Run 1 keeps its 63 s leading offset
#   Run 3 keeps nominal 12° for 240 s
# ============================================================

per_degree_plot_runs = [
    ("Incremental run 1", inc1),
    ("Incremental run 2", inc2),
    ("Incremental run 3", inc3),
]

per_degree_export_rows = []

fig, axes = plt.subplots(
    3,
    1,
    figsize=(15.0, 15.5),
    sharex=True,
)

for run_index, (run_label, table) in enumerate(per_degree_plot_runs):
    ax = axes[run_index]
    plot_table = table[table["actual_live_time_s"] > 0].copy()

    plot_table["calibrated_angle_deg"] = (
        plot_table["nominal_angle_deg"] - TRUE_ZERO_OFFSET_DEG
    )
    plot_table["rate_counts_per_minute"] = (
        plot_table["rate_counts_per_hour"] / 60.0
    )
    plot_table["total_sigma_counts_per_minute"] = (
        plot_table["total_sigma_counts_per_hour"] / 60.0
    )

    per_degree_export_rows.append(
        plot_table[
            [
                "run",
                "nominal_angle_deg",
                "calibrated_angle_deg",
                "angle_sigma_deg",
                "actual_live_time_s",
                "accepted_counts",
                "rate_counts_per_minute",
                "total_sigma_counts_per_minute",
                "run_start_timestamp",
                "N_areal_cm2",
            ]
        ].copy()
    )

    ax.errorbar(
        plot_table["calibrated_angle_deg"],
        plot_table["rate_counts_per_minute"],
        xerr=plot_table["angle_sigma_deg"],
        yerr=plot_table["total_sigma_counts_per_minute"],
        fmt="o-",
        markersize=5,
        linewidth=1.2,
        capsize=3,
        label=run_label,
    )

    # Label every measured per-degree point with its count rate.
    for point_number, (_, row) in enumerate(plot_table.iterrows()):
        dy = 8 if point_number % 2 == 0 else -12
        ax.annotate(
            f"{row['rate_counts_per_minute']:.1f} ± "
            f"{row['total_sigma_counts_per_minute']:.1f}",
            (
                row["calibrated_angle_deg"],
                row["rate_counts_per_minute"],
            ),
            xytext=(4, dy),
            textcoords="offset points",
            fontsize=7,
            rotation=35,
            ha="left",
            va="center",
        )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
        label="Physical 0°",
    )

    # Run 3 special case: nominal 12° was measured for 4 minutes.
    if run_label == "Incremental run 3":
        special = plot_table[plot_table["nominal_angle_deg"] == 12]
        if len(special) == 1:
            s = special.iloc[0]
            ax.scatter(
                [s["calibrated_angle_deg"]],
                [s["rate_counts_per_minute"]],
                marker="s",
                s=85,
                facecolors="none",
                edgecolors="black",
                linewidths=1.4,
                zorder=5,
                label="Nominal 12° / calibrated 10° measured for 4 min",
            )

    ax.set_ylabel("Count rate\n(counts/minute)")
    ax.set_title(
        f"{run_label}: per-degree foil scan (nominal 2° = true 0°)"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

axes[-1].set_xlabel("Calibrated scattering angle (degrees)")

fig.suptitle(
    "Per-degree foil scans — measured count rates only\n"
    "120 s per angle + 5 s movement; Run 3 nominal 12° measured for 240 s",
    fontsize=16,
    y=0.995,
)

fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig(
    OUTPUT_DIR / "per_degree_runs_count_rate_vs_calibrated_angle_labeled.png",
    dpi=240,
    bbox_inches="tight",
)
plt.show()

pd.concat(per_degree_export_rows, ignore_index=True).to_csv(
    OUTPUT_DIR / "per_degree_runs_count_rate_vs_calibrated_angle.csv",
    index=False,
)


# ============================================================
# NO-FOIL ZERO-ANGLE RUN — nominal 0° -> 1° -> 2°
#
# Each setting is held for 4 minutes. The 5 s after the 4-minute
# and 8-minute boundaries is excluded as angle-change time.
# ============================================================

NO_FOIL_BIN_SECONDS = 60.0
NO_FOIL_TRANSITION_WINDOWS = [
    (4.0 * 60.0, 4.0 * 60.0 + 5.0),
    (8.0 * 60.0, 8.0 * 60.0 + 5.0),
]


def overlap_seconds(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def make_no_foil_rate_bins(df):
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    total_s = (end - start).total_seconds()

    elapsed_s = (df["timestamp"] - start).dt.total_seconds().to_numpy()
    passes_voltage_cut = df["pulse_height_V"].to_numpy() >= THRESHOLD_V

    edges = np.arange(
        0.0,
        total_s + NO_FOIL_BIN_SECONDS,
        NO_FOIL_BIN_SECONDS,
    )

    if edges[-1] > total_s:
        edges[-1] = total_s
    if edges[-1] < total_s:
        edges = np.append(edges, total_s)
    edges = np.unique(edges)

    rows = []

    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        live_s = right - left

        for transition_start, transition_end in NO_FOIL_TRANSITION_WINDOWS:
            live_s -= overlap_seconds(
                left,
                right,
                transition_start,
                transition_end,
            )

        live_s = max(live_s, 0.0)
        if live_s <= 0:
            continue

        accepted = (
            (elapsed_s >= left)
            & (elapsed_s < right)
            & passes_voltage_cut
        )

        for transition_start, transition_end in NO_FOIL_TRANSITION_WINDOWS:
            accepted &= ~(
                (elapsed_s >= transition_start)
                & (elapsed_s < transition_end)
            )

        N = int(accepted.sum())
        live_min = live_s / 60.0
        rate = N / live_min
        poisson_sigma = math.sqrt(N) / live_min
        geometry_sigma = GEOMETRY_REL_SIGMA * rate
        total_sigma = math.hypot(poisson_sigma, geometry_sigma)

        center_min = 0.5 * (left + right) / 60.0

        if center_min < 4.0:
            nominal_angle = 0.0
        elif center_min < 8.0:
            nominal_angle = 1.0
        else:
            nominal_angle = 2.0

        rows.append(
            {
                "bin_index": bin_index,
                "elapsed_center_min": center_min,
                "nominal_angle_deg": nominal_angle,
                "calibrated_angle_deg": nominal_angle - TRUE_ZERO_OFFSET_DEG,
                "live_time_s": live_s,
                "accepted_counts": N,
                "rate_counts_per_minute": rate,
                "poisson_sigma_counts_per_minute": poisson_sigma,
                "geometry_sigma_counts_per_minute": geometry_sigma,
                "total_sigma_counts_per_minute": total_sigma,
            }
        )

    return pd.DataFrame(rows)


no_foil_raw = load_run(FILES["no_foil"])
no_foil_rates = make_no_foil_rate_bins(no_foil_raw)

no_foil_rates.to_csv(
    OUTPUT_DIR / "no_foil_0_1_2_count_rate_per_minute.csv",
    index=False,
)

# Average count rate for each 4-minute nominal setting.
no_foil_angle_averages = (
    no_foil_rates
    .groupby(
        ["nominal_angle_deg", "calibrated_angle_deg"],
        as_index=False,
    )
    .agg(
        accepted_counts=("accepted_counts", "sum"),
        live_time_s=("live_time_s", "sum"),
    )
)

no_foil_angle_averages["average_rate_counts_per_minute"] = (
    no_foil_angle_averages["accepted_counts"]
    / (no_foil_angle_averages["live_time_s"] / 60.0)
)

no_foil_angle_averages["poisson_sigma_counts_per_minute"] = (
    np.sqrt(no_foil_angle_averages["accepted_counts"])
    / (no_foil_angle_averages["live_time_s"] / 60.0)
)

no_foil_angle_averages["geometry_sigma_counts_per_minute"] = (
    GEOMETRY_REL_SIGMA
    * no_foil_angle_averages["average_rate_counts_per_minute"]
)

no_foil_angle_averages["total_sigma_counts_per_minute"] = np.hypot(
    no_foil_angle_averages["poisson_sigma_counts_per_minute"],
    no_foil_angle_averages["geometry_sigma_counts_per_minute"],
)

no_foil_angle_averages.to_csv(
    OUTPUT_DIR / "no_foil_0_1_2_average_count_rates.csv",
    index=False,
)

fig, ax = plt.subplots(figsize=(13.5, 7.2))

ax.errorbar(
    no_foil_rates["elapsed_center_min"],
    no_foil_rates["rate_counts_per_minute"],
    yerr=no_foil_rates["total_sigma_counts_per_minute"],
    fmt="o-",
    markersize=5,
    linewidth=1.2,
    capsize=3,
    label="No-foil measured count rate",
)

# Quantitative label on every 1-minute count-rate point.
for point_number, (_, row) in enumerate(no_foil_rates.iterrows()):
    dy = 8 if point_number % 2 == 0 else -12
    ax.annotate(
        f"{row['rate_counts_per_minute']:.1f} ± "
        f"{row['total_sigma_counts_per_minute']:.1f}",
        (
            row["elapsed_center_min"],
            row["rate_counts_per_minute"],
        ),
        xytext=(4, dy),
        textcoords="offset points",
        fontsize=7,
        rotation=35,
        ha="left",
        va="center",
    )

for boundary_min in [4.0, 8.0]:
    ax.axvline(
        boundary_min,
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
    )

# segment_labels = [
#     (2.0, "Nominal 0°\nCalibrated -2°"),
#     (6.0, "Nominal 1°\nCalibrated -1°"),
#     (10.0, "Nominal 2°\nTrue 0°"),
# ]
segment_labels = [
    (2.0, "0°"),
    (6.0, "1°"),
    (10.0, "2°"),
]
for x_position, label_text in segment_labels:
    ax.text(
        x_position,
        0.97,
        label_text,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10,
    )

# Horizontal average line for each 4-minute setting.
for row in no_foil_angle_averages.itertuples(index=False):
    nominal_angle = int(row.nominal_angle_deg)

    if nominal_angle == 0:
        x_start, x_end = 0.0, 4.0
    elif nominal_angle == 1:
        x_start, x_end = 4.0, 8.0
    else:
        x_start, x_end = 8.0, 12.0

    ax.hlines(
        row.average_rate_counts_per_minute,
        x_start,
        x_end,
        linestyles=":",
        linewidth=1.8,
        label=(
            f"Nominal {nominal_angle}° "
            f"(cal. {row.calibrated_angle_deg:g}°) avg = "
            f"{row.average_rate_counts_per_minute:.1f} ± "
            f"{row.total_sigma_counts_per_minute:.1f} counts/min"
        ),
    )

ax.set_xlabel(
    "Elapsed time from raw run start (minutes)",
    fontsize=12,
)
ax.set_ylabel(
    "Count rate after 4 V cut (counts/minute)",
    fontsize=12,
)
ax.set_title(
    "No-foil zero-angle Calibration: \n"
    "Nominal 2° is the physical 0° beam direction",
    fontsize=14,
)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "no_foil_0_1_2_count_rate_per_minute_labeled.png",
    dpi=240,
    bbox_inches="tight",
)
plt.show()


# ============================================================
# CIRCULAR GRADIENT MAPS — VISUAL ONLY
#
# One 2x2 sheet per incremental run:
#     0° offset | 1° offset
#     2° offset | 3° offset
#
# These maps are intentionally kept clean. The quantitative rate
# labels belong on the fixed-angle count-rate plots above.
# ============================================================

profiles = {}
all_rates = []

for key, (label, table) in datasets.items():
    for z0 in ZERO_OFFSETS_DEG:
        p = calibrated_radial_profile(
            table,
            z0,
        )

        profiles[
            (key, z0)
        ] = p

        all_rates.extend(
            p[
                "rate_counts_per_hour"
            ].to_numpy(
                dtype=float
            )
        )

# One shared color scale so the visual intensity means the same thing
# across every run and every zero-offset choice.
vmin = np.nanmin(
    all_rates
)

vmax = np.nanmax(
    all_rates
)

MAP_EXTENT_DEG = 15.0
GRID_SIZE = 700

for key, (label, table) in datasets.items():

    # ONE window per run rather than four separate windows.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.5, 13.0),
    )

    axes = axes.flatten()
    last_im = None

    for ax, z0 in zip(
        axes,
        ZERO_OFFSETS_DEG,
    ):
        p = profiles[
            (key, z0)
        ]

        r_data = p[
            "radial_angle_deg"
        ].to_numpy(
            dtype=float
        )

        rate_data = p[
            "rate_counts_per_hour"
        ].to_numpy(
            dtype=float
        )

        axis = np.linspace(
            -MAP_EXTENT_DEG,
            MAP_EXTENT_DEG,
            GRID_SIZE,
        )

        X, Y = np.meshgrid(
            axis,
            axis,
        )

        R = np.sqrt(
            X**2 + Y**2
        )

        field = np.full_like(
            R,
            np.nan,
            dtype=float,
        )

        max_measured_radius = np.nanmax(
            r_data
        )

        measured_region = (
            R <= max_measured_radius
        )

        field[
            measured_region
        ] = np.interp(
            R[measured_region],
            r_data,
            rate_data,
        )

        last_im = ax.imshow(
            field,
            extent=[
                -MAP_EXTENT_DEG,
                MAP_EXTENT_DEG,
                -MAP_EXTENT_DEG,
                MAP_EXTENT_DEG,
            ],
            origin="lower",
            interpolation="bilinear",
            vmin=vmin,
            vmax=vmax,
        )

        # Visual guide circles only.
        for rr in [
            5,
            10,
            15,
        ]:
            ax.add_patch(
                plt.Circle(
                    (0, 0),
                    rr,
                    fill=False,
                    linewidth=0.8,
                    alpha=0.42,
                )
            )

            ax.text(
                0.45,
                rr - 0.8,
                f"{rr}°",
                fontsize=9,
                alpha=0.75,
            )

        # Center = calibrated zero angle.
        ax.scatter(
            [0],
            [0],
            marker="+",
            s=48,
            linewidths=1.4,
        )

        ax.set_aspect(
            "equal"
        )

        ax.set_xlim(
            -MAP_EXTENT_DEG,
            MAP_EXTENT_DEG,
        )

        ax.set_ylim(
            -MAP_EXTENT_DEG,
            MAP_EXTENT_DEG,
        )

        ax.set_xlabel(
            "Calibrated scattering-angle coordinate (degrees)"
        )

        ax.set_ylabel(
            "Calibrated scattering-angle coordinate (degrees)"
        )

        ax.set_title(
            f"Zero offset = {z0}°",
            fontsize=13,
            pad=10,
        )

        ax.grid(
            alpha=0.10
        )

    fig.suptitle(
        (
            f"{label}: circular count-rate gradients\n"
            f"zero-angle offsets 0°, 1°, 2°, and 3°"
        ),
        fontsize=16,
        y=0.98,
    )

    # Shared colorbar for the whole 2x2 sheet.
    fig.subplots_adjust(
        left=0.06,
        right=0.90,
        bottom=0.06,
        top=0.91,
        wspace=0.24,
        hspace=0.23,
    )

    colorbar_ax = fig.add_axes(
        [
            0.925,
            0.16,
            0.018,
            0.68,
        ]
    )

    colorbar = fig.colorbar(
        last_im,
        cax=colorbar_ax,
    )

    colorbar.set_label(
        "Count rate after 4 V cut (counts/hour)",
        fontsize=11,
    )

    plt.savefig(
        OUTPUT_DIR
        / f"gradient_{key}_offsets_0_1_2_3_one_sheet.png",
        dpi=240,
        bbox_inches="tight",
    )

    # Only ONE window opens for this run.
    plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ============================================================
# # RUTHERFORD LAB — DATA-ONLY VISUALIZATION
# # ZERO-ANGLE CALIBRATION CHECK: 0°, 1°, 2°, 3°
# # ============================================================
# #
# # No Rutherford theory fit.
# # No inferred Z.
# # No residuals.
# #
# # This version tests four possible choices for the true zero angle:
# #
# #       ZERO = nominal 0°
# #       ZERO = nominal 1°
# #       ZERO = nominal 2°
# #       ZERO = nominal 3°
# #
# # If zero is set to z0, then:
# #
# #       calibrated angle = nominal angle - z0
# #
# # Example for ZERO = 2°:
# #
# #       nominal 0° -> calibrated -2°
# #       nominal 1° -> calibrated -1°
# #       nominal 2° -> calibrated  0°
# #       nominal 3° -> calibrated +1°
# #       ...
# #
# # For the CIRCULAR maps, scattering-angle magnitude is radial.
# # Therefore +theta and -theta correspond to the same radius.
# # When both are present, their measured count rates are averaged
# # before interpolation.
# #
# # The only "function" used for the gradient map is piecewise-linear
# # interpolation between the resulting measured radial count rates.
# # No physics equation is fit.
# #
# # OUTPUTS
# # -------
# # 1. One large plot containing all six fixed-angle count-rate-vs-time runs.
# #
# # 2. One 2x2 figure showing count rate vs calibrated scattering angle
# #    for zero choices 0°, 1°, 2°, and 3°.
# #
# # 3-6. Four separate, spacious circular-gradient figures:
# #      - zero = 0°
# #      - zero = 1°
# #      - zero = 2°
# #      - zero = 3°
# #
# #      Each figure contains:
# #          incremental run 1
# #          incremental run 2
# #          combined average
# #
# # ============================================================


# # ============================================================
# # 0. FILES AND SETTINGS
# # ============================================================

# BASE_DIR = Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS")

# PER_DEGREE_RUN1_FILE = (
#     BASE_DIR / "PER_DEGREE_RUN_ONE_30MINUTES_RUN_ARIYANTURNER_RFS.csv"
# )

# PER_DEGREE_RUN2_FILE = (
#     BASE_DIR / "PER_DEGREE_RUN_ONE_HOUR_RUN_ARIYANTURNER_RFS_RUN2.csv"
# )

# FIVE_RUN1_FILE = (
#     BASE_DIR / "5DEGREE_RUN_ONE_HOUR_ARIYANTURNER_RFS.csv"
# )

# FIVE_RUN2_FILE = (
#     BASE_DIR / "5DEGREES_HOUR_RUN_ARIYANTURNER_RFS.csv"
# )

# TEN_RUN1_FILE = (
#     BASE_DIR / "10DEGREE_OVERNIGHT_RUN_ARIYAN_TURNER_RFS_RUN2.csv"
# )

# TEN_RUN2_FILE = (
#     BASE_DIR / "10DEGREES_OVERNIGHT_RUN_ARIYANTURNER_RFS.csv"
# )

# FIFTEEN_RUN1_FILE = (
#     BASE_DIR / "15DEGREES_OVERNIGHT_ARIYANTURNER_RFS.csv"
# )

# FIFTEEN_RUN2_FILE = (
#     BASE_DIR / "15DEGREES_WEEKEND_RUN_ARIYANTURNER_RFS.csv"
# )


# THRESHOLD_V = 4.0


# # ------------------------------------------------------------
# # Incremental timing
# # ------------------------------------------------------------

# RUN1_LEADING_OFFSET_SECONDS = 63.0
# RUN2_LEADING_OFFSET_SECONDS = 0.0

# FIRST_NOMINAL_ANGLE_DEG = 0
# LAST_NOMINAL_ANGLE_DEG = 15

# MEASUREMENT_SECONDS_PER_ANGLE = 120.0
# ANGLE_CHANGE_SECONDS = 5.0

# ANGLE_CYCLE_SECONDS = (
#     MEASUREMENT_SECONDS_PER_ANGLE
#     + ANGLE_CHANGE_SECONDS
# )


# # ------------------------------------------------------------
# # Test these four zero-angle choices
# # ------------------------------------------------------------

# ZERO_ANGLE_CHOICES_DEG = [0, 1, 2, 3]


# # ------------------------------------------------------------
# # Fixed-angle plotting bins
# # ------------------------------------------------------------

# FIVE_DEGREE_TIME_BIN = "1min"
# TEN_DEGREE_TIME_BIN = "1h"
# FIFTEEN_DEGREE_TIME_BIN = "1h"


# # ------------------------------------------------------------
# # Plot/output settings
# # ------------------------------------------------------------

# SHOW_PLOTS = True

# OUTPUT_DIR = BASE_DIR / "rutherford_zero_angle_comparison"

# OUTPUT_DIR.mkdir(
#     parents=True,
#     exist_ok=True,
# )


# # ============================================================
# # 1. DATA LOADING
# # ============================================================

# def load_run(filename):
#     filename = Path(filename)

#     if not filename.exists():
#         raise FileNotFoundError(
#             f"Could not find:\n{filename}"
#         )

#     df = pd.read_csv(filename)

#     required = {
#         "timestamp",
#         "pulse_height_V",
#     }

#     missing = required - set(df.columns)

#     if missing:
#         raise ValueError(
#             f"{filename.name} is missing column(s): {sorted(missing)}\n"
#             f"Columns found: {list(df.columns)}"
#         )

#     df["timestamp"] = pd.to_datetime(
#         df["timestamp"],
#         errors="coerce",
#     )

#     df["pulse_height_V"] = pd.to_numeric(
#         df["pulse_height_V"],
#         errors="coerce",
#     )

#     df = (
#         df
#         .dropna(subset=["timestamp", "pulse_height_V"])
#         .sort_values("timestamp")
#         .reset_index(drop=True)
#     )

#     if len(df) == 0:
#         raise ValueError(
#             f"{filename.name} contains no valid events."
#         )

#     return df


# def apply_voltage_cut(df):
#     return df[
#         df["pulse_height_V"] >= THRESHOLD_V
#     ].copy()


# def finish_plot(filename):
#     plt.savefig(
#         OUTPUT_DIR / filename,
#         dpi=220,
#         bbox_inches="tight",
#     )

#     if SHOW_PLOTS:
#         plt.show()
#     else:
#         plt.close()


# # ============================================================
# # 2. FIXED-ANGLE RUNS
# # ============================================================

# FIXED_RUNS = [
#     {
#         "key": "5_run1",
#         "label": "5° run 1",
#         "angle": 5,
#         "filename": FIVE_RUN1_FILE,
#         "time_bin": FIVE_DEGREE_TIME_BIN,
#     },
#     {
#         "key": "5_run2",
#         "label": "5° run 2",
#         "angle": 5,
#         "filename": FIVE_RUN2_FILE,
#         "time_bin": FIVE_DEGREE_TIME_BIN,
#     },
#     {
#         "key": "10_run1",
#         "label": "10° run 1",
#         "angle": 10,
#         "filename": TEN_RUN1_FILE,
#         "time_bin": TEN_DEGREE_TIME_BIN,
#     },
#     {
#         "key": "10_run2",
#         "label": "10° run 2",
#         "angle": 10,
#         "filename": TEN_RUN2_FILE,
#         "time_bin": TEN_DEGREE_TIME_BIN,
#     },
#     {
#         "key": "15_run1",
#         "label": "15° overnight",
#         "angle": 15,
#         "filename": FIFTEEN_RUN1_FILE,
#         "time_bin": FIFTEEN_DEGREE_TIME_BIN,
#     },
#     {
#         "key": "15_run2",
#         "label": "15° weekend",
#         "angle": 15,
#         "filename": FIFTEEN_RUN2_FILE,
#         "time_bin": FIFTEEN_DEGREE_TIME_BIN,
#     },
# ]


# def time_bin_seconds(frequency):
#     return pd.to_timedelta(
#         frequency
#     ).total_seconds()


# def fixed_run_rate_vs_time(
#     full_run,
#     cut_run,
#     frequency,
# ):
#     """
#     Count rate vs elapsed time for one fixed-angle run.

#     - Zero-count bins are preserved.
#     - Final partial bin is kept.
#     - Final partial bin is divided by its actual duration.
#     - Output is always counts/hour.
#     """
#     start = full_run["timestamp"].min()
#     end = full_run["timestamp"].max()

#     total_duration_s = (
#         end - start
#     ).total_seconds()

#     if total_duration_s <= 0:
#         raise ValueError(
#             "Fixed run has non-positive duration."
#         )

#     bin_width_s = time_bin_seconds(
#         frequency
#     )

#     edges_s = np.arange(
#         0.0,
#         total_duration_s + bin_width_s,
#         bin_width_s,
#     )

#     if edges_s[-1] > total_duration_s:
#         edges_s[-1] = total_duration_s

#     if edges_s[-1] < total_duration_s:
#         edges_s = np.append(
#             edges_s,
#             total_duration_s,
#         )

#     edges_s = np.unique(
#         edges_s
#     )

#     if len(edges_s) < 2:
#         edges_s = np.array(
#             [0.0, total_duration_s]
#         )

#     accepted_elapsed_s = (
#         cut_run["timestamp"] - start
#     ).dt.total_seconds().to_numpy()

#     counts, _ = np.histogram(
#         accepted_elapsed_s,
#         bins=edges_s,
#     )

#     durations_s = np.diff(
#         edges_s
#     )

#     rate_counts_per_hour = (
#         counts
#         / (durations_s / 3600.0)
#     )

#     elapsed_hours = (
#         (edges_s[:-1] + edges_s[1:])
#         / 2.0
#         / 3600.0
#     )

#     return pd.DataFrame(
#         {
#             "elapsed_hours": elapsed_hours,
#             "bin_duration_seconds": durations_s,
#             "accepted_counts": counts,
#             "rate_counts_per_hour": rate_counts_per_hour,
#         }
#     )


# fixed_results = {}

# for config in FIXED_RUNS:

#     full = load_run(
#         config["filename"]
#     )

#     cut = apply_voltage_cut(
#         full
#     )

#     time_series = fixed_run_rate_vs_time(
#         full_run=full,
#         cut_run=cut,
#         frequency=config["time_bin"],
#     )

#     fixed_results[
#         config["key"]
#     ] = {
#         **config,
#         "full": full,
#         "cut": cut,
#         "time_series": time_series,
#     }

#     time_series.to_csv(
#         OUTPUT_DIR
#         / f"{config['key']}_count_rate_vs_time.csv",
#         index=False,
#     )


# # ============================================================
# # FIGURE 1
# # ALL FIXED-ANGLE RUNS — ONE SPACIOUS PLOT
# # ============================================================

# fig, ax = plt.subplots(
#     figsize=(13.5, 7.2)
# )

# for config in FIXED_RUNS:

#     ts = fixed_results[
#         config["key"]
#     ]["time_series"]

#     ax.plot(
#         ts["elapsed_hours"],
#         ts["rate_counts_per_hour"],
#         marker="o",
#         markersize=3,
#         linewidth=1.1,
#         alpha=0.78,
#         label=config["label"],
#     )


# ax.set_xlabel(
#     "Elapsed time from beginning of each trial (hours)",
#     fontsize=12,
# )

# ax.set_ylabel(
#     "Count rate after 4 V cut (counts/hour)",
#     fontsize=12,
# )

# ax.set_title(
#     "All fixed-angle trials: measured count rate over time",
#     fontsize=15,
#     pad=14,
# )

# ax.grid(
#     alpha=0.25
# )

# # Legend outside the data area.
# ax.legend(
#     loc="center left",
#     bbox_to_anchor=(1.02, 0.5),
#     frameon=True,
# )

# fig.subplots_adjust(
#     left=0.09,
#     right=0.79,
#     bottom=0.12,
#     top=0.90,
# )

# finish_plot(
#     "01_all_fixed_angle_runs_count_rate_vs_time.png"
# )


# # ============================================================
# # 3. INCREMENTAL RUNS
# # ============================================================

# INCREMENTAL_RUNS = [
#     {
#         "key": "incremental_run1",
#         "label": "Incremental run 1",
#         "filename": PER_DEGREE_RUN1_FILE,
#         "leading_offset_s": RUN1_LEADING_OFFSET_SECONDS,
#     },
#     {
#         "key": "incremental_run2",
#         "label": "Incremental run 2",
#         "filename": PER_DEGREE_RUN2_FILE,
#         "leading_offset_s": RUN2_LEADING_OFFSET_SECONDS,
#     },
# ]


# def analyze_incremental_run(
#     filename,
#     label,
#     leading_offset_s,
# ):
#     """
#     Experimental timing:

#         120 s data at nominal angle
#         5 s movement
#         120 s data at next nominal angle
#         5 s movement
#         ...

#     Nominal angles run from 0° through 15°.

#     Run 1:
#         first timestamp + 63 s = start of nominal 0° data.

#     Run 2:
#         first timestamp = start of nominal 0° data.
#     """
#     df = load_run(
#         filename
#     )

#     raw_start = df[
#         "timestamp"
#     ].min()

#     nominal_zero_start = (
#         raw_start
#         + pd.to_timedelta(
#             leading_offset_s,
#             unit="s",
#         )
#     )

#     work = df.copy()

#     work[
#         "elapsed_from_nominal_zero_s"
#     ] = (
#         work["timestamp"]
#         - nominal_zero_start
#     ).dt.total_seconds()

#     # Remove anything before nominal 0° actually begins.
#     work = work[
#         work["elapsed_from_nominal_zero_s"] >= 0
#     ].copy()

#     work[
#         "cycle_index"
#     ] = np.floor(
#         work["elapsed_from_nominal_zero_s"]
#         / ANGLE_CYCLE_SECONDS
#     ).astype(int)

#     work[
#         "time_inside_cycle_s"
#     ] = (
#         work["elapsed_from_nominal_zero_s"]
#         - work["cycle_index"]
#         * ANGLE_CYCLE_SECONDS
#     )

#     valid_nominal_angle = (
#         (
#             work["cycle_index"]
#             >= FIRST_NOMINAL_ANGLE_DEG
#         )
#         & (
#             work["cycle_index"]
#             <= LAST_NOMINAL_ANGLE_DEG
#         )
#     )

#     in_measurement_window = (
#         work["time_inside_cycle_s"]
#         < MEASUREMENT_SECONDS_PER_ANGLE
#     )

#     passes_voltage_cut = (
#         work["pulse_height_V"]
#         >= THRESHOLD_V
#     )

#     work["accepted"] = (
#         valid_nominal_angle
#         & in_measurement_window
#         & passes_voltage_cut
#     )

#     accepted = work[
#         work["accepted"]
#     ].copy()

#     accepted[
#         "nominal_angle_deg"
#     ] = accepted[
#         "cycle_index"
#     ].astype(int)

#     rows = []

#     for nominal_angle in range(
#         FIRST_NOMINAL_ANGLE_DEG,
#         LAST_NOMINAL_ANGLE_DEG + 1,
#     ):

#         events = accepted[
#             accepted["nominal_angle_deg"]
#             == nominal_angle
#         ]

#         count = len(
#             events
#         )

#         rate_counts_per_hour = (
#             count
#             / (
#                 MEASUREMENT_SECONDS_PER_ANGLE
#                 / 3600.0
#             )
#         )

#         rows.append(
#             {
#                 "run": label,
#                 "nominal_angle_deg": nominal_angle,
#                 "accepted_counts": count,
#                 "rate_counts_per_hour": rate_counts_per_hour,
#             }
#         )

#     summary = pd.DataFrame(
#         rows
#     )

#     return work, accepted, summary


# incremental_results = {}

# for config in INCREMENTAL_RUNS:

#     assigned, accepted, summary = analyze_incremental_run(
#         filename=config["filename"],
#         label=config["label"],
#         leading_offset_s=config["leading_offset_s"],
#     )

#     incremental_results[
#         config["key"]
#     ] = {
#         **config,
#         "assigned": assigned,
#         "accepted": accepted,
#         "summary": summary,
#     }

#     summary.to_csv(
#         OUTPUT_DIR
#         / f"{config['key']}_nominal_count_rate_by_angle.csv",
#         index=False,
#     )


# # ============================================================
# # 4. NOMINAL ANGLE TABLE
# # ============================================================

# run1_summary = (
#     incremental_results[
#         "incremental_run1"
#     ]["summary"]
#     .set_index("nominal_angle_deg")
# )

# run2_summary = (
#     incremental_results[
#         "incremental_run2"
#     ]["summary"]
#     .set_index("nominal_angle_deg")
# )

# nominal_angles = np.arange(
#     FIRST_NOMINAL_ANGLE_DEG,
#     LAST_NOMINAL_ANGLE_DEG + 1,
#     dtype=float,
# )

# incremental_table = pd.DataFrame(
#     {
#         "nominal_angle_deg": nominal_angles,
#     }
# )

# incremental_table[
#     "run1_rate_counts_per_hour"
# ] = (
#     incremental_table[
#         "nominal_angle_deg"
#     ].map(
#         run1_summary[
#             "rate_counts_per_hour"
#         ]
#     )
# )

# incremental_table[
#     "run2_rate_counts_per_hour"
# ] = (
#     incremental_table[
#         "nominal_angle_deg"
#     ].map(
#         run2_summary[
#             "rate_counts_per_hour"
#         ]
#     )
# )

# incremental_table[
#     "combined_average_rate_counts_per_hour"
# ] = (
#     incremental_table[
#         [
#             "run1_rate_counts_per_hour",
#             "run2_rate_counts_per_hour",
#         ]
#     ].mean(
#         axis=1
#     )
# )

# incremental_table.to_csv(
#     OUTPUT_DIR
#     / "incremental_nominal_count_rates.csv",
#     index=False,
# )


# # ============================================================
# # 5. ZERO-ANGLE CALIBRATION TABLES
# # ============================================================

# def make_calibrated_table(
#     nominal_table,
#     zero_angle_deg,
# ):
#     """
#     Shift the angular coordinate.

#         calibrated = nominal - zero_angle
#     """
#     calibrated = nominal_table.copy()

#     calibrated[
#         "zero_angle_choice_deg"
#     ] = zero_angle_deg

#     calibrated[
#         "calibrated_angle_deg"
#     ] = (
#         calibrated[
#             "nominal_angle_deg"
#         ]
#         - zero_angle_deg
#     )

#     return calibrated


# calibrated_tables = {}

# for zero_angle in ZERO_ANGLE_CHOICES_DEG:

#     table = make_calibrated_table(
#         incremental_table,
#         zero_angle,
#     )

#     calibrated_tables[
#         zero_angle
#     ] = table

#     table.to_csv(
#         OUTPUT_DIR
#         / f"incremental_zero_set_to_{zero_angle}deg.csv",
#         index=False,
#     )


# # ============================================================
# # FIGURE 2
# # COUNT RATE VS CALIBRATED SCATTERING ANGLE
# # FOUR ZERO-ANGLE CHOICES
# # ============================================================

# fig, axes = plt.subplots(
#     2,
#     2,
#     figsize=(14.0, 10.5),
# )

# axes = axes.flatten()

# for ax, zero_angle in zip(
#     axes,
#     ZERO_ANGLE_CHOICES_DEG,
# ):

#     table = calibrated_tables[
#         zero_angle
#     ]

#     ax.plot(
#         table["calibrated_angle_deg"],
#         table["run1_rate_counts_per_hour"],
#         marker="o",
#         markersize=4,
#         linewidth=1.2,
#         label="Incremental run 1",
#     )

#     ax.plot(
#         table["calibrated_angle_deg"],
#         table["run2_rate_counts_per_hour"],
#         marker="s",
#         markersize=4,
#         linewidth=1.2,
#         label="Incremental run 2",
#     )

#     ax.plot(
#         table["calibrated_angle_deg"],
#         table[
#             "combined_average_rate_counts_per_hour"
#         ],
#         marker="o",
#         markersize=6,
#         linewidth=2.0,
#         label="Combined average",
#     )

#     ax.axvline(
#         0,
#         linestyle="--",
#         linewidth=1.0,
#         alpha=0.7,
#     )

#     ax.set_title(
#         f"Nominal {zero_angle}° treated as true 0°",
#         fontsize=13,
#         pad=10,
#     )

#     ax.set_xlabel(
#         "Calibrated scattering angle (degrees)"
#     )

#     ax.set_ylabel(
#         "Count rate (counts/hour)"
#     )

#     ax.grid(
#         alpha=0.25
#     )

#     ax.legend(
#         fontsize=9
#     )


# fig.suptitle(
#     "Incremental scans under four possible zero-angle calibrations",
#     fontsize=16,
#     y=0.98,
# )

# fig.subplots_adjust(
#     left=0.08,
#     right=0.97,
#     bottom=0.08,
#     top=0.91,
#     wspace=0.26,
#     hspace=0.34,
# )

# finish_plot(
#     "02_zero_angle_calibration_count_rate_comparison.png"
# )


# # ============================================================
# # 6. RADIAL PROFILE CONSTRUCTION
# # ============================================================
# #
# # For a circular map we need count rate as a function of
# # scattering-angle MAGNITUDE:
# #
# #       r = |calibrated angle|
# #
# # If the zero-angle choice creates both -theta and +theta
# # measurements, they represent the same radial scattering angle
# # in the circular visualization.
# #
# # So we average them.
# #
# # Example, zero = 2°:
# #
# #       nominal 1° -> calibrated -1°
# #       nominal 3° -> calibrated +1°
# #
# # Both contribute to radial angle = 1°.
# # ============================================================

# def make_radial_profile(
#     calibrated_table,
#     rate_column,
# ):
#     radial = pd.DataFrame(
#         {
#             "radial_angle_deg": np.abs(
#                 calibrated_table[
#                     "calibrated_angle_deg"
#                 ].to_numpy(
#                     dtype=float
#                 )
#             ),
#             "rate_counts_per_hour": calibrated_table[
#                 rate_column
#             ].to_numpy(
#                 dtype=float
#             ),
#         }
#     )

#     radial = (
#         radial
#         .groupby(
#             "radial_angle_deg",
#             as_index=False,
#         )
#         .agg(
#             rate_counts_per_hour=(
#                 "rate_counts_per_hour",
#                 "mean",
#             )
#         )
#         .sort_values(
#             "radial_angle_deg"
#         )
#         .reset_index(
#             drop=True
#         )
#     )

#     return radial


# def make_piecewise_linear_rate_function(
#     radial_profile,
# ):
#     """
#     Data-only function:
#     linearly interpolate between measured radial-angle values.
#     """
#     x = radial_profile[
#         "radial_angle_deg"
#     ].to_numpy(
#         dtype=float
#     )

#     y = radial_profile[
#         "rate_counts_per_hour"
#     ].to_numpy(
#         dtype=float
#     )

#     def rate_function(theta_magnitude_deg):
#         return np.interp(
#             theta_magnitude_deg,
#             x,
#             y,
#         )

#     return rate_function


# # ============================================================
# # 7. CIRCULAR FIELD
# # ============================================================

# MAP_EXTENT_DEG = 15.0
# GRID_SIZE = 750


# def make_circular_field(
#     radial_profile,
# ):
#     """
#     Produce a circle in which physical radius = scattering-angle
#     magnitude in degrees.

#     We always display coordinates out to +/-15°.

#     If a shifted zero choice does not provide measurements all the
#     way to 15° radial angle, the unmeasured outer annulus stays blank.
#     No extrapolation is performed.
#     """
#     rate_function = (
#         make_piecewise_linear_rate_function(
#             radial_profile
#         )
#     )

#     axis = np.linspace(
#         -MAP_EXTENT_DEG,
#         MAP_EXTENT_DEG,
#         GRID_SIZE,
#     )

#     X, Y = np.meshgrid(
#         axis,
#         axis,
#     )

#     radial_angle = np.sqrt(
#         X**2 + Y**2
#     )

#     max_measured_radius = radial_profile[
#         "radial_angle_deg"
#     ].max()

#     values = np.full(
#         radial_angle.shape,
#         np.nan,
#         dtype=float,
#     )

#     measured_region = (
#         radial_angle
#         <= max_measured_radius
#     )

#     values[
#         measured_region
#     ] = rate_function(
#         radial_angle[
#             measured_region
#         ]
#     )

#     return (
#         X,
#         Y,
#         values,
#         max_measured_radius,
#     )


# # ============================================================
# # 8. SHARED COLOR SCALE
# # ============================================================
# #
# # One scale for ALL maps so colors mean the same count rate
# # everywhere.

# all_original_rates = np.concatenate(
#     [
#         incremental_table[
#             "run1_rate_counts_per_hour"
#         ].to_numpy(
#             dtype=float
#         ),
#         incremental_table[
#             "run2_rate_counts_per_hour"
#         ].to_numpy(
#             dtype=float
#         ),
#         incremental_table[
#             "combined_average_rate_counts_per_hour"
#         ].to_numpy(
#             dtype=float
#         ),
#     ]
# )

# GLOBAL_VMIN = np.nanmin(
#     all_original_rates
# )

# GLOBAL_VMAX = np.nanmax(
#     all_original_rates
# )


# # ============================================================
# # FIGURES 3–6
# # ONE LARGE GRADIENT-MAP FIGURE PER ZERO CHOICE
# # ============================================================

# rate_columns = [
#     (
#         "run1_rate_counts_per_hour",
#         "Incremental run 1",
#     ),
#     (
#         "run2_rate_counts_per_hour",
#         "Incremental run 2",
#     ),
#     (
#         "combined_average_rate_counts_per_hour",
#         "Combined average",
#     ),
# ]


# for zero_angle in ZERO_ANGLE_CHOICES_DEG:

#     table = calibrated_tables[
#         zero_angle
#     ]

#     fig, axes = plt.subplots(
#         1,
#         3,
#         figsize=(19.5, 6.8),
#     )

#     last_image = None

#     for ax, (
#         rate_column,
#         title,
#     ) in zip(
#         axes,
#         rate_columns,
#     ):

#         radial_profile = make_radial_profile(
#             calibrated_table=table,
#             rate_column=rate_column,
#         )

#         (
#             X,
#             Y,
#             field,
#             max_measured_radius,
#         ) = make_circular_field(
#             radial_profile
#         )

#         last_image = ax.imshow(
#             field,
#             extent=[
#                 -MAP_EXTENT_DEG,
#                 MAP_EXTENT_DEG,
#                 -MAP_EXTENT_DEG,
#                 MAP_EXTENT_DEG,
#             ],
#             origin="lower",
#             interpolation="bilinear",
#             vmin=GLOBAL_VMIN,
#             vmax=GLOBAL_VMAX,
#         )

#         # Radial guide circles
#         for guide_radius in [
#             5,
#             10,
#             15,
#         ]:
#             circle = plt.Circle(
#                 (0, 0),
#                 guide_radius,
#                 fill=False,
#                 linewidth=0.8,
#                 alpha=0.40,
#             )

#             ax.add_patch(
#                 circle
#             )

#             ax.text(
#                 0.55,
#                 guide_radius - 0.8,
#                 f"{guide_radius}°",
#                 fontsize=9,
#                 alpha=0.72,
#             )

#         # Mark the center / calibrated 0°
#         ax.scatter(
#             [0],
#             [0],
#             marker="+",
#             s=45,
#             linewidths=1.4,
#         )

#         ax.set_title(
#             title,
#             fontsize=14,
#             pad=12,
#         )

#         ax.set_xlabel(
#             "Scattering-angle coordinate (degrees)",
#             fontsize=10,
#         )

#         ax.set_ylabel(
#             "Scattering-angle coordinate (degrees)",
#             fontsize=10,
#         )

#         ax.set_xlim(
#             -MAP_EXTENT_DEG,
#             MAP_EXTENT_DEG,
#         )

#         ax.set_ylim(
#             -MAP_EXTENT_DEG,
#             MAP_EXTENT_DEG,
#         )

#         ax.set_aspect(
#             "equal"
#         )

#         ax.grid(
#             alpha=0.10
#         )

#         # Explicitly state measured radial coverage.
#         ax.text(
#             0.5,
#             -0.15,
#             (
#                 f"Measured radial coverage: "
#                 f"0° to {max_measured_radius:.0f}°"
#             ),
#             transform=ax.transAxes,
#             ha="center",
#             fontsize=9,
#         )


#     fig.suptitle(
#         (
#             f"Circular count-rate maps — nominal {zero_angle}° "
#             f"treated as true 0°\n"
#             f"±θ measurements are averaged at equal radial angle"
#         ),
#         fontsize=16,
#         y=0.96,
#     )

#     # Leave a dedicated right-side strip for the colorbar.
#     fig.subplots_adjust(
#         left=0.045,
#         right=0.89,
#         bottom=0.18,
#         top=0.82,
#         wspace=0.32,
#     )

#     colorbar_ax = fig.add_axes(
#         [0.915, 0.21, 0.018, 0.56]
#     )

#     colorbar = fig.colorbar(
#         last_image,
#         cax=colorbar_ax,
#     )

#     colorbar.set_label(
#         "Count rate after 4 V cut (counts/hour)",
#         fontsize=11,
#     )

#     finish_plot(
#         f"03_gradient_maps_zero_set_to_{zero_angle}deg.png"
#     )


# # ============================================================
# # 9. SAVE RADIAL PROFILES
# # ============================================================

# for zero_angle in ZERO_ANGLE_CHOICES_DEG:

#     table = calibrated_tables[
#         zero_angle
#     ]

#     for rate_column, title in rate_columns:

#         profile = make_radial_profile(
#             calibrated_table=table,
#             rate_column=rate_column,
#         )

#         safe_name = (
#             title
#             .lower()
#             .replace(" ", "_")
#         )

#         profile.to_csv(
#             OUTPUT_DIR
#             / (
#                 f"radial_profile_zero_{zero_angle}deg_"
#                 f"{safe_name}.csv"
#             ),
#             index=False,
#         )


# # ============================================================
# # 10. FINAL SUMMARY
# # ============================================================

# print(
#     "\n============================================================"
# )

# print(
#     "DONE"
# )

# print(
#     "============================================================"
# )

# print(
#     f"Outputs saved to:\n{OUTPUT_DIR}"
# )

# print(
#     "\nZero-angle choices tested:"
# )

# for zero_angle in ZERO_ANGLE_CHOICES_DEG:
#     print(
#         f"  nominal {zero_angle}° -> true 0°"
#     )

# print(
#     "\nMain figures:"
# )

# print(
#     "  01_all_fixed_angle_runs_count_rate_vs_time.png"
# )

# print(
#     "  02_zero_angle_calibration_count_rate_comparison.png"
# )

# for zero_angle in ZERO_ANGLE_CHOICES_DEG:
#     print(
#         f"  03_gradient_maps_zero_set_to_{zero_angle}deg.png"
#     )

# print(
#     "\nGradient-map rule:"
# )

# print(
#     "  calibrated angle = nominal angle - chosen zero angle"
# )

# print(
#     "  radial angle = |calibrated angle|"
# )

# print(
#     "  +theta and -theta measurements are averaged"
# )

# print(
#     "  interpolation is piecewise-linear between measured radial points"
# )

# print(
#     "  no theoretical function or extrapolation is used"
# )




# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ============================================================
# # Rutherford scattering analysis
# #
# # Includes:
# #   1) two incremental "per-degree" runs,
# #   2) two 5-degree runs,
# #   3) two 10-degree runs,
# #   4) two 15-degree runs,
# #   5) 4 V pulse-height analysis cut,
# #   6) Poisson count-rate uncertainties,
# #   7) Rutherford R(theta) ~ csc^4(theta/2) fit,
# #   8) fixed-angle validation,
# #   9) linearized Rutherford plot,
# #  10) time-stability plots,
# #  11) CSV output tables.
# #
# # IMPORTANT timing interpretation for the incremental runs:
# # - Run 1: discard the first 1 minute 3 seconds (63 s), which
# #   corresponds to the initial 0-degree period.
# # - After that, angle starts at 1 degree and increases by
# #   1 degree every 2 minutes.
# # - By default, the first 5 seconds of EVERY MINUTE are removed,
# #   matching the instruction literally.
# #
# # If "first 5 seconds" was intended ONLY after each angle change
# # (once every 2 minutes), set CUT_FIRST_5S_EVERY_MINUTE = False.
# # ============================================================


# # ============================================================
# # 0. USER SETTINGS / FILES
# # ============================================================

# BASE_DIR = Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS")

# PER_DEGREE_RUN1_FILE = BASE_DIR / "PER_DEGREE_RUN_ONE_30MINUTES_RUN_ARIYANTURNER_RFS.csv"
# PER_DEGREE_RUN2_FILE = BASE_DIR / "PER_DEGREE_RUN_ONE_HOUR_RUN_ARIYANTURNER_RFS_RUN2.csv"

# FIVE_RUN1_FILE = BASE_DIR / "5DEGREE_RUN_ONE_HOUR_ARIYANTURNER_RFS.csv"
# FIVE_RUN2_FILE = BASE_DIR / "5DEGREES_HOUR_RUN_ARIYANTURNER_RFS.csv"

# TEN_RUN1_FILE = BASE_DIR / "10DEGREE_OVERNIGHT_RUN_ARIYAN_TURNER_RFS_RUN2.csv"
# TEN_RUN2_FILE = BASE_DIR / "10DEGREES_OVERNIGHT_RUN_ARIYANTURNER_RFS.csv"

# FIFTEEN_RUN1_FILE = BASE_DIR / "15DEGREES_OVERNIGHT_ARIYANTURNER_RFS.csv"
# FIFTEEN_RUN2_FILE = BASE_DIR / "15DEGREES_WEEKEND_RUN_ARIYANTURNER_RFS.csv"


# THRESHOLD = 4.0  # volts

# # Incremental-run timing
# RUN1_ZERO_DEGREE_OFFSET_SECONDS = 63.0
# RUN2_ZERO_DEGREE_OFFSET_SECONDS = 0.0

# ANGLE_START_DEG = 1.0
# ANGLE_STEP_DEG = 1.0
# ANGLE_HOLD_SECONDS = 120.0

# ADJUSTMENT_CUT_SECONDS = 5.0
# CUT_FIRST_5S_EVERY_MINUTE = True

# # Because timestamps are pulse times rather than an independent
# # "run ended at exactly ..." clock, allow a nearly-complete last
# # 2-minute interval to count as complete.
# FULL_INTERVAL_TOLERANCE_SECONDS = 10.0

# # The manual recommends fitting a stable intermediate angular region
# # rather than blindly using the smallest and largest angles.
# # Inspect the plots and change these if your data suggest a better window.
# FIT_MIN_ANGLE_DEG = 5.0
# FIT_MAX_ANGLE_DEG = 15.0

# SHOW_PLOTS = True
# OUTPUT_DIR = BASE_DIR / "rutherford_analysis_outputs"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# # ============================================================
# # 1. BASIC HELPERS
# # ============================================================

# def require_columns(df, filename):
#     required = {"timestamp", "pulse_height_V"}
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(
#             f"{filename} is missing required column(s): {sorted(missing)}. "
#             f"Found columns: {list(df.columns)}"
#         )


# def load_run(filename):
#     filename = Path(filename)

#     if not filename.exists():
#         raise FileNotFoundError(f"Could not find file:\n{filename}")

#     df = pd.read_csv(filename)
#     require_columns(df, filename)

#     df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
#     df["pulse_height_V"] = pd.to_numeric(df["pulse_height_V"], errors="coerce")

#     df = (
#         df.dropna(subset=["timestamp", "pulse_height_V"])
#         .sort_values("timestamp")
#         .reset_index(drop=True)
#     )

#     if len(df) < 2:
#         raise ValueError(f"{filename} does not contain enough valid rows.")

#     return df


# def apply_voltage_cut(df):
#     return df[df["pulse_height_V"] >= THRESHOLD].copy()


# def rutherford_factor(theta_deg):
#     """
#     x(theta) = csc^4(theta/2) = sin(theta/2)^(-4)
#     """
#     theta_deg = np.asarray(theta_deg, dtype=float)

#     if np.any(theta_deg <= 0):
#         raise ValueError("Rutherford factor is singular at theta <= 0 degrees.")

#     theta_rad = np.deg2rad(theta_deg)
#     return np.sin(theta_rad / 2.0) ** (-4)


# def finish_plot(filename):
#     plt.tight_layout()
#     plt.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
#     if SHOW_PLOTS:
#         plt.show()
#     else:
#         plt.close()


# # ============================================================
# # 2. FIXED-ANGLE RUNS
# #
# # Keep EVERY run separate.
# # Do not silently combine the two 5°, two 10°, or two 15° runs.
# # ============================================================

# FIXED_RUN_CONFIG = [
#     {
#         "key": "5_run1",
#         "label": "5° run 1",
#         "angle_deg": 5.0,
#         "filename": FIVE_RUN1_FILE,
#         "stability_frequency": "1min",
#         "stability_ylabel": "Counts per minute",
#         "marker": "o",
#     },
#     {
#         "key": "5_run2",
#         "label": "5° run 2",
#         "angle_deg": 5.0,
#         "filename": FIVE_RUN2_FILE,
#         "stability_frequency": "1min",
#         "stability_ylabel": "Counts per minute",
#         "marker": "s",
#     },
#     {
#         "key": "10_run1",
#         "label": "10° run 1",
#         "angle_deg": 10.0,
#         "filename": TEN_RUN1_FILE,
#         "stability_frequency": "1h",
#         "stability_ylabel": "Counts per hour",
#         "marker": "o",
#     },
#     {
#         "key": "10_run2",
#         "label": "10° run 2",
#         "angle_deg": 10.0,
#         "filename": TEN_RUN2_FILE,
#         "stability_frequency": "1h",
#         "stability_ylabel": "Counts per hour",
#         "marker": "s",
#     },
#     {
#         "key": "15_run1",
#         "label": "15° overnight",
#         "angle_deg": 15.0,
#         "filename": FIFTEEN_RUN1_FILE,
#         "stability_frequency": "1h",
#         "stability_ylabel": "Counts per hour",
#         "marker": "o",
#     },
#     {
#         "key": "15_run2",
#         "label": "15° weekend",
#         "angle_deg": 15.0,
#         "filename": FIFTEEN_RUN2_FILE,
#         "stability_frequency": "1h",
#         "stability_ylabel": "Counts per hour",
#         "marker": "s",
#     },
# ]


# def rate_and_error(full_run, accepted_run):
#     """
#     R = N/T
#     sigma_R = sqrt(N)/T

#     T is inferred from the first and last recorded pulse timestamps.
#     """
#     duration_s = (
#         full_run["timestamp"].max() - full_run["timestamp"].min()
#     ).total_seconds()

#     if duration_s <= 0:
#         raise ValueError("Run duration is not positive.")

#     duration_h = duration_s / 3600.0
#     N = len(accepted_run)

#     rate_h = N / duration_h
#     sigma_rate_h = np.sqrt(N) / duration_h

#     return N, duration_h, rate_h, sigma_rate_h


# fixed_runs = {}
# fixed_summary_rows = []

# for cfg in FIXED_RUN_CONFIG:
#     full = load_run(cfg["filename"])
#     cut = apply_voltage_cut(full)

#     N, T_h, R_h, dR_h = rate_and_error(full, cut)

#     fixed_runs[cfg["key"]] = {
#         **cfg,
#         "full": full,
#         "cut": cut,
#         "N": N,
#         "T_h": T_h,
#         "R_h": R_h,
#         "dR_h": dR_h,
#     }

#     fixed_summary_rows.append(
#         {
#             "run": cfg["label"],
#             "angle_deg": cfg["angle_deg"],
#             "duration_h": T_h,
#             "accepted_counts": N,
#             "rate_per_h": R_h,
#             "sigma_rate_per_h": dR_h,
#         }
#     )

# fixed_summary = pd.DataFrame(fixed_summary_rows)

# print("\n============================================================")
# print("FIXED-ANGLE RUNS AFTER 4 V CUT")
# print("R = N/T,   sigma_R = sqrt(N)/T")
# print("============================================================")
# print(
#     fixed_summary.to_string(
#         index=False,
#         formatters={
#             "duration_h": "{:.6f}".format,
#             "rate_per_h": "{:.3f}".format,
#             "sigma_rate_per_h": "{:.3f}".format,
#         },
#     )
# )

# fixed_summary.to_csv(
#     OUTPUT_DIR / "fixed_angle_rate_summary.csv",
#     index=False,
# )


# # ============================================================
# # 3. INCREMENTAL / PER-DEGREE RUN ANALYSIS
# # ============================================================

# PER_DEGREE_CONFIG = [
#     {
#         "key": "incremental_run1",
#         "label": "Incremental run 1",
#         "filename": PER_DEGREE_RUN1_FILE,
#         "offset_seconds": RUN1_ZERO_DEGREE_OFFSET_SECONDS,
#         "marker": "o",
#     },
#     {
#         "key": "incremental_run2",
#         "label": "Incremental run 2",
#         "filename": PER_DEGREE_RUN2_FILE,
#         "offset_seconds": RUN2_ZERO_DEGREE_OFFSET_SECONDS,
#         "marker": "s",
#     },
# ]


# def stable_time_mask(elapsed_seconds):
#     """
#     Return True for times that are allowed into the incremental analysis.

#     Default interpretation:
#       remove [0, 5) s from every 60-second minute.

#     Alternative interpretation:
#       if CUT_FIRST_5S_EVERY_MINUTE = False,
#       remove only [0, 5) s after each 2-minute angle change.
#     """
#     elapsed_seconds = np.asarray(elapsed_seconds, dtype=float)

#     if CUT_FIRST_5S_EVERY_MINUTE:
#         second_within_minute = np.mod(elapsed_seconds, 60.0)
#         return second_within_minute >= ADJUSTMENT_CUT_SECONDS

#     second_within_angle = np.mod(elapsed_seconds, ANGLE_HOLD_SECONDS)
#     return second_within_angle >= ADJUSTMENT_CUT_SECONDS


# def live_seconds_per_full_angle_bin():
#     if CUT_FIRST_5S_EVERY_MINUTE:
#         n_minutes = ANGLE_HOLD_SECONDS / 60.0

#         if not np.isclose(n_minutes, round(n_minutes)):
#             raise ValueError(
#                 "ANGLE_HOLD_SECONDS must be an integer number of minutes "
#                 "when CUT_FIRST_5S_EVERY_MINUTE=True."
#             )

#         return ANGLE_HOLD_SECONDS - (
#             round(n_minutes) * ADJUSTMENT_CUT_SECONDS
#         )

#     return ANGLE_HOLD_SECONDS - ADJUSTMENT_CUT_SECONDS


# def analyze_incremental_run(filename, label, offset_seconds):
#     """
#     Convert timestamps into 2-minute angle bins.

#     For run 1:
#       analysis_start = first timestamp + 63 s

#     For run 2:
#       analysis_start = first timestamp

#     First post-offset angle = 1 degree.
#     Then angle increases by 1 degree every 120 s.

#     Only complete 2-minute angle intervals are retained.
#     """
#     df = load_run(filename)

#     raw_start = df["timestamp"].min()
#     raw_end = df["timestamp"].max()

#     analysis_start = raw_start + pd.to_timedelta(offset_seconds, unit="s")

#     usable_duration_s = (raw_end - analysis_start).total_seconds()

#     if usable_duration_s <= 0:
#         raise ValueError(
#             f"{label}: zero-angle offset is longer than the recorded run."
#         )

#     # Count near-complete final intervals as complete if the last pulse
#     # falls within the configured tolerance of the nominal boundary.
#     n_full_bins = int(
#         np.floor(
#             (usable_duration_s + FULL_INTERVAL_TOLERANCE_SECONDS)
#             / ANGLE_HOLD_SECONDS
#         )
#     )

#     if n_full_bins < 1:
#         raise ValueError(
#             f"{label}: no complete {ANGLE_HOLD_SECONDS:.0f}-second angle bins."
#         )

#     analysis_end = (
#         analysis_start
#         + pd.to_timedelta(n_full_bins * ANGLE_HOLD_SECONDS, unit="s")
#     )

#     work = df[
#         (df["timestamp"] >= analysis_start)
#         & (df["timestamp"] < analysis_end)
#     ].copy()

#     work["elapsed_s"] = (
#         work["timestamp"] - analysis_start
#     ).dt.total_seconds()

#     work["angle_index"] = np.floor(
#         work["elapsed_s"] / ANGLE_HOLD_SECONDS
#     ).astype(int)

#     work["angle_deg"] = (
#         ANGLE_START_DEG + ANGLE_STEP_DEG * work["angle_index"]
#     )

#     work["stable_time"] = stable_time_mask(work["elapsed_s"])
#     work["passes_voltage_cut"] = work["pulse_height_V"] >= THRESHOLD
#     work["accepted"] = work["stable_time"] & work["passes_voltage_cut"]

#     stable = work[work["stable_time"]].copy()
#     accepted = work[work["accepted"]].copy()

#     live_s = live_seconds_per_full_angle_bin()

#     rows = []

#     for i in range(n_full_bins):
#         angle = ANGLE_START_DEG + i * ANGLE_STEP_DEG

#         interval_start = (
#             analysis_start
#             + pd.to_timedelta(i * ANGLE_HOLD_SECONDS, unit="s")
#         )
#         interval_end = (
#             interval_start
#             + pd.to_timedelta(ANGLE_HOLD_SECONDS, unit="s")
#         )

#         stable_i = stable[stable["angle_index"] == i]
#         accepted_i = accepted[accepted["angle_index"] == i]

#         raw_stable_count = len(stable_i)
#         N = len(accepted_i)

#         live_h = live_s / 3600.0
#         R_h = N / live_h
#         dR_h = np.sqrt(N) / live_h

#         rows.append(
#             {
#                 "run": label,
#                 "angle_deg": angle,
#                 "interval_start": interval_start,
#                 "interval_end": interval_end,
#                 "scheduled_interval_s": ANGLE_HOLD_SECONDS,
#                 "live_time_s": live_s,
#                 "stable_raw_counts": raw_stable_count,
#                 "accepted_counts": N,
#                 "rate_per_h": R_h,
#                 "sigma_rate_per_h": dR_h,
#             }
#         )

#     summary = pd.DataFrame(rows)

#     metadata = {
#         "label": label,
#         "filename": str(filename),
#         "raw_start": raw_start,
#         "raw_end": raw_end,
#         "analysis_start": analysis_start,
#         "analysis_end": analysis_end,
#         "offset_seconds": offset_seconds,
#         "n_full_bins": n_full_bins,
#         "live_s_per_bin": live_s,
#     }

#     return df, work, accepted, summary, metadata


# incremental_runs = {}

# for cfg in PER_DEGREE_CONFIG:
#     full, assigned, accepted, summary, metadata = analyze_incremental_run(
#         filename=cfg["filename"],
#         label=cfg["label"],
#         offset_seconds=cfg["offset_seconds"],
#     )

#     incremental_runs[cfg["key"]] = {
#         **cfg,
#         "full": full,
#         "assigned": assigned,
#         "accepted": accepted,
#         "summary": summary,
#         "metadata": metadata,
#     }

#     summary.to_csv(
#         OUTPUT_DIR / f"{cfg['key']}_angle_summary.csv",
#         index=False,
#     )

#     print("\n============================================================")
#     print(cfg["label"].upper())
#     print("============================================================")
#     print(f"file:            {cfg['filename']}")
#     print(f"raw start:       {metadata['raw_start']}")
#     print(f"analysis start:  {metadata['analysis_start']}")
#     print(f"analysis end:    {metadata['analysis_end']}")
#     print(f"offset removed:  {metadata['offset_seconds']:.1f} s")
#     print(f"full angle bins: {metadata['n_full_bins']}")
#     print(f"live time/bin:   {metadata['live_s_per_bin']:.1f} s")
#     print()
#     print(
#         summary[
#             [
#                 "angle_deg",
#                 "accepted_counts",
#                 "live_time_s",
#                 "rate_per_h",
#                 "sigma_rate_per_h",
#             ]
#         ].to_string(
#             index=False,
#             formatters={
#                 "angle_deg": "{:.0f}".format,
#                 "live_time_s": "{:.1f}".format,
#                 "rate_per_h": "{:.2f}".format,
#                 "sigma_rate_per_h": "{:.2f}".format,
#             },
#         )
#     )


# # ============================================================
# # 4. COMBINE THE TWO INCREMENTAL RUNS ANGLE-BY-ANGLE
# #
# # This is the correct way to combine repeated counting runs:
# #     N_total = sum(N_i)
# #     T_total = sum(T_i)
# #     R = N_total / T_total
# #     sigma_R = sqrt(N_total) / T_total
# #
# # The individual runs are still kept and plotted separately.
# # ============================================================

# all_incremental = pd.concat(
#     [
#         incremental_runs["incremental_run1"]["summary"],
#         incremental_runs["incremental_run2"]["summary"],
#     ],
#     ignore_index=True,
# )

# combined_incremental = (
#     all_incremental.groupby("angle_deg", as_index=False)
#     .agg(
#         accepted_counts=("accepted_counts", "sum"),
#         stable_raw_counts=("stable_raw_counts", "sum"),
#         live_time_s=("live_time_s", "sum"),
#         n_runs=("run", "nunique"),
#     )
#     .sort_values("angle_deg")
#     .reset_index(drop=True)
# )

# combined_incremental["live_time_h"] = (
#     combined_incremental["live_time_s"] / 3600.0
# )

# combined_incremental["rate_per_h"] = (
#     combined_incremental["accepted_counts"]
#     / combined_incremental["live_time_h"]
# )

# combined_incremental["sigma_rate_per_h"] = (
#     np.sqrt(combined_incremental["accepted_counts"])
#     / combined_incremental["live_time_h"]
# )

# combined_incremental["rutherford_x"] = rutherford_factor(
#     combined_incremental["angle_deg"].to_numpy()
# )

# combined_incremental.to_csv(
#     OUTPUT_DIR / "incremental_combined_angle_summary.csv",
#     index=False,
# )

# print("\n============================================================")
# print("COMBINED INCREMENTAL DATA")
# print("============================================================")
# print(
#     combined_incremental[
#         [
#             "angle_deg",
#             "n_runs",
#             "accepted_counts",
#             "live_time_s",
#             "rate_per_h",
#             "sigma_rate_per_h",
#         ]
#     ].to_string(
#         index=False,
#         formatters={
#             "angle_deg": "{:.0f}".format,
#             "live_time_s": "{:.1f}".format,
#             "rate_per_h": "{:.2f}".format,
#             "sigma_rate_per_h": "{:.2f}".format,
#         },
#     )
# )


# # ============================================================
# # 5. WEIGHTED LINEARIZED RUTHERFORD FIT
# #
# # Rutherford:
# #     R(theta) = A csc^4(theta/2)
# #
# # Define:
# #     x = csc^4(theta/2)
# #     y = R
# #
# # Then:
# #     y = A x
# #
# # We fit A with weighted least squares through the origin.
# # ============================================================

# fit_data = combined_incremental[
#     (combined_incremental["angle_deg"] >= FIT_MIN_ANGLE_DEG)
#     & (combined_incremental["angle_deg"] <= FIT_MAX_ANGLE_DEG)
#     & (combined_incremental["accepted_counts"] > 0)
#     & (combined_incremental["sigma_rate_per_h"] > 0)
# ].copy()

# if len(fit_data) < 2:
#     raise ValueError(
#         "Not enough nonzero incremental points in the selected fit window. "
#         "Change FIT_MIN_ANGLE_DEG / FIT_MAX_ANGLE_DEG."
#     )

# x = fit_data["rutherford_x"].to_numpy(dtype=float)
# y = fit_data["rate_per_h"].to_numpy(dtype=float)
# sigma = fit_data["sigma_rate_per_h"].to_numpy(dtype=float)

# weights = 1.0 / sigma**2

# A_fit = np.sum(weights * x * y) / np.sum(weights * x**2)
# dA_fit = 1.0 / np.sqrt(np.sum(weights * x**2))

# y_fit = A_fit * x
# residual_sigma = (y - y_fit) / sigma

# chi2 = np.sum(residual_sigma**2)
# dof = len(fit_data) - 1
# reduced_chi2 = chi2 / dof if dof > 0 else np.nan

# print("\n============================================================")
# print("RUTHERFORD FIT TO COMBINED INCREMENTAL DATA")
# print("============================================================")
# print(
#     f"Fit window: {FIT_MIN_ANGLE_DEG:g}° to "
#     f"{FIT_MAX_ANGLE_DEG:g}°"
# )
# print("Model: R(theta) = A * csc^4(theta/2)")
# print(f"A = {A_fit:.8g} +/- {dA_fit:.3g} counts/hour")
# print(f"chi^2 = {chi2:.3f}")
# print(f"dof = {dof}")
# print(f"reduced chi^2 = {reduced_chi2:.3f}")


# # Add prediction + residual columns to combined table
# combined_incremental["theory_rate_per_h"] = (
#     A_fit * combined_incremental["rutherford_x"]
# )

# combined_incremental["percent_difference"] = (
#     100.0
#     * (
#         combined_incremental["rate_per_h"]
#         - combined_incremental["theory_rate_per_h"]
#     )
#     / combined_incremental["theory_rate_per_h"]
# )

# combined_incremental["pull_sigma"] = np.where(
#     combined_incremental["sigma_rate_per_h"] > 0,
#     (
#         combined_incremental["rate_per_h"]
#         - combined_incremental["theory_rate_per_h"]
#     )
#     / combined_incremental["sigma_rate_per_h"],
#     np.nan,
# )

# combined_incremental.to_csv(
#     OUTPUT_DIR / "incremental_combined_with_theory.csv",
#     index=False,
# )


# # ============================================================
# # 6. COMPARE EVERY FIXED-ANGLE RUN WITH THE INCREMENTAL FIT
# # ============================================================

# fixed_validation_rows = []

# for row in fixed_summary.itertuples(index=False):
#     x_theta = float(rutherford_factor(row.angle_deg))
#     theory_rate = A_fit * x_theta

#     percent_diff = 100.0 * (
#         row.rate_per_h - theory_rate
#     ) / theory_rate

#     pull = (
#         (row.rate_per_h - theory_rate) / row.sigma_rate_per_h
#         if row.sigma_rate_per_h > 0
#         else np.nan
#     )

#     # Relative Z consistency check only.
#     #
#     # Because R ~ Z^2 at fixed experimental normalization:
#     #     Z_relative = 79 * sqrt(R_measured / R_Au_fit)
#     #
#     # This is NOT an independent absolute measurement of Z,
#     # because A was itself obtained from gold-scattering data.
#     Z_relative = (
#         79.0 * np.sqrt(row.rate_per_h / theory_rate)
#         if row.rate_per_h > 0 and theory_rate > 0
#         else np.nan
#     )

#     # Propagate rate + fitted-normalization statistical uncertainty.
#     if row.rate_per_h > 0 and A_fit > 0:
#         dZ_relative = Z_relative * 0.5 * np.sqrt(
#             (row.sigma_rate_per_h / row.rate_per_h) ** 2
#             + (dA_fit / A_fit) ** 2
#         )
#     else:
#         dZ_relative = np.nan

#     fixed_validation_rows.append(
#         {
#             "run": row.run,
#             "angle_deg": row.angle_deg,
#             "measured_rate_per_h": row.rate_per_h,
#             "sigma_rate_per_h": row.sigma_rate_per_h,
#             "theory_rate_per_h": theory_rate,
#             "percent_difference": percent_diff,
#             "pull_sigma": pull,
#             "relative_Z_check": Z_relative,
#             "sigma_relative_Z": dZ_relative,
#         }
#     )

# fixed_validation = pd.DataFrame(fixed_validation_rows)

# print("\n============================================================")
# print("FIXED-ANGLE RUNS VS INCREMENTAL RUTHERFORD FIT")
# print("============================================================")
# print(
#     fixed_validation.to_string(
#         index=False,
#         formatters={
#             "angle_deg": "{:.0f}".format,
#             "measured_rate_per_h": "{:.3f}".format,
#             "sigma_rate_per_h": "{:.3f}".format,
#             "theory_rate_per_h": "{:.3f}".format,
#             "percent_difference": "{:+.2f}".format,
#             "pull_sigma": "{:+.2f}".format,
#             "relative_Z_check": "{:.2f}".format,
#             "sigma_relative_Z": "{:.2f}".format,
#         },
#     )
# )

# fixed_validation.to_csv(
#     OUTPUT_DIR / "fixed_angle_vs_incremental_fit.csv",
#     index=False,
# )


# # ============================================================
# # 7. PLOT — INCREMENTAL COUNT RATE VS ANGLE
# # ============================================================

# plt.figure(figsize=(9.5, 6.0))

# for key in ["incremental_run1", "incremental_run2"]:
#     run = incremental_runs[key]
#     s = run["summary"]

#     plt.errorbar(
#         s["angle_deg"],
#         s["rate_per_h"],
#         yerr=s["sigma_rate_per_h"],
#         fmt=run["marker"] + "-",
#         markersize=4.5,
#         linewidth=1.1,
#         capsize=2,
#         alpha=0.72,
#         label=run["label"],
#     )

# plt.errorbar(
#     combined_incremental["angle_deg"],
#     combined_incremental["rate_per_h"],
#     yerr=combined_incremental["sigma_rate_per_h"],
#     fmt="o",
#     markersize=6.5,
#     capsize=3,
#     label="Combined incremental rate",
# )

# theta_min = max(
#     0.5,
#     float(combined_incremental["angle_deg"].min()),
# )
# theta_max = float(combined_incremental["angle_deg"].max())

# theta_curve = np.linspace(theta_min, theta_max, 700)
# theory_curve = A_fit * rutherford_factor(theta_curve)

# plt.plot(
#     theta_curve,
#     theory_curve,
#     linewidth=2,
#     label=(
#         r"Weighted Rutherford fit: "
#         r"$R=A\,\csc^4(\theta/2)$"
#     ),
# )

# plt.axvspan(
#     FIT_MIN_ANGLE_DEG,
#     FIT_MAX_ANGLE_DEG,
#     alpha=0.10,
#     label="Fit window",
# )

# plt.yscale("log")
# plt.xlabel("Scattering angle θ (degrees)")
# plt.ylabel("Count rate after timing + 4 V cuts (counts/hour)")
# plt.title("Incremental Rutherford scattering run")
# plt.grid(alpha=0.25, which="both")
# plt.legend()
# finish_plot("incremental_rate_vs_angle.png")


# # ============================================================
# # 8. PLOT — LINEARIZED RUTHERFORD RELATION
# # ============================================================

# plt.figure(figsize=(9.0, 6.0))

# fit_plot_x = fit_data["rutherford_x"].to_numpy()
# fit_plot_y = fit_data["rate_per_h"].to_numpy()
# fit_plot_dy = fit_data["sigma_rate_per_h"].to_numpy()

# plt.errorbar(
#     fit_plot_x,
#     fit_plot_y,
#     yerr=fit_plot_dy,
#     fmt="o",
#     markersize=6,
#     capsize=3,
#     label="Combined incremental data in fit window",
# )

# x_line = np.linspace(0, fit_plot_x.max() * 1.05, 500)

# plt.plot(
#     x_line,
#     A_fit * x_line,
#     linewidth=2,
#     label=(
#         rf"$R=Ax$, "
#         rf"$A={A_fit:.4g}\pm{dA_fit:.2g}$ counts/hour"
#     ),
# )

# plt.xlabel(r"$x=\csc^4(\theta/2)$")
# plt.ylabel("Count rate (counts/hour)")
# plt.title("Linearized Rutherford-scattering fit")
# plt.grid(alpha=0.25)
# plt.legend()
# finish_plot("incremental_linearized_rutherford_fit.png")


# # ============================================================
# # 9. PLOT — INCREMENTAL FIT RESIDUALS
# # ============================================================

# plot_residual = combined_incremental[
#     combined_incremental["sigma_rate_per_h"] > 0
# ].copy()

# plt.figure(figsize=(9.0, 5.0))

# plt.errorbar(
#     plot_residual["angle_deg"],
#     plot_residual["percent_difference"],
#     fmt="o",
#     markersize=5.5,
# )

# plt.axhline(0, linestyle="--", linewidth=1.5)
# plt.axvspan(
#     FIT_MIN_ANGLE_DEG,
#     FIT_MAX_ANGLE_DEG,
#     alpha=0.10,
# )

# plt.xlabel("Scattering angle θ (degrees)")
# plt.ylabel("(Measured - Rutherford) / Rutherford (%)")
# plt.title("Incremental-run deviation from Rutherford fit")
# plt.grid(alpha=0.25)
# finish_plot("incremental_percent_residuals.png")


# # ============================================================
# # 10. PLOT — FIXED-ANGLE RUNS ON THE SAME RUTHERFORD CURVE
# #
# # Every repeated run stays separate.
# # ============================================================

# plt.figure(figsize=(9.5, 6.0))

# theta_curve_fixed = np.linspace(
#     min(4.5, fixed_summary["angle_deg"].min()),
#     max(16.0, fixed_summary["angle_deg"].max() + 1),
#     500,
# )

# plt.plot(
#     theta_curve_fixed,
#     A_fit * rutherford_factor(theta_curve_fixed),
#     linewidth=2,
#     label="Rutherford prediction from incremental fit",
# )

# for cfg in FIXED_RUN_CONFIG:
#     run_row = fixed_summary[fixed_summary["run"] == cfg["label"]].iloc[0]

#     plt.errorbar(
#         run_row["angle_deg"],
#         run_row["rate_per_h"],
#         yerr=run_row["sigma_rate_per_h"],
#         fmt=cfg["marker"],
#         markersize=8,
#         capsize=4,
#         label=cfg["label"],
#     )

# plt.yscale("log")
# plt.xlabel("Scattering angle θ (degrees)")
# plt.ylabel("Count rate (counts/hour)")
# plt.title("Fixed-angle runs vs Rutherford relation")
# plt.grid(alpha=0.25, which="both")
# plt.legend()
# finish_plot("fixed_angle_runs_vs_rutherford.png")


# # ============================================================
# # 11. PLOT — FIXED-RUN RELATIVE Z CONSISTENCY
# #
# # This is intentionally labeled "relative".
# # It is not an independent absolute determination of gold Z.
# # ============================================================

# plt.figure(figsize=(9.0, 5.5))

# x_positions = np.arange(len(fixed_validation))

# plt.errorbar(
#     x_positions,
#     fixed_validation["relative_Z_check"],
#     yerr=fixed_validation["sigma_relative_Z"],
#     fmt="o",
#     markersize=7,
#     capsize=4,
# )

# plt.axhline(
#     79.0,
#     linestyle="--",
#     linewidth=1.5,
#     label="Gold reference: Z = 79",
# )

# plt.xticks(
#     x_positions,
#     fixed_validation["run"],
#     rotation=30,
#     ha="right",
# )

# plt.ylabel("Relative inferred Z")
# plt.title("Relative gold-Z consistency of fixed-angle runs")
# plt.grid(alpha=0.25)
# plt.legend()
# finish_plot("fixed_angle_relative_Z_check.png")


# # ============================================================
# # 12. FIXED-RUN PULSE-HEIGHT SPECTRA
# # ============================================================

# bins = np.linspace(
#     THRESHOLD,
#     max(
#         THRESHOLD + 0.2,
#         max(run["cut"]["pulse_height_V"].max() for run in fixed_runs.values()),
#     ),
#     70,
# )

# plt.figure(figsize=(10.0, 6.0))

# for cfg in FIXED_RUN_CONFIG:
#     run = fixed_runs[cfg["key"]]

#     counts, edges = np.histogram(
#         run["cut"]["pulse_height_V"],
#         bins=bins,
#     )

#     centers = 0.5 * (edges[:-1] + edges[1:])
#     rate_per_bin = counts / run["T_h"]

#     plt.step(
#         centers,
#         rate_per_bin,
#         where="mid",
#         linewidth=1.4,
#         label=cfg["label"],
#     )

# plt.axvline(
#     THRESHOLD,
#     linestyle="--",
#     linewidth=1.3,
#     label="4 V analysis cut",
# )

# plt.xlabel("Pulse height (V)")
# plt.ylabel("Count rate per bin (counts/hour)")
# plt.title("Fixed-angle pulse-height spectra after 4 V cut")
# plt.grid(alpha=0.25)
# plt.legend()
# finish_plot("fixed_angle_pulse_height_spectra.png")


# # ============================================================
# # 13. TIME-STABILITY HELPER FOR FIXED-ANGLE RUNS
# #
# # Bins are aligned to the actual first recorded pulse.
# # A clearly incomplete final bin is dropped.
# # ============================================================

# def plot_rate_stability(full_run, cut_run, frequency, title, ylabel, filename):
#     start_time = full_run["timestamp"].min()

#     counts = (
#         cut_run.set_index("timestamp")["pulse_height_V"]
#         .resample(frequency, origin=start_time)
#         .count()
#     )

#     counts = counts[counts > 0]

#     if len(counts) >= 3:
#         median_previous = counts.iloc[:-1].median()

#         if (
#             median_previous > 0
#             and counts.iloc[-1] < 0.70 * median_previous
#         ):
#             counts = counts.iloc[:-1]

#     if len(counts) == 0:
#         print(f"Skipping stability plot with no accepted bins: {title}")
#         return

#     errors = np.sqrt(counts.to_numpy())

#     plt.figure(figsize=(9.5, 5.5))

#     plt.errorbar(
#         counts.index,
#         counts.to_numpy(),
#         yerr=errors,
#         fmt="o-",
#         markersize=4,
#         linewidth=1.2,
#         capsize=2,
#         label=r"Measured counts $\pm\sqrt{N}$",
#     )

#     mean_counts = counts.mean()

#     plt.axhline(
#         mean_counts,
#         linestyle="--",
#         linewidth=1.4,
#         label=f"Mean = {mean_counts:.1f}",
#     )

#     plt.xlabel("Time")
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.grid(alpha=0.25)
#     plt.legend()
#     plt.xticks(rotation=35)
#     finish_plot(filename)


# for cfg in FIXED_RUN_CONFIG:
#     run = fixed_runs[cfg["key"]]

#     plot_rate_stability(
#         full_run=run["full"],
#         cut_run=run["cut"],
#         frequency=cfg["stability_frequency"],
#         title=f"{cfg['label']}: count-rate stability after 4 V cut",
#         ylabel=cfg["stability_ylabel"],
#         filename=f"{cfg['key']}_stability.png",
#     )


# # ============================================================
# # 14. INCREMENTAL-RUN TIMING DIAGNOSTIC
# #
# # This plot is useful for checking that the offset and angle
# # assignment are doing what you intended.
# # ============================================================

# for key in ["incremental_run1", "incremental_run2"]:
#     run = incremental_runs[key]
#     assigned = run["assigned"]

#     plt.figure(figsize=(10.0, 4.8))

#     accepted = assigned[assigned["accepted"]]
#     rejected_timing = assigned[
#         (~assigned["stable_time"])
#         & assigned["passes_voltage_cut"]
#     ]

#     plt.scatter(
#         accepted["timestamp"],
#         accepted["angle_deg"],
#         s=8,
#         alpha=0.65,
#         label="Accepted >= 4 V pulses",
#     )

#     if len(rejected_timing) > 0:
#         plt.scatter(
#             rejected_timing["timestamp"],
#             rejected_timing["angle_deg"],
#             s=8,
#             alpha=0.45,
#             label=">= 4 V but removed during adjustment window",
#         )

#     plt.xlabel("Time")
#     plt.ylabel("Assigned scattering angle (degrees)")
#     plt.title(
#         f"{run['label']}: timestamp-to-angle assignment diagnostic"
#     )
#     plt.grid(alpha=0.25)
#     plt.legend()
#     plt.xticks(rotation=30)

#     finish_plot(f"{key}_timing_assignment_diagnostic.png")


# # ============================================================
# # 15. FINAL SUMMARY
# # ============================================================

# print("\n============================================================")
# print("DONE")
# print("============================================================")
# print(f"Results written to:\n{OUTPUT_DIR}")
# print()
# print("Important timing settings:")
# print(f"  Run 1 zero-degree offset = {RUN1_ZERO_DEGREE_OFFSET_SECONDS:.1f} s")
# print(f"  Run 2 zero-degree offset = {RUN2_ZERO_DEGREE_OFFSET_SECONDS:.1f} s")
# print(f"  First post-offset angle  = {ANGLE_START_DEG:.1f} degrees")
# print(f"  Angle step               = {ANGLE_STEP_DEG:.1f} degree")
# print(f"  Time per angle           = {ANGLE_HOLD_SECONDS:.1f} s")
# print(f"  Adjustment cut           = {ADJUSTMENT_CUT_SECONDS:.1f} s")

# if CUT_FIRST_5S_EVERY_MINUTE:
#     print("  Cut mode                 = first 5 s of EVERY minute")
# else:
#     print("  Cut mode                 = first 5 s after each 2-minute angle change")

# print(
#     f"  Live time per full angle = "
#     f"{live_seconds_per_full_angle_bin():.1f} s"
# )
# print(
#     f"  Rutherford fit window    = "
#     f"{FIT_MIN_ANGLE_DEG:g}° to {FIT_MAX_ANGLE_DEG:g}°"
# )


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ============================================================
# # RUTHERFORD LAB — DATA-ONLY VISUALIZATION
# # ZERO-ANGLE CALIBRATION CHECK: 0°, 1°, 2°, 3°
# # ============================================================
# #
# # No Rutherford theory fit.
# # No inferred Z.
# # No residuals.
# #
# # This version tests four possible choices for the true zero angle:
# #
# #       ZERO = nominal 0°
# #       ZERO = nominal 1°
# #       ZERO = nominal 2°
# #       ZERO = nominal 3°
# #
# # If zero is set to z0, then:
# #
# #       calibrated angle = nominal angle - z0
# #
# # Example for ZERO = 2°:
# #
# #       nominal 0° -> calibrated -2°
# #       nominal 1° -> calibrated -1°
# #       nominal 2° -> calibrated  0°
# #       nominal 3° -> calibrated +1°
# #       ...
# #
# # For the CIRCULAR maps, scattering-angle magnitude is radial.
# # Therefore +theta and -theta correspond to the same radius.
# # When both are present, their measured count rates are averaged
# # before interpolation.
# #
# # The only "function" used for the gradient map is piecewise-linear
# # interpolation between the resulting measured radial count rates.
# # No physics equation is fit.
# #
# # OUTPUTS
# # -------
# # 1. One large plot containing all six fixed-angle count-rate-vs-time runs.
# #
# # 2. One 2x2 figure showing count rate vs calibrated scattering angle
# #    for zero choices 0°, 1°, 2°, and 3°.
# #
# # 3-6. Four separate, spacious circular-gradient figures:
# #      - zero = 0°
# #      - zero = 1°
# #      - zero = 2°
# #      - zero = 3°
# #
# #      Each figure contains:
# #          incremental run 1
# #          incremental run 2
# #          combined average
# #
# # ============================================================


# # ============================================================
# # 0. FILES AND SETTINGS
# # ============================================================

# BASE_DIR = Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS")

# PER_DEGREE_RUN1_FILE = (
#     BASE_DIR / "PER_DEGREE_RUN_ONE_30MINUTES_RUN_ARIYANTURNER_RFS.csv"
# )

# PER_DEGREE_RUN2_FILE = (
#     BASE_DIR / "PER_DEGREE_RUN_ONE_HOUR_RUN_ARIYANTURNER_RFS_RUN2.csv"
# )

# FIVE_RUN1_FILE = (
#     BASE_DIR / "5DEGREE_RUN_ONE_HOUR_ARIYANTURNER_RFS.csv"
# )

# FIVE_RUN2_FILE = (
#     BASE_DIR / "5DEGREES_HOUR_RUN_ARIYANTURNER_RFS.csv"
# )

# TEN_RUN1_FILE = (
#     BASE_DIR / "10DEGREE_OVERNIGHT_RUN_ARIYAN_TURNER_RFS_RUN2.csv"
# )

# TEN_RUN2_FILE = (
#     BASE_DIR / "10DEGREES_OVERNIGHT_RUN_ARIYANTURNER_RFS.csv"
# )

# FIFTEEN_RUN1_FILE = (
#     BASE_DIR / "15DEGREES_OVERNIGHT_ARIYANTURNER_RFS.csv"
# )

# FIFTEEN_RUN2_FILE = (
#     BASE_DIR / "15DEGREES_WEEKEND_RUN_ARIYANTURNER_RFS.csv"
# )


# THRESHOLD_V = 4.0


# # ------------------------------------------------------------
# # Incremental timing
# # ------------------------------------------------------------

# RUN1_LEADING_OFFSET_SECONDS = 63.0
# RUN2_LEADING_OFFSET_SECONDS = 0.0

# FIRST_NOMINAL_ANGLE_DEG = 0
# LAST_NOMINAL_ANGLE_DEG = 15

# MEASUREMENT_SECONDS_PER_ANGLE = 120.0
# ANGLE_CHANGE_SECONDS = 5.0

# ANGLE_CYCLE_SECONDS = (
#     MEASUREMENT_SECONDS_PER_ANGLE
#     + ANGLE_CHANGE_SECONDS
# )


# # ------------------------------------------------------------
# # Test these four zero-angle choices
# # ------------------------------------------------------------

# ZERO_ANGLE_CHOICES_DEG = [0, 1, 2, 3]


# # ------------------------------------------------------------
# # Fixed-angle plotting bins
# # ------------------------------------------------------------

# FIVE_DEGREE_TIME_BIN = "1min"
# TEN_DEGREE_TIME_BIN = "1h"
# FIFTEEN_DEGREE_TIME_BIN = "1h"


# # ------------------------------------------------------------
# # Plot/output settings
# # ------------------------------------------------------------

# SHOW_PLOTS = True

# OUTPUT_DIR = BASE_DIR / "rutherford_zero_angle_comparison"

# OUTPUT_DIR.mkdir(
#     parents=True,
#     exist_ok=True,
# )


# # ============================================================
# # 1. DATA LOADING
# # ============================================================

# def load_run(filename):
#     filename = Path(filename)

#     if not filename.exists():
#         raise FileNotFoundError(
#             f"Could not find:\n{filename}"
#         )

#     df = pd.read_csv(filename)

#     required = {
#         "timestamp",
#         "pulse_height_V",
#     }

#     missing = required - set(df.columns)

#     if missing:
#         raise ValueError(
#             f"{filename.name} is missing column(s): {sorted(missing)}\n"
#             f"Columns found: {list(df.columns)}"
#         )

#     df["timestamp"] = pd.to_datetime(
#         df["timestamp"],
#         errors="coerce",
#     )

#     df["pulse_height_V"] = pd.to_numeric(
#         df["pulse_height_V"],
#         errors="coerce",
#     )

#     df = (
#         df
#         .dropna(subset=["timestamp", "pulse_height_V"])
#         .sort_values("timestamp")
#         .reset_index(drop=True)
#     )

#     if len(df) == 0:
#         raise ValueError(
#             f"{filename.name} contains no valid events."
#         )

#     return df


# def apply_voltage_cut(df):
#     return df[
#         df["pulse_height_V"] >= THRESHOLD_V
#     ].copy()


# def finish_plot(filename):
#     plt.savefig(
#         OUTPUT_DIR / filename,
#         dpi=220,
#         bbox_inches="tight",
#     )

#     if SHOW_PLOTS:
#         plt.show()
#     else:
#         plt.close()


# # ============================================================
# # 2. FIXED-ANGLE RUNS
# # ============================================================

# FIXED_RUNS = [
#     {
#         "key": "5_run1",
#         "label": "5° run 1",
#         "angle": 5,
#         "filename": FIVE_RUN1_FILE,
#         "time_bin": FIVE_DEGREE_TIME_BIN,
#     },
#     {
#         "key": "5_run2",
#         "label": "5° run 2",
#         "angle": 5,
#         "filename": FIVE_RUN2_FILE,
#         "time_bin": FIVE_DEGREE_TIME_BIN,
#     },
#     {
#         "key": "10_run1",
#         "label": "10° run 1",
#         "angle": 10,
#         "filename": TEN_RUN1_FILE,
#         "time_bin": TEN_DEGREE_TIME_BIN,
#     },
#     {
#         "key": "10_run2",
#         "label": "10° run 2",
#         "angle": 10,
#         "filename": TEN_RUN2_FILE,
#         "time_bin": TEN_DEGREE_TIME_BIN,
#     },
#     {
#         "key": "15_run1",
#         "label": "15° overnight",
#         "angle": 15,
#         "filename": FIFTEEN_RUN1_FILE,
#         "time_bin": FIFTEEN_DEGREE_TIME_BIN,
#     },
#     {
#         "key": "15_run2",
#         "label": "15° weekend",
#         "angle": 15,
#         "filename": FIFTEEN_RUN2_FILE,
#         "time_bin": FIFTEEN_DEGREE_TIME_BIN,
#     },
# ]


# def time_bin_seconds(frequency):
#     return pd.to_timedelta(
#         frequency
#     ).total_seconds()


# def fixed_run_rate_vs_time(
#     full_run,
#     cut_run,
#     frequency,
# ):
#     """
#     Count rate vs elapsed time for one fixed-angle run.

#     - Zero-count bins are preserved.
#     - Final partial bin is kept.
#     - Final partial bin is divided by its actual duration.
#     - Output is always counts/hour.
#     """
#     start = full_run["timestamp"].min()
#     end = full_run["timestamp"].max()

#     total_duration_s = (
#         end - start
#     ).total_seconds()

#     if total_duration_s <= 0:
#         raise ValueError(
#             "Fixed run has non-positive duration."
#         )

#     bin_width_s = time_bin_seconds(
#         frequency
#     )

#     edges_s = np.arange(
#         0.0,
#         total_duration_s + bin_width_s,
#         bin_width_s,
#     )

#     if edges_s[-1] > total_duration_s:
#         edges_s[-1] = total_duration_s

#     if edges_s[-1] < total_duration_s:
#         edges_s = np.append(
#             edges_s,
#             total_duration_s,
#         )

#     edges_s = np.unique(
#         edges_s
#     )

#     if len(edges_s) < 2:
#         edges_s = np.array(
#             [0.0, total_duration_s]
#         )

#     accepted_elapsed_s = (
#         cut_run["timestamp"] - start
#     ).dt.total_seconds().to_numpy()

#     counts, _ = np.histogram(
#         accepted_elapsed_s,
#         bins=edges_s,
#     )

#     durations_s = np.diff(
#         edges_s
#     )

#     rate_counts_per_hour = (
#         counts
#         / (durations_s / 3600.0)
#     )

#     elapsed_hours = (
#         (edges_s[:-1] + edges_s[1:])
#         / 2.0
#         / 3600.0
#     )

#     return pd.DataFrame(
#         {
#             "elapsed_hours": elapsed_hours,
#             "bin_duration_seconds": durations_s,
#             "accepted_counts": counts,
#             "rate_counts_per_hour": rate_counts_per_hour,
#         }
#     )


# fixed_results = {}

# for config in FIXED_RUNS:

#     full = load_run(
#         config["filename"]
#     )

#     cut = apply_voltage_cut(
#         full
#     )

#     time_series = fixed_run_rate_vs_time(
#         full_run=full,
#         cut_run=cut,
#         frequency=config["time_bin"],
#     )

#     fixed_results[
#         config["key"]
#     ] = {
#         **config,
#         "full": full,
#         "cut": cut,
#         "time_series": time_series,
#     }

#     time_series.to_csv(
#         OUTPUT_DIR
#         / f"{config['key']}_count_rate_vs_time.csv",
#         index=False,
#     )


# # ============================================================
# # FIGURE 1
# # ALL FIXED-ANGLE RUNS — ONE SPACIOUS PLOT
# # ============================================================

# fig, ax = plt.subplots(
#     figsize=(13.5, 7.2)
# )

# for config in FIXED_RUNS:

#     ts = fixed_results[
#         config["key"]
#     ]["time_series"]

#     ax.plot(
#         ts["elapsed_hours"],
#         ts["rate_counts_per_hour"],
#         marker="o",
#         markersize=3,
#         linewidth=1.1,
#         alpha=0.78,
#         label=config["label"],
#     )


# ax.set_xlabel(
#     "Elapsed time from beginning of each trial (hours)",
#     fontsize=12,
# )

# ax.set_ylabel(
#     "Count rate after 4 V cut (counts/hour)",
#     fontsize=12,
# )

# ax.set_title(
#     "All fixed-angle trials: measured count rate over time",
#     fontsize=15,
#     pad=14,
# )

# ax.grid(
#     alpha=0.25
# )

# # Legend outside the data area.
# ax.legend(
#     loc="center left",
#     bbox_to_anchor=(1.02, 0.5),
#     frameon=True,
# )

# fig.subplots_adjust(
#     left=0.09,
#     right=0.79,
#     bottom=0.12,
#     top=0.90,
# )

# finish_plot(
#     "01_all_fixed_angle_runs_count_rate_vs_time.png"
# )


# # ============================================================
# # 3. INCREMENTAL RUNS
# # ============================================================

# INCREMENTAL_RUNS = [
#     {
#         "key": "incremental_run1",
#         "label": "Incremental run 1",
#         "filename": PER_DEGREE_RUN1_FILE,
#         "leading_offset_s": RUN1_LEADING_OFFSET_SECONDS,
#     },
#     {
#         "key": "incremental_run2",
#         "label": "Incremental run 2",
#         "filename": PER_DEGREE_RUN2_FILE,
#         "leading_offset_s": RUN2_LEADING_OFFSET_SECONDS,
#     },
# ]


# def analyze_incremental_run(
#     filename,
#     label,
#     leading_offset_s,
# ):
#     """
#     Experimental timing:

#         120 s data at nominal angle
#         5 s movement
#         120 s data at next nominal angle
#         5 s movement
#         ...

#     Nominal angles run from 0° through 15°.

#     Run 1:
#         first timestamp + 63 s = start of nominal 0° data.

#     Run 2:
#         first timestamp = start of nominal 0° data.
#     """
#     df = load_run(
#         filename
#     )

#     raw_start = df[
#         "timestamp"
#     ].min()

#     nominal_zero_start = (
#         raw_start
#         + pd.to_timedelta(
#             leading_offset_s,
#             unit="s",
#         )
#     )

#     work = df.copy()

#     work[
#         "elapsed_from_nominal_zero_s"
#     ] = (
#         work["timestamp"]
#         - nominal_zero_start
#     ).dt.total_seconds()

#     # Remove anything before nominal 0° actually begins.
#     work = work[
#         work["elapsed_from_nominal_zero_s"] >= 0
#     ].copy()

#     work[
#         "cycle_index"
#     ] = np.floor(
#         work["elapsed_from_nominal_zero_s"]
#         / ANGLE_CYCLE_SECONDS
#     ).astype(int)

#     work[
#         "time_inside_cycle_s"
#     ] = (
#         work["elapsed_from_nominal_zero_s"]
#         - work["cycle_index"]
#         * ANGLE_CYCLE_SECONDS
#     )

#     valid_nominal_angle = (
#         (
#             work["cycle_index"]
#             >= FIRST_NOMINAL_ANGLE_DEG
#         )
#         & (
#             work["cycle_index"]
#             <= LAST_NOMINAL_ANGLE_DEG
#         )
#     )

#     in_measurement_window = (
#         work["time_inside_cycle_s"]
#         < MEASUREMENT_SECONDS_PER_ANGLE
#     )

#     passes_voltage_cut = (
#         work["pulse_height_V"]
#         >= THRESHOLD_V
#     )

#     work["accepted"] = (
#         valid_nominal_angle
#         & in_measurement_window
#         & passes_voltage_cut
#     )

#     accepted = work[
#         work["accepted"]
#     ].copy()

#     accepted[
#         "nominal_angle_deg"
#     ] = accepted[
#         "cycle_index"
#     ].astype(int)

#     rows = []

#     for nominal_angle in range(
#         FIRST_NOMINAL_ANGLE_DEG,
#         LAST_NOMINAL_ANGLE_DEG + 1,
#     ):

#         events = accepted[
#             accepted["nominal_angle_deg"]
#             == nominal_angle
#         ]

#         count = len(
#             events
#         )

#         rate_counts_per_hour = (
#             count
#             / (
#                 MEASUREMENT_SECONDS_PER_ANGLE
#                 / 3600.0
#             )
#         )

#         rows.append(
#             {
#                 "run": label,
#                 "nominal_angle_deg": nominal_angle,
#                 "accepted_counts": count,
#                 "rate_counts_per_hour": rate_counts_per_hour,
#             }
#         )

#     summary = pd.DataFrame(
#         rows
#     )

#     return work, accepted, summary


# incremental_results = {}

# for config in INCREMENTAL_RUNS:

#     assigned, accepted, summary = analyze_incremental_run(
#         filename=config["filename"],
#         label=config["label"],
#         leading_offset_s=config["leading_offset_s"],
#     )

#     incremental_results[
#         config["key"]
#     ] = {
#         **config,
#         "assigned": assigned,
#         "accepted": accepted,
#         "summary": summary,
#     }

#     summary.to_csv(
#         OUTPUT_DIR
#         / f"{config['key']}_nominal_count_rate_by_angle.csv",
#         index=False,
#     )


# # ============================================================
# # 4. NOMINAL ANGLE TABLE
# # ============================================================

# run1_summary = (
#     incremental_results[
#         "incremental_run1"
#     ]["summary"]
#     .set_index("nominal_angle_deg")
# )

# run2_summary = (
#     incremental_results[
#         "incremental_run2"
#     ]["summary"]
#     .set_index("nominal_angle_deg")
# )

# nominal_angles = np.arange(
#     FIRST_NOMINAL_ANGLE_DEG,
#     LAST_NOMINAL_ANGLE_DEG + 1,
#     dtype=float,
# )

# incremental_table = pd.DataFrame(
#     {
#         "nominal_angle_deg": nominal_angles,
#     }
# )

# incremental_table[
#     "run1_rate_counts_per_hour"
# ] = (
#     incremental_table[
#         "nominal_angle_deg"
#     ].map(
#         run1_summary[
#             "rate_counts_per_hour"
#         ]
#     )
# )

# incremental_table[
#     "run2_rate_counts_per_hour"
# ] = (
#     incremental_table[
#         "nominal_angle_deg"
#     ].map(
#         run2_summary[
#             "rate_counts_per_hour"
#         ]
#     )
# )

# incremental_table[
#     "combined_average_rate_counts_per_hour"
# ] = (
#     incremental_table[
#         [
#             "run1_rate_counts_per_hour",
#             "run2_rate_counts_per_hour",
#         ]
#     ].mean(
#         axis=1
#     )
# )

# incremental_table.to_csv(
#     OUTPUT_DIR
#     / "incremental_nominal_count_rates.csv",
#     index=False,
# )


# # ============================================================
# # 5. ZERO-ANGLE CALIBRATION TABLES
# # ============================================================

# def make_calibrated_table(
#     nominal_table,
#     zero_angle_deg,
# ):
#     """
#     Shift the angular coordinate.

#         calibrated = nominal - zero_angle
#     """
#     calibrated = nominal_table.copy()

#     calibrated[
#         "zero_angle_choice_deg"
#     ] = zero_angle_deg

#     calibrated[
#         "calibrated_angle_deg"
#     ] = (
#         calibrated[
#             "nominal_angle_deg"
#         ]
#         - zero_angle_deg
#     )

#     return calibrated


# calibrated_tables = {}

# for zero_angle in ZERO_ANGLE_CHOICES_DEG:

#     table = make_calibrated_table(
#         incremental_table,
#         zero_angle,
#     )

#     calibrated_tables[
#         zero_angle
#     ] = table

#     table.to_csv(
#         OUTPUT_DIR
#         / f"incremental_zero_set_to_{zero_angle}deg.csv",
#         index=False,
#     )


# # ============================================================
# # FIGURE 2
# # COUNT RATE VS CALIBRATED SCATTERING ANGLE
# # FOUR ZERO-ANGLE CHOICES
# # ============================================================

# fig, axes = plt.subplots(
#     2,
#     2,
#     figsize=(14.0, 10.5),
# )

# axes = axes.flatten()

# for ax, zero_angle in zip(
#     axes,
#     ZERO_ANGLE_CHOICES_DEG,
# ):

#     table = calibrated_tables[
#         zero_angle
#     ]

#     ax.plot(
#         table["calibrated_angle_deg"],
#         table["run1_rate_counts_per_hour"],
#         marker="o",
#         markersize=4,
#         linewidth=1.2,
#         label="Incremental run 1",
#     )

#     ax.plot(
#         table["calibrated_angle_deg"],
#         table["run2_rate_counts_per_hour"],
#         marker="s",
#         markersize=4,
#         linewidth=1.2,
#         label="Incremental run 2",
#     )

#     ax.plot(
#         table["calibrated_angle_deg"],
#         table[
#             "combined_average_rate_counts_per_hour"
#         ],
#         marker="o",
#         markersize=6,
#         linewidth=2.0,
#         label="Combined average",
#     )

#     ax.axvline(
#         0,
#         linestyle="--",
#         linewidth=1.0,
#         alpha=0.7,
#     )

#     ax.set_title(
#         f"Nominal {zero_angle}° treated as true 0°",
#         fontsize=13,
#         pad=10,
#     )

#     ax.set_xlabel(
#         "Calibrated scattering angle (degrees)"
#     )

#     ax.set_ylabel(
#         "Count rate (counts/hour)"
#     )

#     ax.grid(
#         alpha=0.25
#     )

#     ax.legend(
#         fontsize=9
#     )


# fig.suptitle(
#     "Incremental scans under four possible zero-angle calibrations",
#     fontsize=16,
#     y=0.98,
# )

# fig.subplots_adjust(
#     left=0.08,
#     right=0.97,
#     bottom=0.08,
#     top=0.91,
#     wspace=0.26,
#     hspace=0.34,
# )

# finish_plot(
#     "02_zero_angle_calibration_count_rate_comparison.png"
# )


# # ============================================================
# # 6. RADIAL PROFILE CONSTRUCTION
# # ============================================================
# #
# # For a circular map we need count rate as a function of
# # scattering-angle MAGNITUDE:
# #
# #       r = |calibrated angle|
# #
# # If the zero-angle choice creates both -theta and +theta
# # measurements, they represent the same radial scattering angle
# # in the circular visualization.
# #
# # So we average them.
# #
# # Example, zero = 2°:
# #
# #       nominal 1° -> calibrated -1°
# #       nominal 3° -> calibrated +1°
# #
# # Both contribute to radial angle = 1°.
# # ============================================================

# def make_radial_profile(
#     calibrated_table,
#     rate_column,
# ):
#     radial = pd.DataFrame(
#         {
#             "radial_angle_deg": np.abs(
#                 calibrated_table[
#                     "calibrated_angle_deg"
#                 ].to_numpy(
#                     dtype=float
#                 )
#             ),
#             "rate_counts_per_hour": calibrated_table[
#                 rate_column
#             ].to_numpy(
#                 dtype=float
#             ),
#         }
#     )

#     radial = (
#         radial
#         .groupby(
#             "radial_angle_deg",
#             as_index=False,
#         )
#         .agg(
#             rate_counts_per_hour=(
#                 "rate_counts_per_hour",
#                 "mean",
#             )
#         )
#         .sort_values(
#             "radial_angle_deg"
#         )
#         .reset_index(
#             drop=True
#         )
#     )

#     return radial


# def make_piecewise_linear_rate_function(
#     radial_profile,
# ):
#     """
#     Data-only function:
#     linearly interpolate between measured radial-angle values.
#     """
#     x = radial_profile[
#         "radial_angle_deg"
#     ].to_numpy(
#         dtype=float
#     )

#     y = radial_profile[
#         "rate_counts_per_hour"
#     ].to_numpy(
#         dtype=float
#     )

#     def rate_function(theta_magnitude_deg):
#         return np.interp(
#             theta_magnitude_deg,
#             x,
#             y,
#         )

#     return rate_function


# # ============================================================
# # 7. CIRCULAR FIELD
# # ============================================================

# MAP_EXTENT_DEG = 15.0
# GRID_SIZE = 750


# def make_circular_field(
#     radial_profile,
# ):
#     """
#     Produce a circle in which physical radius = scattering-angle
#     magnitude in degrees.

#     We always display coordinates out to +/-15°.

#     If a shifted zero choice does not provide measurements all the
#     way to 15° radial angle, the unmeasured outer annulus stays blank.
#     No extrapolation is performed.
#     """
#     rate_function = (
#         make_piecewise_linear_rate_function(
#             radial_profile
#         )
#     )

#     axis = np.linspace(
#         -MAP_EXTENT_DEG,
#         MAP_EXTENT_DEG,
#         GRID_SIZE,
#     )

#     X, Y = np.meshgrid(
#         axis,
#         axis,
#     )

#     radial_angle = np.sqrt(
#         X**2 + Y**2
#     )

#     max_measured_radius = radial_profile[
#         "radial_angle_deg"
#     ].max()

#     values = np.full(
#         radial_angle.shape,
#         np.nan,
#         dtype=float,
#     )

#     measured_region = (
#         radial_angle
#         <= max_measured_radius
#     )

#     values[
#         measured_region
#     ] = rate_function(
#         radial_angle[
#             measured_region
#         ]
#     )

#     return (
#         X,
#         Y,
#         values,
#         max_measured_radius,
#     )


# # ============================================================
# # 8. SHARED COLOR SCALE
# # ============================================================
# #
# # One scale for ALL maps so colors mean the same count rate
# # everywhere.

# all_original_rates = np.concatenate(
#     [
#         incremental_table[
#             "run1_rate_counts_per_hour"
#         ].to_numpy(
#             dtype=float
#         ),
#         incremental_table[
#             "run2_rate_counts_per_hour"
#         ].to_numpy(
#             dtype=float
#         ),
#         incremental_table[
#             "combined_average_rate_counts_per_hour"
#         ].to_numpy(
#             dtype=float
#         ),
#     ]
# )

# GLOBAL_VMIN = np.nanmin(
#     all_original_rates
# )

# GLOBAL_VMAX = np.nanmax(
#     all_original_rates
# )


# # ============================================================
# # FIGURES 3–6
# # ONE LARGE GRADIENT-MAP FIGURE PER ZERO CHOICE
# # ============================================================

# rate_columns = [
#     (
#         "run1_rate_counts_per_hour",
#         "Incremental run 1",
#     ),
#     (
#         "run2_rate_counts_per_hour",
#         "Incremental run 2",
#     ),
#     (
#         "combined_average_rate_counts_per_hour",
#         "Combined average",
#     ),
# ]


# for zero_angle in ZERO_ANGLE_CHOICES_DEG:

#     table = calibrated_tables[
#         zero_angle
#     ]

#     fig, axes = plt.subplots(
#         1,
#         3,
#         figsize=(19.5, 6.8),
#     )

#     last_image = None

#     for ax, (
#         rate_column,
#         title,
#     ) in zip(
#         axes,
#         rate_columns,
#     ):

#         radial_profile = make_radial_profile(
#             calibrated_table=table,
#             rate_column=rate_column,
#         )

#         (
#             X,
#             Y,
#             field,
#             max_measured_radius,
#         ) = make_circular_field(
#             radial_profile
#         )

#         last_image = ax.imshow(
#             field,
#             extent=[
#                 -MAP_EXTENT_DEG,
#                 MAP_EXTENT_DEG,
#                 -MAP_EXTENT_DEG,
#                 MAP_EXTENT_DEG,
#             ],
#             origin="lower",
#             interpolation="bilinear",
#             vmin=GLOBAL_VMIN,
#             vmax=GLOBAL_VMAX,
#         )

#         # Radial guide circles
#         for guide_radius in [
#             5,
#             10,
#             15,
#         ]:
#             circle = plt.Circle(
#                 (0, 0),
#                 guide_radius,
#                 fill=False,
#                 linewidth=0.8,
#                 alpha=0.40,
#             )

#             ax.add_patch(
#                 circle
#             )

#             ax.text(
#                 0.55,
#                 guide_radius - 0.8,
#                 f"{guide_radius}°",
#                 fontsize=9,
#                 alpha=0.72,
#             )

#         # Mark the center / calibrated 0°
#         ax.scatter(
#             [0],
#             [0],
#             marker="+",
#             s=45,
#             linewidths=1.4,
#         )

#         ax.set_title(
#             title,
#             fontsize=14,
#             pad=12,
#         )

#         ax.set_xlabel(
#             "Scattering-angle coordinate (degrees)",
#             fontsize=10,
#         )

#         ax.set_ylabel(
#             "Scattering-angle coordinate (degrees)",
#             fontsize=10,
#         )

#         ax.set_xlim(
#             -MAP_EXTENT_DEG,
#             MAP_EXTENT_DEG,
#         )

#         ax.set_ylim(
#             -MAP_EXTENT_DEG,
#             MAP_EXTENT_DEG,
#         )

#         ax.set_aspect(
#             "equal"
#         )

#         ax.grid(
#             alpha=0.10
#         )

#         # Explicitly state measured radial coverage.
#         ax.text(
#             0.5,
#             -0.15,
#             (
#                 f"Measured radial coverage: "
#                 f"0° to {max_measured_radius:.0f}°"
#             ),
#             transform=ax.transAxes,
#             ha="center",
#             fontsize=9,
#         )


#     fig.suptitle(
#         (
#             f"Circular count-rate maps — nominal {zero_angle}° "
#             f"treated as true 0°\n"
#             f"±θ measurements are averaged at equal radial angle"
#         ),
#         fontsize=16,
#         y=0.96,
#     )

#     # Leave a dedicated right-side strip for the colorbar.
#     fig.subplots_adjust(
#         left=0.045,
#         right=0.89,
#         bottom=0.18,
#         top=0.82,
#         wspace=0.32,
#     )

#     colorbar_ax = fig.add_axes(
#         [0.915, 0.21, 0.018, 0.56]
#     )

#     colorbar = fig.colorbar(
#         last_image,
#         cax=colorbar_ax,
#     )

#     colorbar.set_label(
#         "Count rate after 4 V cut (counts/hour)",
#         fontsize=11,
#     )

#     finish_plot(
#         f"03_gradient_maps_zero_set_to_{zero_angle}deg.png"
#     )


# # ============================================================
# # 9. SAVE RADIAL PROFILES
# # ============================================================

# for zero_angle in ZERO_ANGLE_CHOICES_DEG:

#     table = calibrated_tables[
#         zero_angle
#     ]

#     for rate_column, title in rate_columns:

#         profile = make_radial_profile(
#             calibrated_table=table,
#             rate_column=rate_column,
#         )

#         safe_name = (
#             title
#             .lower()
#             .replace(" ", "_")
#         )

#         profile.to_csv(
#             OUTPUT_DIR
#             / (
#                 f"radial_profile_zero_{zero_angle}deg_"
#                 f"{safe_name}.csv"
#             ),
#             index=False,
#         )


# # ============================================================
# # 10. FINAL SUMMARY
# # ============================================================

# print(
#     "\n============================================================"
# )

# print(
#     "DONE"
# )

# print(
#     "============================================================"
# )

# print(
#     f"Outputs saved to:\n{OUTPUT_DIR}"
# )

# print(
#     "\nZero-angle choices tested:"
# )

# for zero_angle in ZERO_ANGLE_CHOICES_DEG:
#     print(
#         f"  nominal {zero_angle}° -> true 0°"
#     )

# print(
#     "\nMain figures:"
# )

# print(
#     "  01_all_fixed_angle_runs_count_rate_vs_time.png"
# )

# print(
#     "  02_zero_angle_calibration_count_rate_comparison.png"
# )

# for zero_angle in ZERO_ANGLE_CHOICES_DEG:
#     print(
#         f"  03_gradient_maps_zero_set_to_{zero_angle}deg.png"
#     )

# print(
#     "\nGradient-map rule:"
# )

# print(
#     "  calibrated angle = nominal angle - chosen zero angle"
# )

# print(
#     "  radial angle = |calibrated angle|"
# )

# print(
#     "  +theta and -theta measurements are averaged"
# )

# print(
#     "  interpolation is piecewise-linear between measured radial points"
# )

# print(
#     "  no theoretical function or extrapolation is used"
# )




# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ============================================================
# # Rutherford scattering analysis
# #
# # Includes:
# #   1) two incremental "per-degree" runs,
# #   2) two 5-degree runs,
# #   3) two 10-degree runs,
# #   4) two 15-degree runs,
# #   5) 4 V pulse-height analysis cut,
# #   6) Poisson count-rate uncertainties,
# #   7) Rutherford R(theta) ~ csc^4(theta/2) fit,
# #   8) fixed-angle validation,
# #   9) linearized Rutherford plot,
# #  10) time-stability plots,
# #  11) CSV output tables.
# #
# # IMPORTANT timing interpretation for the incremental runs:
# # - Run 1: discard the first 1 minute 3 seconds (63 s), which
# #   corresponds to the initial 0-degree period.
# # - After that, angle starts at 1 degree and increases by
# #   1 degree every 2 minutes.
# # - By default, the first 5 seconds of EVERY MINUTE are removed,
# #   matching the instruction literally.
# #
# # If "first 5 seconds" was intended ONLY after each angle change
# # (once every 2 minutes), set CUT_FIRST_5S_EVERY_MINUTE = False.
# # ============================================================


# # ============================================================
# # 0. USER SETTINGS / FILES
# # ============================================================

# BASE_DIR = Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS")

# PER_DEGREE_RUN1_FILE = BASE_DIR / "PER_DEGREE_RUN_ONE_30MINUTES_RUN_ARIYANTURNER_RFS.csv"
# PER_DEGREE_RUN2_FILE = BASE_DIR / "PER_DEGREE_RUN_ONE_HOUR_RUN_ARIYANTURNER_RFS_RUN2.csv"

# FIVE_RUN1_FILE = BASE_DIR / "5DEGREE_RUN_ONE_HOUR_ARIYANTURNER_RFS.csv"
# FIVE_RUN2_FILE = BASE_DIR / "5DEGREES_HOUR_RUN_ARIYANTURNER_RFS.csv"

# TEN_RUN1_FILE = BASE_DIR / "10DEGREE_OVERNIGHT_RUN_ARIYAN_TURNER_RFS_RUN2.csv"
# TEN_RUN2_FILE = BASE_DIR / "10DEGREES_OVERNIGHT_RUN_ARIYANTURNER_RFS.csv"

# FIFTEEN_RUN1_FILE = BASE_DIR / "15DEGREES_OVERNIGHT_ARIYANTURNER_RFS.csv"
# FIFTEEN_RUN2_FILE = BASE_DIR / "15DEGREES_WEEKEND_RUN_ARIYANTURNER_RFS.csv"


# THRESHOLD = 4.0  # volts

# # Incremental-run timing
# RUN1_ZERO_DEGREE_OFFSET_SECONDS = 63.0
# RUN2_ZERO_DEGREE_OFFSET_SECONDS = 0.0

# ANGLE_START_DEG = 1.0
# ANGLE_STEP_DEG = 1.0
# ANGLE_HOLD_SECONDS = 120.0

# ADJUSTMENT_CUT_SECONDS = 5.0
# CUT_FIRST_5S_EVERY_MINUTE = True

# # Because timestamps are pulse times rather than an independent
# # "run ended at exactly ..." clock, allow a nearly-complete last
# # 2-minute interval to count as complete.
# FULL_INTERVAL_TOLERANCE_SECONDS = 10.0

# # The manual recommends fitting a stable intermediate angular region
# # rather than blindly using the smallest and largest angles.
# # Inspect the plots and change these if your data suggest a better window.
# FIT_MIN_ANGLE_DEG = 5.0
# FIT_MAX_ANGLE_DEG = 15.0

# SHOW_PLOTS = True
# OUTPUT_DIR = BASE_DIR / "rutherford_analysis_outputs"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# # ============================================================
# # 1. BASIC HELPERS
# # ============================================================

# def require_columns(df, filename):
#     required = {"timestamp", "pulse_height_V"}
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(
#             f"{filename} is missing required column(s): {sorted(missing)}. "
#             f"Found columns: {list(df.columns)}"
#         )


# def load_run(filename):
#     filename = Path(filename)

#     if not filename.exists():
#         raise FileNotFoundError(f"Could not find file:\n{filename}")

#     df = pd.read_csv(filename)
#     require_columns(df, filename)

#     df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
#     df["pulse_height_V"] = pd.to_numeric(df["pulse_height_V"], errors="coerce")

#     df = (
#         df.dropna(subset=["timestamp", "pulse_height_V"])
#         .sort_values("timestamp")
#         .reset_index(drop=True)
#     )

#     if len(df) < 2:
#         raise ValueError(f"{filename} does not contain enough valid rows.")

#     return df


# def apply_voltage_cut(df):
#     return df[df["pulse_height_V"] >= THRESHOLD].copy()


# def rutherford_factor(theta_deg):
#     """
#     x(theta) = csc^4(theta/2) = sin(theta/2)^(-4)
#     """
#     theta_deg = np.asarray(theta_deg, dtype=float)

#     if np.any(theta_deg <= 0):
#         raise ValueError("Rutherford factor is singular at theta <= 0 degrees.")

#     theta_rad = np.deg2rad(theta_deg)
#     return np.sin(theta_rad / 2.0) ** (-4)


# def finish_plot(filename):
#     plt.tight_layout()
#     plt.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
#     if SHOW_PLOTS:
#         plt.show()
#     else:
#         plt.close()


# # ============================================================
# # 2. FIXED-ANGLE RUNS
# #
# # Keep EVERY run separate.
# # Do not silently combine the two 5°, two 10°, or two 15° runs.
# # ============================================================

# FIXED_RUN_CONFIG = [
#     {
#         "key": "5_run1",
#         "label": "5° run 1",
#         "angle_deg": 5.0,
#         "filename": FIVE_RUN1_FILE,
#         "stability_frequency": "1min",
#         "stability_ylabel": "Counts per minute",
#         "marker": "o",
#     },
#     {
#         "key": "5_run2",
#         "label": "5° run 2",
#         "angle_deg": 5.0,
#         "filename": FIVE_RUN2_FILE,
#         "stability_frequency": "1min",
#         "stability_ylabel": "Counts per minute",
#         "marker": "s",
#     },
#     {
#         "key": "10_run1",
#         "label": "10° run 1",
#         "angle_deg": 10.0,
#         "filename": TEN_RUN1_FILE,
#         "stability_frequency": "1h",
#         "stability_ylabel": "Counts per hour",
#         "marker": "o",
#     },
#     {
#         "key": "10_run2",
#         "label": "10° run 2",
#         "angle_deg": 10.0,
#         "filename": TEN_RUN2_FILE,
#         "stability_frequency": "1h",
#         "stability_ylabel": "Counts per hour",
#         "marker": "s",
#     },
#     {
#         "key": "15_run1",
#         "label": "15° overnight",
#         "angle_deg": 15.0,
#         "filename": FIFTEEN_RUN1_FILE,
#         "stability_frequency": "1h",
#         "stability_ylabel": "Counts per hour",
#         "marker": "o",
#     },
#     {
#         "key": "15_run2",
#         "label": "15° weekend",
#         "angle_deg": 15.0,
#         "filename": FIFTEEN_RUN2_FILE,
#         "stability_frequency": "1h",
#         "stability_ylabel": "Counts per hour",
#         "marker": "s",
#     },
# ]


# def rate_and_error(full_run, accepted_run):
#     """
#     R = N/T
#     sigma_R = sqrt(N)/T

#     T is inferred from the first and last recorded pulse timestamps.
#     """
#     duration_s = (
#         full_run["timestamp"].max() - full_run["timestamp"].min()
#     ).total_seconds()

#     if duration_s <= 0:
#         raise ValueError("Run duration is not positive.")

#     duration_h = duration_s / 3600.0
#     N = len(accepted_run)

#     rate_h = N / duration_h
#     sigma_rate_h = np.sqrt(N) / duration_h

#     return N, duration_h, rate_h, sigma_rate_h


# fixed_runs = {}
# fixed_summary_rows = []

# for cfg in FIXED_RUN_CONFIG:
#     full = load_run(cfg["filename"])
#     cut = apply_voltage_cut(full)

#     N, T_h, R_h, dR_h = rate_and_error(full, cut)

#     fixed_runs[cfg["key"]] = {
#         **cfg,
#         "full": full,
#         "cut": cut,
#         "N": N,
#         "T_h": T_h,
#         "R_h": R_h,
#         "dR_h": dR_h,
#     }

#     fixed_summary_rows.append(
#         {
#             "run": cfg["label"],
#             "angle_deg": cfg["angle_deg"],
#             "duration_h": T_h,
#             "accepted_counts": N,
#             "rate_per_h": R_h,
#             "sigma_rate_per_h": dR_h,
#         }
#     )

# fixed_summary = pd.DataFrame(fixed_summary_rows)

# print("\n============================================================")
# print("FIXED-ANGLE RUNS AFTER 4 V CUT")
# print("R = N/T,   sigma_R = sqrt(N)/T")
# print("============================================================")
# print(
#     fixed_summary.to_string(
#         index=False,
#         formatters={
#             "duration_h": "{:.6f}".format,
#             "rate_per_h": "{:.3f}".format,
#             "sigma_rate_per_h": "{:.3f}".format,
#         },
#     )
# )

# fixed_summary.to_csv(
#     OUTPUT_DIR / "fixed_angle_rate_summary.csv",
#     index=False,
# )


# # ============================================================
# # 3. INCREMENTAL / PER-DEGREE RUN ANALYSIS
# # ============================================================

# PER_DEGREE_CONFIG = [
#     {
#         "key": "incremental_run1",
#         "label": "Incremental run 1",
#         "filename": PER_DEGREE_RUN1_FILE,
#         "offset_seconds": RUN1_ZERO_DEGREE_OFFSET_SECONDS,
#         "marker": "o",
#     },
#     {
#         "key": "incremental_run2",
#         "label": "Incremental run 2",
#         "filename": PER_DEGREE_RUN2_FILE,
#         "offset_seconds": RUN2_ZERO_DEGREE_OFFSET_SECONDS,
#         "marker": "s",
#     },
# ]


# def stable_time_mask(elapsed_seconds):
#     """
#     Return True for times that are allowed into the incremental analysis.

#     Default interpretation:
#       remove [0, 5) s from every 60-second minute.

#     Alternative interpretation:
#       if CUT_FIRST_5S_EVERY_MINUTE = False,
#       remove only [0, 5) s after each 2-minute angle change.
#     """
#     elapsed_seconds = np.asarray(elapsed_seconds, dtype=float)

#     if CUT_FIRST_5S_EVERY_MINUTE:
#         second_within_minute = np.mod(elapsed_seconds, 60.0)
#         return second_within_minute >= ADJUSTMENT_CUT_SECONDS

#     second_within_angle = np.mod(elapsed_seconds, ANGLE_HOLD_SECONDS)
#     return second_within_angle >= ADJUSTMENT_CUT_SECONDS


# def live_seconds_per_full_angle_bin():
#     if CUT_FIRST_5S_EVERY_MINUTE:
#         n_minutes = ANGLE_HOLD_SECONDS / 60.0

#         if not np.isclose(n_minutes, round(n_minutes)):
#             raise ValueError(
#                 "ANGLE_HOLD_SECONDS must be an integer number of minutes "
#                 "when CUT_FIRST_5S_EVERY_MINUTE=True."
#             )

#         return ANGLE_HOLD_SECONDS - (
#             round(n_minutes) * ADJUSTMENT_CUT_SECONDS
#         )

#     return ANGLE_HOLD_SECONDS - ADJUSTMENT_CUT_SECONDS


# def analyze_incremental_run(filename, label, offset_seconds):
#     """
#     Convert timestamps into 2-minute angle bins.

#     For run 1:
#       analysis_start = first timestamp + 63 s

#     For run 2:
#       analysis_start = first timestamp

#     First post-offset angle = 1 degree.
#     Then angle increases by 1 degree every 120 s.

#     Only complete 2-minute angle intervals are retained.
#     """
#     df = load_run(filename)

#     raw_start = df["timestamp"].min()
#     raw_end = df["timestamp"].max()

#     analysis_start = raw_start + pd.to_timedelta(offset_seconds, unit="s")

#     usable_duration_s = (raw_end - analysis_start).total_seconds()

#     if usable_duration_s <= 0:
#         raise ValueError(
#             f"{label}: zero-angle offset is longer than the recorded run."
#         )

#     # Count near-complete final intervals as complete if the last pulse
#     # falls within the configured tolerance of the nominal boundary.
#     n_full_bins = int(
#         np.floor(
#             (usable_duration_s + FULL_INTERVAL_TOLERANCE_SECONDS)
#             / ANGLE_HOLD_SECONDS
#         )
#     )

#     if n_full_bins < 1:
#         raise ValueError(
#             f"{label}: no complete {ANGLE_HOLD_SECONDS:.0f}-second angle bins."
#         )

#     analysis_end = (
#         analysis_start
#         + pd.to_timedelta(n_full_bins * ANGLE_HOLD_SECONDS, unit="s")
#     )

#     work = df[
#         (df["timestamp"] >= analysis_start)
#         & (df["timestamp"] < analysis_end)
#     ].copy()

#     work["elapsed_s"] = (
#         work["timestamp"] - analysis_start
#     ).dt.total_seconds()

#     work["angle_index"] = np.floor(
#         work["elapsed_s"] / ANGLE_HOLD_SECONDS
#     ).astype(int)

#     work["angle_deg"] = (
#         ANGLE_START_DEG + ANGLE_STEP_DEG * work["angle_index"]
#     )

#     work["stable_time"] = stable_time_mask(work["elapsed_s"])
#     work["passes_voltage_cut"] = work["pulse_height_V"] >= THRESHOLD
#     work["accepted"] = work["stable_time"] & work["passes_voltage_cut"]

#     stable = work[work["stable_time"]].copy()
#     accepted = work[work["accepted"]].copy()

#     live_s = live_seconds_per_full_angle_bin()

#     rows = []

#     for i in range(n_full_bins):
#         angle = ANGLE_START_DEG + i * ANGLE_STEP_DEG

#         interval_start = (
#             analysis_start
#             + pd.to_timedelta(i * ANGLE_HOLD_SECONDS, unit="s")
#         )
#         interval_end = (
#             interval_start
#             + pd.to_timedelta(ANGLE_HOLD_SECONDS, unit="s")
#         )

#         stable_i = stable[stable["angle_index"] == i]
#         accepted_i = accepted[accepted["angle_index"] == i]

#         raw_stable_count = len(stable_i)
#         N = len(accepted_i)

#         live_h = live_s / 3600.0
#         R_h = N / live_h
#         dR_h = np.sqrt(N) / live_h

#         rows.append(
#             {
#                 "run": label,
#                 "angle_deg": angle,
#                 "interval_start": interval_start,
#                 "interval_end": interval_end,
#                 "scheduled_interval_s": ANGLE_HOLD_SECONDS,
#                 "live_time_s": live_s,
#                 "stable_raw_counts": raw_stable_count,
#                 "accepted_counts": N,
#                 "rate_per_h": R_h,
#                 "sigma_rate_per_h": dR_h,
#             }
#         )

#     summary = pd.DataFrame(rows)

#     metadata = {
#         "label": label,
#         "filename": str(filename),
#         "raw_start": raw_start,
#         "raw_end": raw_end,
#         "analysis_start": analysis_start,
#         "analysis_end": analysis_end,
#         "offset_seconds": offset_seconds,
#         "n_full_bins": n_full_bins,
#         "live_s_per_bin": live_s,
#     }

#     return df, work, accepted, summary, metadata


# incremental_runs = {}

# for cfg in PER_DEGREE_CONFIG:
#     full, assigned, accepted, summary, metadata = analyze_incremental_run(
#         filename=cfg["filename"],
#         label=cfg["label"],
#         offset_seconds=cfg["offset_seconds"],
#     )

#     incremental_runs[cfg["key"]] = {
#         **cfg,
#         "full": full,
#         "assigned": assigned,
#         "accepted": accepted,
#         "summary": summary,
#         "metadata": metadata,
#     }

#     summary.to_csv(
#         OUTPUT_DIR / f"{cfg['key']}_angle_summary.csv",
#         index=False,
#     )

#     print("\n============================================================")
#     print(cfg["label"].upper())
#     print("============================================================")
#     print(f"file:            {cfg['filename']}")
#     print(f"raw start:       {metadata['raw_start']}")
#     print(f"analysis start:  {metadata['analysis_start']}")
#     print(f"analysis end:    {metadata['analysis_end']}")
#     print(f"offset removed:  {metadata['offset_seconds']:.1f} s")
#     print(f"full angle bins: {metadata['n_full_bins']}")
#     print(f"live time/bin:   {metadata['live_s_per_bin']:.1f} s")
#     print()
#     print(
#         summary[
#             [
#                 "angle_deg",
#                 "accepted_counts",
#                 "live_time_s",
#                 "rate_per_h",
#                 "sigma_rate_per_h",
#             ]
#         ].to_string(
#             index=False,
#             formatters={
#                 "angle_deg": "{:.0f}".format,
#                 "live_time_s": "{:.1f}".format,
#                 "rate_per_h": "{:.2f}".format,
#                 "sigma_rate_per_h": "{:.2f}".format,
#             },
#         )
#     )


# # ============================================================
# # 4. COMBINE THE TWO INCREMENTAL RUNS ANGLE-BY-ANGLE
# #
# # This is the correct way to combine repeated counting runs:
# #     N_total = sum(N_i)
# #     T_total = sum(T_i)
# #     R = N_total / T_total
# #     sigma_R = sqrt(N_total) / T_total
# #
# # The individual runs are still kept and plotted separately.
# # ============================================================

# all_incremental = pd.concat(
#     [
#         incremental_runs["incremental_run1"]["summary"],
#         incremental_runs["incremental_run2"]["summary"],
#     ],
#     ignore_index=True,
# )

# combined_incremental = (
#     all_incremental.groupby("angle_deg", as_index=False)
#     .agg(
#         accepted_counts=("accepted_counts", "sum"),
#         stable_raw_counts=("stable_raw_counts", "sum"),
#         live_time_s=("live_time_s", "sum"),
#         n_runs=("run", "nunique"),
#     )
#     .sort_values("angle_deg")
#     .reset_index(drop=True)
# )

# combined_incremental["live_time_h"] = (
#     combined_incremental["live_time_s"] / 3600.0
# )

# combined_incremental["rate_per_h"] = (
#     combined_incremental["accepted_counts"]
#     / combined_incremental["live_time_h"]
# )

# combined_incremental["sigma_rate_per_h"] = (
#     np.sqrt(combined_incremental["accepted_counts"])
#     / combined_incremental["live_time_h"]
# )

# combined_incremental["rutherford_x"] = rutherford_factor(
#     combined_incremental["angle_deg"].to_numpy()
# )

# combined_incremental.to_csv(
#     OUTPUT_DIR / "incremental_combined_angle_summary.csv",
#     index=False,
# )

# print("\n============================================================")
# print("COMBINED INCREMENTAL DATA")
# print("============================================================")
# print(
#     combined_incremental[
#         [
#             "angle_deg",
#             "n_runs",
#             "accepted_counts",
#             "live_time_s",
#             "rate_per_h",
#             "sigma_rate_per_h",
#         ]
#     ].to_string(
#         index=False,
#         formatters={
#             "angle_deg": "{:.0f}".format,
#             "live_time_s": "{:.1f}".format,
#             "rate_per_h": "{:.2f}".format,
#             "sigma_rate_per_h": "{:.2f}".format,
#         },
#     )
# )


# # ============================================================
# # 5. WEIGHTED LINEARIZED RUTHERFORD FIT
# #
# # Rutherford:
# #     R(theta) = A csc^4(theta/2)
# #
# # Define:
# #     x = csc^4(theta/2)
# #     y = R
# #
# # Then:
# #     y = A x
# #
# # We fit A with weighted least squares through the origin.
# # ============================================================

# fit_data = combined_incremental[
#     (combined_incremental["angle_deg"] >= FIT_MIN_ANGLE_DEG)
#     & (combined_incremental["angle_deg"] <= FIT_MAX_ANGLE_DEG)
#     & (combined_incremental["accepted_counts"] > 0)
#     & (combined_incremental["sigma_rate_per_h"] > 0)
# ].copy()

# if len(fit_data) < 2:
#     raise ValueError(
#         "Not enough nonzero incremental points in the selected fit window. "
#         "Change FIT_MIN_ANGLE_DEG / FIT_MAX_ANGLE_DEG."
#     )

# x = fit_data["rutherford_x"].to_numpy(dtype=float)
# y = fit_data["rate_per_h"].to_numpy(dtype=float)
# sigma = fit_data["sigma_rate_per_h"].to_numpy(dtype=float)

# weights = 1.0 / sigma**2

# A_fit = np.sum(weights * x * y) / np.sum(weights * x**2)
# dA_fit = 1.0 / np.sqrt(np.sum(weights * x**2))

# y_fit = A_fit * x
# residual_sigma = (y - y_fit) / sigma

# chi2 = np.sum(residual_sigma**2)
# dof = len(fit_data) - 1
# reduced_chi2 = chi2 / dof if dof > 0 else np.nan

# print("\n============================================================")
# print("RUTHERFORD FIT TO COMBINED INCREMENTAL DATA")
# print("============================================================")
# print(
#     f"Fit window: {FIT_MIN_ANGLE_DEG:g}° to "
#     f"{FIT_MAX_ANGLE_DEG:g}°"
# )
# print("Model: R(theta) = A * csc^4(theta/2)")
# print(f"A = {A_fit:.8g} +/- {dA_fit:.3g} counts/hour")
# print(f"chi^2 = {chi2:.3f}")
# print(f"dof = {dof}")
# print(f"reduced chi^2 = {reduced_chi2:.3f}")


# # Add prediction + residual columns to combined table
# combined_incremental["theory_rate_per_h"] = (
#     A_fit * combined_incremental["rutherford_x"]
# )

# combined_incremental["percent_difference"] = (
#     100.0
#     * (
#         combined_incremental["rate_per_h"]
#         - combined_incremental["theory_rate_per_h"]
#     )
#     / combined_incremental["theory_rate_per_h"]
# )

# combined_incremental["pull_sigma"] = np.where(
#     combined_incremental["sigma_rate_per_h"] > 0,
#     (
#         combined_incremental["rate_per_h"]
#         - combined_incremental["theory_rate_per_h"]
#     )
#     / combined_incremental["sigma_rate_per_h"],
#     np.nan,
# )

# combined_incremental.to_csv(
#     OUTPUT_DIR / "incremental_combined_with_theory.csv",
#     index=False,
# )


# # ============================================================
# # 6. COMPARE EVERY FIXED-ANGLE RUN WITH THE INCREMENTAL FIT
# # ============================================================

# fixed_validation_rows = []

# for row in fixed_summary.itertuples(index=False):
#     x_theta = float(rutherford_factor(row.angle_deg))
#     theory_rate = A_fit * x_theta

#     percent_diff = 100.0 * (
#         row.rate_per_h - theory_rate
#     ) / theory_rate

#     pull = (
#         (row.rate_per_h - theory_rate) / row.sigma_rate_per_h
#         if row.sigma_rate_per_h > 0
#         else np.nan
#     )

#     # Relative Z consistency check only.
#     #
#     # Because R ~ Z^2 at fixed experimental normalization:
#     #     Z_relative = 79 * sqrt(R_measured / R_Au_fit)
#     #
#     # This is NOT an independent absolute measurement of Z,
#     # because A was itself obtained from gold-scattering data.
#     Z_relative = (
#         79.0 * np.sqrt(row.rate_per_h / theory_rate)
#         if row.rate_per_h > 0 and theory_rate > 0
#         else np.nan
#     )

#     # Propagate rate + fitted-normalization statistical uncertainty.
#     if row.rate_per_h > 0 and A_fit > 0:
#         dZ_relative = Z_relative * 0.5 * np.sqrt(
#             (row.sigma_rate_per_h / row.rate_per_h) ** 2
#             + (dA_fit / A_fit) ** 2
#         )
#     else:
#         dZ_relative = np.nan

#     fixed_validation_rows.append(
#         {
#             "run": row.run,
#             "angle_deg": row.angle_deg,
#             "measured_rate_per_h": row.rate_per_h,
#             "sigma_rate_per_h": row.sigma_rate_per_h,
#             "theory_rate_per_h": theory_rate,
#             "percent_difference": percent_diff,
#             "pull_sigma": pull,
#             "relative_Z_check": Z_relative,
#             "sigma_relative_Z": dZ_relative,
#         }
#     )

# fixed_validation = pd.DataFrame(fixed_validation_rows)

# print("\n============================================================")
# print("FIXED-ANGLE RUNS VS INCREMENTAL RUTHERFORD FIT")
# print("============================================================")
# print(
#     fixed_validation.to_string(
#         index=False,
#         formatters={
#             "angle_deg": "{:.0f}".format,
#             "measured_rate_per_h": "{:.3f}".format,
#             "sigma_rate_per_h": "{:.3f}".format,
#             "theory_rate_per_h": "{:.3f}".format,
#             "percent_difference": "{:+.2f}".format,
#             "pull_sigma": "{:+.2f}".format,
#             "relative_Z_check": "{:.2f}".format,
#             "sigma_relative_Z": "{:.2f}".format,
#         },
#     )
# )

# fixed_validation.to_csv(
#     OUTPUT_DIR / "fixed_angle_vs_incremental_fit.csv",
#     index=False,
# )


# # ============================================================
# # 7. PLOT — INCREMENTAL COUNT RATE VS ANGLE
# # ============================================================

# plt.figure(figsize=(9.5, 6.0))

# for key in ["incremental_run1", "incremental_run2"]:
#     run = incremental_runs[key]
#     s = run["summary"]

#     plt.errorbar(
#         s["angle_deg"],
#         s["rate_per_h"],
#         yerr=s["sigma_rate_per_h"],
#         fmt=run["marker"] + "-",
#         markersize=4.5,
#         linewidth=1.1,
#         capsize=2,
#         alpha=0.72,
#         label=run["label"],
#     )

# plt.errorbar(
#     combined_incremental["angle_deg"],
#     combined_incremental["rate_per_h"],
#     yerr=combined_incremental["sigma_rate_per_h"],
#     fmt="o",
#     markersize=6.5,
#     capsize=3,
#     label="Combined incremental rate",
# )

# theta_min = max(
#     0.5,
#     float(combined_incremental["angle_deg"].min()),
# )
# theta_max = float(combined_incremental["angle_deg"].max())

# theta_curve = np.linspace(theta_min, theta_max, 700)
# theory_curve = A_fit * rutherford_factor(theta_curve)

# plt.plot(
#     theta_curve,
#     theory_curve,
#     linewidth=2,
#     label=(
#         r"Weighted Rutherford fit: "
#         r"$R=A\,\csc^4(\theta/2)$"
#     ),
# )

# plt.axvspan(
#     FIT_MIN_ANGLE_DEG,
#     FIT_MAX_ANGLE_DEG,
#     alpha=0.10,
#     label="Fit window",
# )

# plt.yscale("log")
# plt.xlabel("Scattering angle θ (degrees)")
# plt.ylabel("Count rate after timing + 4 V cuts (counts/hour)")
# plt.title("Incremental Rutherford scattering run")
# plt.grid(alpha=0.25, which="both")
# plt.legend()
# finish_plot("incremental_rate_vs_angle.png")


# # ============================================================
# # 8. PLOT — LINEARIZED RUTHERFORD RELATION
# # ============================================================

# plt.figure(figsize=(9.0, 6.0))

# fit_plot_x = fit_data["rutherford_x"].to_numpy()
# fit_plot_y = fit_data["rate_per_h"].to_numpy()
# fit_plot_dy = fit_data["sigma_rate_per_h"].to_numpy()

# plt.errorbar(
#     fit_plot_x,
#     fit_plot_y,
#     yerr=fit_plot_dy,
#     fmt="o",
#     markersize=6,
#     capsize=3,
#     label="Combined incremental data in fit window",
# )

# x_line = np.linspace(0, fit_plot_x.max() * 1.05, 500)

# plt.plot(
#     x_line,
#     A_fit * x_line,
#     linewidth=2,
#     label=(
#         rf"$R=Ax$, "
#         rf"$A={A_fit:.4g}\pm{dA_fit:.2g}$ counts/hour"
#     ),
# )

# plt.xlabel(r"$x=\csc^4(\theta/2)$")
# plt.ylabel("Count rate (counts/hour)")
# plt.title("Linearized Rutherford-scattering fit")
# plt.grid(alpha=0.25)
# plt.legend()
# finish_plot("incremental_linearized_rutherford_fit.png")


# # ============================================================
# # 9. PLOT — INCREMENTAL FIT RESIDUALS
# # ============================================================

# plot_residual = combined_incremental[
#     combined_incremental["sigma_rate_per_h"] > 0
# ].copy()

# plt.figure(figsize=(9.0, 5.0))

# plt.errorbar(
#     plot_residual["angle_deg"],
#     plot_residual["percent_difference"],
#     fmt="o",
#     markersize=5.5,
# )

# plt.axhline(0, linestyle="--", linewidth=1.5)
# plt.axvspan(
#     FIT_MIN_ANGLE_DEG,
#     FIT_MAX_ANGLE_DEG,
#     alpha=0.10,
# )

# plt.xlabel("Scattering angle θ (degrees)")
# plt.ylabel("(Measured - Rutherford) / Rutherford (%)")
# plt.title("Incremental-run deviation from Rutherford fit")
# plt.grid(alpha=0.25)
# finish_plot("incremental_percent_residuals.png")


# # ============================================================
# # 10. PLOT — FIXED-ANGLE RUNS ON THE SAME RUTHERFORD CURVE
# #
# # Every repeated run stays separate.
# # ============================================================

# plt.figure(figsize=(9.5, 6.0))

# theta_curve_fixed = np.linspace(
#     min(4.5, fixed_summary["angle_deg"].min()),
#     max(16.0, fixed_summary["angle_deg"].max() + 1),
#     500,
# )

# plt.plot(
#     theta_curve_fixed,
#     A_fit * rutherford_factor(theta_curve_fixed),
#     linewidth=2,
#     label="Rutherford prediction from incremental fit",
# )

# for cfg in FIXED_RUN_CONFIG:
#     run_row = fixed_summary[fixed_summary["run"] == cfg["label"]].iloc[0]

#     plt.errorbar(
#         run_row["angle_deg"],
#         run_row["rate_per_h"],
#         yerr=run_row["sigma_rate_per_h"],
#         fmt=cfg["marker"],
#         markersize=8,
#         capsize=4,
#         label=cfg["label"],
#     )

# plt.yscale("log")
# plt.xlabel("Scattering angle θ (degrees)")
# plt.ylabel("Count rate (counts/hour)")
# plt.title("Fixed-angle runs vs Rutherford relation")
# plt.grid(alpha=0.25, which="both")
# plt.legend()
# finish_plot("fixed_angle_runs_vs_rutherford.png")


# # ============================================================
# # 11. PLOT — FIXED-RUN RELATIVE Z CONSISTENCY
# #
# # This is intentionally labeled "relative".
# # It is not an independent absolute determination of gold Z.
# # ============================================================

# plt.figure(figsize=(9.0, 5.5))

# x_positions = np.arange(len(fixed_validation))

# plt.errorbar(
#     x_positions,
#     fixed_validation["relative_Z_check"],
#     yerr=fixed_validation["sigma_relative_Z"],
#     fmt="o",
#     markersize=7,
#     capsize=4,
# )

# plt.axhline(
#     79.0,
#     linestyle="--",
#     linewidth=1.5,
#     label="Gold reference: Z = 79",
# )

# plt.xticks(
#     x_positions,
#     fixed_validation["run"],
#     rotation=30,
#     ha="right",
# )

# plt.ylabel("Relative inferred Z")
# plt.title("Relative gold-Z consistency of fixed-angle runs")
# plt.grid(alpha=0.25)
# plt.legend()
# finish_plot("fixed_angle_relative_Z_check.png")


# # ============================================================
# # 12. FIXED-RUN PULSE-HEIGHT SPECTRA
# # ============================================================

# bins = np.linspace(
#     THRESHOLD,
#     max(
#         THRESHOLD + 0.2,
#         max(run["cut"]["pulse_height_V"].max() for run in fixed_runs.values()),
#     ),
#     70,
# )

# plt.figure(figsize=(10.0, 6.0))

# for cfg in FIXED_RUN_CONFIG:
#     run = fixed_runs[cfg["key"]]

#     counts, edges = np.histogram(
#         run["cut"]["pulse_height_V"],
#         bins=bins,
#     )

#     centers = 0.5 * (edges[:-1] + edges[1:])
#     rate_per_bin = counts / run["T_h"]

#     plt.step(
#         centers,
#         rate_per_bin,
#         where="mid",
#         linewidth=1.4,
#         label=cfg["label"],
#     )

# plt.axvline(
#     THRESHOLD,
#     linestyle="--",
#     linewidth=1.3,
#     label="4 V analysis cut",
# )

# plt.xlabel("Pulse height (V)")
# plt.ylabel("Count rate per bin (counts/hour)")
# plt.title("Fixed-angle pulse-height spectra after 4 V cut")
# plt.grid(alpha=0.25)
# plt.legend()
# finish_plot("fixed_angle_pulse_height_spectra.png")


# # ============================================================
# # 13. TIME-STABILITY HELPER FOR FIXED-ANGLE RUNS
# #
# # Bins are aligned to the actual first recorded pulse.
# # A clearly incomplete final bin is dropped.
# # ============================================================

# def plot_rate_stability(full_run, cut_run, frequency, title, ylabel, filename):
#     start_time = full_run["timestamp"].min()

#     counts = (
#         cut_run.set_index("timestamp")["pulse_height_V"]
#         .resample(frequency, origin=start_time)
#         .count()
#     )

#     counts = counts[counts > 0]

#     if len(counts) >= 3:
#         median_previous = counts.iloc[:-1].median()

#         if (
#             median_previous > 0
#             and counts.iloc[-1] < 0.70 * median_previous
#         ):
#             counts = counts.iloc[:-1]

#     if len(counts) == 0:
#         print(f"Skipping stability plot with no accepted bins: {title}")
#         return

#     errors = np.sqrt(counts.to_numpy())

#     plt.figure(figsize=(9.5, 5.5))

#     plt.errorbar(
#         counts.index,
#         counts.to_numpy(),
#         yerr=errors,
#         fmt="o-",
#         markersize=4,
#         linewidth=1.2,
#         capsize=2,
#         label=r"Measured counts $\pm\sqrt{N}$",
#     )

#     mean_counts = counts.mean()

#     plt.axhline(
#         mean_counts,
#         linestyle="--",
#         linewidth=1.4,
#         label=f"Mean = {mean_counts:.1f}",
#     )

#     plt.xlabel("Time")
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.grid(alpha=0.25)
#     plt.legend()
#     plt.xticks(rotation=35)
#     finish_plot(filename)


# for cfg in FIXED_RUN_CONFIG:
#     run = fixed_runs[cfg["key"]]

#     plot_rate_stability(
#         full_run=run["full"],
#         cut_run=run["cut"],
#         frequency=cfg["stability_frequency"],
#         title=f"{cfg['label']}: count-rate stability after 4 V cut",
#         ylabel=cfg["stability_ylabel"],
#         filename=f"{cfg['key']}_stability.png",
#     )


# # ============================================================
# # 14. INCREMENTAL-RUN TIMING DIAGNOSTIC
# #
# # This plot is useful for checking that the offset and angle
# # assignment are doing what you intended.
# # ============================================================

# for key in ["incremental_run1", "incremental_run2"]:
#     run = incremental_runs[key]
#     assigned = run["assigned"]

#     plt.figure(figsize=(10.0, 4.8))

#     accepted = assigned[assigned["accepted"]]
#     rejected_timing = assigned[
#         (~assigned["stable_time"])
#         & assigned["passes_voltage_cut"]
#     ]

#     plt.scatter(
#         accepted["timestamp"],
#         accepted["angle_deg"],
#         s=8,
#         alpha=0.65,
#         label="Accepted >= 4 V pulses",
#     )

#     if len(rejected_timing) > 0:
#         plt.scatter(
#             rejected_timing["timestamp"],
#             rejected_timing["angle_deg"],
#             s=8,
#             alpha=0.45,
#             label=">= 4 V but removed during adjustment window",
#         )

#     plt.xlabel("Time")
#     plt.ylabel("Assigned scattering angle (degrees)")
#     plt.title(
#         f"{run['label']}: timestamp-to-angle assignment diagnostic"
#     )
#     plt.grid(alpha=0.25)
#     plt.legend()
#     plt.xticks(rotation=30)

#     finish_plot(f"{key}_timing_assignment_diagnostic.png")


# # ============================================================
# # 15. FINAL SUMMARY
# # ============================================================

# print("\n============================================================")
# print("DONE")
# print("============================================================")
# print(f"Results written to:\n{OUTPUT_DIR}")
# print()
# print("Important timing settings:")
# print(f"  Run 1 zero-degree offset = {RUN1_ZERO_DEGREE_OFFSET_SECONDS:.1f} s")
# print(f"  Run 2 zero-degree offset = {RUN2_ZERO_DEGREE_OFFSET_SECONDS:.1f} s")
# print(f"  First post-offset angle  = {ANGLE_START_DEG:.1f} degrees")
# print(f"  Angle step               = {ANGLE_STEP_DEG:.1f} degree")
# print(f"  Time per angle           = {ANGLE_HOLD_SECONDS:.1f} s")
# print(f"  Adjustment cut           = {ADJUSTMENT_CUT_SECONDS:.1f} s")

# if CUT_FIRST_5S_EVERY_MINUTE:
#     print("  Cut mode                 = first 5 s of EVERY minute")
# else:
#     print("  Cut mode                 = first 5 s after each 2-minute angle change")

# print(
#     f"  Live time per full angle = "
#     f"{live_seconds_per_full_angle_bin():.1f} s"
# )
# print(
#     f"  Rutherford fit window    = "
#     f"{FIT_MIN_ANGLE_DEG:g}° to {FIT_MAX_ANGLE_DEG:g}°"
# )
