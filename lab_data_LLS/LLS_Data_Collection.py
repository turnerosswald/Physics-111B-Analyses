import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# EXPERIMENT IX: JOHNSON NOISE — FINAL PLOTTING SCRIPT
# ============================================================
# What this script does:
#   1. Loads all complete runs for BOTH resistor data sets.
#   2. Computes the noise of EACH run as
#          sqrt(<(R - <R>)^2>)
#      and then averages those per-run values.
#   3. Calculates the SR830 ENBW from time constant + rolloff.
#   4. Includes SR830 input noise in the theory comparison:
#          V_total^2 = V_Johnson^2 + V_input^2
#   5. Makes simple Experiment IX plots:
#        A. Raw runs for each condition
#        B. Sample-rate comparison at fixed tau = 30 ms
#        C. Noise vs ENBW (main checkpoint plot)
#        D. Noise vs time constant
#   6. Saves a CSV summary table.
#
# The exact Experiment IX noise statistic used by the lab manual is
#          <dR^2>^(1/2)
# i.e. the RMS fluctuation about <R>.  (Not <dR^(1/2)>.)
#
# IMPORTANT:
#   - The sample-rate comparison is a DATA-ACQUISITION check.
#     Changing 64 -> 128 -> 256 Hz does NOT change the SR830 ENBW
#     when the time constant and rolloff are unchanged.
#
#   - The bandwidth/time-constant comparison uses ONLY the 64 Hz
#     runs so that sample rate is held fixed while tau changes.
#
#   - We do NOT use the point-by-point averaged waveform's RMS as
#     the experimental noise because averaging waveforms suppresses
#     random noise.
#
#   - The lab manual asks you to think about a possible sqrt(2)
#     effect because the SR830 output R is a magnitude. This script
#     leaves the measured data uncorrected so the raw measurement can
#     be compared transparently to theory. See notes at the bottom.
# ============================================================


# ------------------------------------------------------------
# Physical / instrument constants
# ------------------------------------------------------------
K_B = 1.380649e-23  # Boltzmann constant, J/K
TEMPERATURE_K = 295.0  # change if you recorded room temperature

# LLS manual / SR830 approximate input noise density
SR830_INPUT_NOISE_DENSITY = 5.0e-9  # V_rms / sqrt(Hz)

# Experiment IX uses the highest low-pass rolloff
ROLLOFF_DB_PER_OCT = 24

# SR830 ENBW = factor/tau
ENBW_FACTORS = {
    6: 1.0 / 4.0,
    12: 1.0 / 8.0,
    18: 3.0 / 32.0,
    24: 5.0 / 64.0,
}


# ------------------------------------------------------------
# Resistance values
# ------------------------------------------------------------
# The TEST5 files are explicitly labeled "10kOhm".
# Your earlier script used 9.90 kOhm, so that measured value is kept.
R_10K_OHM = 9.90e3

# Experiment IX starts with a 100 kOhm resistor before repeating with
# the ~10 kOhm resistor. Replace this with your measured DMM value if
# you wrote down something more precise (e.g. 99.7e3).
R_100K_OHM = 100.0e3


# ------------------------------------------------------------
# Data locations
# ------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\turno\OneDrive\Documents\lab_data_LLS")
OUTPUT_DIR = BASE_DIR / "EXP9_final_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Final Experiment IX data set
# ------------------------------------------------------------
# The two older files named EXP9JohnsonNoise100Hz... are left out of
# the FINAL analysis because the TURNERARIYAN... files form the full,
# matched set across sample rate/time constant/resistance.
# You can add them back as preliminary runs if needed.
# ------------------------------------------------------------
TESTS = {
    # ======================== 100 kOhm ========================
    "100k | 30 ms | 64 Hz": {
        "file": BASE_DIR / "TURNERARIYANEXP9TEST2(T 30.0E-3 ).xls",
        "resistor_group": "100 kOhm",
        "resistance_ohm": R_100K_OHM,
        "sample_rate_hz": 64,
        "span_s": 10,
        "time_constant_s": 30e-3,
    },
    "100k | 30 ms | 128 Hz": {
        "file": BASE_DIR / "TURNERARIYANEXP9TEST1(T 30.0E-3 ).xls",
        "resistor_group": "100 kOhm",
        "resistance_ohm": R_100K_OHM,
        "sample_rate_hz": 128,
        "span_s": 5,
        "time_constant_s": 30e-3,
    },
    "100k | 30 ms | 256 Hz": {
        "file": BASE_DIR / "TURNERARIYANEXP9TEST3(T 30.0E-3 ).xls",
        "resistor_group": "100 kOhm",
        "resistance_ohm": R_100K_OHM,
        "sample_rate_hz": 256,
        "span_s": 3,
        "time_constant_s": 30e-3,
    },
    "100k | 100 ms | 64 Hz": {
        "file": BASE_DIR / "TURNERARIYANEXP9TEST4TIMECONST_100mS(T 100.0E-3 ).xls",
        "resistor_group": "100 kOhm",
        "resistance_ohm": R_100K_OHM,
        "sample_rate_hz": 64,
        "span_s": 10,
        "time_constant_s": 100e-3,
    },
    "100k | 300 ms | 64 Hz": {
        "file": BASE_DIR / "TURNERARIYANEXP9TEST4TIMECONST_300mS(T 300.0E-3 ).xls",
        "resistor_group": "100 kOhm",
        "resistance_ohm": R_100K_OHM,
        "sample_rate_hz": 64,
        "span_s": 10,
        "time_constant_s": 300e-3,
    },
    "100k | 1 s | 64 Hz": {
        "file": BASE_DIR / "TURNERARIYANEXP9TEST4TIMECONST_1S(T 1.0E+0 ).xls",
        "resistor_group": "100 kOhm",
        "resistance_ohm": R_100K_OHM,
        "sample_rate_hz": 64,
        "span_s": 10,
        "time_constant_s": 1.0,
    },
    "100k | 3 s | 64 Hz": {
        "file": BASE_DIR / "TURNERARIYANEXP9TEST4TIMECONST_3S(T 3.0E+0 ).xls",
        "resistor_group": "100 kOhm",
        "resistance_ohm": R_100K_OHM,
        "sample_rate_hz": 64,
        "span_s": 10,
        "time_constant_s": 3.0,
    },

    # ========================= 10 kOhm ========================
    "10k | 30 ms | 64 Hz": {
        "file": BASE_DIR / "TURNERARIYANEXP9TEST5TIMECONST_30ms_10kOhm_64Hz_10s(T = 30.0E-3 ).xls",
        "resistor_group": "10 kOhm",
        "resistance_ohm": R_10K_OHM,
        "sample_rate_hz": 64,
        "span_s": 10,
        "time_constant_s": 30e-3,
    },
    "10k | 30 ms | 128 Hz": {
        "file": BASE_DIR / "TURNERARIYANEXP9TEST5TIMECONST_30ms_10kOhm_128Hz_5s(T = 30.0E-3 ).xls",
        "resistor_group": "10 kOhm",
        "resistance_ohm": R_10K_OHM,
        "sample_rate_hz": 128,
        "span_s": 5,
        "time_constant_s": 30e-3,
    },
    "10k | 30 ms | 256 Hz": {
        "file": BASE_DIR / "TURNERARIYANEXP9TEST5TIMECONST_30ms_10kOhm_256Hz_3s(T = 30.0E-3 ).xls",
        "resistor_group": "10 kOhm",
        "resistance_ohm": R_10K_OHM,
        "sample_rate_hz": 256,
        "span_s": 3,
        "time_constant_s": 30e-3,
    },
    "10k | 100 ms | 64 Hz": {
        "file": BASE_DIR / "TURNERARIYANEXP9TEST5TIMECONST_100ms_10kOhm_64Hz_10s(T = 100.0E-3 ).xls",
        "resistor_group": "10 kOhm",
        "resistance_ohm": R_10K_OHM,
        "sample_rate_hz": 64,
        "span_s": 10,
        "time_constant_s": 100e-3,
    },
    "10k | 300 ms | 64 Hz": {
        "file": BASE_DIR / "TURNERARIYANEXP9TEST5TIMECONST_300ms_10kOhm_64Hz_10s(T = 300.0E-3 ).xls",
        "resistor_group": "10 kOhm",
        "resistance_ohm": R_10K_OHM,
        "sample_rate_hz": 64,
        "span_s": 10,
        "time_constant_s": 300e-3,
    },
    "10k | 1 s | 64 Hz": {
        "file": BASE_DIR / "TURNERARIYANEXP9TEST5TIMECONST_1s_10kOhm_64Hz_10s(T = 1.0E+0 ).xls",
        "resistor_group": "10 kOhm",
        "resistance_ohm": R_10K_OHM,
        "sample_rate_hz": 64,
        "span_s": 10,
        "time_constant_s": 1.0,
    },
    "10k | 3 s | 64 Hz": {
        "file": BASE_DIR / "TURNERARIYANEXP9TEST5TIMECONST_3s_10kOhm_64Hz_10s(T 3.0E+0 ).xls",
        "resistor_group": "10 kOhm",
        "resistance_ohm": R_10K_OHM,
        "sample_rate_hz": 64,
        "span_s": 10,
        "time_constant_s": 3.0,
    },
}


# ------------------------------------------------------------
# Plot-control switches
# ------------------------------------------------------------
MAKE_RAW_RUN_PLOTS = True
MAKE_PER_RUN_DIAGNOSTIC_PLOTS = False  # turn on if you want 14 extra plots
SHOW_PLOTS = True


# ============================================================
# Helper functions
# ============================================================
def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def format_tau(time_constant_s: float) -> str:
    """Pretty label for the SR830 time constant."""
    if time_constant_s < 1.0:
        return f"{time_constant_s * 1e3:g} ms"
    return f"{time_constant_s:g} s"


def enbw_hz(time_constant_s: float, rolloff_db_per_oct: int = ROLLOFF_DB_PER_OCT) -> float:
    if rolloff_db_per_oct not in ENBW_FACTORS:
        raise ValueError(f"Unsupported rolloff: {rolloff_db_per_oct} dB/oct")
    return ENBW_FACTORS[rolloff_db_per_oct] / time_constant_s


def johnson_noise_rms(resistance_ohm: float, bandwidth_hz: float, temperature_k: float = TEMPERATURE_K) -> float:
    return np.sqrt(4.0 * K_B * temperature_k * resistance_ohm * bandwidth_hz)


def sr830_input_noise_rms(bandwidth_hz: float) -> float:
    return SR830_INPUT_NOISE_DENSITY * np.sqrt(bandwidth_hz)


def total_theory_noise_rms(resistance_ohm: float, bandwidth_hz: float) -> float:
    v_j = johnson_noise_rms(resistance_ohm, bandwidth_hz)
    v_in = sr830_input_noise_rms(bandwidth_hz)
    return np.sqrt(v_j**2 + v_in**2)


def load_numeric_vector(file_path: Path) -> np.ndarray:
    """Load the lab's numeric .xls text export robustly."""
    try:
        raw = np.loadtxt(file_path)
    except ValueError:
        raw = np.genfromtxt(file_path)

    data = np.asarray(raw, dtype=float).reshape(-1)
    data = data[np.isfinite(data)]

    if data.size == 0:
        raise ValueError(f"{file_path.name}: no numeric data found")

    return data


def load_complete_runs(file_path: Path, sample_rate_hz: float, span_s: float):
    data = load_numeric_vector(file_path)

    points_per_run = int(round(sample_rate_hz * span_s))
    n_runs = len(data) // points_per_run
    leftover = len(data) % points_per_run

    if n_runs == 0:
        raise ValueError(
            f"{file_path.name}: {len(data)} points is not enough for one "
            f"{span_s:g} s run at {sample_rate_hz:g} Hz ({points_per_run} points)."
        )

    if leftover:
        print(
            f"  WARNING: {file_path.name}: ignoring {leftover} leftover "
            "point(s) that do not form a complete run."
        )
        data = data[: n_runs * points_per_run]

    runs = data.reshape(n_runs, points_per_run)
    time_s = np.arange(points_per_run) / sample_rate_hz
    return runs, time_s, leftover


def save_and_maybe_show(filename: str):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# ============================================================
# Analyze every data file
# ============================================================
summary_rows = []
run_cache = {}

for test_name, info in TESTS.items():
    file_path = Path(info["file"])

    if not file_path.exists():
        print(f"\nSKIPPING {test_name}: file not found")
        print(f"  {file_path}")
        continue

    sample_rate = float(info["sample_rate_hz"])
    span_s = float(info["span_s"])
    tau = float(info["time_constant_s"])
    resistance = float(info["resistance_ohm"])

    print(f"\n{test_name}")
    runs, time_s, leftover = load_complete_runs(file_path, sample_rate, span_s)
    n_runs = runs.shape[0]

    # Per-run statistics
    run_means = np.mean(runs, axis=1)
    run_dR = runs - run_means[:, None]
    run_rms = np.sqrt(np.mean(run_dR**2, axis=1))

    # Manual quantities:
    #   <R> for each run
    #   <dR^2>^(1/2) for each run, where dR = R - <R>
    # The checkpoint then asks for the average of the per-run <dR^2>^(1/2) values.
    mean_R_across_runs = np.mean(run_means)
    measured_noise = np.mean(run_rms)
    run_to_run_std = np.std(run_rms, ddof=1) if n_runs > 1 else 0.0
    sem = run_to_run_std / np.sqrt(n_runs) if n_runs > 1 else 0.0

    # Theory
    bandwidth = enbw_hz(tau)
    v_j = johnson_noise_rms(resistance, bandwidth)
    v_in = sr830_input_noise_rms(bandwidth)
    v_total = np.sqrt(v_j**2 + v_in**2)

    ratio = measured_noise / v_total
    pct_diff = 100.0 * (measured_noise - v_total) / v_total

    print(f"  File:                         {file_path.name}")
    print(f"  Runs found:                   {n_runs}")
    print(f"  Sample rate:                  {sample_rate:g} Hz")
    print(f"  Span per run:                 {span_s:g} s")
    print(f"  Time constant:                {tau:g} s")
    print(f"  Resistance:                   {resistance/1e3:g} kOhm")
    print(f"  ENBW:                         {bandwidth:.6g} Hz")
    print(f"  Johnson theory:               {v_j:.6e} V_rms")
    print(f"  SR830 input-noise contribution:{v_in:.6e} V_rms")
    print(f"  Total theory:                 {v_total:.6e} V_rms")
    print(f"  Mean of <R> across runs:      {mean_R_across_runs:.6e} V")
    print(f"  Mean <dR^2>^(1/2):            {measured_noise:.6e} V")
    print(f"  SEM of <dR^2>^(1/2):          {sem:.6e} V")
    print(f"  Measured / theory:            {ratio:.4f}")
    print(f"  Percent difference:           {pct_diff:+.2f}%")

    row = {
        "test_name": test_name,
        "file": str(file_path),
        "resistor_group": info["resistor_group"],
        "resistance_ohm": resistance,
        "sample_rate_hz": sample_rate,
        "span_s": span_s,
        "time_constant_s": tau,
        "rolloff_db_per_oct": ROLLOFF_DB_PER_OCT,
        "temperature_k": TEMPERATURE_K,
        "n_runs": n_runs,
        "enbw_hz": bandwidth,
        "johnson_theory_v_rms": v_j,
        "sr830_input_noise_v_rms": v_in,
        "total_theory_v_rms": v_total,
        "mean_R_across_runs_v": mean_R_across_runs,
        "measured_mean_per_run_rms_v": measured_noise,
        "run_to_run_std_v": run_to_run_std,
        "sem_v": sem,
        "measured_over_theory": ratio,
        "percent_difference": pct_diff,
    }
    summary_rows.append(row)

    run_cache[test_name] = {
        "runs": runs,
        "time_s": time_s,
        "run_means": run_means,
        "run_rms": run_rms,
        "row": row,
    }

    # --------------------------------------------------------
    # RAW-RUN PLOT
    # --------------------------------------------------------
    # The legend uses the same quantities shown in the Experiment IX
    # analyzed-data table: <R> and <dR^2>^(1/2).
    if MAKE_RAW_RUN_PLOTS:
        plt.figure(figsize=(13, 6))
        for i, run in enumerate(runs, start=1):
            plt.plot(
                time_s,
                run,
                linewidth=1.0,
                alpha=0.75,
                label=(
                    rf"Run {i}: $\langle R\rangle$={run_means[i-1]:.3e} V, "
                    rf"$\langle dR^2\rangle^{{1/2}}$={run_rms[i-1]:.3e} V"
                ),
            )

        # Across-run averages requested by the checkpoint.
        stats_text = (
            f"Number of runs = {n_runs}\n"
            rf"Mean of $\langle R\rangle$ = {mean_R_across_runs:.3e} V" "\n"
            rf"Mean of $\langle dR^2\rangle^{{1/2}}$ = {measured_noise:.3e} V" "\n"
            rf"SEM of $\langle dR^2\rangle^{{1/2}}$ = {sem:.2e} V" "\n"
            rf"ENBW = {bandwidth:.3g} Hz"
        )
        plt.text(
            0.98,
            0.97,
            stats_text,
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.80),
        )

        plt.xlabel("Time (s)")
        plt.ylabel("SR830 magnitude R (V)")
        plt.title(
            f"Experiment IX raw runs — {info['resistor_group']}, "
            f"tau={format_tau(tau)}, sample rate={sample_rate:g} Hz"
        )
        plt.grid(alpha=0.3)
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
        save_and_maybe_show(f"raw_{safe_filename(test_name)}.png")

    # --------------------------------------------------------
    # OPTIONAL PER-RUN DIAGNOSTIC
    # --------------------------------------------------------
    if MAKE_PER_RUN_DIAGNOSTIC_PLOTS:
        run_numbers = np.arange(1, n_runs + 1)

        plt.figure(figsize=(9, 5))
        plt.plot(
            run_numbers,
            run_rms,
            marker="o",
            label=r"Measured $\langle dR^2\rangle^{1/2}$",
        )
        plt.axhline(v_j, linestyle="--", label=f"Johnson theory = {v_j:.2e} V")
        plt.axhline(
            v_total,
            linestyle="-.",
            label=f"Johnson + SR830 input = {v_total:.2e} V",
        )
        for run_number, noise_value in zip(run_numbers, run_rms):
            plt.annotate(
                f"{noise_value:.2e}",
                (run_number, noise_value),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )
        plt.xlabel("Run number")
        plt.ylabel(r"$\langle dR^2\rangle^{1/2}$ (V)")
        plt.title(f"Per-run noise diagnostic — {test_name}")
        plt.grid(alpha=0.3)
        plt.legend()
        save_and_maybe_show(f"per_run_{safe_filename(test_name)}.png")


# ============================================================
# Save summary CSV
# ============================================================
if not summary_rows:
    raise RuntimeError("No Experiment IX data files were found. Check BASE_DIR and filenames.")

summary_csv = OUTPUT_DIR / "EXP9_final_summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"\nSaved summary table:\n  {summary_csv}")


# ============================================================
# Convenience helpers for combined plots
# ============================================================
def rows_for(group=None, tau=None, sample_rate=None):
    rows = summary_rows
    if group is not None:
        rows = [r for r in rows if r["resistor_group"] == group]
    if tau is not None:
        rows = [r for r in rows if np.isclose(r["time_constant_s"], tau)]
    if sample_rate is not None:
        rows = [r for r in rows if np.isclose(r["sample_rate_hz"], sample_rate)]
    return rows


def sort_by(rows, key):
    return sorted(rows, key=lambda r: r[key])


# ============================================================
# PLOT A: SAMPLE-RATE COMPARISON AT FIXED tau = 30 ms
# ============================================================
# This checks whether 64, 128, or 256 Hz acquisition materially changes
# the measured noise. The theoretical SR830 bandwidth is IDENTICAL for
# all three because tau and rolloff are fixed.
# ============================================================
for group in ["100 kOhm", "10 kOhm"]:
    rows = sort_by(rows_for(group=group, tau=30e-3), "sample_rate_hz")
    if not rows:
        continue

    x = np.array([r["sample_rate_hz"] for r in rows])
    y = np.array([r["measured_mean_per_run_rms_v"] for r in rows])
    yerr = np.array([r["sem_v"] for r in rows])
    theory = rows[0]["total_theory_v_rms"]
    johnson = rows[0]["johnson_theory_v_rms"]
    enbw = rows[0]["enbw_hz"]

    plt.figure(figsize=(9, 5.5))
    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=4,
        label=r"Mean measured $\langle dR^2\rangle^{1/2}$ $\pm$ SEM",
    )
    plt.axhline(
        theory,
        linestyle="-.",
        label=f"Theory: Johnson + SR830 input = {theory:.2e} V",
    )
    plt.axhline(
        johnson,
        linestyle="--",
        label=f"Johnson theory only = {johnson:.2e} V",
    )

    # Put the actual measured average next to each point.
    for xi, yi, row in zip(x, y, rows):
        plt.annotate(
            f"{yi:.2e} V\nN={row['n_runs']}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    info_text = (
        rf"$\tau$ = 30 ms" "\n"
        rf"ENBW = {enbw:.3g} Hz" "\n"
        rf"$R_{{source}}$ = {rows[0]['resistance_ohm']/1e3:g} k$\Omega$"
    )
    plt.text(
        0.98,
        0.03,
        info_text,
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.80),
    )

    plt.xlabel("Computer sample rate (Hz)")
    plt.ylabel(r"Mean per-run $\langle dR^2\rangle^{1/2}$ (V)")
    plt.title(f"Experiment IX sample-rate check — {group}, tau = 30 ms")
    plt.xticks(x)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    save_and_maybe_show(f"sample_rate_comparison_{safe_filename(group)}.png")


# ============================================================
# PLOT B: MAIN CHECKPOINT PLOT — NOISE VS ENBW
# ============================================================
# Use the 64 Hz acquisition runs only, so sample rate is held fixed.
# Each measured point is the checkpoint quantity: the average, across
# runs, of <dR^2>^(1/2).
# ============================================================
plt.figure(figsize=(10.5, 6.5))

for group in ["100 kOhm", "10 kOhm"]:
    rows = sort_by(rows_for(group=group, sample_rate=64), "enbw_hz")
    if not rows:
        continue

    enbw = np.array([r["enbw_hz"] for r in rows])
    measured = np.array([r["measured_mean_per_run_rms_v"] for r in rows])
    sem = np.array([r["sem_v"] for r in rows])
    theory = np.array([r["total_theory_v_rms"] for r in rows])

    plt.errorbar(
        enbw,
        measured,
        yerr=sem,
        marker="o",
        capsize=4,
        linestyle="none",
        label=rf"{group}: mean measured $\langle dR^2\rangle^{{1/2}}$",
    )
    plt.plot(
        enbw,
        theory,
        marker="s",
        label=f"{group}: total theory (Johnson + SR830 input)",
    )

    # Label every experimental point with its time constant and measured average.
    for row, x_value, y_value in zip(rows, enbw, measured):
        plt.annotate(
            f"{format_tau(row['time_constant_s'])}\n{y_value:.2e} V",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(6, 7),
            fontsize=7.5,
        )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Effective noise bandwidth, ENBW (Hz)")
plt.ylabel(r"Mean per-run $\langle dR^2\rangle^{1/2}$ (V)")
plt.title(
    "Experiment IX: measured noise vs effective noise bandwidth\n"
    "64 Hz acquisition; error bars are SEM across repeated runs"
)
plt.grid(alpha=0.3, which="both")
plt.legend(fontsize=8)
save_and_maybe_show("EXP9_main_noise_vs_ENBW.png")


# ============================================================
# PLOT C: SAME PHYSICS, SHOWN VS TIME CONSTANT
# ============================================================
plt.figure(figsize=(10.5, 6.5))

for group in ["100 kOhm", "10 kOhm"]:
    rows = sort_by(rows_for(group=group, sample_rate=64), "time_constant_s")
    if not rows:
        continue

    tau = np.array([r["time_constant_s"] for r in rows])
    measured = np.array([r["measured_mean_per_run_rms_v"] for r in rows])
    sem = np.array([r["sem_v"] for r in rows])
    theory = np.array([r["total_theory_v_rms"] for r in rows])

    plt.errorbar(
        tau,
        measured,
        yerr=sem,
        marker="o",
        capsize=4,
        linestyle="none",
        label=rf"{group}: mean measured $\langle dR^2\rangle^{{1/2}}$",
    )
    plt.plot(
        tau,
        theory,
        marker="s",
        label=f"{group}: total theory (Johnson + SR830 input)",
    )

    # Numeric labels make the averages readable without looking at the CSV.
    for x_value, y_value in zip(tau, measured):
        plt.annotate(
            f"{y_value:.2e} V",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(6, 7),
            fontsize=7.5,
        )

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"SR830 time constant $\tau$ (s)")
plt.ylabel(r"Mean per-run $\langle dR^2\rangle^{1/2}$ (V)")
plt.title(
    "Experiment IX: measured noise vs lock-in time constant\n"
    "64 Hz acquisition; error bars are SEM across repeated runs"
)
plt.grid(alpha=0.3, which="both")
plt.legend(fontsize=8)
save_and_maybe_show("EXP9_noise_vs_time_constant.png")


# ============================================================
# Console summary
# ============================================================
print("\n============================================================")
print("EXPERIMENT IX FINAL OUTPUT COMPLETE")
print("============================================================")
print(f"Plots saved to:\n  {OUTPUT_DIR}")
print(f"Summary CSV:\n  {summary_csv}")
print("\nMain plots to use in the checkpoint/write-up:")
print("  1. EXP9_main_noise_vs_ENBW.png")
print("  2. EXP9_noise_vs_time_constant.png")
print("  3. sample_rate_comparison_100_kOhm.png")
print("  4. sample_rate_comparison_10_kOhm.png")
print("  + raw_*.png files, now labeled with <R> and <dR^2>^(1/2) for every run")
print("\nInterpretation:")
print("  - Raw plots show <R> and <dR^2>^(1/2) for every individual run.")
print("  - Sample-rate plots show the mean <dR^2>^(1/2) ± SEM at 64/128/256 Hz.")
print("  - ENBW/time-constant plots test the expected white-noise bandwidth dependence.")
print("  - Both resistor data sets remain on the ENBW/time-constant plots.")
print("  - Theory includes SR830 input noise in quadrature.")
print("\nNOTE ABOUT sqrt(2):")
print("  The SR830 output R is a magnitude, and the lab manual explicitly asks")
print("  whether the measured value is displaced by about sqrt(2). This script")
print("  intentionally does not apply a hidden sqrt(2) correction. If your")
print("  instructor wants the R-magnitude correction applied to the comparison,")
print("  do that as an explicitly labeled analysis step rather than silently")
print("  modifying the raw measured noise.")


# ============================================================
# END-OF-SCRIPT THEORY COMPARISON FOR THE TWO MAIN PLOTS
# ============================================================
# This section does NOT make any new plots. It quantitatively compares
# the same 64 Hz points used in:
#   1. EXP9_main_noise_vs_ENBW.png
#   2. EXP9_noise_vs_time_constant.png
# against the total theoretical noise
#
#       V_total = sqrt(V_Johnson^2 + V_input^2).
#
# For white noise at fixed resistance and 24 dB/oct rolloff,
#
#       V_noise ∝ sqrt(ENBW) ∝ tau^(-1/2),
#
# so the expected log-log slopes are +0.5 versus ENBW and -0.5
# versus time constant tau.
# ============================================================

comparison_lines = []
comparison_lines.append("=" * 72)
comparison_lines.append("EXPERIMENT IX — COMPARISON OF MEASURED NOISE WITH THEORY")
comparison_lines.append("=" * 72)
comparison_lines.append(
    "Theory used: V_total = sqrt(V_Johnson^2 + V_SR830,input^2)"
)
comparison_lines.append(
    "Expected white-noise scaling: V ∝ ENBW^(1/2) ∝ tau^(-1/2)"
)
comparison_lines.append(
    "Measured quantity: mean across runs of <dR^2>^(1/2)"
)
comparison_lines.append("")

for group in ["100 kOhm", "10 kOhm"]:
    # Use exactly the same 64 Hz subset as the two main plots.
    rows = sort_by(rows_for(group=group, sample_rate=64), "time_constant_s")
    if len(rows) < 2:
        continue

    tau = np.array([r["time_constant_s"] for r in rows], dtype=float)
    enbw = np.array([r["enbw_hz"] for r in rows], dtype=float)
    measured = np.array(
        [r["measured_mean_per_run_rms_v"] for r in rows], dtype=float
    )
    theory = np.array([r["total_theory_v_rms"] for r in rows], dtype=float)
    sem = np.array([r["sem_v"] for r in rows], dtype=float)

    ratio = measured / theory
    percent_difference = 100.0 * (measured - theory) / theory

    # Log-log power-law slopes measured from the data.
    # Theory predicts +0.5 for noise vs ENBW and -0.5 for noise vs tau.
    slope_enbw = np.polyfit(np.log(enbw), np.log(measured), 1)[0]
    slope_tau = np.polyfit(np.log(tau), np.log(measured), 1)[0]

    mean_ratio = np.mean(ratio)
    mean_abs_percent_difference = np.mean(np.abs(percent_difference))
    rms_percent_difference = np.sqrt(np.mean(percent_difference**2))

    # The manual asks whether the R-magnitude result is displaced by ~sqrt(2).
    # We REPORT the ratio but do not automatically apply a correction.
    sqrt2 = np.sqrt(2.0)

    comparison_lines.append(f"{group}")
    comparison_lines.append("-" * len(group))
    comparison_lines.append(
        "tau        ENBW (Hz)    measured (V)       theory (V)         "
        "meas/theory    % difference"
    )

    for t, bw, vm, vt, s, r, pd in zip(
        tau, enbw, measured, theory, sem, ratio, percent_difference
    ):
        comparison_lines.append(
            f"{format_tau(t):>8s}   "
            f"{bw:10.4g}   "
            f"{vm:12.4e} ± {s:8.2e}   "
            f"{vt:12.4e}   "
            f"{r:10.3f}   "
            f"{pd:+10.2f}%"
        )

    comparison_lines.append("")
    comparison_lines.append(
        f"Mean measured/theory ratio = {mean_ratio:.3f} "
        f"(perfect agreement = 1.000; sqrt(2) = {sqrt2:.3f})"
    )
    comparison_lines.append(
        f"Mean absolute percent difference = {mean_abs_percent_difference:.2f}%"
    )
    comparison_lines.append(
        f"RMS percent difference = {rms_percent_difference:.2f}%"
    )
    comparison_lines.append(
        f"Measured log-log slope vs ENBW = {slope_enbw:.3f} "
        f"(white-noise theory = +0.500)"
    )
    comparison_lines.append(
        f"Measured log-log slope vs tau = {slope_tau:.3f} "
        f"(white-noise theory = -0.500)"
    )
    comparison_lines.append("")

    # Simple, transparent interpretation without hiding a sqrt(2) correction.
    if mean_abs_percent_difference <= 10:
        agreement_text = "very close quantitative agreement with total theory"
    elif mean_abs_percent_difference <= 25:
        agreement_text = "reasonable quantitative agreement with total theory"
    else:
        agreement_text = "a noticeable quantitative discrepancy from total theory"

    comparison_lines.append(
        f"Summary: the {group} data show {agreement_text}."
    )
    comparison_lines.append(
        "The ENBW slope should be compared with +0.5 and the time-constant "
        "slope with -0.5 to test the expected white-noise dependence."
    )
    comparison_lines.append(
        "The measured/theory ratio is also listed next to sqrt(2), because "
        "Experiment IX explicitly asks you to consider the SR830 R-magnitude effect."
    )
    comparison_lines.append("")

# Print the comparison to the terminal.
comparison_text = "\n".join(comparison_lines)
print("\n" + comparison_text)

# Also save it as a text file next to the plots and CSV.
theory_comparison_file = OUTPUT_DIR / "EXP9_theory_comparison.txt"
with theory_comparison_file.open("w", encoding="utf-8") as f:
    f.write(comparison_text + "\n")

print(f"\nTheory comparison saved to:\n  {theory_comparison_file}")
