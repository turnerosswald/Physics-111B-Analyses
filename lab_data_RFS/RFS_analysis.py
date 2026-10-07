import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

import math

# ============================================================

# RFS_analysis.py

# Rutherford Scattering — THEORY + MEASUREMENTS

#

# This file first draws the theory curves, then loads the experimental data.

#

# Rutherford angular dependence:

#

#       dσ/dΩ ∝ sin^(-4)(θ/2)

#

# For a fixed experimental geometry/source/foil/detector setup,

# the measured count rate should therefore have the form

#

#       R(θ) = A sin^(-4)(θ/2)

#

# where A contains all angle-independent normalization factors.

#

# Later, when experimental rates are added, the manual-recommended

# linearized fit is:

#

#       x_i = sin^(-4)(θ_i/2)

#       y_i = R_i

#

#       y_i = A x_i

#

# or, if an approximately angle-independent background is retained,

#

#       y_i = A x_i + B

#

# For now: THEORY ONLY. No measured values are inserted.

# ============================================================





# ------------------------------------------------------------

# Output settings

# ------------------------------------------------------------



OUTPUT_DIR = Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS\analysis_outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



SHOW_PLOTS = True





# ------------------------------------------------------------

# Theory settings

# ------------------------------------------------------------



# Rutherford theory is singular at exactly theta = 0 degrees,

# so the theoretical curve must begin at a small positive angle.

THETA_MIN_DEG = 0.5

THETA_MAX_DEG = 20.0



# A normalization is required to put the pure theory on a finite,

# dimensionless vertical scale without using experimental data.

#

# After the measured 2-degree zero offset, the physical fixed angles are

# 3, 8, and 13 degrees.  Normalize the dimensionless theory at 3 degrees

# so every theory plot uses the same physical-angle convention as the data.

#

#       R(theta) / R(3 deg)

#

# This does NOT use the measured 3-degree count rate.

REFERENCE_ANGLE_DEG = 3.0



# Physical scattering-angle guide lines after the 2-degree zero correction.

GUIDE_ANGLES_DEG = [3.0, 8.0, 13.0]





# ============================================================

# 1. RUTHERFORD THEORY FUNCTIONS

# ============================================================



def rutherford_factor(theta_deg):

    """

    Pure Rutherford angular factor:



        f(theta) = sin^(-4)(theta/2)

                 = csc^4(theta/2)



    theta_deg:

        scattering angle in degrees.



    The function diverges at theta = 0, so all input angles

    must be strictly positive.

    """

    theta_deg = np.asarray(theta_deg, dtype=float)



    if np.any(theta_deg <= 0):

        raise ValueError(

            "Rutherford sin^(-4)(theta/2) is singular at "

            "theta <= 0 degrees."

        )



    theta_rad = np.deg2rad(theta_deg)



    return np.sin(theta_rad / 2.0) ** (-4)





def rutherford_rate_model(theta_deg, A):

    """

    Future experimental fit model:



        R(theta) = A sin^(-4)(theta/2)



    A is the angle-independent normalization.



    No value of A is inferred in this theory-only file.

    """

    return A * rutherford_factor(theta_deg)





def rutherford_rate_model_with_background(theta_deg, A, B):

    """

    Optional future model if an approximately angle-independent

    background remains:



        R(theta) = A sin^(-4)(theta/2) + B

    """

    return A * rutherford_factor(theta_deg) + B





def normalized_rutherford_rate(theta_deg, reference_angle_deg=REFERENCE_ANGLE_DEG):

    """

    Dimensionless Rutherford curve normalized at a reference angle:



        R(theta) / R(theta_ref)

          = f(theta) / f(theta_ref)



    This lets us visualize the theoretical shape without using

    any measured count-rate normalization.

    """

    return (

        rutherford_factor(theta_deg)

        / rutherford_factor(reference_angle_deg)

    )





def linearized_x(theta_deg):

    """

    Manual-recommended x-coordinate for the eventual straight-line fit:



        x = sin^(-4)(theta/2)



    Then ideal Rutherford scattering gives



        R = A x.

    """

    return rutherford_factor(theta_deg)





# ============================================================

# 2. PURE-THEORY RUTHERFORD CURVE

# ============================================================



theta = np.linspace(

    THETA_MIN_DEG,

    THETA_MAX_DEG,

    2000,

)



relative_rate = normalized_rutherford_rate(

    theta,

    REFERENCE_ANGLE_DEG,

)



fig, ax = plt.subplots(

    figsize=(10.5, 6.5)

)



ax.plot(

    theta,

    relative_rate,

    linewidth=2.2,

    label=(

        r"Rutherford theory: "

        r"$R(\theta)\propto\sin^{-4}(\theta/2)$"

    ),

)



# Physical fixed-angle guides after the 2-degree zero correction.

for angle in GUIDE_ANGLES_DEG:

    ax.axvline(

        angle,

        linestyle="--",

        linewidth=0.9,

        alpha=0.45,

    )



    theory_ratio = float(

        normalized_rutherford_rate(

            angle,

            REFERENCE_ANGLE_DEG,

        )

    )



    ax.scatter(

        [angle],

        [theory_ratio],

        s=42,

        zorder=4,

    )



    ax.annotate(

        (

            rf"$\theta={angle:g}^\circ$"

            "\n"

            rf"$R/R_{{3^\circ}}={theory_ratio:.4g}$"

        ),

        xy=(

            angle,

            theory_ratio,

        ),

        xytext=(

            7,

            7,

        ),

        textcoords="offset points",

        fontsize=9,

    )



ax.set_yscale(

    "log"

)



ax.set_xlim(

    THETA_MIN_DEG,

    THETA_MAX_DEG,

)



ax.set_xlabel(

    r"Scattering angle $\theta$ (degrees)",

    fontsize=12,

)



ax.set_ylabel(

    r"Relative theoretical rate $R(\theta)/R(3^\circ)$",

    fontsize=12,

)



ax.set_title(

    r"Pure Rutherford angular dependence: "

    r"$R(\theta)\propto\sin^{-4}(\theta/2)$",

    fontsize=14,

)



ax.grid(

    alpha=0.25,

    which="both",

)



ax.legend()



fig.tight_layout()



fig.savefig(

    OUTPUT_DIR / "01_rutherford_theory_sin_minus4.png",

    dpi=240,

    bbox_inches="tight",

)



if SHOW_PLOTS:

    plt.show()

else:

    plt.close(fig)





# ============================================================

# 3. PURE-THEORY LINEARIZED VIEW

#

# This is the form that will later be used for a least-squares fit:

#

#       x = sin^(-4)(theta/2)

#       y = R

#

#       y = A x

#

# Since there are no measured rates yet, we plot only the

# dimensionless theoretical relation:

#

#       y = x / f(3 deg)

#

# so y is again R(theta)/R(3 deg).

# ============================================================



theta_linear = np.linspace(

    REFERENCE_ANGLE_DEG,

    THETA_MAX_DEG,

    1000,

)



x_theory = linearized_x(

    theta_linear

)



normalization = float(

    rutherford_factor(

        REFERENCE_ANGLE_DEG

    )

)



y_theory = (

    x_theory

    / normalization

)



fig, ax = plt.subplots(

    figsize=(8.5, 6.2)

)



ax.plot(

    x_theory,

    y_theory,

    linewidth=2.2,

    label=(

        r"Pure theory: "

        r"$R/R_{3^\circ}=x/"

        r"\sin^{-4}(3^\circ/2)$"

    ),

)



ax.set_xlabel(

    r"$x=\sin^{-4}(\theta/2)$",

    fontsize=12,

)



ax.set_ylabel(

    r"Relative theoretical rate $R(\theta)/R(3^\circ)$",

    fontsize=12,

)



ax.set_title(

    "Linearized Rutherford relation — theory only",

    fontsize=14,

)



ax.grid(

    alpha=0.25

)



ax.legend()



fig.tight_layout()



fig.savefig(

    OUTPUT_DIR / "02_rutherford_theory_linearized.png",

    dpi=240,

    bbox_inches="tight",

)



if SHOW_PLOTS:

    plt.show()

else:

    plt.close(fig)







# ============================================================

# 4. THEORY CURVE FOR THE INFERRED GOLD ATOMIC NUMBER Z_2(theta)

#

# Using the equation currently written in the report:

#

#       dσ/dΩ

#       = [(-Z1 Z2)/2]^2 * [1/(4 E^2)]

#         * 1/sin^4(theta/2)

#

# which simplifies to

#

#       dσ/dΩ

#       = (Z1^2 Z2^2)/(16 E^2)

#         * csc^4(theta/2).

#

# Solving algebraically for Z2 gives

#

#       Z2(theta)

#       = (4 E / Z1)

#         sin^2(theta/2)

#         sqrt(dσ/dΩ).

#

# Here we remain THEORY ONLY:

#   - Z1 = 2 for the alpha particle.

#   - E = 3.77 MeV, following the experimental write-up.

#   - Z2 = 79 is used to generate the ideal gold Rutherford

#     differential cross section.

#   - We then feed that theoretical dσ/dΩ back into the inverted

#     equation above.

#

# Therefore an ideal Rutherford experiment on gold must recover

#

#       Z2(theta) = 79

#

# independent of theta. Any theta-dependence later seen after

# inserting measured dσ/dΩ would represent experimental/systematic

# deviations from the ideal Rutherford model.

#

# IMPORTANT:

# This section follows the simplified equation exactly as it is

# currently written in the report. The full SI Rutherford expression

# contains the electrostatic factor k e^2. That factor must be restored

# when converting a physical cross section in SI units into an absolute

# atomic number.

# ============================================================



Z1_ALPHA = 2.0

Z2_GOLD = 79.0

ALPHA_ENERGY_MEV = 3.77





def rutherford_cross_section_reduced(

    theta_deg,

    Z1=Z1_ALPHA,

    Z2=Z2_GOLD,

    E=ALPHA_ENERGY_MEV,

):

    """

    Rutherford differential cross section in the simplified

    normalization used by the current report:



        dσ/dΩ

        = (Z1^2 Z2^2)/(16 E^2)

          * sin^(-4)(theta/2)



    This is used only as a theory-consistency curve here.

    """

    theta_deg = np.asarray(theta_deg, dtype=float)



    if np.any(theta_deg <= 0):

        raise ValueError(

            "The Rutherford expression is singular at theta <= 0."

        )



    theta_rad = np.deg2rad(theta_deg)



    return (

        (Z1**2 * Z2**2)

        / (16.0 * E**2)

        * np.sin(theta_rad / 2.0) ** (-4)

    )





def infer_Z2_from_cross_section_reduced(

    theta_deg,

    dsigma_domega,

    Z1=Z1_ALPHA,

    E=ALPHA_ENERGY_MEV,

):

    """

    Invert the report's Rutherford equation:



        Z2(theta)

        = (4 E / Z1)

          sin^2(theta/2)

          sqrt(dσ/dΩ).

    """

    theta_deg = np.asarray(theta_deg, dtype=float)

    dsigma_domega = np.asarray(dsigma_domega, dtype=float)



    if np.any(theta_deg <= 0):

        raise ValueError(

            "The Rutherford expression is singular at theta <= 0."

        )



    if np.any(dsigma_domega < 0):

        raise ValueError(

            "dσ/dΩ must be non-negative."

        )



    theta_rad = np.deg2rad(theta_deg)



    return (

        (4.0 * E / Z1)

        * np.sin(theta_rad / 2.0) ** 2

        * np.sqrt(dsigma_domega)

    )





theta_Z2 = np.linspace(

    THETA_MIN_DEG,

    THETA_MAX_DEG,

    2000,

)



theory_dsigma_domega = rutherford_cross_section_reduced(

    theta_Z2,

)



Z2_theory = infer_Z2_from_cross_section_reduced(

    theta_Z2,

    theory_dsigma_domega,

)



fig, ax = plt.subplots(

    figsize=(9.5, 6.2)

)



ax.plot(

    theta_Z2,

    Z2_theory,

    linewidth=2.2,

    label=r"Ideal Rutherford prediction: $Z_2(\theta)$",

)



ax.axhline(

    Z2_GOLD,

    linestyle="--",

    linewidth=1.4,

    alpha=0.75,

    label=r"Gold: $Z_2=79$",

)



ax.set_xlim(

    THETA_MIN_DEG,

    THETA_MAX_DEG,

)



ax.set_ylim(

    75,

    83,

)



ax.set_xlabel(

    r"Scattering angle $\theta$ (degrees)",

    fontsize=12,

)



ax.set_ylabel(

    r"Inferred atomic number $Z_2(\theta)$",

    fontsize=12,

)



ax.set_title(

    r"Theoretical gold atomic number from Rutherford scattering",

    fontsize=14,

)



ax.grid(

    alpha=0.25

)



ax.legend()



fig.tight_layout()



fig.savefig(

    OUTPUT_DIR / "03_gold_Z2_theory_vs_theta.png",

    dpi=240,

    bbox_inches="tight",

)



if SHOW_PLOTS:

    plt.show()

else:

    plt.close(fig)





# ============================================================

# 5. PRINT THEORY-ONLY SUMMARY

# ============================================================



print(

    "\n============================================================"

)



print(

    "RUTHERFORD THEORY ONLY"

)



print(

    "============================================================"

)



print(

    r"Angular model: R(theta) = A sin^(-4)(theta/2)"

)



print(

    r"Linearized coordinate: x = sin^(-4)(theta/2)"

)



print(

    r"Ideal future fit: R = A x"

)



print(

    r"Optional background model: R = A x + B"

)



print(

    r"Inverted Rutherford relation: "

    r"Z2(theta) = (4E/Z1) sin^2(theta/2) sqrt(dsigma/dOmega)"

)



print(

    r"Ideal gold theory: Z2(theta) = 79"

)



print(

    "\nTheory-only plots complete; experimental measurement section follows."

)



print(

    f"\nTheory plots saved to:\n{OUTPUT_DIR}"

)

BASE_DIR = Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS")

OUTPUT_DIR = BASE_DIR / "analysis_outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



SHOW_PLOTS = True



FILES = {

    "5_run1": Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS\5DEGREE_RUN_ONE_HOUR_ARIYANTURNER_RFS.csv"),

    "5_run2": Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS\5DEGREES_HOUR_RUN_ARIYANTURNER_RFS.csv"),



    "10_run1": Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS\10DEGREE_OVERNIGHT_RUN_ARIYAN_TURNER_RFS_RUN2.csv"),

    "10_run2": Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS\10DEGREES_OVERNIGHT_RUN_ARIYANTURNER_RFS.csv"),



    "15_run1": Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS\15DEGREES_OVERNIGHT_ARIYANTURNER_RFS.csv"),

    "15_run2": Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS\15DEGREES_WEEKEND_RUN_ARIYANTURNER_RFS.csv"),

    "15_run3": Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS\15DEGREES_OVERNIGHT_ARIYANTURNER_RUN3.csv"),



    "no_foil": Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS\NO_FOIL_TEN_MINUTE_RUN_ARIYANTURNER_RFS.csv"),



    "inc_run1": Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS\PER_DEGREE_RUN_ONE_30MINUTES_RUN_ARIYANTURNER_RFS.csv"),

    "inc_run2": Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS\PER_DEGREE_RUN_ONE_HOUR_RUN_ARIYANTURNER_RFS_RUN2.csv"),

    "inc_run3": Path(r"C:\Users\turno\OneDrive\Documents\lab_data_RFS\PER_DEGREE_RUN_ONE_HOUR_RUN_ARIYANTURNER_RFS_RUN3.csv"),

}





# ------------------------------------------------------------

# Experimental constants

# ------------------------------------------------------------



THRESHOLD_V = 4.0

TRUE_ZERO_OFFSET_DEG = 2.0



# User-specified correction for the physical 3-degree measurement:
# retain 98.17% of the measured signal (remove 1.83%).
# Keep this explicit so the raw rate remains available in CSV outputs.

THREE_DEGREE_REMOVED_FRACTION = 0.0183

THREE_DEGREE_RETAINED_FRACTION = 1.0 - THREE_DEGREE_REMOVED_FRACTION



# Alpha particle / gold

Z1_ALPHA = 2.0

Z2_GOLD = 79.0

ALPHA_ENERGY_MEV = 3.77

# Coulomb constant combination in nuclear units:

#     (1 / 4 pi eps0) e^2

KE2_MEV_FM = 1.4399645478



# Target areal number densities.
# The original (heavier) foil was used through 2026-09-01.
# Experiments after 2026-09-01 used the replacement lighter foil.

N_AREAL_CM2 = 8.26e18
N_AREAL_CM2_NEW = 4.72e18
FOIL_REPLACEMENT_DATE = pd.Timestamp("2026-09-01")


def areal_density_for_timestamp(timestamp):
    """Return the correct foil areal density for a run timestamp."""
    ts = pd.Timestamp(timestamp)

    if pd.isna(ts):
        raise ValueError("Cannot choose foil areal density from a missing timestamp.")

    # Compare by calendar date even if a CSV timestamp includes a timezone.
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    run_date = ts.normalize()

    if run_date <= FOIL_REPLACEMENT_DATE:
        return N_AREAL_CM2
    elif run_date > FOIL_REPLACEMENT_DATE:
        return N_AREAL_CM2_NEW

    raise ValueError(f"Could not determine foil areal density for {timestamp!r}.")



# Experimental solid angle used in the cross-section normalization.

# Use the value adopted in the report directly rather than recomputing it

# from the detector dimensions.

SOLID_ANGLE_SR = 0.14
SOLID_ANGLE_SIGMA_SR = 0.006

# No independent uncertainty was supplied for the two areal densities.
# Leave this at zero rather than inventing one.  The error-analysis output
# explicitly flags this as an unquantified systematic; update it if the foil
# log gives a mass / areal-density uncertainty.
N_AREAL_REL_SIGMA = 0.0

# Add an empirical repeatability systematic to the vertical error bars.
# This is estimated FROM THE DATA at each angle from the scatter among
# repeated runs after normalizing out the foil areal density.  It is treated
# as a systematic repeatability term (sample SD / mean), so it is NOT divided
# by sqrt(n).  This captures run-to-run changes such as alignment, foil
# positioning/condition, gain/threshold drift, and vacuum/reassembly effects.
USE_REPEATABILITY_SYSTEMATIC = True

# Quantitative Rutherford fit window.
#
# Primary Z extraction / goodness-of-fit region:
#   - below 4 deg: do not use quantitatively because the report's
#     multiple-scattering estimate reaches the <1% regime only above
#     about 3.09 deg, and the Rutherford curve is exceptionally sensitive
#     to angle near zero;
#   - 4--9 deg: use as the primary quantitative region;
#   - 10--13 deg: KEEP AND PLOT these measurements, but treat them as a
#     high-angle systematic diagnostic rather than allowing them to set Z.
#
# The fit-window-sensitivity table below still explicitly tests windows
# extending through 13 deg, so the excluded high-angle points are not hidden.
FIT_MIN_ANGLE_DEG = 4.0
FIT_MAX_ANGLE_DEG = 9.0

# The user requested numerical chi-square output rather than extra residual
# plots. Set this True later if you want the pull plots back.
MAKE_CHI_SQUARE_PLOTS = False



# Detector dimensions are retained only for the pre-existing geometry

# uncertainty calculation below.

DETECTOR_RADIUS_CM = 0.375 * 2.54

DETECTOR_DISTANCE_CM = 4.5





# ------------------------------------------------------------

# Experimental angular / geometry uncertainties

# ------------------------------------------------------------

# Central angle calibration stays at nominal 2 deg = physical 0 deg.
# We DO NOT move the central angles to make the data agree with Z = 79.
#
# Instead, keep separate uncertainty components so the propagated error
# honestly reflects how difficult the mechanical/angular measurement is.
#
# PROTRACTOR_SIGMA_DEG:
#     Direct detector-angle readout / setting uncertainty.
#
# ZERO_CALIBRATION_SIGMA_DEG:
#     Uncertainty in locating the center of the no-foil beam peak around
#     the adopted 2 deg zero.
#
# ANGLE_ALIGNMENT_SYSTEMATIC_SIGMA_DEG:
#     Conservative extra allowance for repeatability / finite alignment
#     effects visible in the angular-gradient measurements.  This changes
#     ERROR BARS ONLY; it does not shift any measured angle.
#
# If you later decide this extra allowance is not justified, set it to 0.

PROTRACTOR_SIGMA_DEG = 1.0
ZERO_CALIBRATION_SIGMA_DEG = 0.5
ANGLE_ALIGNMENT_SYSTEMATIC_SIGMA_DEG = 1.0


COLLIMATOR_SEPARATION_CM = 1.99
COLLIMATOR_SEPARATION_SIGMA_CM = 0.005

DETECTOR_DISTANCE_SIGMA_CM = 0.05

APERTURE_DIAMETER_CM = (
    (1.0 / 16.0) * 2.54
)


# Small uncertainty in the nominal collimator angle that comes only from
# the measured uncertainty in the collimator separation.  This is NOT the
# full beam divergence; it is just the geometry-measurement contribution.
alpha_sigma_rad = (
    APERTURE_DIAMETER_CM
    / (
        COLLIMATOR_SEPARATION_CM**2
        + APERTURE_DIAMETER_CM**2
    )
    * COLLIMATOR_SEPARATION_SIGMA_CM
)

COLLIMATOR_ANGLE_SIGMA_DEG = np.degrees(
    alpha_sigma_rad
)


# "Measurement-only" angle uncertainty before the additional conservative
# alignment allowance.
BASE_ANGLE_SIGMA_DEG = np.sqrt(
    PROTRACTOR_SIGMA_DEG**2
    + ZERO_CALIBRATION_SIGMA_DEG**2
    + COLLIMATOR_ANGLE_SIGMA_DEG**2
)


# Total 1-sigma angle uncertainty used for x-error bars, the Rutherford
# linearized fit, and the propagated uncertainty in inferred Z.
TOTAL_ANGLE_SIGMA_DEG = np.sqrt(
    BASE_ANGLE_SIGMA_DEG**2
    + ANGLE_ALIGNMENT_SYSTEMATIC_SIGMA_DEG**2
)

# ------------------------------------------------------------
# Uncertainties used specifically for statistical fitting
# ------------------------------------------------------------
#
# IMPORTANT:
# TOTAL_ANGLE_SIGMA_DEG above remains the full conservative error bar
# shown on the measurement / inferred-Z plots.
#
# For chi-square, however, we must not pretend that every systematic is
# an independent random error on every point:
#
#   * PROTRACTOR_SIGMA_DEG is treated as point-to-point angle-setting
#     uncertainty.
#
#   * ZERO_CALIBRATION_SIGMA_DEG is ONE common angular offset shared by
#     all points. It enters the covariance matrix as a correlated term.
#
#   * ANGLE_ALIGNMENT_SYSTEMATIC_SIGMA_DEG is NOT inserted again into
#     the chi-square because the empirical run-to-run repeatability term
#     already captures observed alignment/reassembly scatter. Including
#     both independently would double count the same effect.
#
#   * The tiny collimator-separation measurement term is also common and
#     is combined with the zero-calibration term.
#
POINT_TO_POINT_ANGLE_SIGMA_DEG = PROTRACTOR_SIGMA_DEG
COMMON_ANGLE_OFFSET_SIGMA_DEG = np.hypot(
    ZERO_CALIBRATION_SIGMA_DEG,
    COLLIMATOR_ANGLE_SIGMA_DEG,
)


DETECTOR_GEOM_REL_SIGMA = (
    2.0
    * DETECTOR_DISTANCE_SIGMA_CM
    / DETECTOR_DISTANCE_CM
)

COLLIMATOR_GEOM_REL_SIGMA = (
    2.0
    * COLLIMATOR_SEPARATION_SIGMA_CM
    / COLLIMATOR_SEPARATION_CM
)

GEOMETRY_REL_SIGMA = np.hypot(
    DETECTOR_GEOM_REL_SIGMA,
    COLLIMATOR_GEOM_REL_SIGMA,
)


# Solid-angle geometry diagnostics only.  The analysis still uses the
# adopted SOLID_ANGLE_SR above, so these lines do NOT change the plots.
SOLID_ANGLE_SMALL_ANGLE_SR = (
    np.pi * DETECTOR_RADIUS_CM**2 / DETECTOR_DISTANCE_CM**2
)

SOLID_ANGLE_EXACT_DISK_SR = (
    2.0
    * np.pi
    * (
        1.0
        - DETECTOR_DISTANCE_CM
        / np.sqrt(
            DETECTOR_DISTANCE_CM**2
            + DETECTOR_RADIUS_CM**2
        )
    )
)



# ============================================================

# 1. THEORY

# ============================================================



def rutherford_factor(theta_deg):

    theta_deg = np.asarray(

        theta_deg,

        dtype=float,

    )



    if np.any(theta_deg <= 0):

        raise ValueError(

            "Rutherford factor is singular at theta <= 0."

        )



    theta_rad = np.deg2rad(

        theta_deg

    )



    return (

        np.sin(theta_rad / 2.0)

        ** (-4)

    )





def rutherford_cross_section_cm2_per_sr(

    theta_deg,

    Z2=Z2_GOLD,

):

    """

    Physical Rutherford differential cross section:



        dσ/dΩ

        = [k Z1 Z2 e^2 / (4E)]^2

          csc^4(theta/2)



    Returned in cm^2 / sr.

    """

    theta_deg = np.asarray(

        theta_deg,

        dtype=float,

    )



    prefactor_fm = (

        KE2_MEV_FM

        * Z1_ALPHA

        * Z2

        / (

            4.0

            * ALPHA_ENERGY_MEV

        )

    )



    dsigma_fm2 = (

        prefactor_fm**2

        * rutherford_factor(

            theta_deg

        )

    )



    # 1 fm = 1e-13 cm, so 1 fm^2 = 1e-26 cm^2.

    return (

        dsigma_fm2

        * 1.0e-26

    )





def infer_Z2(

    theta_deg,

    dsigma_cm2_per_sr,

):

    """

    Invert the physical Rutherford expression:



        Z2(theta)

        = [4E / (Z1 k e^2)]

          sin^2(theta/2)

          sqrt(dσ/dΩ)



    dσ/dΩ is converted from cm^2 to fm^2 internally.

    """

    theta_deg = np.asarray(

        theta_deg,

        dtype=float,

    )



    dsigma_cm2_per_sr = np.asarray(

        dsigma_cm2_per_sr,

        dtype=float,

    )



    if np.any(theta_deg <= 0):

        raise ValueError(

            "Z2(theta) requires theta > 0."

        )



    theta_rad = np.deg2rad(

        theta_deg

    )



    dsigma_fm2_per_sr = (

        dsigma_cm2_per_sr

        * 1.0e26

    )



    return (

        (

            4.0

            * ALPHA_ENERGY_MEV

        )

        / (

            Z1_ALPHA

            * KE2_MEV_FM

        )

        * np.sin(

            theta_rad / 2.0

        )**2

        * np.sqrt(

            dsigma_fm2_per_sr

        )

    )





# Pure-theory Z2(theta): should be exactly 79.

theta_theory = np.linspace(

    0.5,

    15.0,

    1500,

)



theory_cross_section = (

    rutherford_cross_section_cm2_per_sr(

        theta_theory,

        Z2=Z2_GOLD,

    )

)



Z2_theory = infer_Z2(

    theta_theory,

    theory_cross_section,

)





# ============================================================

# 2. DATA LOADING HELPERS

import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

import math

# ============================================================



def load_run(path):

    df = pd.read_csv(

        path

    )



    df["timestamp"] = pd.to_datetime(

        df["timestamp"],

        errors="coerce",

    )



    df["pulse_height_V"] = pd.to_numeric(

        df["pulse_height_V"],

        errors="coerce",

    )



    return (

        df

        .dropna(

            subset=[

                "timestamp",

                "pulse_height_V",

            ]

        )

        .sort_values(

            "timestamp"

        )

        .reset_index(

            drop=True

        )

    )





# ============================================================

# 3. FIXED-ANGLE AVERAGES

#

# This copies the averaging method from the plotting file:

#   - bin the whole run

#   - estimate a per-run 3-sigma baseline

#   - only reject >3-sigma points near the first/last 15%

#   - average retained counts / retained live time

# ============================================================



EDGE_FRACTION = 0.15





def make_rate_bins(

    df,

    bin_width_s,

    rate_unit_s,

):

    start = df[

        "timestamp"

    ].min()



    end = df[

        "timestamp"

    ].max()



    total_s = (

        end - start

    ).total_seconds()



    elapsed = (

        df["timestamp"]

        - start

    ).dt.total_seconds().to_numpy()



    accepted = (

        df["pulse_height_V"].to_numpy()

        >= THRESHOLD_V

    )



    edges = np.arange(

        0.0,

        total_s + bin_width_s,

        bin_width_s,

    )



    if edges[-1] > total_s:

        edges[-1] = total_s



    if edges[-1] < total_s:

        edges = np.append(

            edges,

            total_s,

        )



    edges = np.unique(

        edges

    )



    rows = []



    for i, (

        left,

        right,

    ) in enumerate(

        zip(

            edges[:-1],

            edges[1:],

        )

    ):

        live_s = (

            right - left

        )



        mask = (

            (elapsed >= left)

            & (elapsed < right)

            & accepted

        )



        N = int(

            mask.sum()

        )



        live_units = (

            live_s / rate_unit_s

        )



        rate = (

            N / live_units

        )



        rows.append({

            "bin_index": i,

            "live_time_s": live_s,

            "accepted_counts": N,

            "rate": rate,

        })



    return pd.DataFrame(

        rows

    )





def iterative_sigma_baseline(

    values

):

    values = np.asarray(

        values,

        dtype=float,

    )



    mask = np.isfinite(

        values

    )



    for _ in range(20):

        current = values[

            mask

        ]



        mean = current.mean()

        std = current.std(

            ddof=1

        )



        if std <= 0:

            break



        new_mask = (

            np.isfinite(values)

            & (

                np.abs(

                    values - mean

                )

                <= 3.0 * std

            )

        )



        if np.array_equal(

            new_mask,

            mask,

        ):

            mask = new_mask

            break



        mask = new_mask



    current = values[

        mask

    ]



    return (

        current.mean(),

        current.std(

            ddof=1

        ),

    )





def mark_edge_outliers(

    table

):

    table = table.copy()



    mean, std = (

        iterative_sigma_baseline(

            table[

                "rate"

            ].to_numpy()

        )

    )



    n = len(

        table

    )



    edge_n = max(

        1,

        int(

            np.ceil(

                EDGE_FRACTION

                * n

            )

        ),

    )



    near_edge = np.zeros(

        n,

        dtype=bool,

    )



    near_edge[

        :edge_n

    ] = True



    near_edge[

        -edge_n:

    ] = True



    z = (

        np.abs(

            table[

                "rate"

            ].to_numpy()

            - mean

        )

        / std

    )



    table[

        "excluded_edge_outlier"

    ] = (

        near_edge

        & (

            z > 3.0

        )

    )



    return table





FIXED_GROUPS = {

    # physical_angle_deg: (source_nominal_angle_deg, bin_s, rate_unit_s, runs)

    3.0: (

        5.0,

        60.0,

        60.0,

        [

            ("3 deg run 1", FILES["5_run1"]),

            ("3 deg run 2", FILES["5_run2"]),

        ],

    ),

    8.0: (

        10.0,

        3600.0,

        3600.0,

        [

            ("8 deg run 1", FILES["10_run1"]),

            ("8 deg run 2", FILES["10_run2"]),

        ],

    ),

    13.0: (

        15.0,

        3600.0,

        3600.0,

        [

            ("13 deg overnight run 1", FILES["15_run1"]),

            ("13 deg weekend run", FILES["15_run2"]),

            ("13 deg overnight run 3", FILES["15_run3"]),

        ],

    ),

}





fixed_run_rows = []


for theta_deg, (
    source_nominal_angle,
    bin_s,
    rate_unit_s,
    runs,
) in FIXED_GROUPS.items():

    for label, path in runs:
        run_df = load_run(path)
        run_start = run_df["timestamp"].min()
        run_N_areal = areal_density_for_timestamp(run_start)

        table = mark_edge_outliers(
            make_rate_bins(
                run_df,
                bin_s,
                rate_unit_s,
            )
        )

        kept = table[
            ~table["excluded_edge_outlier"]
        ].copy()

        counts = kept["accepted_counts"].sum()
        live_s = kept["live_time_s"].sum()
        live_h = live_s / 3600.0
        rate_h = counts / live_h
        poisson_sigma_h = np.sqrt(counts) / live_h

        total_sigma_h = np.hypot(
            poisson_sigma_h,
            GEOMETRY_REL_SIGMA * rate_h,
        )

        # The first nominal-15-degree run is the punctured-foil run
        # described in the report. Keep it visible, but exclude it from
        # the quantitative combined 13-degree point.
        punctured_foil_run = (
            np.isclose(theta_deg, 13.0)
            and Path(path) == FILES["15_run1"]
        )

        fixed_run_rows.append({
            "run": label,
            "run_start_timestamp": run_start,
            "source_nominal_angle_deg": source_nominal_angle,
            "theta_deg": theta_deg,
            "accepted_counts": counts,
            "live_time_s": live_s,
            "rate_counts_per_hour": rate_h,
            "sigma_rate_counts_per_hour": total_sigma_h,
            "N_areal_cm2": run_N_areal,
            "areal_exposure_cm2_s": run_N_areal * live_s,
            "punctured_foil_run": punctured_foil_run,
            "use_for_quantitative_combination": not punctured_foil_run,
        })


fixed_runs = pd.DataFrame(
    fixed_run_rows
)



fixed_runs.to_csv(

    OUTPUT_DIR

    / "fixed_angle_Z2_individual_runs.csv",

    index=False,

)





# Combine repeated fixed-angle runs by summing counts and live time.

fixed_combined = (

    fixed_runs[
        fixed_runs["use_for_quantitative_combination"]
    ]

    .groupby(

        "theta_deg",

        as_index=False,

    )

    .agg(
        accepted_counts=(
            "accepted_counts",
            "sum",
        ),
        live_time_s=(
            "live_time_s",
            "sum",
        ),
        areal_exposure_cm2_s=(
            "areal_exposure_cm2_s",
            "sum",
        ),
        n_runs=(
            "run",
            "count",
        ),
    )
)



fixed_combined[
    "N_areal_cm2"
] = (
    fixed_combined["areal_exposure_cm2_s"]
    / fixed_combined["live_time_s"]
)


fixed_combined[

    "rate_counts_per_hour"

] = (

    fixed_combined[

        "accepted_counts"

    ]

    / (

        fixed_combined[

            "live_time_s"

        ]

        / 3600.0

    )

)



fixed_combined[

    "poisson_sigma_counts_per_hour"

] = (

    np.sqrt(

        fixed_combined[

            "accepted_counts"

        ]

    )

    / (

        fixed_combined[

            "live_time_s"

        ]

        / 3600.0

    )

)



fixed_combined[

    "sigma_rate_counts_per_hour"

] = np.hypot(

    fixed_combined[

        "poisson_sigma_counts_per_hour"

    ],

    GEOMETRY_REL_SIGMA

    * fixed_combined[

        "rate_counts_per_hour"

    ],

)


# ------------------------------------------------------------
# Fixed-angle empirical run-to-run repeatability
# ------------------------------------------------------------
#
# Compare the repeated measurements at the SAME physical angle after
# dividing by the foil areal density.  This prevents a known foil change
# from being misidentified as "bad reproducibility".
#
# The punctured-foil 13-degree run is deliberately excluded from this
# estimate, because it is a known damaged-target condition rather than
# ordinary repeatability scatter.

_fixed_rep = fixed_runs[
    fixed_runs["use_for_quantitative_combination"]
].copy()

_fixed_rep["three_degree_signal_factor"] = 1.0
_fixed_rep.loc[
    np.isclose(_fixed_rep["theta_deg"], 3.0),
    "three_degree_signal_factor",
] = THREE_DEGREE_RETAINED_FRACTION

_fixed_rep["normalized_repeatability_signal"] = (
    _fixed_rep["rate_counts_per_hour"]
    * _fixed_rep["three_degree_signal_factor"]
    / _fixed_rep["N_areal_cm2"]
)

_fixed_rep_rows = []
for _theta, _group in _fixed_rep.groupby("theta_deg"):
    _vals = _group["normalized_repeatability_signal"].to_numpy(dtype=float)
    _vals = _vals[np.isfinite(_vals) & (_vals > 0)]

    if len(_vals) >= 2 and np.mean(_vals) > 0:
        _mean = float(np.mean(_vals))
        _sd = float(np.std(_vals, ddof=1))
        _rel = _sd / abs(_mean)
    else:
        _mean = float(np.mean(_vals)) if len(_vals) else np.nan
        _sd = np.nan
        _rel = 0.0

    _fixed_rep_rows.append({
        "theta_deg": float(_theta),
        "n_repeats": int(len(_vals)),
        "normalized_signal_mean": _mean,
        "normalized_signal_sd": _sd,
        "reproducibility_rel_sigma": float(_rel),
    })

fixed_repeatability = pd.DataFrame(_fixed_rep_rows)
fixed_repeatability.to_csv(
    OUTPUT_DIR / "fixed_angle_repeatability_by_angle.csv",
    index=False,
)

_fixed_rep_map = dict(
    zip(
        fixed_repeatability["theta_deg"],
        fixed_repeatability["reproducibility_rel_sigma"],
    )
)

fixed_runs["reproducibility_rel_sigma"] = (
    fixed_runs["theta_deg"].map(_fixed_rep_map).fillna(0.0)
)

fixed_combined["reproducibility_rel_sigma"] = (
    fixed_combined["theta_deg"].map(_fixed_rep_map).fillna(0.0)
)

if not USE_REPEATABILITY_SYSTEMATIC:
    fixed_runs["reproducibility_rel_sigma"] = 0.0
    fixed_combined["reproducibility_rel_sigma"] = 0.0





# ============================================================

# 4. PER-DEGREE RUNS

# ============================================================



RUN1_OFFSET_S = 63.0

ANGLE_MEASUREMENT_S = 120.0

ANGLE_CHANGE_S = 5.0





def incremental_summary(

    path,

    label,

    leading_offset_s,

    extended_angle12=False,

):

    df = load_run(

        path

    )



    raw_start = df[

        "timestamp"

    ].min()

    run_N_areal = areal_density_for_timestamp(raw_start)



    raw_end = df[

        "timestamp"

    ].max()



    zero_start = (

        raw_start

        + pd.to_timedelta(

            leading_offset_s,

            unit="s",

        )

    )



    elapsed = (

        df[

            "timestamp"

        ]

        - zero_start

    ).dt.total_seconds().to_numpy()



    accepted_v = (

        df[

            "pulse_height_V"

        ].to_numpy()

        >= THRESHOLD_V

    )



    available_end_s = (

        raw_end

        - zero_start

    ).total_seconds()



    rows = []

    t = 0.0



    for nominal_angle in range(16):



        measurement_s = (

            240.0

            if (

                extended_angle12

                and nominal_angle == 12

            )

            else

            ANGLE_MEASUREMENT_S

        )



        start_s = t

        scheduled_end_s = (

            t + measurement_s

        )



        actual_end_s = min(

            scheduled_end_s,

            available_end_s,

        )



        live_s = max(

            0.0,

            actual_end_s

            - start_s,

        )



        if live_s > 0:

            mask = (

                (elapsed >= start_s)

                & (

                    elapsed

                    < actual_end_s

                )

                & accepted_v

            )



            N = int(

                mask.sum()

            )



            live_h = (

                live_s / 3600.0

            )



            rate_h = (

                N / live_h

            )



            poisson_sigma_h = (

                np.sqrt(N)

                / live_h

            )



            total_sigma_h = np.hypot(

                poisson_sigma_h,

                GEOMETRY_REL_SIGMA

                * rate_h,

            )



        else:

            N = 0

            rate_h = np.nan

            total_sigma_h = np.nan



        calibrated = (

            nominal_angle

            - TRUE_ZERO_OFFSET_DEG

        )



        rows.append({

            "run": label,

            "nominal_angle_deg": nominal_angle,

            "calibrated_angle_deg": calibrated,

            "radial_angle_deg": abs(calibrated),

            "accepted_counts": N,

            "live_time_s": live_s,

            "rate_counts_per_hour": rate_h,

            "sigma_rate_counts_per_hour": total_sigma_h,
            "N_areal_cm2": run_N_areal,
            "run_start_timestamp": raw_start,
            "areal_exposure_cm2_s": run_N_areal * live_s,

        })



        t = (

            scheduled_end_s

            + ANGLE_CHANGE_S

        )



    return pd.DataFrame(

        rows

    )





inc1 = incremental_summary(

    FILES[

        "inc_run1"

    ],

    "Incremental run 1",

    63.0,

    False,

)



inc2 = incremental_summary(

    FILES[

        "inc_run2"

    ],

    "Incremental run 2",

    0.0,

    False,

)



inc3 = incremental_summary(

    FILES[

        "inc_run3"

    ],

    "Incremental run 3",

    0.0,

    True,

)



incremental_all = pd.concat(

    [

        inc1,

        inc2,

        inc3,

    ],

    ignore_index=True,

)



# For Rutherford scattering, +theta and -theta have the same

# scattering-angle magnitude.  Combine equal radial angles by

# summing counts and live time.

incremental_radial = (

    incremental_all[

        incremental_all[

            "live_time_s"

        ] > 0

    ]

    .groupby(

        "radial_angle_deg",

        as_index=False,

    )

    .agg(
        accepted_counts=(
            "accepted_counts",
            "sum",
        ),
        live_time_s=(
            "live_time_s",
            "sum",
        ),
        areal_exposure_cm2_s=(
            "areal_exposure_cm2_s",
            "sum",
        ),
    )
)



incremental_radial[
    "N_areal_cm2"
] = (
    incremental_radial["areal_exposure_cm2_s"]
    / incremental_radial["live_time_s"]
)


incremental_radial[

    "rate_counts_per_hour"

] = (

    incremental_radial[

        "accepted_counts"

    ]

    / (

        incremental_radial[

            "live_time_s"

        ]

        / 3600.0

    )

)



incremental_radial[

    "poisson_sigma_counts_per_hour"

] = (

    np.sqrt(

        incremental_radial[

            "accepted_counts"

        ]

    )

    / (

        incremental_radial[

            "live_time_s"

        ]

        / 3600.0

    )

)



incremental_radial[

    "sigma_rate_counts_per_hour"

] = np.hypot(

    incremental_radial[

        "poisson_sigma_counts_per_hour"

    ],

    GEOMETRY_REL_SIGMA

    * incremental_radial[

        "rate_counts_per_hour"

    ],

)


# ------------------------------------------------------------
# Per-degree empirical run-to-run repeatability
# ------------------------------------------------------------
#
# First combine +theta and -theta WITHIN EACH run, then compare the
# independently repeated runs at the same radial scattering angle.
# Normalize by N_areal so the foil replacement is not counted as
# repeatability scatter.

_inc_run_radial = (
    incremental_all[
        incremental_all["live_time_s"] > 0
    ]
    .groupby(
        ["run", "radial_angle_deg"],
        as_index=False,
    )
    .agg(
        accepted_counts=("accepted_counts", "sum"),
        live_time_s=("live_time_s", "sum"),
        areal_exposure_cm2_s=("areal_exposure_cm2_s", "sum"),
    )
)

_inc_run_radial["N_areal_cm2"] = (
    _inc_run_radial["areal_exposure_cm2_s"]
    / _inc_run_radial["live_time_s"]
)

_inc_run_radial["rate_counts_per_hour"] = (
    _inc_run_radial["accepted_counts"]
    / (_inc_run_radial["live_time_s"] / 3600.0)
)

_inc_run_radial["three_degree_signal_factor"] = 1.0
_inc_run_radial.loc[
    np.isclose(_inc_run_radial["radial_angle_deg"], 3.0),
    "three_degree_signal_factor",
] = THREE_DEGREE_RETAINED_FRACTION

_inc_run_radial["normalized_repeatability_signal"] = (
    _inc_run_radial["rate_counts_per_hour"]
    * _inc_run_radial["three_degree_signal_factor"]
    / _inc_run_radial["N_areal_cm2"]
)

_inc_rep_rows = []
for _theta, _group in _inc_run_radial.groupby("radial_angle_deg"):
    _vals = _group["normalized_repeatability_signal"].to_numpy(dtype=float)
    _vals = _vals[np.isfinite(_vals) & (_vals > 0)]

    if len(_vals) >= 2 and np.mean(_vals) > 0:
        _mean = float(np.mean(_vals))
        _sd = float(np.std(_vals, ddof=1))
        _rel = _sd / abs(_mean)
    else:
        _mean = float(np.mean(_vals)) if len(_vals) else np.nan
        _sd = np.nan
        _rel = 0.0

    _inc_rep_rows.append({
        "theta_deg": float(_theta),
        "n_repeats": int(len(_vals)),
        "normalized_signal_mean": _mean,
        "normalized_signal_sd": _sd,
        "reproducibility_rel_sigma": float(_rel),
    })

incremental_repeatability = pd.DataFrame(_inc_rep_rows)
incremental_repeatability.to_csv(
    OUTPUT_DIR / "per_degree_repeatability_by_angle.csv",
    index=False,
)

_inc_rep_map = dict(
    zip(
        incremental_repeatability["theta_deg"],
        incremental_repeatability["reproducibility_rel_sigma"],
    )
)

incremental_radial["reproducibility_rel_sigma"] = (
    incremental_radial["radial_angle_deg"].map(_inc_rep_map).fillna(0.0)
)

if not USE_REPEATABILITY_SYSTEMATIC:
    incremental_radial["reproducibility_rel_sigma"] = 0.0





# ============================================================

# 5. NO-FOIL I0

#

# Use the nominal 2 deg block, because nominal 2 deg is the

# physical beam zero.

# ============================================================



NO_FOIL_TRANSITIONS = [

    (

        4.0 * 60.0,

        4.0 * 60.0 + 5.0,

    ),

    (

        8.0 * 60.0,

        8.0 * 60.0 + 5.0,

    ),

]





def overlap_seconds(

    a0,

    a1,

    b0,

    b1,

):

    return max(

        0.0,

        min(

            a1,

            b1,

        )

        - max(

            a0,

            b0,

        ),

    )





def no_foil_bins(

    df

):

    start = df[

        "timestamp"

    ].min()



    end = df[

        "timestamp"

    ].max()



    total_s = (

        end - start

    ).total_seconds()



    elapsed = (

        df[

            "timestamp"

        ]

        - start

    ).dt.total_seconds().to_numpy()



    accepted_v = (

        df[

            "pulse_height_V"

        ].to_numpy()

        >= THRESHOLD_V

    )



    edges = np.arange(

        0.0,

        total_s + 60.0,

        60.0,

    )



    if edges[-1] > total_s:

        edges[-1] = total_s



    edges = np.unique(

        edges

    )



    rows = []



    for left, right in zip(

        edges[:-1],

        edges[1:],

    ):

        live_s = (

            right - left

        )



        for w0, w1 in NO_FOIL_TRANSITIONS:

            live_s -= overlap_seconds(

                left,

                right,

                w0,

                w1,

            )



        if live_s <= 0:

            continue



        mask = (

            (elapsed >= left)

            & (elapsed < right)

            & accepted_v

        )



        for w0, w1 in NO_FOIL_TRANSITIONS:

            mask &= ~(

                (elapsed >= w0)

                & (elapsed < w1)

            )



        N = int(

            mask.sum()

        )



        center_min = (

            0.5

            * (

                left + right

            )

            / 60.0

        )



        if center_min < 4.0:

            nominal_angle = 0

        elif center_min < 8.0:

            nominal_angle = 1

        else:

            nominal_angle = 2



        rows.append({

            "nominal_angle_deg": nominal_angle,

            "accepted_counts": N,

            "live_time_s": live_s,

        })



    return pd.DataFrame(

        rows

    )





no_foil = no_foil_bins(

    load_run(

        FILES[

            "no_foil"

        ]

    )

)



zero_block = no_foil[

    no_foil[

        "nominal_angle_deg"

    ] == 2

]



I0_counts = (

    zero_block[

        "accepted_counts"

    ].sum()

)



I0_live_s = (

    zero_block[

        "live_time_s"

    ].sum()

)



I0_rate_h = (

    I0_counts

    / (

        I0_live_s

        / 3600.0

    )

)



I0_poisson_sigma_h = (

    np.sqrt(

        I0_counts

    )

    / (

        I0_live_s

        / 3600.0

    )

)



I0_sigma_h = np.hypot(

    I0_poisson_sigma_h,

    GEOMETRY_REL_SIGMA

    * I0_rate_h,

)





# ============================================================

# 6. CONVERT FIXED-ANGLE COUNT RATES -> DIFFERENTIAL CROSS SECTION

#

# Experimental relation used in the report:

#

#       dσ/dΩ = Is / (ΔΩ I0 N)

#

# where

#   Is = measured count rate WITH the gold foil for each fixed-angle run,

#   I0 = incident-beam count rate WITHOUT the foil at nominal 2°

#        (the experimentally identified physical beam zero),

#   ΔΩ = 0.14 sr,

#   N = 8.26e18 cm^-2 through 9/1/2026, then 4.72e18 cm^-2 after 9/1/2026.

#

# The same conversion is applied below to both the relabeled fixed-angle

# measurements and the combined per-degree scans.

# ============================================================





def add_cross_section_columns(table):

    """

    Add experimental dσ/dΩ columns using



        dσ/dΩ = Is / (ΔΩ I0 N).



    The physical 3-degree measurement retains 98.17% of the measured
    signal (1.83% removed). Raw and corrected rates are both retained
    in the output for auditing.

    """

    out = table.copy()



    theta_values = out["theta_deg"].to_numpy(dtype=float)

    raw_Is = out["rate_counts_per_hour"].to_numpy(dtype=float)



    live_h = out["live_time_s"].to_numpy(dtype=float) / 3600.0

    counts = out["accepted_counts"].to_numpy(dtype=float)



    raw_sigma_Is = np.sqrt(counts) / live_h



    correction_factor = np.ones_like(raw_Is, dtype=float)

    correction_factor[np.isclose(theta_values, 3.0)] = THREE_DEGREE_RETAINED_FRACTION



    Is = raw_Is * correction_factor

    sigma_Is = raw_sigma_Is * correction_factor



    if "N_areal_cm2" not in out.columns:
        raise ValueError(
            "Cross-section table is missing N_areal_cm2. "
            "Each run must be tagged with the correct foil based on its timestamp."
        )

    N_areal_values = out["N_areal_cm2"].to_numpy(dtype=float)

    dsigma_cm2_per_sr = (
        Is
        / (
            SOLID_ANGLE_SR
            * I0_rate_h
            * N_areal_values
        )
    )



    # --------------------------------------------------------
    # Vertical uncertainty budget
    # --------------------------------------------------------
    #
    # Measured cross section:
    #
    #   dσ/dΩ = Is / (ΔΩ I0 N)
    #
    # The y-error therefore contains:
    #   1) Poisson uncertainty of the scattered counts Is,
    #   2) Poisson uncertainty of the no-foil beam rate I0,
    #   3) detector solid-angle uncertainty,
    #   4) foil areal-density uncertainty (0 until independently known),
    #   5) empirical run-to-run repeatability scatter at this angle.
    #
    # Angle uncertainty is NOT folded into this measured y-error.  It is
    # retained as x-error and projected into y only when evaluating the
    # Rutherford fit / chi-square.

    if "reproducibility_rel_sigma" in out.columns:
        reproducibility_rel = (
            out["reproducibility_rel_sigma"]
            .to_numpy(dtype=float)
        )
    else:
        reproducibility_rel = np.zeros_like(Is, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_Is_poisson = sigma_Is / Is
        rel_I0_poisson = np.full_like(
            Is,
            I0_poisson_sigma_h / I0_rate_h,
            dtype=float,
        )
        rel_solid_angle = np.full_like(
            Is,
            SOLID_ANGLE_SIGMA_SR / SOLID_ANGLE_SR,
            dtype=float,
        )
        rel_areal_density = np.full_like(
            Is,
            N_AREAL_REL_SIGMA,
            dtype=float,
        )

        relative_sigma = np.sqrt(
            rel_Is_poisson**2
            + rel_I0_poisson**2
            + rel_solid_angle**2
            + rel_areal_density**2
            + reproducibility_rel**2
        )

    sigma_dsigma_cm2_per_sr = dsigma_cm2_per_sr * relative_sigma

    # For goodness-of-fit / shape statistics, separate independent
    # point-to-point scatter from common normalization systematics.
    #
    # Point-to-point:
    #   - scattered-count Poisson uncertainty
    #   - measured run-to-run repeatability at the same angle
    #
    # Common normalization:
    #   - I0 normalization
    #   - detector solid angle
    #   - foil areal-density normalization (when an uncertainty is supplied)
    #
    # The FULL uncertainty above is still used for the plotted y-error bars.
    rel_sigma_point_to_point = np.sqrt(
        rel_Is_poisson**2
        + reproducibility_rel**2
    )

    rel_sigma_common_normalization = np.sqrt(
        rel_I0_poisson**2
        + rel_solid_angle**2
        + rel_areal_density**2
    )

    sigma_dsigma_point_to_point = (
        dsigma_cm2_per_sr
        * rel_sigma_point_to_point
    )


    out["raw_rate_counts_per_hour"] = raw_Is
    out["three_degree_signal_factor"] = correction_factor
    out["analysis_rate_counts_per_hour"] = Is
    out["sigma_analysis_rate_counts_per_hour"] = sigma_Is

    out["differential_cross_section_cm2_per_sr"] = dsigma_cm2_per_sr

    # Keep every contribution explicit for the error-analysis table.
    out["rel_sigma_Is_poisson"] = rel_Is_poisson
    out["rel_sigma_I0_poisson"] = rel_I0_poisson
    out["rel_sigma_solid_angle"] = rel_solid_angle
    out["rel_sigma_areal_density"] = rel_areal_density
    out["rel_sigma_repeatability"] = reproducibility_rel
    out["rel_sigma_cross_section_total"] = relative_sigma
    out["rel_sigma_cross_section_point_to_point"] = (
        rel_sigma_point_to_point
    )
    out["rel_sigma_cross_section_common_normalization"] = (
        rel_sigma_common_normalization
    )
    out["sigma_differential_cross_section_point_to_point"] = (
        sigma_dsigma_point_to_point
    )

    out["sigma_cross_section_from_Is"] = (
        dsigma_cm2_per_sr * rel_Is_poisson
    )
    out["sigma_cross_section_from_I0"] = (
        dsigma_cm2_per_sr * rel_I0_poisson
    )
    out["sigma_cross_section_from_solid_angle"] = (
        dsigma_cm2_per_sr * rel_solid_angle
    )
    out["sigma_cross_section_from_areal_density"] = (
        dsigma_cm2_per_sr * rel_areal_density
    )
    out["sigma_cross_section_from_repeatability"] = (
        dsigma_cm2_per_sr * reproducibility_rel
    )
    out["sigma_differential_cross_section_cm2_per_sr"] = (
        sigma_dsigma_cm2_per_sr
    )

    out["angle_sigma_deg"] = TOTAL_ANGLE_SIGMA_DEG



    return out





def add_inferred_Z_columns(table):

    """Infer target atomic number Z2 from each measured cross section."""

    out = table.copy()



    theta_deg = out["theta_deg"].to_numpy(dtype=float)

    dsigma = out["differential_cross_section_cm2_per_sr"].to_numpy(dtype=float)

    sigma_dsigma = out["sigma_differential_cross_section_cm2_per_sr"].to_numpy(dtype=float)



    Z2 = infer_Z2(theta_deg, dsigma)



    # Z ∝ sin^2(theta/2) * sqrt(dσ/dΩ).

    # Propagate cross-section uncertainty plus the existing angle uncertainty.

    theta_rad = np.deg2rad(theta_deg)

    sigma_theta_rad = np.deg2rad(out["angle_sigma_deg"].to_numpy(dtype=float))



    with np.errstate(divide="ignore", invalid="ignore"):

        rel_from_cross_section = 0.5 * sigma_dsigma / dsigma

        rel_from_angle = np.abs(1.0 / np.tan(theta_rad / 2.0)) * sigma_theta_rad

        relative_sigma_Z = np.sqrt(

            rel_from_cross_section**2

            + rel_from_angle**2

        )



    out["inferred_Z2"] = Z2

    # Keep the two contributions separate in the CSV so the error budget
    # is auditable.  The plotted sigma_inferred_Z2 remains the quadrature
    # total used everywhere else.
    out["sigma_Z_from_cross_section"] = Z2 * rel_from_cross_section
    out["sigma_Z_from_angle"] = Z2 * rel_from_angle
    out["sigma_inferred_Z2"] = Z2 * relative_sigma_Z

    return out





# Keep every fixed-angle run separate for the measurement plot.

fixed_cross_sections = add_inferred_Z_columns(

    add_cross_section_columns(fixed_runs)

)



fixed_cross_sections.to_csv(

    OUTPUT_DIR / "fixed_angle_differential_cross_sections_individual_runs.csv",

    index=False,

)



# Also combine the repeated fixed-angle runs into one experimental point

# at each physical angle (3, 8, and 13 degrees) for the Z inversion plots.

fixed_combined_cross_sections = add_inferred_Z_columns(

    add_cross_section_columns(fixed_combined)

)



fixed_combined_cross_sections.to_csv(

    OUTPUT_DIR / "fixed_angle_differential_cross_sections_combined.csv",

    index=False,

)



# Per-degree scans: use the already combined radial-angle count rates.

# Drop theta = 0 because Rutherford theory is singular there.

incremental_for_cross_section = (

    incremental_radial[

        incremental_radial["radial_angle_deg"] > 0

    ]

    .rename(columns={"radial_angle_deg": "theta_deg"})

    .copy()

)



incremental_cross_sections = add_inferred_Z_columns(

    add_cross_section_columns(incremental_for_cross_section)

)



incremental_cross_sections.to_csv(

    OUTPUT_DIR / "per_degree_differential_cross_sections.csv",

    index=False,

)


# Diagnostic ratio to the absolute Z=79 Rutherford prediction.
# A constant ratio below 1 would point mainly to a normalization/efficiency
# issue; strong angle dependence points to angle-dependent systematics.
for _table in (
    incremental_cross_sections,
    fixed_cross_sections,
    fixed_combined_cross_sections,
):
    _table["gold_theory_cross_section_cm2_per_sr"] = (
        rutherford_cross_section_cm2_per_sr(
            _table["theta_deg"].to_numpy(dtype=float),
            Z2=Z2_GOLD,
        )
    )
    _table["measured_to_gold_theory_ratio"] = (
        _table["differential_cross_section_cm2_per_sr"]
        / _table["gold_theory_cross_section_cm2_per_sr"]
    )

# Re-save with the diagnostic columns included.
incremental_cross_sections.to_csv(
    OUTPUT_DIR / "per_degree_differential_cross_sections.csv",
    index=False,
)
fixed_cross_sections.to_csv(
    OUTPUT_DIR / "fixed_angle_differential_cross_sections_individual_runs.csv",
    index=False,
)
fixed_combined_cross_sections.to_csv(
    OUTPUT_DIR / "fixed_angle_differential_cross_sections_combined.csv",
    index=False,
)






# ============================================================
# 6.5 MANUAL-STYLE LINEARIZED RUTHERFORD FIT
#
# The lab manual recommends turning the angular dependence into a
# straight-line problem:
#
#       x = csc^4(theta/2)
#       y = dσ/dΩ
#       y = m x
#
# The slope m contains the Rutherford normalization and therefore Z.
# A y = m x + b fit is also shown only as a diagnostic for a residual
# angle-independent background/offset.
#
# Low-angle points below the multiple-scattering limit in the report
# are excluded. High-angle points are NOT silently discarded.
# ============================================================

fit_points = incremental_cross_sections[
    (incremental_cross_sections["theta_deg"] >= FIT_MIN_ANGLE_DEG)
    & (incremental_cross_sections["theta_deg"] <= FIT_MAX_ANGLE_DEG)
    & np.isfinite(
        incremental_cross_sections["differential_cross_section_cm2_per_sr"]
    )
    & (incremental_cross_sections["differential_cross_section_cm2_per_sr"] > 0)
].copy()

if len(fit_points) < 3:
    raise ValueError(
        "Not enough valid points for the Rutherford linearized fit. "
        f"Need at least 3 points in {FIT_MIN_ANGLE_DEG:g}°–{FIT_MAX_ANGLE_DEG:g}°."
    )


def rutherford_linearized_x_and_sigma(theta_deg, sigma_theta_deg):
    """
    x = csc^4(theta/2), including first-order angular uncertainty.

    For theta in radians:
        dx/dtheta = -2 x cot(theta/2)
    """
    theta_deg = np.asarray(theta_deg, dtype=float)
    sigma_theta_deg = np.asarray(sigma_theta_deg, dtype=float)

    theta_rad = np.deg2rad(theta_deg)
    sigma_theta_rad = np.deg2rad(sigma_theta_deg)

    x = rutherford_factor(theta_deg)
    dx_dtheta = -2.0 * x / np.tan(theta_rad / 2.0)
    sigma_x = np.abs(dx_dtheta) * sigma_theta_rad
    return x, sigma_x


def fit_line_with_x_uncertainty(
    x,
    y,
    sigma_x,
    sigma_y,
    intercept=False,
    iterations=30,
    sigma_x_common=None,
):
    """
    Generalized least-squares fit with both independent and correlated
    angular uncertainty.

    Independent x error is projected into y point-by-point:

        Var_i += (m sigma_x,i)^2

    A common zero-angle uncertainty is represented by a rank-one
    covariance term:

        C_common = v v^T,
        v_i = m sigma_x_common,i.

    This prevents a single shared angle calibration uncertainty from
    being counted as though it were six independent random errors.

    intercept=False -> y = m x
    intercept=True  -> y = m x + b
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma_x = np.asarray(sigma_x, dtype=float)
    sigma_y = np.asarray(sigma_y, dtype=float)

    if sigma_x_common is None:
        sigma_x_common = np.zeros_like(x, dtype=float)
    else:
        sigma_x_common = np.asarray(
            sigma_x_common,
            dtype=float,
        )

    good = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(sigma_x)
        & np.isfinite(sigma_x_common)
        & np.isfinite(sigma_y)
        & (sigma_y > 0)
    )

    x = x[good]
    y = y[good]
    sigma_x = sigma_x[good]
    sigma_x_common = sigma_x_common[good]
    sigma_y = sigma_y[good]

    # Initial diagonal weighted fit.
    w = 1.0 / sigma_y**2

    if intercept:
        X = np.column_stack([x, np.ones_like(x)])
        cov_beta = np.linalg.inv(
            X.T @ (w[:, None] * X)
        )
        slope, offset = (
            cov_beta
            @ (X.T @ (w * y))
        )
    else:
        slope = float(
            np.sum(w * x * y)
            / np.sum(w * x**2)
        )
        offset = 0.0

    slope = float(slope)
    offset = float(offset)

    for _ in range(iterations):
        diag_var = (
            sigma_y**2
            + (slope * sigma_x)**2
        )

        common_vector = (
            slope
            * sigma_x_common
        )

        C = np.diag(diag_var)
        C += np.outer(
            common_vector,
            common_vector,
        )

        C_inv = np.linalg.pinv(C)

        if intercept:
            X = np.column_stack(
                [x, np.ones_like(x)]
            )
            normal = X.T @ C_inv @ X
            cov_beta = np.linalg.pinv(normal)
            beta = (
                cov_beta
                @ (X.T @ C_inv @ y)
            )
            new_slope = float(beta[0])
            new_offset = float(beta[1])
        else:
            denom = float(
                x.T @ C_inv @ x
            )
            new_slope = float(
                (x.T @ C_inv @ y)
                / denom
            )
            new_offset = 0.0

        if (
            np.isclose(
                new_slope,
                slope,
                rtol=1e-10,
                atol=0.0,
            )
            and np.isclose(
                new_offset,
                offset,
                rtol=1e-10,
                atol=1e-30,
            )
        ):
            slope = new_slope
            offset = new_offset
            break

        slope = new_slope
        offset = new_offset

    # Final covariance at the converged solution.
    diag_var = (
        sigma_y**2
        + (slope * sigma_x)**2
    )
    common_vector = (
        slope
        * sigma_x_common
    )

    C = np.diag(diag_var)
    C += np.outer(
        common_vector,
        common_vector,
    )
    C_inv = np.linalg.pinv(C)

    if intercept:
        X = np.column_stack(
            [x, np.ones_like(x)]
        )
        cov_beta = np.linalg.pinv(
            X.T @ C_inv @ X
        )
        sigma_slope = float(
            np.sqrt(cov_beta[0, 0])
        )
        sigma_offset = float(
            np.sqrt(cov_beta[1, 1])
        )
        model = slope * x + offset
        dof = max(len(x) - 2, 0)
    else:
        sigma_slope = float(
            1.0
            / np.sqrt(
                x.T @ C_inv @ x
            )
        )
        sigma_offset = 0.0
        model = slope * x
        dof = max(len(x) - 1, 0)

    residual = y - model
    chi2 = float(
        residual.T
        @ C_inv
        @ residual
    )

    return {
        "slope": slope,
        "sigma_slope": sigma_slope,
        "intercept": offset,
        "sigma_intercept": sigma_offset,
        "chi2": chi2,
        "dof": dof,
        "covariance": C,
        "covariance_inverse": C_inv,
        "model": model,
    }


def Z_from_linearized_slope(slope, sigma_slope):
    if slope <= 0:
        return np.nan, np.nan

    Z = (
        4.0 * ALPHA_ENERGY_MEV
        / (Z1_ALPHA * KE2_MEV_FM)
        * np.sqrt(slope / 1.0e-26)
    )
    sigma_Z = 0.5 * Z * sigma_slope / slope
    return float(Z), float(sigma_Z)


# Full errors remain available for plotting/reporting.
fit_sigma_y_full = fit_points[
    "sigma_differential_cross_section_cm2_per_sr"
].to_numpy(dtype=float)

# Shape chi-square uses only point-to-point vertical scatter.
fit_sigma_y = fit_points[
    "sigma_differential_cross_section_point_to_point"
].to_numpy(dtype=float)

# Independent angle-setting uncertainty.
_fit_angle_point_deg = np.full(
    len(fit_points),
    POINT_TO_POINT_ANGLE_SIGMA_DEG,
    dtype=float,
)

fit_x, fit_sigma_x = rutherford_linearized_x_and_sigma(
    fit_points["theta_deg"],
    _fit_angle_point_deg,
)

# Common zero-angle uncertainty: one shared offset across all points.
_, fit_sigma_x_common = rutherford_linearized_x_and_sigma(
    fit_points["theta_deg"],
    np.full(
        len(fit_points),
        COMMON_ANGLE_OFFSET_SIGMA_DEG,
        dtype=float,
    ),
)

fit_y = fit_points[
    "differential_cross_section_cm2_per_sr"
].to_numpy(dtype=float)

linear_fit_origin = fit_line_with_x_uncertainty(
    fit_x,
    fit_y,
    fit_sigma_x,
    fit_sigma_y,
    intercept=False,
    sigma_x_common=fit_sigma_x_common,
)
linear_fit_background = fit_line_with_x_uncertainty(
    fit_x,
    fit_y,
    fit_sigma_x,
    fit_sigma_y,
    intercept=True,
    sigma_x_common=fit_sigma_x_common,
)
linear_fit_Z, linear_fit_sigma_Z = Z_from_linearized_slope(
    linear_fit_origin["slope"],
    linear_fit_origin["sigma_slope"],
)

# Common normalization uncertainties do not belong in the shape chi-square,
# because the fitted slope absorbs an overall scale. They DO matter for the
# absolute inferred Z. Since Z ∝ sqrt(dσ/dΩ), a fractional normalization
# uncertainty contributes half as much fractionally to Z.
COMMON_NORMALIZATION_REL_SIGMA = np.sqrt(
    (I0_poisson_sigma_h / I0_rate_h) ** 2
    + (SOLID_ANGLE_SIGMA_SR / SOLID_ANGLE_SR) ** 2
    + N_AREAL_REL_SIGMA ** 2
)

linear_fit_sigma_Z_common_norm = (
    0.5
    * linear_fit_Z
    * COMMON_NORMALIZATION_REL_SIGMA
)

linear_fit_sigma_Z_total = np.hypot(
    linear_fit_sigma_Z,
    linear_fit_sigma_Z_common_norm,
)

# Robustness check: vary fit boundaries instead of selecting one window
# because it "looks good". Large changes in fitted Z indicate that the
# window choice is itself a systematic uncertainty.
fit_window_rows = []
for lower in [4.0, 5.0, 6.0]:
    for upper in [8.0, 9.0, 10.0, 11.0, 12.0, 13.0]:
        if upper <= lower:
            continue

        window = incremental_cross_sections[
            (incremental_cross_sections["theta_deg"] >= lower)
            & (incremental_cross_sections["theta_deg"] <= upper)
            & np.isfinite(
                incremental_cross_sections[
                    "differential_cross_section_cm2_per_sr"
                ]
            )
            & (
                incremental_cross_sections[
                    "differential_cross_section_cm2_per_sr"
                ] > 0
            )
        ].copy()

        if len(window) < 3:
            continue

        _w_point_angle = np.full(
            len(window),
            POINT_TO_POINT_ANGLE_SIGMA_DEG,
            dtype=float,
        )
        wx, wsx = rutherford_linearized_x_and_sigma(
            window["theta_deg"],
            _w_point_angle,
        )
        _, wsx_common = rutherford_linearized_x_and_sigma(
            window["theta_deg"],
            np.full(
                len(window),
                COMMON_ANGLE_OFFSET_SIGMA_DEG,
                dtype=float,
            ),
        )

        wy = window[
            "differential_cross_section_cm2_per_sr"
        ].to_numpy(dtype=float)
        wsy = window[
            "sigma_differential_cross_section_point_to_point"
        ].to_numpy(dtype=float)

        wfit = fit_line_with_x_uncertainty(
            wx,
            wy,
            wsx,
            wsy,
            intercept=False,
            sigma_x_common=wsx_common,
        )
        wz, wsz = Z_from_linearized_slope(
            wfit["slope"], wfit["sigma_slope"]
        )

        fit_window_rows.append({
            "fit_min_angle_deg": lower,
            "fit_max_angle_deg": upper,
            "n_points": len(window),
            "Z_fit": wz,
            "sigma_Z_fit": wsz,
            "chi2": wfit["chi2"],
            "dof": wfit["dof"],
            "reduced_chi2": (
                wfit["chi2"] / wfit["dof"]
                if wfit["dof"] > 0
                else np.nan
            ),
        })

fit_window_sensitivity = pd.DataFrame(fit_window_rows)
fit_window_sensitivity.to_csv(
    OUTPUT_DIR / "rutherford_fit_window_sensitivity.csv",
    index=False,
)

fit_export = fit_points[
    [
        "theta_deg",
        "differential_cross_section_cm2_per_sr",
        "sigma_differential_cross_section_cm2_per_sr",
        "angle_sigma_deg",
        "measured_to_gold_theory_ratio",
    ]
].copy()
fit_export["linearized_x_csc4"] = fit_x
fit_export["sigma_linearized_x_csc4"] = fit_sigma_x
fit_export.to_csv(
    OUTPUT_DIR / "rutherford_linearized_fit_points.csv",
    index=False,
)

fig_linear, ax_linear = plt.subplots(figsize=(10.5, 7.0))

ax_linear.errorbar(
    fit_x,
    fit_y,
    xerr=fit_sigma_x,
    yerr=fit_sigma_y,
    fmt="o",
    markersize=6,
    capsize=3,
    linewidth=1.0,
    label="Per-degree measurements (point-to-point errors used in fit)",
)

x_line = np.linspace(0.0, np.nanmax(fit_x) * 1.05, 500)

ax_linear.plot(
    x_line,
    linear_fit_origin["slope"] * x_line,
    linewidth=2.0,
    label=(
        rf"Rutherford $y=mx$: "
        rf"$Z={linear_fit_Z:.1f}\pm{linear_fit_sigma_Z:.1f}$"
    ),
)

ax_linear.plot(
    x_line,
    linear_fit_background["slope"] * x_line
    + linear_fit_background["intercept"],
    linestyle="--",
    linewidth=1.6,
    label="Diagnostic $y=mx+b$",
)

for _, row in fit_points.iterrows():
    x_point = float(rutherford_factor(row["theta_deg"]))
    ax_linear.annotate(
        f"{row['theta_deg']:g}°",
        (
            x_point,
            row["differential_cross_section_cm2_per_sr"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

ax_linear.set_xlabel(r"$x=\csc^4(\theta/2)$", fontsize=12)
ax_linear.set_ylabel(
    r"Differential cross section $d\sigma/d\Omega$ (cm$^2$/sr)",
    fontsize=12,
)
ax_linear.set_title(
    "Linearized Rutherford test of the measured angular dependence",
    fontsize=14,
)
ax_linear.grid(alpha=0.25)
ax_linear.legend(fontsize=8)
fig_linear.tight_layout()
fig_linear.savefig(
    OUTPUT_DIR / "07_rutherford_linearized_experimental_fit.png",
    dpi=240,
    bbox_inches="tight",
)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig_linear)


# ============================================================
# 6.6 ERROR ANALYSIS / GOODNESS OF FIT
# ============================================================
#
# This section keeps the distinction between measured-y uncertainty and
# angular uncertainty:
#
#   measured y-error:
#       counting + I0 + solid angle + N + repeatability
#
#   angle error:
#       kept as sigma_theta and projected into y through the slope of the
#       Rutherford model when calculating chi-square.
#
# Two goodness-of-fit questions are reported:
#
#   A) Absolute gold test:
#      Is the measured cross section compatible with the Z = 79
#      Rutherford prediction, with NO free normalization parameter?
#
#   B) Angular-shape test:
#      Does the data follow y = m csc^4(theta/2) when m is fitted?
#
# The second test is the cleaner test of Rutherford angular dependence,
# because a free m absorbs an angle-independent normalization mismatch.


def _regularized_gamma_q(a, x):
    """
    Regularized upper incomplete gamma Q(a, x), using only the Python
    standard library.  This gives chi-square survival probabilities
    without requiring scipy.
    """
    if a <= 0 or x < 0:
        return np.nan
    if x == 0:
        return 1.0

    eps = 3.0e-14
    tiny = 1.0e-300
    max_iter = 10000
    gln = math.lgamma(a)

    # Series for P(a,x), then Q = 1-P.
    if x < a + 1.0:
        ap = a
        summation = 1.0 / a
        delta = summation

        for _ in range(max_iter):
            ap += 1.0
            delta *= x / ap
            summation += delta
            if abs(delta) < abs(summation) * eps:
                break

        P = summation * math.exp(-x + a * math.log(x) - gln)
        return float(max(0.0, min(1.0, 1.0 - P)))

    # Continued fraction directly for Q(a,x).
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / max(b, tiny)
    h = d

    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break

    Q = math.exp(-x + a * math.log(x) - gln) * h
    return float(max(0.0, min(1.0, Q)))


def chi_square_p_value(chi2_value, dof):
    if dof <= 0 or not np.isfinite(chi2_value):
        return np.nan
    return _regularized_gamma_q(0.5 * dof, 0.5 * chi2_value)


def chi_square_label(reduced_chi2, p_value):
    """
    Short descriptive label only; interpretation still belongs in the report.
    """
    if not np.isfinite(reduced_chi2):
        return "undefined"
    if reduced_chi2 < 0.5:
        return "scatter smaller than assigned uncertainties"
    if reduced_chi2 <= 2.0:
        return "order-unity reduced chi-square"
    return "scatter larger than expected from assigned uncertainties"


_error_theta = fit_points["theta_deg"].to_numpy(dtype=float)
_error_theta_rad = np.deg2rad(_error_theta)
_error_sigma_theta_rad = np.deg2rad(
    fit_points["angle_sigma_deg"].to_numpy(dtype=float)
)
_error_y = fit_y.copy()
_error_sigma_y = fit_sigma_y.copy()

# ------------------------------------------------------------
# A) Absolute Z=79 Rutherford curve
# ------------------------------------------------------------
_gold_model = rutherford_cross_section_cm2_per_sr(
    _error_theta,
    Z2=Z2_GOLD,
)

# dy/dtheta for Rutherford y ~ csc^4(theta/2), with theta in radians.
_gold_dydtheta = (
    -2.0
    * _gold_model
    / np.tan(_error_theta_rad / 2.0)
)

# Independent, point-to-point angle-setting contribution.
_gold_sigma_from_point_angle = (
    np.abs(_gold_dydtheta)
    * np.deg2rad(
        POINT_TO_POINT_ANGLE_SIGMA_DEG
    )
)

# Point-to-point vertical cross-section uncertainty.
_gold_sigma_point_y = fit_sigma_y.copy()

# Start covariance with independent diagonal terms.
_gold_C = np.diag(
    _gold_sigma_point_y**2
    + _gold_sigma_from_point_angle**2
)

# Common normalization uncertainties are rank-one covariance terms.
_gold_common_norm_vector = (
    _gold_model
    * np.sqrt(
        (I0_poisson_sigma_h / I0_rate_h) ** 2
        + (SOLID_ANGLE_SIGMA_SR / SOLID_ANGLE_SR) ** 2
    )
)
_gold_C += np.outer(
    _gold_common_norm_vector,
    _gold_common_norm_vector,
)

# If a foil-density uncertainty is later supplied, it is correlated only
# among points that used the same foil density.
if N_AREAL_REL_SIGMA > 0:
    _N_values = fit_points[
        "N_areal_cm2"
    ].to_numpy(dtype=float)

    for _N_unique in np.unique(_N_values):
        _mask = np.isclose(
            _N_values,
            _N_unique,
        )
        _vN = np.zeros_like(
            _gold_model,
            dtype=float,
        )
        _vN[_mask] = (
            _gold_model[_mask]
            * N_AREAL_REL_SIGMA
        )
        _gold_C += np.outer(
            _vN,
            _vN,
        )

# Common zero-angle calibration uncertainty: one shared shift, not one
# independent 0.5-degree error for every point.
_gold_zero_vector = (
    _gold_dydtheta
    * np.deg2rad(
        COMMON_ANGLE_OFFSET_SIGMA_DEG
    )
)
_gold_C += np.outer(
    _gold_zero_vector,
    _gold_zero_vector,
)

_gold_C_inv = np.linalg.pinv(
    _gold_C
)

_gold_residual = (
    _error_y
    - _gold_model
)

gold_chi2 = float(
    _gold_residual.T
    @ _gold_C_inv
    @ _gold_residual
)
gold_dof = int(
    len(_gold_residual)
)  # no fitted parameters
gold_reduced_chi2 = (
    gold_chi2 / gold_dof
    if gold_dof > 0
    else np.nan
)
gold_p_value = chi_square_p_value(
    gold_chi2,
    gold_dof,
)

# A diagonal-only diagnostic "pull" can still be exported, but it is NOT
# what is used to calculate chi-square because the covariance is correlated.
_gold_diag_sigma = np.sqrt(
    np.diag(_gold_C)
)
_gold_pull = (
    _gold_residual
    / _gold_diag_sigma
)


# ------------------------------------------------------------
# B) Best-fit Rutherford angular shape y = m x
# ------------------------------------------------------------
_best_model = linear_fit_origin[
    "model"
]
_best_C = linear_fit_origin[
    "covariance"
]
_best_C_inv = linear_fit_origin[
    "covariance_inverse"
]

_best_residual = (
    fit_y
    - _best_model
)

best_chi2 = float(
    linear_fit_origin["chi2"]
)
best_dof = int(
    linear_fit_origin["dof"]
)
best_reduced_chi2 = (
    best_chi2 / best_dof
    if best_dof > 0
    else np.nan
)
best_p_value = chi_square_p_value(
    best_chi2,
    best_dof,
)

_best_diag_sigma = np.sqrt(
    np.diag(_best_C)
)
_best_pull = (
    _best_residual
    / _best_diag_sigma
)


# ------------------------------------------------------------
# C) Diagnostic background model y = m x + b
# ------------------------------------------------------------
_bg_model = linear_fit_background[
    "model"
]
_bg_C = linear_fit_background[
    "covariance"
]
_bg_C_inv = linear_fit_background[
    "covariance_inverse"
]

_bg_residual = (
    fit_y
    - _bg_model
)

bg_chi2 = float(
    linear_fit_background["chi2"]
)
bg_dof = int(
    linear_fit_background["dof"]
)
bg_reduced_chi2 = (
    bg_chi2 / bg_dof
    if bg_dof > 0
    else np.nan
)
bg_p_value = chi_square_p_value(
    bg_chi2,
    bg_dof,
)

# ============================================================
# 6.7 FIXED-ANGLE / OVERNIGHT / WEEKEND ERROR ANALYSIS
# ============================================================
#
# Analyze the long fixed-angle runs separately from the per-degree scan.
# The punctured 13-degree run stays visible in tables/plots but is excluded
# from all quantitative fixed-run chi-square calculations.

# ---------- A) Individual run vs Z=79 ----------
fixed_individual_error_analysis = fixed_cross_sections.copy()

_fixed_theta_all = fixed_individual_error_analysis[
    "theta_deg"
].to_numpy(dtype=float)

_fixed_theta_all_rad = np.deg2rad(_fixed_theta_all)

_fixed_gold_model_all = rutherford_cross_section_cm2_per_sr(
    _fixed_theta_all,
    Z2=Z2_GOLD,
)

_fixed_gold_dydtheta_all = (
    -2.0
    * _fixed_gold_model_all
    / np.tan(_fixed_theta_all_rad / 2.0)
)

# For a single-run descriptive comparison, use the full plotted
# uncertainties (not the covariance chi-square treatment).
_fixed_gold_sigma_angle_full = (
    np.abs(_fixed_gold_dydtheta_all)
    * np.deg2rad(TOTAL_ANGLE_SIGMA_DEG)
)

_fixed_gold_sigma_full = np.sqrt(
    fixed_individual_error_analysis[
        "sigma_differential_cross_section_cm2_per_sr"
    ].to_numpy(dtype=float) ** 2
    + _fixed_gold_sigma_angle_full ** 2
)

_fixed_gold_residual_all = (
    fixed_individual_error_analysis[
        "differential_cross_section_cm2_per_sr"
    ].to_numpy(dtype=float)
    - _fixed_gold_model_all
)

fixed_individual_error_analysis[
    "gold_theory_cross_section_cm2_per_sr"
] = _fixed_gold_model_all

fixed_individual_error_analysis[
    "gold_sigma_from_full_angle_uncertainty"
] = _fixed_gold_sigma_angle_full

fixed_individual_error_analysis[
    "gold_total_sigma_for_single_run_comparison"
] = _fixed_gold_sigma_full

fixed_individual_error_analysis[
    "gold_residual_sigma"
] = (
    _fixed_gold_residual_all
    / _fixed_gold_sigma_full
)

fixed_individual_error_analysis[
    "Z_difference_from_79_sigma"
] = (
    (
        fixed_individual_error_analysis["inferred_Z2"]
        - Z2_GOLD
    )
    / fixed_individual_error_analysis["sigma_inferred_Z2"]
)

fixed_individual_error_analysis.to_csv(
    OUTPUT_DIR / "fixed_individual_run_error_analysis.csv",
    index=False,
)


# ---------- B) Run-to-run consistency at the same angle ----------
#
# Compare q = Is/N between repeated runs.  Common I0 and DeltaOmega cancel.
# Use Poisson uncertainty only here; otherwise we'd compare the run scatter
# against a repeatability uncertainty estimated from that same scatter.

_fixed_consistency_rows = []

for _theta, _group in fixed_cross_sections.groupby("theta_deg"):
    _usable = _group[
        _group["use_for_quantitative_combination"]
    ].copy()

    if len(_usable) < 2:
        continue

    _live_h = (
        _usable["live_time_s"].to_numpy(dtype=float)
        / 3600.0
    )
    _counts = _usable["accepted_counts"].to_numpy(dtype=float)
    _correction = _usable[
        "three_degree_signal_factor"
    ].to_numpy(dtype=float)
    _N = _usable["N_areal_cm2"].to_numpy(dtype=float)

    _q = (
        _usable["analysis_rate_counts_per_hour"]
        .to_numpy(dtype=float)
        / _N
    )

    _sigma_rate_poisson = (
        np.sqrt(_counts)
        / _live_h
        * _correction
    )
    _sigma_q = _sigma_rate_poisson / _N

    _good = (
        np.isfinite(_q)
        & np.isfinite(_sigma_q)
        & (_sigma_q > 0)
    )
    _q = _q[_good]
    _sigma_q = _sigma_q[_good]

    if len(_q) < 2:
        continue

    _w = 1.0 / _sigma_q**2
    _q_mean = float(np.sum(_w * _q) / np.sum(_w))
    _chi2 = float(
        np.sum(((_q - _q_mean) / _sigma_q) ** 2)
    )
    _dof = int(len(_q) - 1)
    _p = chi_square_p_value(_chi2, _dof)

    _sample_mean = float(np.mean(_q))
    _sample_sd = float(np.std(_q, ddof=1))
    _fractional_spread = (
        _sample_sd / abs(_sample_mean)
        if _sample_mean != 0
        else np.nan
    )

    _fixed_consistency_rows.append({
        "theta_deg": float(_theta),
        "n_runs": int(len(_q)),
        "weighted_mean_normalized_signal": _q_mean,
        "sample_fractional_spread": _fractional_spread,
        "chi2": _chi2,
        "dof": _dof,
        "reduced_chi2": (
            _chi2 / _dof if _dof > 0 else np.nan
        ),
        "p_value": _p,
    })

fixed_run_consistency = pd.DataFrame(_fixed_consistency_rows)
fixed_run_consistency.to_csv(
    OUTPUT_DIR / "fixed_run_consistency_chi_square.csv",
    index=False,
)


# ---------- C) All undamaged long runs vs absolute Z=79 ----------
_fixed_valid = fixed_cross_sections[
    fixed_cross_sections["use_for_quantitative_combination"]
].copy()

_fixed_theta = _fixed_valid["theta_deg"].to_numpy(dtype=float)
_fixed_theta_rad = np.deg2rad(_fixed_theta)
_fixed_y = _fixed_valid[
    "differential_cross_section_cm2_per_sr"
].to_numpy(dtype=float)

_fixed_sigma_y_point = _fixed_valid[
    "sigma_differential_cross_section_point_to_point"
].to_numpy(dtype=float)

_fixed_gold_model = rutherford_cross_section_cm2_per_sr(
    _fixed_theta,
    Z2=Z2_GOLD,
)

_fixed_gold_dydtheta = (
    -2.0
    * _fixed_gold_model
    / np.tan(_fixed_theta_rad / 2.0)
)

_fixed_sigma_from_point_angle = (
    np.abs(_fixed_gold_dydtheta)
    * np.deg2rad(POINT_TO_POINT_ANGLE_SIGMA_DEG)
)

_fixed_gold_C = np.diag(
    _fixed_sigma_y_point**2
    + _fixed_sigma_from_point_angle**2
)

# Common normalization covariance: I0 + detector solid angle.
_fixed_common_norm_vector = (
    _fixed_gold_model
    * np.sqrt(
        (I0_poisson_sigma_h / I0_rate_h) ** 2
        + (SOLID_ANGLE_SIGMA_SR / SOLID_ANGLE_SR) ** 2
    )
)
_fixed_gold_C += np.outer(
    _fixed_common_norm_vector,
    _fixed_common_norm_vector,
)

# Common zero-angle covariance.
_fixed_zero_vector = (
    _fixed_gold_dydtheta
    * np.deg2rad(COMMON_ANGLE_OFFSET_SIGMA_DEG)
)
_fixed_gold_C += np.outer(
    _fixed_zero_vector,
    _fixed_zero_vector,
)

# If supplied later, foil-density uncertainty is correlated only among
# runs that used the same foil.
if N_AREAL_REL_SIGMA > 0:
    _fixed_N_values = _fixed_valid[
        "N_areal_cm2"
    ].to_numpy(dtype=float)

    for _N_unique in np.unique(_fixed_N_values):
        _mask = np.isclose(_fixed_N_values, _N_unique)
        _vN = np.zeros_like(_fixed_gold_model, dtype=float)
        _vN[_mask] = (
            _fixed_gold_model[_mask]
            * N_AREAL_REL_SIGMA
        )
        _fixed_gold_C += np.outer(_vN, _vN)

_fixed_gold_C_inv = np.linalg.pinv(_fixed_gold_C)

_fixed_gold_residual = _fixed_y - _fixed_gold_model

fixed_gold_chi2 = float(
    _fixed_gold_residual.T
    @ _fixed_gold_C_inv
    @ _fixed_gold_residual
)
fixed_gold_dof = int(len(_fixed_gold_residual))
fixed_gold_reduced_chi2 = (
    fixed_gold_chi2 / fixed_gold_dof
    if fixed_gold_dof > 0
    else np.nan
)
fixed_gold_p_value = chi_square_p_value(
    fixed_gold_chi2,
    fixed_gold_dof,
)


# ---------- D) Diagnostic Rutherford shape fit to long runs ----------
#
# This uses the undamaged 3°, 8°, and 13° long runs.  It is diagnostic only:
# the per-degree 4°–9° result remains the primary Z extraction.

_fixed_point_angle = np.full(
    len(_fixed_valid),
    POINT_TO_POINT_ANGLE_SIGMA_DEG,
    dtype=float,
)

_fixed_x, _fixed_sigma_x = rutherford_linearized_x_and_sigma(
    _fixed_valid["theta_deg"],
    _fixed_point_angle,
)

_, _fixed_sigma_x_common = rutherford_linearized_x_and_sigma(
    _fixed_valid["theta_deg"],
    np.full(
        len(_fixed_valid),
        COMMON_ANGLE_OFFSET_SIGMA_DEG,
        dtype=float,
    ),
)

fixed_shape_fit = fit_line_with_x_uncertainty(
    _fixed_x,
    _fixed_y,
    _fixed_sigma_x,
    _fixed_sigma_y_point,
    intercept=False,
    sigma_x_common=_fixed_sigma_x_common,
)

fixed_shape_Z, fixed_shape_sigma_Z = Z_from_linearized_slope(
    fixed_shape_fit["slope"],
    fixed_shape_fit["sigma_slope"],
)

fixed_shape_sigma_Z_common_norm = (
    0.5
    * fixed_shape_Z
    * COMMON_NORMALIZATION_REL_SIGMA
)

fixed_shape_sigma_Z_total = np.hypot(
    fixed_shape_sigma_Z,
    fixed_shape_sigma_Z_common_norm,
)

fixed_shape_chi2 = float(fixed_shape_fit["chi2"])
fixed_shape_dof = int(fixed_shape_fit["dof"])
fixed_shape_reduced_chi2 = (
    fixed_shape_chi2 / fixed_shape_dof
    if fixed_shape_dof > 0
    else np.nan
)
fixed_shape_p_value = chi_square_p_value(
    fixed_shape_chi2,
    fixed_shape_dof,
)

fixed_run_goodness_of_fit = pd.DataFrame([
    {
        "test": "Fixed runs vs absolute gold theory Z=79",
        "n_runs": len(_fixed_valid),
        "chi2": fixed_gold_chi2,
        "dof": fixed_gold_dof,
        "reduced_chi2": fixed_gold_reduced_chi2,
        "p_value": fixed_gold_p_value,
        "fitted_Z": np.nan,
        "sigma_Z_total": np.nan,
    },
    {
        "test": "Fixed runs diagnostic Rutherford shape fit",
        "n_runs": len(_fixed_valid),
        "chi2": fixed_shape_chi2,
        "dof": fixed_shape_dof,
        "reduced_chi2": fixed_shape_reduced_chi2,
        "p_value": fixed_shape_p_value,
        "fitted_Z": fixed_shape_Z,
        "sigma_Z_total": fixed_shape_sigma_Z_total,
    },
])

fixed_run_goodness_of_fit.to_csv(
    OUTPUT_DIR / "fixed_run_goodness_of_fit.csv",
    index=False,
)


# ------------------------------------------------------------
# Point-by-point error-analysis table
# ------------------------------------------------------------
error_analysis_points = fit_points.copy()

error_analysis_points["gold_model_cross_section"] = _gold_model
error_analysis_points["gold_sigma_from_point_angle"] = (
    _gold_sigma_from_point_angle
)
error_analysis_points["gold_covariance_diagonal_sigma"] = (
    _gold_diag_sigma
)
error_analysis_points["gold_pull_diagonal_diagnostic"] = (
    _gold_pull
)

error_analysis_points["best_fit_model_cross_section"] = (
    _best_model
)
error_analysis_points["best_fit_covariance_diagonal_sigma"] = (
    _best_diag_sigma
)
error_analysis_points["best_fit_pull_diagonal_diagnostic"] = (
    _best_pull
)

error_analysis_points.to_csv(
    OUTPUT_DIR / "error_analysis_fit_points.csv",
    index=False,
)


# Full per-degree vertical uncertainty budget.
_error_budget_columns = [
    "theta_deg",
    "differential_cross_section_cm2_per_sr",
    "rel_sigma_Is_poisson",
    "rel_sigma_I0_poisson",
    "rel_sigma_solid_angle",
    "rel_sigma_areal_density",
    "rel_sigma_repeatability",
    "rel_sigma_cross_section_total",
    "rel_sigma_cross_section_point_to_point",
    "rel_sigma_cross_section_common_normalization",
    "sigma_differential_cross_section_point_to_point",
    "angle_sigma_deg",
    "inferred_Z2",
    "sigma_Z_from_cross_section",
    "sigma_Z_from_angle",
    "sigma_inferred_Z2",
]

incremental_cross_sections[
    _error_budget_columns
].to_csv(
    OUTPUT_DIR / "error_budget_per_degree.csv",
    index=False,
)

fixed_combined_cross_sections[
    _error_budget_columns
].to_csv(
    OUTPUT_DIR / "error_budget_fixed_combined.csv",
    index=False,
)


# ------------------------------------------------------------
# Goodness-of-fit summary table
# ------------------------------------------------------------
goodness_of_fit = pd.DataFrame([
    {
        "test": "Absolute Rutherford gold theory (Z=79)",
        "fit_min_angle_deg": FIT_MIN_ANGLE_DEG,
        "fit_max_angle_deg": FIT_MAX_ANGLE_DEG,
        "n_points": len(fit_points),
        "n_fitted_parameters": 0,
        "chi2": gold_chi2,
        "dof": gold_dof,
        "reduced_chi2": gold_reduced_chi2,
        "p_value": gold_p_value,
        "description": chi_square_label(
            gold_reduced_chi2,
            gold_p_value,
        ),
    },
    {
        "test": "Rutherford angular shape y=m*csc^4(theta/2)",
        "fit_min_angle_deg": FIT_MIN_ANGLE_DEG,
        "fit_max_angle_deg": FIT_MAX_ANGLE_DEG,
        "n_points": len(fit_points),
        "n_fitted_parameters": 1,
        "chi2": best_chi2,
        "dof": best_dof,
        "reduced_chi2": best_reduced_chi2,
        "p_value": best_p_value,
        "description": chi_square_label(
            best_reduced_chi2,
            best_p_value,
        ),
    },
    {
        "test": "Diagnostic y=m*csc^4(theta/2)+b",
        "fit_min_angle_deg": FIT_MIN_ANGLE_DEG,
        "fit_max_angle_deg": FIT_MAX_ANGLE_DEG,
        "n_points": len(fit_points),
        "n_fitted_parameters": 2,
        "chi2": bg_chi2,
        "dof": bg_dof,
        "reduced_chi2": bg_reduced_chi2,
        "p_value": bg_p_value,
        "description": chi_square_label(
            bg_reduced_chi2,
            bg_p_value,
        ),
    },
])

goodness_of_fit.to_csv(
    OUTPUT_DIR / "rutherford_goodness_of_fit.csv",
    index=False,
)


# ------------------------------------------------------------
# Optional pull / normalized-residual plots
# ------------------------------------------------------------
if MAKE_CHI_SQUARE_PLOTS:
    # ------------------------------------------------------------
    # Pull / normalized-residual plots
    # ------------------------------------------------------------
    fig_pull, ax_pull = plt.subplots(figsize=(10.0, 6.0))
    ax_pull.axhline(0.0, linewidth=1.2)
    ax_pull.axhline(1.0, linestyle="--", linewidth=0.8, alpha=0.6)
    ax_pull.axhline(-1.0, linestyle="--", linewidth=0.8, alpha=0.6)
    ax_pull.axhline(2.0, linestyle=":", linewidth=0.8, alpha=0.5)
    ax_pull.axhline(-2.0, linestyle=":", linewidth=0.8, alpha=0.5)

    ax_pull.errorbar(
        _error_theta,
        _gold_pull,
        xerr=fit_points["angle_sigma_deg"],
        fmt="o",
        capsize=3,
        linewidth=1.0,
    )

    ax_pull.set_xlabel(
        r"Physical scattering angle $\theta$ (degrees)",
        fontsize=12,
    )
    ax_pull.set_ylabel(
        r"Normalized residual $(y-y_{\rm theory})/\sigma_{\rm eff}$",
        fontsize=12,
    )
    ax_pull.set_title(
        (
            r"Pulls relative to absolute Rutherford gold theory ($Z=79$)"
            "\n"
            rf"$\chi^2/\nu={gold_chi2:.2f}/{gold_dof}"
            rf"={gold_reduced_chi2:.2f}$, "
            rf"$p={gold_p_value:.3g}$"
        ),
        fontsize=13,
    )
    ax_pull.grid(alpha=0.25)
    fig_pull.tight_layout()
    fig_pull.savefig(
        OUTPUT_DIR / "08_gold_theory_pulls.png",
        dpi=240,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig_pull)


    fig_pull_fit, ax_pull_fit = plt.subplots(figsize=(10.0, 6.0))
    ax_pull_fit.axhline(0.0, linewidth=1.2)
    ax_pull_fit.axhline(1.0, linestyle="--", linewidth=0.8, alpha=0.6)
    ax_pull_fit.axhline(-1.0, linestyle="--", linewidth=0.8, alpha=0.6)
    ax_pull_fit.axhline(2.0, linestyle=":", linewidth=0.8, alpha=0.5)
    ax_pull_fit.axhline(-2.0, linestyle=":", linewidth=0.8, alpha=0.5)

    ax_pull_fit.errorbar(
        _error_theta,
        _best_pull,
        xerr=fit_points["angle_sigma_deg"],
        fmt="o",
        capsize=3,
        linewidth=1.0,
    )

    ax_pull_fit.set_xlabel(
        r"Physical scattering angle $\theta$ (degrees)",
        fontsize=12,
    )
    ax_pull_fit.set_ylabel(
        r"Normalized residual $(y-y_{\rm fit})/\sigma_{\rm eff}$",
        fontsize=12,
    )
    ax_pull_fit.set_title(
        (
            r"Pulls relative to best-fit Rutherford angular dependence"
            "\n"
            rf"$\chi^2/\nu={best_chi2:.2f}/{best_dof}"
            rf"={best_reduced_chi2:.2f}$, "
            rf"$p={best_p_value:.3g}$"
        ),
        fontsize=13,
    )
    ax_pull_fit.grid(alpha=0.25)
    fig_pull_fit.tight_layout()
    fig_pull_fit.savefig(
        OUTPUT_DIR / "09_best_fit_rutherford_pulls.png",
        dpi=240,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig_pull_fit)



# ------------------------------------------------------------
# Human-readable error-analysis report
# ------------------------------------------------------------
_fit_rep = fit_points["reproducibility_rel_sigma"].to_numpy(dtype=float)
_fit_rep = _fit_rep[np.isfinite(_fit_rep)]

_median_rep = (
    float(np.median(_fit_rep))
    if len(_fit_rep)
    else np.nan
)
_max_rep = (
    float(np.max(_fit_rep))
    if len(_fit_rep)
    else np.nan
)

_z_compat_sigma = (
    abs(linear_fit_Z - Z2_GOLD)
    / linear_fit_sigma_Z_total
    if np.isfinite(linear_fit_sigma_Z_total)
    and linear_fit_sigma_Z_total > 0
    else np.nan
)

error_report_lines = [
    "RUTHERFORD SCATTERING ERROR ANALYSIS",
    "=" * 55,
    "",
    f"Fit interval: {FIT_MIN_ANGLE_DEG:g} to {FIT_MAX_ANGLE_DEG:g} deg",
    f"Number of per-degree fit points: {len(fit_points)}",
    "",
    "VERTICAL CROSS-SECTION UNCERTAINTY:",
    "  Included in quadrature:",
    "    - Poisson counting uncertainty of Is",
    "    - Poisson counting uncertainty of I0",
    "    - detector solid-angle uncertainty",
    "    - foil areal-density uncertainty (currently 0 because no independent value was supplied)",
    "    - empirical run-to-run repeatability scatter",
    "",
    f"  DeltaOmega = {SOLID_ANGLE_SR:.4g} +/- {SOLID_ANGLE_SIGMA_SR:.4g} sr",
    f"  Relative DeltaOmega uncertainty = {SOLID_ANGLE_SIGMA_SR / SOLID_ANGLE_SR:.3%}",
    f"  Foil areal-density relative uncertainty used = {N_AREAL_REL_SIGMA:.3%}",
    f"  Median repeatability term in fit range = {_median_rep:.3%}",
    f"  Maximum repeatability term in fit range = {_max_rep:.3%}",
    "",
    "ANGULAR UNCERTAINTY:",
    f"  Full plotted sigma_theta = {TOTAL_ANGLE_SIGMA_DEG:.3f} deg (1 sigma)",
    f"  Point-to-point angle sigma used in chi-square = {POINT_TO_POINT_ANGLE_SIGMA_DEG:.3f} deg",
    f"  Common zero-offset sigma in covariance = {COMMON_ANGLE_OFFSET_SIGMA_DEG:.3f} deg",
    "",
    "GOODNESS OF FIT:",
    f"  Absolute Z=79 theory: chi2 = {gold_chi2:.4g}, dof = {gold_dof}, "
    f"reduced chi2 = {gold_reduced_chi2:.4g}, p = {gold_p_value:.4g}",
    f"  Best-fit Rutherford shape y=m*csc^4(theta/2): chi2 = {best_chi2:.4g}, "
    f"dof = {best_dof}, reduced chi2 = {best_reduced_chi2:.4g}, p = {best_p_value:.4g}",
    f"  Diagnostic y=m*csc^4(theta/2)+b: chi2 = {bg_chi2:.4g}, dof = {bg_dof}, "
    f"reduced chi2 = {bg_reduced_chi2:.4g}, p = {bg_p_value:.4g}",
    "",
    "BEST-FIT NORMALIZATION:",
    f"  Z_fit = {linear_fit_Z:.4g}",
    f"  point-to-point + common-angle sigma_Z = {linear_fit_sigma_Z:.4g}",
    f"  common-normalization sigma_Z = {linear_fit_sigma_Z_common_norm:.4g}",
    f"  combined sigma_Z = {linear_fit_sigma_Z_total:.4g}",
    f"  |Z_fit - 79| / combined sigma_Z = {abs(linear_fit_Z - Z2_GOLD) / linear_fit_sigma_Z_total:.4g}",
    "",
    "IMPORTANT CAVEAT:",
    "  The primary quantitative fit uses 4--9 deg; 10--13 deg remain as high-angle diagnostics.",
    "  The chi-square now separates point-to-point from correlated systematics.",
    "  I0, DeltaOmega, and the common zero-angle calibration are carried through",
    "  covariance terms rather than repeated independently at every angle.",
    "  The free-normalization Rutherford shape fit does not count I0/DeltaOmega",
    "  as point-to-point scatter because the fitted slope absorbs a common scale.",
    "  The foil areal-density uncertainty is not included until an independent uncertainty",
    "  is available from the foil log / weighing procedure.",
]

(OUTPUT_DIR / "error_analysis_summary.txt").write_text(
    "\n".join(error_report_lines),
    encoding="utf-8",
)



# ============================================================
# 7. PLOT: DIFFERENTIAL CROSS SECTION — ALL EXPERIMENTAL DATA

# ============================================================



fig, ax = plt.subplots(figsize=(11.5, 7.2))



# Gold Rutherford theory for reference.

theta_plot = np.linspace(0.5, 15.0, 1200)

ax.plot(

    theta_plot,

    rutherford_cross_section_cm2_per_sr(theta_plot, Z2=Z2_GOLD),

    linewidth=2.0,

    label=r"Rutherford theory for gold ($Z=79$)",

)

# Combined per-degree scan points.

ax.errorbar(

    incremental_cross_sections["theta_deg"],

    incremental_cross_sections["differential_cross_section_cm2_per_sr"],

    xerr=incremental_cross_sections["angle_sigma_deg"],

    yerr=incremental_cross_sections["sigma_differential_cross_section_cm2_per_sr"],

    fmt="o",

    markersize=5,

    capsize=3,

    linewidth=1.0,

    label="Per-degree runs (combined)",

)


for point_number, row in incremental_cross_sections.iterrows():
    dy = 8 if point_number % 2 == 0 else -12
    ax.annotate(
        f"{row['differential_cross_section_cm2_per_sr']:.2e}",
        (
            row["theta_deg"],
            row["differential_cross_section_cm2_per_sr"],
        ),
        xytext=(4, dy),
        textcoords="offset points",
        fontsize=6.5,
        rotation=35,
        ha="left",
        va="center",
    )



# Individual long fixed-angle runs, now labeled only by physical angle.

markers = ["s", "^", "D", "v", "P", "X", "<"]

for i, row in enumerate(fixed_cross_sections.itertuples(index=False)):

    ax.errorbar(

        row.theta_deg,

        row.differential_cross_section_cm2_per_sr,

        xerr=row.angle_sigma_deg,

        yerr=row.sigma_differential_cross_section_cm2_per_sr,

        fmt=markers[i % len(markers)],

        markersize=7,

        capsize=4,

        linewidth=1.2,

        label=row.run,

    )

    ax.annotate(
        f"{row.differential_cross_section_cm2_per_sr:.2e}",
        (row.theta_deg, row.differential_cross_section_cm2_per_sr),
        xytext=(6, 7 if i % 2 == 0 else -13),
        textcoords="offset points",
        fontsize=6.5,
        rotation=35,
        ha="left",
        va="center",
    )



ax.set_yscale("log")

ax.set_xlim(0.5, 15.0)

ax.set_xlabel(r"Physical scattering angle $\theta$ (degrees)", fontsize=12)

ax.set_ylabel(r"Differential cross section $d\sigma/d\Omega$ (cm$^2$/sr)", fontsize=12)

ax.set_title(

    "Measured Rutherford differential cross sections\n"

    r"$d\sigma/d\Omega = I_s/(\Delta\Omega\,I_0\,N)$",

    fontsize=14,

)

ax.grid(alpha=0.25, which="both")

ax.legend(fontsize=8, loc="best")

fig.tight_layout()

fig.savefig(

    OUTPUT_DIR / "04_differential_cross_section_all_measurements.png",

    dpi=240,

    bbox_inches="tight",

)



if SHOW_PLOTS:

    plt.show()

else:

    plt.close(fig)





# ============================================================

# 8. PLOT: INFERRED GOLD ATOMIC NUMBER Z2 VS ANGLE

# ============================================================



fig, ax = plt.subplots(figsize=(11.0, 7.0))



ax.axhline(

    Z2_GOLD,

    linestyle="--",

    linewidth=1.6,

    alpha=0.8,

    label=r"Accepted gold value: $Z=79$",

)

ax.errorbar(

    incremental_cross_sections["theta_deg"],

    incremental_cross_sections["inferred_Z2"],

    xerr=incremental_cross_sections["angle_sigma_deg"],

    yerr=incremental_cross_sections["sigma_inferred_Z2"],

    fmt="o",

    markersize=5,

    capsize=3,

    linewidth=1.0,

    label="Per-degree runs (derived from measured cross sections)",

)


for point_number, row in incremental_cross_sections.iterrows():
    dy = 8 if point_number % 2 == 0 else -12
    ax.annotate(
        f"{row['inferred_Z2']:.1f}",
        (row["theta_deg"], row["inferred_Z2"]),
        xytext=(4, dy),
        textcoords="offset points",
        fontsize=7,
        ha="left",
        va="center",
    )



ax.errorbar(

    fixed_combined_cross_sections["theta_deg"],

    fixed_combined_cross_sections["inferred_Z2"],

    xerr=fixed_combined_cross_sections["angle_sigma_deg"],

    yerr=fixed_combined_cross_sections["sigma_inferred_Z2"],

    fmt="s",

    markersize=8,

    capsize=4,

    linewidth=1.2,

    label="Fixed-angle runs combined (3°, 8°, 13°)",

)



for row in fixed_combined_cross_sections.itertuples(index=False):

    ax.annotate(

        rf"{row.theta_deg:g}°: $Z={row.inferred_Z2:.1f}$",

        (row.theta_deg, row.inferred_Z2),

        xytext=(7, 7),

        textcoords="offset points",

        fontsize=9,

    )



ax.set_xlim(0.5, 15.0)

ax.set_xlabel(r"Physical scattering angle $\theta$ (degrees)", fontsize=12)

ax.set_ylabel(r"Inferred target atomic number $Z_2$", fontsize=12)

ax.set_title(

    "Experimental gold atomic number inferred from measured cross sections",

    fontsize=14,

)

ax.grid(alpha=0.25)

ax.legend(fontsize=8, loc="best")

fig.tight_layout()

fig.savefig(

    OUTPUT_DIR / "05_gold_Z2_theory_and_measurements.png",

    dpi=240,

    bbox_inches="tight",

)



if SHOW_PLOTS:

    plt.show()

else:

    plt.close(fig)





# ============================================================

# 9. PLOT: CROSS SECTION AS A FUNCTION OF Z — DIRECT Z INVERSION

#

# For the three fixed physical angles, draw dσ/dΩ(Z).  Each measured

# cross section is placed at the Z value obtained by inversion.  This

# makes the experimental determination of gold's atomic number visible

# directly on the Z curves.

# ============================================================



fig, ax = plt.subplots(figsize=(11.5, 7.4))

Z_grid = np.linspace(1.0, 120.0, 1200)



for theta_value in [3.0, 8.0, 13.0]:

    theory_vs_Z = rutherford_cross_section_cm2_per_sr(

        theta_value,

        Z2=Z_grid,

    )

    ax.plot(

        Z_grid,

        theory_vs_Z,

        linewidth=1.8,

        label=rf"Rutherford curve at $\theta={theta_value:g}^\circ$",

    )



    # Fixed-angle combined measurement at this angle.

    fixed_match = fixed_combined_cross_sections[

        np.isclose(fixed_combined_cross_sections["theta_deg"], theta_value)

    ]

    if not fixed_match.empty:

        row = fixed_match.iloc[0]

        ax.errorbar(

            row["inferred_Z2"],

            row["differential_cross_section_cm2_per_sr"],

            xerr=row["sigma_inferred_Z2"],

            yerr=row["sigma_differential_cross_section_cm2_per_sr"],

            fmt="s",

            markersize=7,

            capsize=4,

            linewidth=1.1,

        )

        ax.annotate(

            (
                rf"fixed {theta_value:g}°: $Z={row['inferred_Z2']:.1f}$"
                "\n"
                rf"$d\sigma/d\Omega={row['differential_cross_section_cm2_per_sr']:.2e}$"
            ),

            (row["inferred_Z2"], row["differential_cross_section_cm2_per_sr"]),

            xytext=(6, 6),

            textcoords="offset points",

            fontsize=8,

        )



    # Per-degree combined measurement at this same physical angle.

    inc_match = incremental_cross_sections[

        np.isclose(incremental_cross_sections["theta_deg"], theta_value)

    ]

    if not inc_match.empty:

        row = inc_match.iloc[0]

        ax.errorbar(

            row["inferred_Z2"],

            row["differential_cross_section_cm2_per_sr"],

            xerr=row["sigma_inferred_Z2"],

            yerr=row["sigma_differential_cross_section_cm2_per_sr"],

            fmt="o",

            markersize=6,

            capsize=3,

            linewidth=1.0,

        )

        ax.annotate(

            (
                rf"per-degree {theta_value:g}°: $Z={row['inferred_Z2']:.1f}$"
                "\n"
                rf"$d\sigma/d\Omega={row['differential_cross_section_cm2_per_sr']:.2e}$"
            ),

            (row["inferred_Z2"], row["differential_cross_section_cm2_per_sr"]),

            xytext=(6, -13),

            textcoords="offset points",

            fontsize=8,

        )



ax.axvline(

    Z2_GOLD,

    linestyle="--",

    linewidth=1.5,

    alpha=0.8,

    label=r"Gold: $Z=79$",

)

ax.set_yscale("log")

ax.set_xlim(0.0, 120.0)

ax.set_xlabel(r"Target atomic number $Z_2$", fontsize=12)

ax.set_ylabel(r"Differential cross section $d\sigma/d\Omega$ (cm$^2$/sr)", fontsize=12)

ax.set_title(

    "Measured cross sections placed on Rutherford Z-curves",

    fontsize=14,

)

ax.grid(alpha=0.25, which="both")

ax.legend(fontsize=8, loc="best")

fig.tight_layout()

fig.savefig(

    OUTPUT_DIR / "06_cross_section_vs_Z_inversion.png",

    dpi=240,

    bbox_inches="tight",

)



if SHOW_PLOTS:

    plt.show()

else:

    plt.close(fig)





# ============================================================

# 10. PRINT SUMMARY

# ============================================================



print("\n============================================================")

print("EXPERIMENTAL DIFFERENTIAL CROSS SECTION + GOLD Z INFERENCE")

print("============================================================")

print(f"Physical zero offset = {TRUE_ZERO_OFFSET_DEG:g} deg")

print("Fixed source angles relabeled physically: 5->3 deg, 10->8 deg, 15->13 deg")

print(

    f"3 deg correction: remove {100.0 * THREE_DEGREE_REMOVED_FRACTION:.2f}% "

    f"and retain {100.0 * THREE_DEGREE_RETAINED_FRACTION:.2f}% of the signal"

)

print(f"I0 (no foil, nominal 2 deg = physical 0 deg) = {I0_rate_h / 60.0:.3f} counts/min")

print(f"Delta Omega = {SOLID_ANGLE_SR:.3f} ± {SOLID_ANGLE_SIGMA_SR:.3f} sr")
print(
    "Solid-angle geometry check (not substituted into analysis): "
    f"small-angle = {SOLID_ANGLE_SMALL_ANGLE_SR:.4f} sr, "
    f"exact circular disk = {SOLID_ANGLE_EXACT_DISK_SR:.4f} sr"
)

print(
    "Angle uncertainty (1 sigma): "
    f"protractor = {PROTRACTOR_SIGMA_DEG:.2f} deg, "
    f"zero calibration = {ZERO_CALIBRATION_SIGMA_DEG:.2f} deg, "
    f"extra alignment systematic = {ANGLE_ALIGNMENT_SYSTEMATIC_SIGMA_DEG:.2f} deg, "
    f"collimator-geometry measurement = {COLLIMATOR_ANGLE_SIGMA_DEG:.3f} deg"
)
print(
    f"  base sigma_theta = {BASE_ANGLE_SIGMA_DEG:.3f} deg; "
    f"total sigma_theta used = {TOTAL_ANGLE_SIGMA_DEG:.3f} deg"
)

print(
    f"Foil N rule: through 2026-09-01 -> {N_AREAL_CM2:.4e} cm^-2; "
    f"after 2026-09-01 -> {N_AREAL_CM2_NEW:.4e} cm^-2"
)



print("\nPrimary Rutherford fit:")
print(
    f"  fit range = {FIT_MIN_ANGLE_DEG:g}° to {FIT_MAX_ANGLE_DEG:g}° "
    f"({len(fit_points)} per-degree points)"
)
print(
    f"  fitted Z = {linear_fit_Z:.2f} "
    f"+/- {linear_fit_sigma_Z_total:.2f} total"
)
print(
    f"    fit/stat+angle component = +/- {linear_fit_sigma_Z:.2f}"
)
print(
    f"    common normalization component = "
    f"+/- {linear_fit_sigma_Z_common_norm:.2f}"
)

print("\n============================================================")
print("CHI-SQUARED SUMMARY")
print("============================================================")
print(
    f"Primary analysis range: {FIT_MIN_ANGLE_DEG:g}° to "
    f"{FIT_MAX_ANGLE_DEG:g}° ({len(fit_points)} points)"
)
print(
    f"1) Absolute gold theory (Z = 79):\n"
    f"   chi^2 = {gold_chi2:.3f}\n"
    f"   dof = {gold_dof}\n"
    f"   reduced chi^2 = {gold_reduced_chi2:.3f}\n"
    f"   p-value = {gold_p_value:.4g}"
)
print(
    f"2) Best-fit Rutherford SHAPE, y = m csc^4(theta/2):\n"
    f"   chi^2 = {best_chi2:.3f}\n"
    f"   dof = {best_dof}\n"
    f"   reduced chi^2 = {best_reduced_chi2:.3f}\n"
    f"   p-value = {best_p_value:.4g}\n"
    f"   fitted Z = {linear_fit_Z:.2f} +/- "
    f"{linear_fit_sigma_Z_total:.2f} total"
)
print(
    f"3) Diagnostic fit with constant background, y = m csc^4(theta/2) + b:\n"
    f"   chi^2 = {bg_chi2:.3f}\n"
    f"   dof = {bg_dof}\n"
    f"   reduced chi^2 = {bg_reduced_chi2:.3f}\n"
    f"   p-value = {bg_p_value:.4g}"
)

print("\nQuick reading guide:")
print("  reduced chi^2 ~ 1  -> point-to-point scatter matches the uncertainty model")
print("  reduced chi^2 >> 1 -> remaining point-to-point systematics/model disagreement")
print("  reduced chi^2 << 1 -> point-to-point uncertainty is still conservative")
print(
    "  The 4°-9° SHAPE test is the main Rutherford test; "
    "the Z=79 test also includes absolute-normalization systematics."
)
print(
    "  Angles 10°-13° remain on the plots and in the fit-window "
    "sensitivity table, but do not set the primary fitted Z."
)
print(
    f"  Full plotted sigma_theta = {TOTAL_ANGLE_SIGMA_DEG:.3f}°"
)
print(
    f"  Chi-square point-to-point sigma_theta = "
    f"{POINT_TO_POINT_ANGLE_SIGMA_DEG:.3f}°"
)
print(
    f"  Common zero-angle sigma carried as covariance = "
    f"{COMMON_ANGLE_OFFSET_SIGMA_DEG:.3f}°"
)
print(
    "  Shape chi-square y-errors use counting + run-to-run repeatability; "
    "I0/solid-angle normalization is not double-counted as independent scatter."
)
print(
    "  NOTE: foil-density uncertainty remains 0 until an independent "
    "uncertainty is supplied."
)
if len(fit_window_sensitivity):
    finite_Z = fit_window_sensitivity[
        np.isfinite(fit_window_sensitivity["Z_fit"])
    ]
    if len(finite_Z):
        print(
            "  fit-window sensitivity: "
            f"Z ranges from {finite_Z['Z_fit'].min():.2f} "
            f"to {finite_Z['Z_fit'].max():.2f} across tested windows"
        )
        print(
            "  full table saved as rutherford_fit_window_sensitivity.csv"
        )

print("\n============================================================")
print("FIXED-ANGLE / OVERNIGHT / WEEKEND ERROR ANALYSIS")
print("============================================================")
print(
    f"Usable undamaged long runs = {len(_fixed_valid)} "
    "(punctured 13° run excluded quantitatively)"
)
print("\n1) All usable fixed runs vs absolute gold theory (Z = 79):")
print(f"   chi^2 = {fixed_gold_chi2:.3f}")
print(f"   dof = {fixed_gold_dof}")
print(f"   reduced chi^2 = {fixed_gold_reduced_chi2:.3f}")
print(f"   p-value = {fixed_gold_p_value:.4g}")

print("\n2) Diagnostic Rutherford SHAPE fit to the usable fixed runs:")
print(f"   chi^2 = {fixed_shape_chi2:.3f}")
print(f"   dof = {fixed_shape_dof}")
print(f"   reduced chi^2 = {fixed_shape_reduced_chi2:.3f}")
print(f"   p-value = {fixed_shape_p_value:.4g}")
print(
    f"   diagnostic fitted Z = {fixed_shape_Z:.2f} "
    f"+/- {fixed_shape_sigma_Z_total:.2f} total"
)
print(
    "   NOTE: diagnostic only because these long runs are at 3°, 8°, and 13°."
)

print("\n3) Repeated-run consistency at each fixed angle:")
if len(fixed_run_consistency):
    for _row in fixed_run_consistency.itertuples(index=False):
        print(
            f"   {_row.theta_deg:g}°: "
            f"chi^2 = {_row.chi2:.3f}, "
            f"dof = {_row.dof}, "
            f"reduced chi^2 = {_row.reduced_chi2:.3f}, "
            f"p = {_row.p_value:.4g}, "
            f"fractional run spread = "
            f"{100.0 * _row.sample_fractional_spread:.1f}%"
        )
else:
    print("   Not enough repeated runs for a consistency test.")

_punctured_rows = fixed_individual_error_analysis[
    fixed_individual_error_analysis["punctured_foil_run"]
]
if len(_punctured_rows):
    print("\nExcluded damaged-target run:")
    for _row in _punctured_rows.itertuples(index=False):
        print(
            f"   {_row.run}: theta = {_row.theta_deg:g}°, "
            f"Z = {_row.inferred_Z2:.2f} +/- "
            f"{_row.sigma_inferred_Z2:.2f}"
        )

print(
    "\nSaved: fixed_individual_run_error_analysis.csv, "
    "fixed_run_consistency_chi_square.csv, "
    "fixed_run_goodness_of_fit.csv"
)

print("\nCombined fixed-angle results:")

print(

    fixed_combined_cross_sections[

        [

            "theta_deg",

            "raw_rate_counts_per_hour",

            "analysis_rate_counts_per_hour",

            "N_areal_cm2",

            "differential_cross_section_cm2_per_sr",

            "sigma_differential_cross_section_cm2_per_sr",

            "rel_sigma_repeatability",

            "rel_sigma_cross_section_total",

            "rel_sigma_cross_section_point_to_point",

            "rel_sigma_cross_section_common_normalization",

            "inferred_Z2",

            "sigma_Z_from_cross_section",

            "sigma_Z_from_angle",

            "sigma_inferred_Z2",

        ]

    ].to_string(index=False)

)



print("\nPer-degree results:")

print(

    incremental_cross_sections[

        [

            "theta_deg",

            "raw_rate_counts_per_hour",

            "analysis_rate_counts_per_hour",

            "N_areal_cm2",

            "differential_cross_section_cm2_per_sr",

            "sigma_differential_cross_section_cm2_per_sr",

            "rel_sigma_repeatability",

            "rel_sigma_cross_section_total",

            "rel_sigma_cross_section_point_to_point",

            "rel_sigma_cross_section_common_normalization",

            "inferred_Z2",

            "sigma_Z_from_cross_section",

            "sigma_Z_from_angle",

            "sigma_inferred_Z2",

        ]

    ].to_string(index=False)

)



print(f"\nPlots and CSV tables saved to:\n{OUTPUT_DIR}")


# ============================================================
# 11. REQUESTED CHI-SQUARED COMPARISON
#
# Requested datasets:
#   1) Per-degree scan, 4--9 deg
#   2) Per-degree scan, requested 0--13 deg
#      NOTE: theta = 0 cannot be included in Rutherford theory because
#      csc^4(theta/2) diverges there.  The actual usable range is 1--13 deg.
#   3) Combined fixed-angle points at 3, 8, and 13 deg
#
# For each dataset, report:
#   A) Absolute Z=79 test (no fitted parameters)
#   B) Rutherford SHAPE test y = m*csc^4(theta/2) (one fitted parameter)
#
# The same covariance treatment used elsewhere in this file is retained:
#   - counting + repeatability are point-to-point y errors
#   - protractor setting uncertainty is point-to-point angle error
#   - zero calibration is one common correlated angle offset
#   - I0 and solid-angle normalization are common correlated errors
# ============================================================


def requested_chi_square_analysis(
    table,
    label,
    min_angle_deg,
    max_angle_deg,
):
    """Run absolute-gold and free-normalization Rutherford chi-square tests."""

    data = table[
        (table["theta_deg"] >= min_angle_deg)
        & (table["theta_deg"] <= max_angle_deg)
        & (table["theta_deg"] > 0.0)
        & np.isfinite(
            table["differential_cross_section_cm2_per_sr"]
        )
        & (
            table["differential_cross_section_cm2_per_sr"] > 0.0
        )
        & np.isfinite(
            table["sigma_differential_cross_section_point_to_point"]
        )
        & (
            table["sigma_differential_cross_section_point_to_point"] > 0.0
        )
    ].copy()

    if len(data) < 2:
        raise ValueError(
            f"{label}: need at least 2 valid nonzero-angle points "
            "for a chi-square comparison."
        )

    theta = data["theta_deg"].to_numpy(dtype=float)
    theta_rad = np.deg2rad(theta)
    y = data[
        "differential_cross_section_cm2_per_sr"
    ].to_numpy(dtype=float)
    sigma_y_point = data[
        "sigma_differential_cross_section_point_to_point"
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # A) Absolute gold theory, Z = 79
    # --------------------------------------------------------
    gold_model = rutherford_cross_section_cm2_per_sr(
        theta,
        Z2=Z2_GOLD,
    )

    gold_dydtheta = (
        -2.0
        * gold_model
        / np.tan(theta_rad / 2.0)
    )

    sigma_gold_from_point_angle = (
        np.abs(gold_dydtheta)
        * np.deg2rad(POINT_TO_POINT_ANGLE_SIGMA_DEG)
    )

    gold_C = np.diag(
        sigma_y_point**2
        + sigma_gold_from_point_angle**2
    )

    # Common normalization: same I0 and detector solid angle affect all points.
    common_norm_rel = np.sqrt(
        (I0_poisson_sigma_h / I0_rate_h) ** 2
        + (SOLID_ANGLE_SIGMA_SR / SOLID_ANGLE_SR) ** 2
    )
    common_norm_vector = gold_model * common_norm_rel
    gold_C += np.outer(
        common_norm_vector,
        common_norm_vector,
    )

    # Common angular zero calibration.
    common_zero_vector = (
        gold_dydtheta
        * np.deg2rad(COMMON_ANGLE_OFFSET_SIGMA_DEG)
    )
    gold_C += np.outer(
        common_zero_vector,
        common_zero_vector,
    )

    # Optional areal-density systematic, if supplied later.
    # For the current file N_AREAL_REL_SIGMA = 0, so this contributes nothing.
    if N_AREAL_REL_SIGMA > 0:
        if "N_areal_cm2" in data.columns:
            N_values = data["N_areal_cm2"].to_numpy(dtype=float)
            for N_unique in np.unique(N_values):
                mask = np.isclose(N_values, N_unique)
                vN = np.zeros_like(gold_model, dtype=float)
                vN[mask] = (
                    gold_model[mask]
                    * N_AREAL_REL_SIGMA
                )
                gold_C += np.outer(vN, vN)
        else:
            vN = gold_model * N_AREAL_REL_SIGMA
            gold_C += np.outer(vN, vN)

    gold_C_inv = np.linalg.pinv(gold_C)
    gold_residual = y - gold_model

    gold_chi2_local = float(
        gold_residual.T
        @ gold_C_inv
        @ gold_residual
    )
    gold_dof_local = int(len(data))  # no fitted parameters
    gold_reduced_local = (
        gold_chi2_local / gold_dof_local
        if gold_dof_local > 0
        else np.nan
    )
    gold_p_local = chi_square_p_value(
        gold_chi2_local,
        gold_dof_local,
    )

    # --------------------------------------------------------
    # B) Rutherford angular SHAPE, y = m*csc^4(theta/2)
    # --------------------------------------------------------
    point_angle_sigma = np.full(
        len(data),
        POINT_TO_POINT_ANGLE_SIGMA_DEG,
        dtype=float,
    )
    x, sigma_x = rutherford_linearized_x_and_sigma(
        theta,
        point_angle_sigma,
    )

    _, sigma_x_common = rutherford_linearized_x_and_sigma(
        theta,
        np.full(
            len(data),
            COMMON_ANGLE_OFFSET_SIGMA_DEG,
            dtype=float,
        ),
    )

    shape_fit = fit_line_with_x_uncertainty(
        x,
        y,
        sigma_x,
        sigma_y_point,
        intercept=False,
        sigma_x_common=sigma_x_common,
    )

    Z_fit_local, sigma_Z_shape_local = Z_from_linearized_slope(
        shape_fit["slope"],
        shape_fit["sigma_slope"],
    )

    # Common normalization affects the absolute fitted Z, but not the
    # free-normalization shape chi-square itself.
    sigma_Z_common_norm_local = (
        0.5
        * Z_fit_local
        * COMMON_NORMALIZATION_REL_SIGMA
    )
    sigma_Z_total_local = np.hypot(
        sigma_Z_shape_local,
        sigma_Z_common_norm_local,
    )

    shape_chi2_local = float(shape_fit["chi2"])
    shape_dof_local = int(shape_fit["dof"])
    shape_reduced_local = (
        shape_chi2_local / shape_dof_local
        if shape_dof_local > 0
        else np.nan
    )
    shape_p_local = chi_square_p_value(
        shape_chi2_local,
        shape_dof_local,
    )

    angle_string = ", ".join(
        f"{value:g}" for value in theta
    )

    rows = [
        {
            "dataset": label,
            "angle_min_requested_deg": min_angle_deg,
            "angle_max_requested_deg": max_angle_deg,
            "actual_angles_used_deg": angle_string,
            "n_points": len(data),
            "test": "Absolute Rutherford theory, Z=79",
            "chi2": gold_chi2_local,
            "dof": gold_dof_local,
            "reduced_chi2": gold_reduced_local,
            "p_value": gold_p_local,
            "fitted_Z": np.nan,
            "sigma_Z_total": np.nan,
        },
        {
            "dataset": label,
            "angle_min_requested_deg": min_angle_deg,
            "angle_max_requested_deg": max_angle_deg,
            "actual_angles_used_deg": angle_string,
            "n_points": len(data),
            "test": "Rutherford shape, y=m*csc^4(theta/2)",
            "chi2": shape_chi2_local,
            "dof": shape_dof_local,
            "reduced_chi2": shape_reduced_local,
            "p_value": shape_p_local,
            "fitted_Z": Z_fit_local,
            "sigma_Z_total": sigma_Z_total_local,
        },
    ]

    return pd.DataFrame(rows)


requested_chi_square_summary = pd.concat(
    [
        requested_chi_square_analysis(
            incremental_cross_sections,
            "Per-degree scan: 4-9 deg",
            4.0,
            9.0,
        ),
        requested_chi_square_analysis(
            incremental_cross_sections,
            "Per-degree scan: requested 0-13 deg (0 excluded; uses nonzero angles)",
            0.0,
            13.0,
        ),
        requested_chi_square_analysis(
            fixed_combined_cross_sections,
            "Combined fixed-angle runs: 3, 8, 13 deg",
            3.0,
            13.0,
        ),
    ],
    ignore_index=True,
)

requested_chi_square_summary.to_csv(
    OUTPUT_DIR / "requested_chi_square_summary.csv",
    index=False,
)


print("\n============================================================")
print("REQUESTED CHI-SQUARED ANALYSIS")
print("============================================================")
print(
    "NOTE: the requested 0-13 deg Rutherford test excludes theta=0, "
    "where csc^4(theta/2) is singular; it therefore uses the available "
    "nonzero points through 13 deg."
)

for dataset_name, group in requested_chi_square_summary.groupby(
    "dataset",
    sort=False,
):
    first = group.iloc[0]
    print(f"\n{dataset_name}")
    print(
        f"  points used ({int(first['n_points'])}): "
        f"{first['actual_angles_used_deg']} deg"
    )

    for row in group.itertuples(index=False):
        print(f"  {row.test}:")
        print(f"    chi^2 = {row.chi2:.4f}")
        print(f"    dof = {row.dof}")
        print(f"    reduced chi^2 = {row.reduced_chi2:.4f}")
        print(f"    p-value = {row.p_value:.6g}")

        if np.isfinite(row.fitted_Z):
            print(
                f"    fitted Z = {row.fitted_Z:.3f} "
                f"+/- {row.sigma_Z_total:.3f}"
            )

print(
    "\nSaved requested summary to: "
    f"{OUTPUT_DIR / 'requested_chi_square_summary.csv'}"
)
