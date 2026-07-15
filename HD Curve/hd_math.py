"""
hd_math.py — Pure math for the H&D Inspector DCTL.

Every function here has a 1:1 DCTL equivalent. Keep it that way:
no Python-only tricks, no fancy NumPy-only operations except where
explicitly noted as a "vectorised wrapper" for testing.

Reference for ARRI LogC3 EI800: ARRI white paper "ALEXA Log C Curve",
SUP 3.x, AlexaWideGamut3, EI800.
"""

import numpy as np


# ---------------------------------------------------------------------------
#  ARRI LogC3 EI800 — published coefficients
# ---------------------------------------------------------------------------
LOGC3_CUT = 0.010591
LOGC3_A   = 5.555556
LOGC3_B   = 0.052272
LOGC3_C   = 0.247190
LOGC3_D   = 0.385537
LOGC3_E   = 5.367655
LOGC3_F   = 0.092809


def linear_to_logc3(x):
    """Scene-linear → LogC3 (EI800). Vectorised."""
    x = np.asarray(x, dtype=np.float64)
    out = np.where(
        x >= LOGC3_CUT,
        LOGC3_C * np.log10(LOGC3_A * x + LOGC3_B) + LOGC3_D,
        LOGC3_E * x + LOGC3_F,
    )
    return out


def logc3_to_linear(x):
    """LogC3 (EI800) → scene-linear. Vectorised."""
    x = np.asarray(x, dtype=np.float64)
    logc3_cut = LOGC3_E * LOGC3_CUT + LOGC3_F  # ~0.149427
    out = np.where(
        x >= logc3_cut,
        (np.power(10.0, (x - LOGC3_D) / LOGC3_C) - LOGC3_B) / LOGC3_A,
        (x - LOGC3_F) / LOGC3_E,
    )
    return out


# ---------------------------------------------------------------------------
#  ARRI LogC4 — published coefficients (white paper, March 2022)
# ---------------------------------------------------------------------------
# y = (log2(a*x + b) - log2(b)) / D + c   for x >= 0
# Constants designed so y(0) = c (noise floor) and y supports ~16 stops above
# 18% mid-grey before saturating to 1.0. Below x=0 we use a small linear
# extension; for our probe range [-8, +8] stops above 0.18 the input is always
# strictly positive, so the linear segment is never exercised.
LOGC4_A      = (2.0**18 - 16.0) / 117.45            # ≈ 2231.8285
LOGC4_B      = (1023.0 - 95.0) / 1023.0             # ≈ 0.9071
LOGC4_C      = 95.0 / 1023.0                        # ≈ 0.0928
LOGC4_DIV    = 28.0
LOGC4_LOG2B  = np.log2(LOGC4_B)


def linear_to_logc4(x):
    """Scene-linear → ARRI LogC4. Vectorised."""
    x = np.asarray(x, dtype=np.float64)
    safe = np.maximum(x, 0.0)
    y = (np.log2(LOGC4_A * safe + LOGC4_B) - LOGC4_LOG2B) / LOGC4_DIV + LOGC4_C
    # Linear extension for x < 0 (rarely needed; clamp toward c)
    y = np.where(x >= 0.0, y, LOGC4_C + x * 0.05)
    return y


def stops_to_logc4(stops, mid_gray=0.18):
    """Stops above mid-grey → LogC4 code value."""
    return linear_to_logc4(stops_to_linear(stops, mid_gray))


# ---------------------------------------------------------------------------
#  Stop / exposure helpers
# ---------------------------------------------------------------------------

def stops_to_linear(stops, mid_gray=0.18):
    """0 stops = mid_gray."""
    return mid_gray * np.power(2.0, stops)


def linear_to_stops(linear, mid_gray=0.18):
    """Inverse of stops_to_linear. Returns NaN for non-positive input."""
    linear = np.asarray(linear, dtype=np.float64)
    safe = np.where(linear > 0.0, linear, np.nan)
    return np.log2(safe / mid_gray)


def stops_to_logc3(stops, mid_gray=0.18):
    """Convenience: a given stop value → its LogC3 code value."""
    return linear_to_logc3(stops_to_linear(stops, mid_gray))


# ---------------------------------------------------------------------------
#  Probe ramp generator
# ---------------------------------------------------------------------------

ENCODING_LOGC3 = "logc3"
ENCODING_LOGC4 = "logc4"


def generate_probe_ramp(width, stop_min, stop_max, mid_gray=0.18,
                        encoding=ENCODING_LOGC3):
    """
    Generate a 1D probe ramp of length `width`, mapping pixel x in [0, width-1]
    linearly across stops in [stop_min, stop_max], encoded in the requested log
    space.

    encoding: ENCODING_LOGC3 (default — for AWG3/LogC3 LUTs like the Kodak 2383
              and the K64 sRGB LUT), or ENCODING_LOGC4 (for AWG4/LogC4 LUTs like
              the spectral K64 AWG4 sRGB LUT).

    Returns: array of shape (width, 3), float32, identical RGB.
    """
    x = np.linspace(0.0, 1.0, width, dtype=np.float64)
    stops = stop_min + x * (stop_max - stop_min)
    if encoding == ENCODING_LOGC3:
        coded = stops_to_logc3(stops, mid_gray)
    elif encoding == ENCODING_LOGC4:
        coded = stops_to_logc4(stops, mid_gray)
    else:
        raise ValueError(f"unknown encoding: {encoding!r}")
    rgb = np.stack([coded, coded, coded], axis=-1).astype(np.float32)
    return rgb


# ---------------------------------------------------------------------------
#  Y-axis transforms (output value → display value)
# ---------------------------------------------------------------------------

def output_to_percent(v):
    """0..1 → 0..100"""
    return np.asarray(v, dtype=np.float64) * 100.0


def output_to_density(v, eps=1e-5):
    """
    Kodak H&D density: D = -log10(transmittance).
    Assumes input is a transmittance / linear value in (0, 1].
    """
    v = np.asarray(v, dtype=np.float64)
    safe = np.maximum(v, eps)
    return -np.log10(safe)


def output_to_linear(v):
    """Pass-through; named for symmetry."""
    return np.asarray(v, dtype=np.float64)


# ---------------------------------------------------------------------------
#  X-axis transforms (probe position → display value)
# ---------------------------------------------------------------------------

def x_position_to_stops(x_norm, stop_min, stop_max):
    """x_norm in [0,1] → stops, linearly."""
    return stop_min + x_norm * (stop_max - stop_min)


def x_position_to_log_exposure(x_norm, stop_min, stop_max, mid_gray=0.18):
    """Stops → log10 exposure. Stops × log10(2) gives log10 exposure offset."""
    stops = x_position_to_stops(x_norm, stop_min, stop_max)
    # log10(linear / mid_gray) = stops × log10(2)
    return stops * np.log10(2.0)


def x_position_to_linear(x_norm, stop_min, stop_max, mid_gray=0.18):
    """Stops → linear input value."""
    stops = x_position_to_stops(x_norm, stop_min, stop_max)
    return stops_to_linear(stops, mid_gray)
