"""
test_hd_math.py — Verify the math module against known reference values.
Run with: python -m pytest test_hd_math.py -v
Or just: python test_hd_math.py
"""

import numpy as np
import hd_math as m


def approx_eq(a, b, tol=1e-5):
    return abs(float(a) - float(b)) < tol


def test_logc3_round_trip_at_mid_gray():
    """Mid grey (0.18 linear) round-trips through LogC3."""
    lin = 0.18
    logc3 = m.linear_to_logc3(lin)
    back  = m.logc3_to_linear(logc3)
    assert approx_eq(back, lin, tol=1e-6), f"round-trip failed: {lin} -> {logc3} -> {back}"


def test_logc3_known_values():
    """
    ARRI publishes that 0.18 linear → ~0.391 in LogC3 EI800.
    18% grey is the canonical reference.
    """
    logc3_at_mid = m.linear_to_logc3(0.18)
    # ARRI white paper reports LogC3 mid grey at EI800 as 0.391 (3-decimal).
    assert approx_eq(logc3_at_mid, 0.391, tol=0.001), f"mid grey LogC3 = {logc3_at_mid}"


def test_logc3_round_trip_across_range():
    """Round-trip across the full sensible exposure range."""
    test_lin = np.array([0.0, 0.001, 0.01, 0.05, 0.18, 0.5, 1.0, 2.0, 8.0, 32.0])
    logc3 = m.linear_to_logc3(test_lin)
    back  = m.logc3_to_linear(logc3)
    assert np.allclose(back, test_lin, atol=1e-5), \
        f"round-trip failed: {test_lin} vs {back}"


def test_logc3_continuity_at_cut():
    """LogC3 is piecewise-defined; check continuity at the cut point."""
    cut = m.LOGC3_CUT
    # evaluate just above and just below
    above = m.linear_to_logc3(cut + 1e-9)
    below = m.linear_to_logc3(cut - 1e-9)
    assert approx_eq(above, below, tol=1e-6), \
        f"discontinuity at cut: above={above}, below={below}"


def test_stops_round_trip():
    """stops_to_linear and linear_to_stops are inverses."""
    test_stops = np.array([-7.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    lin = m.stops_to_linear(test_stops)
    back = m.linear_to_stops(lin)
    assert np.allclose(back, test_stops, atol=1e-9)


def test_zero_stops_is_mid_gray():
    """0 stops = mid grey by definition."""
    assert approx_eq(m.stops_to_linear(0.0), 0.18)
    assert approx_eq(m.stops_to_linear(0.0, mid_gray=0.20), 0.20)


def test_one_stop_is_doubling():
    """+1 stop = 2x linear, -1 stop = 0.5x linear."""
    assert approx_eq(m.stops_to_linear(1.0), 0.36)
    assert approx_eq(m.stops_to_linear(-1.0), 0.09)


def test_density_known_values():
    """
    Density = -log10(transmittance).
    transmittance 1.0 -> D=0
    transmittance 0.1 -> D=1
    transmittance 0.01 -> D=2
    """
    assert approx_eq(m.output_to_density(1.0), 0.0)
    assert approx_eq(m.output_to_density(0.1), 1.0)
    assert approx_eq(m.output_to_density(0.01), 2.0)
    assert approx_eq(m.output_to_density(0.001), 3.0)


def test_probe_ramp_endpoints():
    """Probe ramp at x=0 should be LogC3 of stop_min, at x=W-1 of stop_max."""
    width = 1920
    stop_min, stop_max = -7.0, 5.0
    ramp = m.generate_probe_ramp(width, stop_min, stop_max)
    assert ramp.shape == (width, 3)
    # endpoints
    expected_left  = m.stops_to_logc3(stop_min)
    expected_right = m.stops_to_logc3(stop_max)
    assert approx_eq(ramp[0, 0], expected_left, tol=1e-5)
    assert approx_eq(ramp[-1, 0], expected_right, tol=1e-5)
    # mid grey should appear at x where stops=0
    zero_x = int((0.0 - stop_min) / (stop_max - stop_min) * (width - 1))
    expected_mid = m.stops_to_logc3(0.0)
    assert approx_eq(ramp[zero_x, 0], expected_mid, tol=1e-3)


def test_x_axis_transforms_consistent():
    """Stops, log exposure, and linear x-axis transforms should be consistent."""
    x_norm = 0.5
    stop_min, stop_max = -7.0, 5.0
    stops = m.x_position_to_stops(x_norm, stop_min, stop_max)
    log_exp = m.x_position_to_log_exposure(x_norm, stop_min, stop_max)
    linear = m.x_position_to_linear(x_norm, stop_min, stop_max)

    # log_exp = stops * log10(2)
    assert approx_eq(log_exp, stops * np.log10(2.0))
    # linear = mid_gray * 2^stops
    assert approx_eq(linear, 0.18 * 2**stops)


def test_log_exposure_one_stop_is_log10_2():
    """Moving 1 stop on the stop axis = log10(2) ≈ 0.301 on the log exposure axis."""
    log_exp_0 = m.x_position_to_log_exposure(0.5, -1.0, 1.0)  # stops=0
    log_exp_1 = m.x_position_to_log_exposure(1.0, -1.0, 1.0)  # stops=1
    diff = log_exp_1 - log_exp_0
    assert approx_eq(diff, np.log10(2.0), tol=1e-6)


# Run as a script
if __name__ == "__main__":
    import sys
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failures.append(t.__name__)
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failures.append(t.__name__)
    print()
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
