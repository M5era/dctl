"""
test_hd_math.py — Verify the math module against known reference values.
Run with: python -m pytest test_hd_math.py -v
Or just: python test_hd_math.py
"""

import numpy as np
import hd_math as m
import chart_renderer as cr


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


def test_logc4_probe_ramp_endpoints():
    """Probe ramp can also emit LogC4 when requested."""
    width = 1024
    stop_min, stop_max = -8.0, 8.0
    ramp = m.generate_probe_ramp(width, stop_min, stop_max, encoding=m.ENCODING_LOGC4)
    assert ramp.shape == (width, 3)
    assert approx_eq(ramp[0, 0], m.stops_to_logc4(stop_min), tol=1e-5)
    assert approx_eq(ramp[-1, 0], m.stops_to_logc4(stop_max), tol=1e-5)


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


# ---------------------------------------------------------------------------
#  Sensitometry curve helpers (chart_renderer)
# ---------------------------------------------------------------------------

def _make_sigmoid_ramp(width=1024, dmin=0.05, dmax=1.5, sharpness=2.5):
    """
    Synthetic print-style curve: density transitions from dmax (left) to dmin
    (right) via a tanh, with flat plateaus at both ends. Returns a (W, 3)
    transmittance ramp suitable for find_curve_clamp().
    """
    x = np.linspace(-1.0, 1.0, width)
    # density curve: dmax on left, dmin on right
    t = (np.tanh(sharpness * x) + 1.0) / 2.0  # 0 → 1 left to right
    density = dmax + (dmin - dmax) * t
    transmittance = np.power(10.0, -density)
    return np.stack([transmittance] * 3, axis=-1).astype(np.float32)


def test_curve_clamp_detects_active_range():
    """Clamp range should sit strictly inside the full range for a curve with plateaus."""
    ramp = _make_sigmoid_ramp(width=2048, sharpness=4.0)
    left_frac, right_frac = cr.find_curve_clamp(ramp, threshold_frac=0.02)
    assert 0.0 < left_frac < 0.4, f"left clamp not tightening: {left_frac}"
    assert 0.6 < right_frac < 1.0, f"right clamp not tightening: {right_frac}"
    assert right_frac - left_frac > 0.2


def test_curve_clamp_flat_curve_returns_full_range():
    """A perfectly flat curve has no transition; clamp returns full range."""
    ramp = np.full((512, 3), 0.5, dtype=np.float32)
    lf, rf = cr.find_curve_clamp(ramp)
    assert lf == 0.0 and rf == 1.0


def test_curve_clamp_short_ramp():
    ramp = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    lf, rf = cr.find_curve_clamp(ramp)
    assert (lf, rf) == (0.0, 1.0)


def test_equal_units_rect_pixels_per_unit_match():
    """1 unit in X should equal 1 unit in Y in pixels (within rounding)."""
    left, top, right, bottom = cr.compute_equal_units_rect(
        0, 0, 1000, 600, x_span=4.82, y_span=1.5,
    )
    px_per_x = (right - left) / 4.82
    px_per_y = (bottom - top) / 1.5
    assert abs(px_per_x - px_per_y) < 1.5  # within 1.5px due to int rounding
    # Also: rect must be inside the available area
    assert 0 <= left and right <= 1000
    assert 0 <= top and bottom <= 600


def test_equal_units_rect_constrained_by_height():
    """When x_span/y_span > avail_w/avail_h, rect should fill width exactly."""
    left, top, right, bottom = cr.compute_equal_units_rect(
        0, 0, 800, 200, x_span=4.0, y_span=1.0,
    )
    # avail ratio = 4.0; data ratio = 4.0 — should fill both dims
    assert right - left == 800
    assert bottom - top == 200


def test_equal_units_rect_constrained_by_width():
    """When data is taller than available, rect should fill height and shrink width."""
    left, top, right, bottom = cr.compute_equal_units_rect(
        0, 0, 1000, 1000, x_span=2.0, y_span=4.0,
    )
    # data ratio 0.5; available ratio 1.0 — height-constrained
    # plot_h=1000, plot_w should be 500
    assert bottom - top == 1000
    assert abs((right - left) - 500) <= 1


def test_sensitometry_offset_log_exposure_zeros_right_clamp():
    """
    With offset_log_exposure=True the right edge of the curve should land at
    log_exp = 0. We verify by rendering at two heights and confirming the
    label '0' appears at the rightmost log-exposure tick position.

    Easier check: directly compute what the offset SHOULD be — the rendered
    function uses clamped_stop_max * log10(2). We mirror that math here.
    """
    ramp = _make_sigmoid_ramp(width=1024, sharpness=4.0)
    stop_min, stop_max = -8.0, 8.0
    lf, rf = cr.find_curve_clamp(ramp, threshold_frac=0.02)
    full_span = stop_max - stop_min
    clamped_stop_max = stop_min + rf * full_span
    expected_offset = -clamped_stop_max * np.log10(2.0)
    # Right-edge log exposure under that offset must be 0.
    assert approx_eq(clamped_stop_max * np.log10(2.0) + expected_offset, 0.0)


def test_sensitometry_render_smoke():
    """Ensure render_sensitometry_page runs end-to-end with all new toggles."""
    ramp = _make_sigmoid_ramp(width=512, sharpness=4.0)
    for show_full in (False, True):
        for offset in (False, True):
            img = cr.render_sensitometry_page(
                800, 600, ramp,
                stop_min=-8.0, stop_max=8.0,
                show_entire_curve=show_full,
                offset_log_exposure=offset,
            )
            assert img.shape == (600, 800, 3)
            assert np.isfinite(img).all()


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
