"""
chart_renderer.py — Render the H&D Inspector chart in Python.

This is intentionally written to mirror what the DCTL will do per-pixel.
We use NumPy for vectorisation (so it runs fast in Python), but the
*logic* is per-pixel and translates directly to DCTL.

Output: an RGB image array that matches what the DCTL should produce
in Resolve given the same inputs.
"""

import numpy as np
import hd_math as m


# -----------------------------------------------------------------------------
# X / Y axis modes
# -----------------------------------------------------------------------------
X_MODE_STOPS         = 0   # x-axis: photometric stops (default)
X_MODE_LOG_EXPOSURE  = 1   # x-axis: log10 exposure (Kodak datasheet style)
X_MODE_LINEAR        = 2   # x-axis: linear input 0..1 (LUT Inspector style)

Y_MODE_PERCENT       = 0   # y-axis: 0..100% output
Y_MODE_DENSITY       = 1   # y-axis: 0..3.0 density (-log10)
Y_MODE_LINEAR        = 2   # y-axis: 0..1 linear


def get_x_axis_label(mode):
    return {
        X_MODE_STOPS:        "Stops (rel. mid grey)",
        X_MODE_LOG_EXPOSURE: "Log Exposure",
        X_MODE_LINEAR:       "Linear Input",
    }[mode]


def get_y_axis_label(mode):
    return {
        Y_MODE_PERCENT: "Output %",
        Y_MODE_DENSITY: "Density",
        Y_MODE_LINEAR:  "Linear Output",
    }[mode]


def get_y_range(mode):
    """Default Y-axis range for each mode."""
    return {
        Y_MODE_PERCENT: (0.0, 100.0),
        Y_MODE_DENSITY: (0.0, 3.0),
        Y_MODE_LINEAR:  (0.0, 1.0),
    }[mode]


def get_x_range(mode, stop_min, stop_max, mid_gray=0.18, zero_lux_offset=0.0):
    """
    Default X-axis range (in display units) for each mode.

    zero_lux_offset: shift in log10(lux-seconds) units, only applied to
    LOG_EXPOSURE mode. Use to place 0 lux-seconds at the desired X position.
    For ISO N reversal film: offset ≈ log10(8 / N).
    """
    if mode == X_MODE_STOPS:
        return (stop_min, stop_max)
    elif mode == X_MODE_LOG_EXPOSURE:
        return (stop_min * np.log10(2.0) + zero_lux_offset,
                stop_max * np.log10(2.0) + zero_lux_offset)
    else:  # LINEAR
        return (m.stops_to_linear(stop_min, mid_gray),
                m.stops_to_linear(stop_max, mid_gray))


def iso_to_zero_lux_offset(iso):
    """
    Convert a film ISO speed to the log10 lux-seconds offset that places
    0 lux-seconds at X=0. Uses the ISO speed-point convention H_ref ≈ 8 / ISO
    lux-sec for an 18% reflectance reference.
    """
    return np.log10(8.0 / float(iso))


def iso_to_datasheet_offset(iso):
    """Backward-compatible alias for iso_to_zero_lux_offset."""
    return iso_to_zero_lux_offset(iso)


# -----------------------------------------------------------------------------
# Y-axis value transform (output value → display value on Y axis)
# -----------------------------------------------------------------------------

def output_to_y_display(value, y_mode):
    if y_mode == Y_MODE_PERCENT:
        return m.output_to_percent(value)
    elif y_mode == Y_MODE_DENSITY:
        return m.output_to_density(value)
    else:
        return m.output_to_linear(value)


# -----------------------------------------------------------------------------
# Probe-aware sampling
# -----------------------------------------------------------------------------

def chart_x_to_probe_x_norm(chart_x_norm, x_mode, stop_min, stop_max, mid_gray=0.18):
    """
    Given a normalised X position on the chart [0, 1], return the
    normalised X position [0, 1] on the *probe ramp* that corresponds
    to that chart X.

    The probe ramp is always linear-in-stops, so:
    - X_MODE_STOPS:        chart_x_norm == probe_x_norm
    - X_MODE_LOG_EXPOSURE: same as stops (since log_exp = stops × log10(2))
    - X_MODE_LINEAR:       chart x is linear, must convert to stops then to probe pos
    """
    if x_mode == X_MODE_STOPS or x_mode == X_MODE_LOG_EXPOSURE:
        return chart_x_norm
    else:  # LINEAR
        # chart_x_norm in [0, 1] maps linearly to [linear_min, linear_max]
        linear_min = m.stops_to_linear(stop_min, mid_gray)
        linear_max = m.stops_to_linear(stop_max, mid_gray)
        linear_val = linear_min + chart_x_norm * (linear_max - linear_min)
        # convert to stops, then to normalised probe position
        if linear_val <= 0:
            return 0.0
        stops = np.log2(linear_val / mid_gray)
        return np.clip((stops - stop_min) / (stop_max - stop_min), 0.0, 1.0)


# -----------------------------------------------------------------------------
# Bitmap font — full ASCII 0..126, ported from Thatcher Freeman's
# Exposure Strip.dctl. 10x16 cell; each row is an int whose bits encode pixels
# (LSB = leftmost). The DCTL port will use the exact same arrays, so what we
# render here matches what the DCTL produces in Resolve.
# -----------------------------------------------------------------------------
import os as _os
import re as _re

FONT_HEIGHT = 16
FONT_WIDTH  = 10
FONT_LENGTH = 127

_FONT_DCTL_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "thatcher-freeman", "Exposure Strip.dctl",
)


def _load_font_from_dctl(path):
    """Parse the font_widths and font arrays from Thatcher's DCTL source."""
    with open(path, "r") as f:
        src = f.read()
    # Widths
    m_w = _re.search(r"font_widths\[FONT_LENGTH\]\s*=\s*\{([^}]*)\}", src, _re.S)
    widths = [int(x) for x in _re.findall(r"-?\d+", m_w.group(1))]
    assert len(widths) == FONT_LENGTH, f"expected {FONT_LENGTH} widths, got {len(widths)}"
    # Glyphs — match the outer block then split into 127 inner {...} groups
    m_g = _re.search(
        r"font\[FONT_LENGTH\]\[FONT_HEIGHT\]\s*=\s*\{(.*?)\}\s*;",
        src, _re.S,
    )
    inner = m_g.group(1)
    groups = _re.findall(r"\{([^{}]*)\}", inner)
    assert len(groups) == FONT_LENGTH, f"expected {FONT_LENGTH} glyphs, got {len(groups)}"
    glyphs = []
    for g in groups:
        rows = [int(x) for x in _re.findall(r"-?\d+", g)]
        assert len(rows) == FONT_HEIGHT
        glyphs.append(rows)
    return widths, glyphs


_FONT_WIDTHS_ARR, _FONT_GLYPHS_ARR = _load_font_from_dctl(_FONT_DCTL_PATH)


def _glyph_for(ch):
    code = ord(ch)
    if 0 <= code < FONT_LENGTH:
        return _FONT_GLYPHS_ARR[code], _FONT_WIDTHS_ARR[code]
    return None, _FONT_WIDTHS_ARR[ord(' ')]


def draw_text(img, text, x_px, y_px, scale=1.0, color=(1.0, 1.0, 1.0)):
    """
    Draw text into img (H, W, 3 array, float).
    x_px, y_px: pixel position of top-left of text.
    scale: multiplier on glyph size (1.0 = 10x16 px).
    """
    H, W = img.shape[:2]
    cx = x_px

    for ch in text:
        glyph, glyph_w = _glyph_for(ch)
        if glyph is None:
            cx += int(glyph_w * scale)
            continue
        for row in range(FONT_HEIGHT):
            bits = glyph[row]
            if bits == 0:
                continue
            for col in range(FONT_WIDTH):
                if bits & (1 << col):
                    x0 = cx + int(col * scale)
                    x1 = cx + int((col + 1) * scale)
                    y0 = y_px + int(row * scale)
                    y1 = y_px + int((row + 1) * scale)
                    if x0 >= 0 and x1 < W and y0 >= 0 and y1 < H:
                        img[y0:y1, x0:x1] = color
        cx += int(glyph_w * scale)


def measure_text(text, scale=1.0):
    """Return pixel width of a text string."""
    total = 0
    for ch in text:
        _, w = _glyph_for(ch)
        total += int(w * scale)
    return total


def measure_text_height(scale=1.0):
    """Glyph cell height in pixels at a given scale."""
    return int(FONT_HEIGHT * scale)


def draw_hline(img, y_px, x0, x1, color, thickness=1):
    """Draw a horizontal line segment."""
    h, w = img.shape[:2]
    if y_px < 0 or y_px >= h:
        return
    xa = max(0, min(x0, x1))
    xb = min(w - 1, max(x0, x1))
    if xa > xb:
        return
    half = max(0, thickness // 2)
    y0 = max(0, y_px - half)
    y1 = min(h, y_px + half + 1)
    img[y0:y1, xa:xb + 1] = color


def draw_vline(img, x_px, y0, y1, color, thickness=1):
    """Draw a vertical line segment."""
    h, w = img.shape[:2]
    if x_px < 0 or x_px >= w:
        return
    ya = max(0, min(y0, y1))
    yb = min(h - 1, max(y0, y1))
    if ya > yb:
        return
    half = max(0, thickness // 2)
    x0 = max(0, x_px - half)
    x1 = min(w, x_px + half + 1)
    img[ya:yb + 1, x0:x1] = color


def draw_rect_outline(img, x0, y0, x1, y1, color, thickness=1):
    """Draw a rectangle outline."""
    draw_hline(img, y0, x0, x1, color, thickness)
    draw_hline(img, y1, x0, x1, color, thickness)
    draw_vline(img, x0, y0, y1, color, thickness)
    draw_vline(img, x1, y0, y1, color, thickness)


def draw_text_centered(img, text, center_x, y_px, scale=1.0, color=(1.0, 1.0, 1.0)):
    """Draw text centered around a given X position."""
    w = measure_text(text, scale)
    draw_text(img, text, int(center_x - w // 2), y_px, scale, color)


def draw_text_rotated_ccw(img, text, x_px, y_px, scale=1.0, color=(1.0, 1.0, 1.0)):
    """
    Draw text rotated 90° counter-clockwise (reads bottom-to-top).
    (x_px, y_px) is the bottom-left anchor of the rotated string.
    """
    H, W = img.shape[:2]
    cy = y_px  # advance upward
    for ch in text:
        glyph, glyph_w = _glyph_for(ch)
        if glyph is None:
            cy -= int(glyph_w * scale)
            continue
        for row in range(FONT_HEIGHT):
            bits = glyph[row]
            if bits == 0:
                continue
            for col in range(FONT_WIDTH):
                if bits & (1 << col):
                    # Rotate CCW: glyph (col, row) → image (row, -col)
                    # Anchor: glyph col axis runs upward from cy; row axis runs rightward from x_px.
                    x0 = x_px + int(row * scale)
                    x1 = x_px + int((row + 1) * scale)
                    y0 = cy - int((col + 1) * scale)
                    y1 = cy - int(col * scale)
                    if x0 >= 0 and x1 < W and y0 >= 0 and y1 < H:
                        img[y0:y1, x0:x1] = color
        cy -= int(glyph_w * scale)


# -----------------------------------------------------------------------------
# Format helpers
# -----------------------------------------------------------------------------

def fmt_stop(s):
    """Format a stop number: '+3', '-7', '0'."""
    if s == 0:
        return "0"
    if s > 0:
        return f"+{int(s)}" if s == int(s) else f"+{s:.1f}"
    return f"{int(s)}" if s == int(s) else f"{s:.1f}"


def fmt_y(value, y_mode):
    """Format a Y-axis tick value."""
    if y_mode == Y_MODE_PERCENT:
        return f"{int(round(value))}"
    elif y_mode == Y_MODE_DENSITY:
        return f"{value:.1f}"
    else:
        return f"{value:.2f}"


def fmt_x(value, x_mode):
    """Format an X-axis tick value."""
    if x_mode == X_MODE_STOPS:
        return fmt_stop(value)
    elif x_mode == X_MODE_LOG_EXPOSURE:
        return f"{value:.2f}"
    else:
        return f"{value:.2f}"


# -----------------------------------------------------------------------------
# Sensitometry-curve helpers
# -----------------------------------------------------------------------------

def find_curve_clamp(transformed_ramp, threshold_frac=0.02):
    """
    Detect where a sensitometry curve is actively transitioning between its
    plateau values (Dmin / Dmax). Returns (left_frac, right_frac) in [0, 1]
    along the sample axis.

    For each channel, the threshold is `threshold_frac` of (Dmax - Dmin). The
    "active" range is where the curve has moved more than `threshold` away
    from BOTH endpoint plateau values. Takes the union across channels so no
    channel's transition gets cut off.
    """
    densities = m.output_to_density(np.asarray(transformed_ramp, dtype=np.float64))
    if densities.ndim == 1:
        densities = densities[:, None]
    W = densities.shape[0]
    if W < 2:
        return 0.0, 1.0

    left_indices = []
    right_indices = []
    for ch in range(densities.shape[1]):
        d = densities[:, ch]
        d_min = float(np.min(d))
        d_max = float(np.max(d))
        if d_max - d_min < 1e-3:
            continue
        thresh = threshold_frac * (d_max - d_min)
        left_plateau = float(d[0])
        right_plateau = float(d[-1])

        deviates_left = np.abs(d - left_plateau) > thresh
        if deviates_left.any():
            left_indices.append(int(np.argmax(deviates_left)))

        deviates_right = np.abs(d - right_plateau) > thresh
        if deviates_right.any():
            right_indices.append(int(W - 1 - np.argmax(deviates_right[::-1])))

    if not left_indices or not right_indices:
        return 0.0, 1.0
    left_idx = min(left_indices)
    right_idx = max(right_indices)
    if right_idx <= left_idx:
        return 0.0, 1.0
    return left_idx / (W - 1), right_idx / (W - 1)


def compute_equal_units_rect(left, top, right, bottom, x_span, y_span):
    """
    Inside the available rect, return a centred sub-rect such that
    pixels-per-x-unit equals pixels-per-y-unit (i.e. one density unit and one
    log-exposure unit cover the same pixel distance).
    """
    avail_w = right - left
    avail_h = bottom - top
    if avail_w <= 0 or avail_h <= 0 or x_span <= 0 or y_span <= 0:
        return left, top, right, bottom

    target_w = avail_h * x_span / y_span
    if target_w <= avail_w:
        plot_w = int(round(target_w))
        plot_h = int(avail_h)
    else:
        plot_w = int(avail_w)
        plot_h = int(round(avail_w * y_span / x_span))

    plot_left = left + (avail_w - plot_w) // 2
    plot_top = top + (avail_h - plot_h) // 2
    plot_right = plot_left + plot_w
    plot_bottom = plot_top + plot_h
    return plot_left, plot_top, plot_right, plot_bottom


# -----------------------------------------------------------------------------
# Main chart renderer
# -----------------------------------------------------------------------------

def render_chart(
    width, height,
    transformed_ramp,         # (probe_W, 3) — output of LUT applied to log-encoded ramp
    stop_min=-7.0, stop_max=5.0, mid_gray=0.18,
    x_mode=X_MODE_STOPS,
    y_mode=Y_MODE_PERCENT,
    zero_lux_offset=0.0,      # log10 lux-sec offset for X axis (LOG_EXPOSURE mode)
    show_image=False,
    background_image=None,    # (H, W, 3) for show_image=True
    overlay_height_frac=1.0,  # 1.0 = fullscreen, 0.2 = bottom 20%
    bg_color=(0.05, 0.05, 0.05),
    grid_color=(0.22, 0.22, 0.22),
    grid_major_color=(0.40, 0.40, 0.40),
    zero_color=(0.85, 0.65, 0.20),
    label_color=(0.85, 0.85, 0.85),
    curve_thickness_px=3,
    label_scale=2.4,
    title_scale=3.0,
    margin_left=130,
    margin_bottom=90,
    margin_top=30,
    margin_right=30,
):
    """
    Render the chart. Returns (H, W, 3) float array.

    The chart is rendered into the bottom `overlay_height_frac` of the frame.
    If show_image=True and background_image is provided, the area above the
    chart shows the background_image.
    """
    # Output image
    img = np.zeros((height, width, 3), dtype=np.float64)
    if show_image and background_image is not None:
        bg = np.asarray(background_image, dtype=np.float64)
        if bg.shape[:2] != (height, width):
            # If user passed a different-sized bg, just tile/skip
            img[:] = 0
        else:
            img[:] = bg

    # Determine chart area (in pixels, top-left origin)
    chart_top_y    = int(height * (1.0 - overlay_height_frac))
    chart_bottom_y = height - 1
    chart_left_x   = 0
    chart_right_x  = width - 1

    # Fill chart background
    img[chart_top_y:chart_bottom_y + 1, chart_left_x:chart_right_x + 1] = bg_color

    # Plot area inside the chart (after axis margins).
    # Kodak datasheet style: the plot region is always SQUARE, regardless of
    # the frame aspect ratio. We compute the largest square that fits inside
    # the available bounding box, then centre it horizontally inside the chart.
    avail_left   = chart_left_x + margin_left
    avail_right  = chart_right_x - margin_right
    avail_top    = chart_top_y + margin_top
    avail_bottom = chart_bottom_y - margin_bottom
    avail_w = avail_right - avail_left
    avail_h = avail_bottom - avail_top

    plot_size = min(avail_w, avail_h)
    if plot_size < 50:
        return img  # too small

    plot_left   = avail_left + (avail_w - plot_size) // 2
    plot_right  = plot_left + plot_size
    plot_bottom = avail_bottom
    plot_top    = plot_bottom - plot_size
    plot_w = plot_size
    plot_h = plot_size

    # Determine X / Y display ranges
    x_disp_min, x_disp_max = get_x_range(x_mode, stop_min, stop_max, mid_gray,
                                         zero_lux_offset)
    y_disp_min, y_disp_max = get_y_range(y_mode)

    # ---- Grid lines ----
    # X grid: choose nice tick spacing
    x_ticks = _nice_ticks(x_disp_min, x_disp_max, target_count=10)
    y_ticks = _nice_ticks(y_disp_min, y_disp_max, target_count=8)

    for tick in x_ticks:
        # Map tick value to pixel x
        frac = (tick - x_disp_min) / (x_disp_max - x_disp_min)
        if frac < 0 or frac > 1:
            continue
        px = plot_left + int(frac * plot_w)
        # "zero" detection: highlight the mid-grey reference column.
        # In LOG_EXPOSURE mode the axis may be shifted by zero_lux_offset, so
        # mid-grey lands at tick == zero_lux_offset, not tick == 0.
        is_zero = False
        if x_mode == X_MODE_STOPS and abs(tick) < 1e-6:
            is_zero = True
        elif x_mode == X_MODE_LOG_EXPOSURE and abs(tick - zero_lux_offset) < 1e-6:
            is_zero = True
        elif x_mode == X_MODE_LINEAR and abs(tick - mid_gray) < mid_gray * 0.05:
            is_zero = True
        col = zero_color if is_zero else grid_color
        # Make zero line 2px wide
        if is_zero:
            if px - 1 >= plot_left:
                img[plot_top:plot_bottom + 1, px - 1] = col
            img[plot_top:plot_bottom + 1, px] = col
            if px + 1 <= plot_right:
                img[plot_top:plot_bottom + 1, px + 1] = col
        else:
            img[plot_top:plot_bottom + 1, px] = col
        # Tick label below axis
        label = fmt_x(tick, x_mode)
        lw = measure_text(label, label_scale)
        draw_text(img, label, px - lw // 2, plot_bottom + 6, label_scale, label_color)

    for tick in y_ticks:
        frac = (tick - y_disp_min) / (y_disp_max - y_disp_min)
        if frac < 0 or frac > 1:
            continue
        py = plot_bottom - int(frac * plot_h)
        col = grid_color
        img[py, plot_left:plot_right + 1] = col
        # Tick label to the left of axis
        label = fmt_y(tick, y_mode)
        lw = measure_text(label, label_scale)
        draw_text(img, label, plot_left - lw - 8, py - 8, label_scale, label_color)

    # Plot axes (left and bottom borders)
    img[plot_top:plot_bottom + 1, plot_left] = grid_major_color
    img[plot_bottom, plot_left:plot_right + 1] = grid_major_color

    # ---- Plot curves ----
    # For each pixel column in the plot area, compute the chart's x-axis value,
    # find the corresponding probe x, sample the transformed ramp, transform
    # to Y display value, and draw a pixel.
    probe_w = transformed_ramp.shape[0]
    curve_R = []
    curve_G = []
    curve_B = []

    for px in range(plot_left, plot_right + 1):
        # chart x in [0, 1]
        chart_x_norm = (px - plot_left) / plot_w
        # probe x in [0, 1]
        probe_x_norm = chart_x_to_probe_x_norm(chart_x_norm, x_mode, stop_min, stop_max, mid_gray)
        # Sample the transformed ramp at this position (linear interpolation)
        probe_idx_f = probe_x_norm * (probe_w - 1)
        i0 = int(np.floor(probe_idx_f))
        i1 = min(i0 + 1, probe_w - 1)
        f = probe_idx_f - i0
        rgb = transformed_ramp[i0] * (1 - f) + transformed_ramp[i1] * f
        # Transform to Y display
        y_r = output_to_y_display(rgb[0], y_mode)
        y_g = output_to_y_display(rgb[1], y_mode)
        y_b = output_to_y_display(rgb[2], y_mode)
        # Pixel positions
        def y_to_px(y):
            frac = (y - y_disp_min) / (y_disp_max - y_disp_min)
            frac = np.clip(frac, 0.0, 1.0)
            return plot_bottom - int(frac * plot_h)
        curve_R.append((px, y_to_px(y_r)))
        curve_G.append((px, y_to_px(y_g)))
        curve_B.append((px, y_to_px(y_b)))

    def draw_curve_points(points, color):
        t = curve_thickness_px
        for (px, py) in points:
            for dy in range(-t // 2, t // 2 + 1):
                yy = py + dy
                if plot_top <= yy <= plot_bottom and plot_left <= px <= plot_right:
                    # additive blend so overlapping curves brighten
                    img[yy, px] = np.minimum(img[yy, px] + np.array(color), 1.0)

    draw_curve_points(curve_B, (0.10, 0.55, 1.0))
    draw_curve_points(curve_G, (0.05, 1.00, 0.05))
    draw_curve_points(curve_R, (1.00, 0.15, 0.05))

    # ---- Axis titles ----
    x_title = get_x_axis_label(x_mode)
    y_title = get_y_axis_label(y_mode)
    # X title centred below the tick labels
    tw = measure_text(x_title, title_scale)
    th = measure_text_height(title_scale)
    tick_h = measure_text_height(label_scale)
    x_title_y = plot_bottom + 8 + tick_h + 12
    if x_title_y + th < height:
        draw_text(img, x_title, (plot_left + plot_right) // 2 - tw // 2,
                  x_title_y, title_scale, label_color)
    # Y title rotated 90° CCW, centred vertically to the left of tick labels
    yt_w = measure_text(y_title, title_scale)
    yt_anchor_y = (plot_top + plot_bottom) // 2 + yt_w // 2
    yt_anchor_x = max(4, plot_left - margin_left + 4)
    draw_text_rotated_ccw(img, y_title, yt_anchor_x, yt_anchor_y,
                          title_scale, label_color)

    return img


def render_sensitometry_page(
    width, height,
    transformed_ramp,
    stop_min=-10.0, stop_max=6.0, mid_gray=0.18,
    zero_lux_offset=0.0,
    show_entire_curve=False,
    clamp_threshold=0.02,
    offset_log_exposure=False,
    y_min=0.0, y_max=None,
    pixels_per_unit=240,
    page_bg=(1.0, 1.0, 1.0),
    text_color=(0.10, 0.10, 0.10),
    subtext_color=(0.22, 0.22, 0.22),
    grid_color=(0.66, 0.66, 0.66),
    border_color=(0.14, 0.14, 0.14),
    curve_r_color=(0.90, 0.18, 0.18),
    curve_g_color=(0.12, 0.60, 0.34),
    curve_b_color=(0.24, 0.34, 0.72),
    title="Sensitometry",
    description_lines=None,
    footer_lines=None,
):
    """
    Render a Kodak-style sensitometry sheet:
        Y = density, top X = log exposure, bottom X = camera stops.

    The plot uses *equal units*: 1.0 in density and 1.0 in log exposure cover
    the same pixel distance.

    By default, the X range is clamped to the active part of the curve
    (toe → shoulder), hiding the asymptotic plateaus. Pass
    `show_entire_curve=True` to show the full input range.

    `offset_log_exposure=True` shifts the log-exposure axis so the right
    clamp position lands at 0. (Overrides `zero_lux_offset` when active.)
    """
    img = np.full((height, width, 3), page_bg, dtype=np.float64)

    title_scale = 3.0
    body_scale = 1.7
    axis_title_scale = 2.8
    tick_scale = 2.4
    footer_scale = 1.55

    if description_lines is None:
        description_lines = [
            "Density (Y) vs. log exposure (top X) is plotted with isotropic 1:1 scaling,",
            "so 1.0 density unit and 1.0 decade of log exposure cover the same distance.",
            "The bottom camera-stops axis is a convenience label (1 stop = log10(2) decade).",
        ]
    if footer_lines is None:
        footer_lines = [
            '"0" on the camera-stops axis marks the nominal exposure of an 18-percent gray card.',
            "Positive values indicate more exposure; negative values indicate less.",
        ]

    # ---- Determine clamp window over the input ramp ----
    full_stop_span = max(stop_max - stop_min, 1.0e-6)
    if show_entire_curve:
        clamp_left_frac, clamp_right_frac = 0.0, 1.0
    else:
        clamp_left_frac, clamp_right_frac = find_curve_clamp(
            transformed_ramp, clamp_threshold
        )
    clamped_stop_min = stop_min + clamp_left_frac * full_stop_span
    clamped_stop_max = stop_min + clamp_right_frac * full_stop_span
    stop_span = max(clamped_stop_max - clamped_stop_min, 1.0e-6)

    # Effective log-exposure offset.
    log10_2 = np.log10(2.0)
    if offset_log_exposure:
        effective_offset = -clamped_stop_max * log10_2
    else:
        effective_offset = zero_lux_offset
    log_min = clamped_stop_min * log10_2 + effective_offset
    log_max = clamped_stop_max * log10_2 + effective_offset

    # Auto y_max: round up curve max within clamp window to nearest 0.1, plus headroom.
    probe_w = transformed_ramp.shape[0]
    i_left = int(round(clamp_left_frac * (probe_w - 1)))
    i_right = int(round(clamp_right_frac * (probe_w - 1)))
    window_densities = m.output_to_density(transformed_ramp[i_left:i_right + 1])
    curve_d_max = float(np.max(window_densities)) if window_densities.size else 1.0

    if y_max is None:
        y_max_auto = float(np.ceil((curve_d_max + 0.05) * 10.0) / 10.0)
        y_max = max(y_max_auto, 0.5)
    y_span = max(y_max - y_min, 1.0e-6)

    # ---- Page header ----
    page_margin_x = int(width * 0.055)
    title_y = int(height * 0.035)
    draw_text(img, title, page_margin_x, title_y, title_scale, text_color)

    body_y = title_y + measure_text_height(title_scale) + 18
    body_line_h = measure_text_height(body_scale) + 10
    for line in description_lines:
        draw_text(img, line, page_margin_x, body_y, body_scale, subtext_color)
        body_y += body_line_h

    footer_line_h = measure_text_height(footer_scale) + 10
    footer_block_h = footer_line_h * len(footer_lines)
    footer_y = height - int(height * 0.08) - footer_block_h

    # Reserve enough vertical clearance above the plot for the top X-axis title
    # ("LOG EXPOSURE …") AND its tick labels so they don't collide with the
    # body description.
    top_axis_clearance = (
        measure_text_height(axis_title_scale)
        + measure_text_height(tick_scale)
        + 60
    )
    plot_avail_top = body_y + max(int(height * 0.06), top_axis_clearance)
    plot_avail_bottom = footer_y - int(height * 0.09)
    plot_avail_left = page_margin_x + int(width * 0.10)
    plot_avail_right = width - page_margin_x - int(width * 0.04)
    if plot_avail_right - plot_avail_left < 50 or plot_avail_bottom - plot_avail_top < 50:
        return img

    # Predetermined coordinate system: plot dimensions are derived from the
    # axis ranges times a fixed pixels-per-unit. This guarantees isotropic
    # 1:1 scaling (1 density unit = 1 log-exposure decade) by construction.
    # If the resulting plot doesn't fit the page, scale uniformly to fit.
    x_span_data = log_max - log_min
    plot_w = int(round(x_span_data * pixels_per_unit))
    plot_h = int(round(y_span * pixels_per_unit))
    avail_w = plot_avail_right - plot_avail_left
    avail_h = plot_avail_bottom - plot_avail_top
    fit_scale = min(avail_w / max(plot_w, 1), avail_h / max(plot_h, 1), 1.0)
    if fit_scale < 1.0:
        plot_w = int(round(plot_w * fit_scale))
        plot_h = int(round(plot_h * fit_scale))
    plot_left = plot_avail_left + (avail_w - plot_w) // 2
    plot_top = plot_avail_top + (avail_h - plot_h) // 2
    plot_right = plot_left + plot_w
    plot_bottom = plot_top + plot_h

    def stop_to_px(stop_value):
        frac = np.clip((stop_value - clamped_stop_min) / stop_span, 0.0, 1.0)
        return plot_left + int(round(frac * plot_w))

    def y_to_px(y_value):
        frac = np.clip((y_value - y_min) / y_span, 0.0, 1.0)
        return plot_bottom - int(round(frac * plot_h))

    # ---- Plot border and Y grid ----
    draw_rect_outline(img, plot_left, plot_top, plot_right, plot_bottom, border_color, thickness=3)
    y_tick_step = 0.5 if y_span <= 2.5 else 1.0
    tick = y_min
    while tick <= y_max + 1.0e-6:
        py = y_to_px(tick)
        draw_hline(img, py, plot_left, plot_right, grid_color, thickness=1)
        label = f"{tick:.1f}"
        draw_text(
            img, label,
            plot_left - measure_text(label, tick_scale) - 18,
            py - measure_text_height(tick_scale) // 2,
            tick_scale, text_color,
        )
        tick += y_tick_step

    # ---- Top log-exposure ticks (0.5-decade minor grid, integer major + label) ----
    minor_grid_color = tuple(min(1.0, c + 0.18) for c in grid_color)
    half_start = int(np.ceil(log_min * 2.0))
    half_end = int(np.floor(log_max * 2.0))
    for half in range(half_start, half_end + 1):
        log_tick = half * 0.5
        stop_tick = (log_tick - effective_offset) / log10_2
        if stop_tick < clamped_stop_min - 1.0e-6 or stop_tick > clamped_stop_max + 1.0e-6:
            continue
        px = stop_to_px(stop_tick)
        is_integer = (half % 2) == 0
        col = grid_color if is_integer else minor_grid_color
        draw_vline(img, px, plot_top, plot_bottom, col, thickness=1)
        if is_integer:
            draw_text_centered(
                img, f"{int(log_tick):d}", px,
                plot_top - measure_text_height(tick_scale) - 16,
                tick_scale, text_color,
            )

    # ---- Bottom camera-stops ticks ----
    minor_tick_h = max(10, plot_h // 18)
    major_tick_h = max(18, plot_h // 9)
    minor_tick_start = int(np.ceil(clamped_stop_min))
    minor_tick_end = int(np.floor(clamped_stop_max))
    for stop_tick in range(minor_tick_start, minor_tick_end + 1):
        px = stop_to_px(float(stop_tick))
        is_major = (stop_tick % 2) == 0
        tick_h = major_tick_h if is_major else minor_tick_h
        draw_vline(img, px, plot_bottom - tick_h, plot_bottom, grid_color, thickness=1)
        if is_major:
            draw_text_centered(
                img, f"{stop_tick:d}", px,
                plot_bottom + 18,
                tick_scale, text_color,
            )

    # ---- Curves (sample only the clamped sub-window of the ramp) ----
    sub_count = max(i_right - i_left + 1, 2)
    curve_points = [[], [], []]
    for px in range(plot_left, plot_right + 1):
        chart_x_norm = (px - plot_left) / plot_w
        probe_idx_f = i_left + chart_x_norm * (sub_count - 1)
        i0 = int(np.floor(probe_idx_f))
        i1 = min(i0 + 1, probe_w - 1)
        f = probe_idx_f - i0
        rgb = transformed_ramp[i0] * (1.0 - f) + transformed_ramp[i1] * f
        curve_points[0].append((px, y_to_px(output_to_y_display(rgb[0], Y_MODE_DENSITY))))
        curve_points[1].append((px, y_to_px(output_to_y_display(rgb[1], Y_MODE_DENSITY))))
        curve_points[2].append((px, y_to_px(output_to_y_display(rgb[2], Y_MODE_DENSITY))))

    def draw_curve(points, color):
        thickness = 5
        for px, py in points:
            for dy in range(-thickness // 2, thickness // 2 + 1):
                yy = py + dy
                if plot_top <= yy <= plot_bottom:
                    img[yy, px] = color

    draw_curve(curve_points[2], curve_b_color)
    draw_curve(curve_points[1], curve_g_color)
    draw_curve(curve_points[0], curve_r_color)

    # ---- Axis titles ----
    log_title = "LOG EXPOSURE  (decades, isotropic with density)"
    if offset_log_exposure:
        log_title = "LOG EXPOSURE  (decades, zeroed at right clamp)"
    draw_text_centered(
        img, log_title,
        (plot_left + plot_right) // 2,
        plot_top - measure_text_height(tick_scale) - measure_text_height(axis_title_scale) - 34,
        axis_title_scale, subtext_color,
    )
    draw_text_centered(
        img, "CAMERA STOPS  (1 stop = 0.301 dec, not isotropic)",
        (plot_left + plot_right) // 2,
        plot_bottom + 18 + measure_text_height(tick_scale) + 26,
        axis_title_scale, subtext_color,
    )
    density_w = measure_text("DENSITY", axis_title_scale)
    draw_text_rotated_ccw(
        img, "DENSITY",
        max(10, plot_left - int(width * 0.085)),
        (plot_top + plot_bottom) // 2 + density_w // 2,
        axis_title_scale, subtext_color,
    )

    # ---- Footer ----
    y = footer_y
    for line in footer_lines:
        draw_text(img, line, page_margin_x, y, footer_scale, subtext_color)
        y += footer_line_h

    return img


def _nice_ticks(vmin, vmax, target_count=8):
    """Return a list of nicely-spaced tick values."""
    span = vmax - vmin
    if span <= 0:
        return [vmin]
    raw_step = span / target_count
    # Round to nearest "nice" number: 1, 2, 5 × 10^k
    mag = 10 ** np.floor(np.log10(raw_step))
    norm = raw_step / mag
    if norm < 1.5:
        step = 1 * mag
    elif norm < 3.5:
        step = 2 * mag
    elif norm < 7.5:
        step = 5 * mag
    else:
        step = 10 * mag
    # Generate ticks
    start = np.ceil(vmin / step) * step
    ticks = []
    v = start
    while v <= vmax + step * 1e-6:
        ticks.append(round(v, 10))
        v += step
    return ticks
