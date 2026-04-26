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


def get_x_range(mode, stop_min, stop_max, mid_gray=0.18, datasheet_offset=0.0):
    """
    Default X-axis range (in display units) for each mode.

    datasheet_offset: shift in log10(lux-seconds) units, only applied to
    LOG_EXPOSURE mode. Use to align our X axis with Kodak datasheet absolute
    exposure scale. For ISO N reversal film: offset ≈ log10(8 / N).
    """
    if mode == X_MODE_STOPS:
        return (stop_min, stop_max)
    elif mode == X_MODE_LOG_EXPOSURE:
        return (stop_min * np.log10(2.0) + datasheet_offset,
                stop_max * np.log10(2.0) + datasheet_offset)
    else:  # LINEAR
        return (m.stops_to_linear(stop_min, mid_gray),
                m.stops_to_linear(stop_max, mid_gray))


def iso_to_datasheet_offset(iso):
    """
    Convert a film ISO speed to a datasheet X-axis offset (log10 lux-seconds).
    Uses the ISO speed-point convention H_ref ≈ 8 / ISO lux-sec for an 18%
    reflectance reference. K64 → ~−0.903; E100 → ~−1.097; ISO 800 → −2.0.
    """
    return np.log10(8.0 / float(iso))


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
# Main chart renderer
# -----------------------------------------------------------------------------

def render_chart(
    width, height,
    transformed_ramp,         # (probe_W, 3) — output of LUT applied to log-encoded ramp
    stop_min=-7.0, stop_max=5.0, mid_gray=0.18,
    x_mode=X_MODE_STOPS,
    y_mode=Y_MODE_PERCENT,
    datasheet_offset=0.0,     # log10 lux-sec offset for X axis (LOG_EXPOSURE mode)
    show_image=False,
    background_image=None,    # (H, W, 3) for show_image=True
    overlay_height_frac=1.0,  # 1.0 = fullscreen, 0.2 = bottom 20%
    bg_color=(0.05, 0.05, 0.05),
    grid_color=(0.22, 0.22, 0.22),
    grid_major_color=(0.40, 0.40, 0.40),
    zero_color=(0.85, 0.65, 0.20),
    label_color=(0.85, 0.85, 0.85),
    curve_thickness_px=2,
    label_scale=1.8,        # tick labels (was 1.2)
    title_scale=2.2,        # axis titles (X / Y legends)
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
                                         datasheet_offset)
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
        # In LOG_EXPOSURE mode the axis may be shifted by datasheet_offset, so
        # mid-grey lands at tick == datasheet_offset, not tick == 0.
        is_zero = False
        if x_mode == X_MODE_STOPS and abs(tick) < 1e-6:
            is_zero = True
        elif x_mode == X_MODE_LOG_EXPOSURE and abs(tick - datasheet_offset) < 1e-6:
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
