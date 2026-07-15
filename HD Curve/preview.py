"""
preview.py — Generate preview PNGs of the H&D Inspector chart for all axis
combinations, using the synthetic test LUT.

This is the validation harness. The output of this script is what the
DCTL should produce when given the same probe ramp and the same LUT.
"""

import os
import numpy as np
from PIL import Image

import hd_math as m
import cube_lut
import chart_renderer as cr


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def gamma_encode_for_display(img, gamma=2.2):
    """sRGB-ish encoding for display PNGs."""
    return np.power(np.clip(img, 0, 1), 1.0 / gamma)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "preview_out")
    os.makedirs(out_dir, exist_ok=True)

    # Use the real Kodak 2383 print LUT (lives one level up alongside this folder)
    lut_path = os.path.join(os.path.dirname(script_dir), "Kodak250D_2383Print.cube")
    lut = cube_lut.CubeLUT.from_file(lut_path)
    print(f"Loaded LUT: size={lut.size}")

    # Generate probe ramp — fixed range matching the DCTL pair contract.
    stop_min, stop_max = -8.0, 8.0
    probe_w = 2048
    probe = m.generate_probe_ramp(probe_w, stop_min, stop_max)
    print(f"Probe ramp: shape={probe.shape}, range=[{probe.min():.3f}, {probe.max():.3f}]")

    # Apply LUT
    transformed = lut.apply(probe)
    print(f"Transformed: range=[{transformed.min():.3f}, {transformed.max():.3f}]")

    # Sanity check: at stops=0 (mid grey), output should be ~0.5 (sigmoid centre)
    zero_idx = int((0.0 - stop_min) / (stop_max - stop_min) * (probe_w - 1))
    print(f"  At stops=0 (idx {zero_idx}): probe={probe[zero_idx, 0]:.4f}, "
          f"out={transformed[zero_idx, 0]:.4f}")

    # Render all 9 combinations
    width, height = 1920, 1080
    combos = [
        (cr.X_MODE_STOPS,        cr.Y_MODE_PERCENT,  "stops_percent"),
        (cr.X_MODE_STOPS,        cr.Y_MODE_DENSITY,  "stops_density"),
        (cr.X_MODE_STOPS,        cr.Y_MODE_LINEAR,   "stops_linear"),
        (cr.X_MODE_LOG_EXPOSURE, cr.Y_MODE_PERCENT,  "logexp_percent"),
        (cr.X_MODE_LOG_EXPOSURE, cr.Y_MODE_DENSITY,  "logexp_density"),
        (cr.X_MODE_LOG_EXPOSURE, cr.Y_MODE_LINEAR,   "logexp_linear"),
        (cr.X_MODE_LINEAR,       cr.Y_MODE_PERCENT,  "linear_percent"),
        (cr.X_MODE_LINEAR,       cr.Y_MODE_DENSITY,  "linear_density"),
        (cr.X_MODE_LINEAR,       cr.Y_MODE_LINEAR,   "linear_linear"),
    ]

    for x_mode, y_mode, name in combos:
        print(f"  Rendering {name}...")
        img = cr.render_chart(
            width, height, transformed,
            stop_min=stop_min, stop_max=stop_max,
            x_mode=x_mode, y_mode=y_mode,
            overlay_height_frac=1.0,  # fullscreen
        )
        # Gamma encode for visual preview
        img_disp = gamma_encode_for_display(img)
        Image.fromarray(to_uint8(img_disp)).save(os.path.join(out_dir, f"{name}.png"))

    # Render an additional logexp_density view with the ISO-64 (Kodachrome 64)
    # speed-point offset, aligning 0 lux-seconds with the X = 0 column.
    print(f"  Rendering logexp_density_k64offset...")
    img = cr.render_chart(
        width, height, transformed,
        stop_min=stop_min, stop_max=stop_max,
        x_mode=cr.X_MODE_LOG_EXPOSURE, y_mode=cr.Y_MODE_DENSITY,
        zero_lux_offset=cr.iso_to_zero_lux_offset(64),
        overlay_height_frac=1.0,
    )
    img_disp = gamma_encode_for_display(img)
    Image.fromarray(to_uint8(img_disp)).save(
        os.path.join(out_dir, "logexp_density_k64offset.png"))

    # Also render bottom-strip overlay version
    print(f"  Rendering bottom_strip_overlay...")
    # Make a fake background image
    bg = np.full((height, width, 3), 0.15, dtype=np.float64)
    # Add a gradient to simulate footage
    grad_x = np.linspace(0, 1, width)
    bg[:, :, 0] = grad_x[None, :] * 0.5
    bg[:, :, 1] = grad_x[None, :] * 0.5
    bg[:, :, 2] = grad_x[None, :] * 0.5
    img = cr.render_chart(
        width, height, transformed,
        stop_min=stop_min, stop_max=stop_max,
        x_mode=cr.X_MODE_STOPS, y_mode=cr.Y_MODE_PERCENT,
        show_image=True, background_image=bg,
        overlay_height_frac=0.40,
    )
    img_disp = gamma_encode_for_display(img)
    Image.fromarray(to_uint8(img_disp)).save(os.path.join(out_dir, "bottom_strip_overlay.png"))

    # Kodak-style sensitometry sheet: dual X axes with density on Y.
    # Render three variants exercising the new toggles.
    sensi_stop_min, sensi_stop_max = -10.0, 8.0
    sensi_probe = m.generate_probe_ramp(probe_w, sensi_stop_min, sensi_stop_max)
    sensi_transformed = lut.apply(sensi_probe)
    sensi_w, sensi_h = 1920, 1080

    print(f"  Rendering sensitometry_page (clamped, default)...")
    img_clamped = cr.render_sensitometry_page(
        sensi_w, sensi_h, sensi_transformed,
        stop_min=sensi_stop_min, stop_max=sensi_stop_max,
    )
    Image.fromarray(to_uint8(img_clamped)).save(
        os.path.join(out_dir, "sensitometry_page.png"))

    print(f"  Rendering sensitometry_page_full (show_entire_curve=True)...")
    img_full = cr.render_sensitometry_page(
        sensi_w, sensi_h, sensi_transformed,
        stop_min=sensi_stop_min, stop_max=sensi_stop_max,
        show_entire_curve=True,
    )
    Image.fromarray(to_uint8(img_full)).save(
        os.path.join(out_dir, "sensitometry_page_full.png"))

    print(f"  Rendering sensitometry_page_offset (offset_log_exposure=True)...")
    img_offset = cr.render_sensitometry_page(
        sensi_w, sensi_h, sensi_transformed,
        stop_min=sensi_stop_min, stop_max=sensi_stop_max,
        offset_log_exposure=True,
    )
    Image.fromarray(to_uint8(img_offset)).save(
        os.path.join(out_dir, "sensitometry_page_offset.png"))

    # Same three variants for Kodachrome 64 (Spectral K64 AWG4 LUT — needs LogC4 probe).
    k64_lut_path = os.path.join(
        os.path.dirname(script_dir),
        "Spectral_K64_AWG4_sRGB.cube",
    )
    if os.path.exists(k64_lut_path):
        k64_lut = cube_lut.CubeLUT.from_file(k64_lut_path)
        print(f"Loaded LUT: K64 size={k64_lut.size}")
        k64_probe = m.generate_probe_ramp(
            probe_w, sensi_stop_min, sensi_stop_max,
            encoding=m.ENCODING_LOGC4,
        )
        k64_transformed = k64_lut.apply(k64_probe)

        print(f"  Rendering k64_sensitometry (clamped)...")
        img = cr.render_sensitometry_page(
            sensi_w, sensi_h, k64_transformed,
            stop_min=sensi_stop_min, stop_max=sensi_stop_max,
            title="Sensitometry — Kodachrome 64",
        )
        Image.fromarray(to_uint8(img)).save(
            os.path.join(out_dir, "k64_sensitometry.png"))

        print(f"  Rendering k64_sensitometry_full (show_entire_curve=True)...")
        img = cr.render_sensitometry_page(
            sensi_w, sensi_h, k64_transformed,
            stop_min=sensi_stop_min, stop_max=sensi_stop_max,
            show_entire_curve=True,
            title="Sensitometry — Kodachrome 64",
        )
        Image.fromarray(to_uint8(img)).save(
            os.path.join(out_dir, "k64_sensitometry_full.png"))

        print(f"  Rendering k64_sensitometry_offset (offset_log_exposure=True)...")
        img = cr.render_sensitometry_page(
            sensi_w, sensi_h, k64_transformed,
            stop_min=sensi_stop_min, stop_max=sensi_stop_max,
            offset_log_exposure=True,
            title="Sensitometry — Kodachrome 64",
        )
        Image.fromarray(to_uint8(img)).save(
            os.path.join(out_dir, "k64_sensitometry_offset.png"))
    else:
        print(f"  (K64 LUT not found at {k64_lut_path}, skipping)")

    # Same three variants for MSGenesisE100 (different LUT character).
    msg_lut_path = os.path.join(
        os.path.dirname(script_dir),
        "MSGenesisE100_5.chart_arri_alexa_awg3_5600K00000001.cube",
    )
    if os.path.exists(msg_lut_path):
        msg_lut = cube_lut.CubeLUT.from_file(msg_lut_path)
        print(f"Loaded LUT: MSGenesisE100 size={msg_lut.size}")
        msg_transformed = msg_lut.apply(sensi_probe)

        print(f"  Rendering msgenesis_sensitometry (clamped)...")
        img = cr.render_sensitometry_page(
            sensi_w, sensi_h, msg_transformed,
            stop_min=sensi_stop_min, stop_max=sensi_stop_max,
            title="Sensitometry — MSGenesisE100",
        )
        Image.fromarray(to_uint8(img)).save(
            os.path.join(out_dir, "msgenesis_sensitometry.png"))

        print(f"  Rendering msgenesis_sensitometry_full (show_entire_curve=True)...")
        img = cr.render_sensitometry_page(
            sensi_w, sensi_h, msg_transformed,
            stop_min=sensi_stop_min, stop_max=sensi_stop_max,
            show_entire_curve=True,
            title="Sensitometry — MSGenesisE100",
        )
        Image.fromarray(to_uint8(img)).save(
            os.path.join(out_dir, "msgenesis_sensitometry_full.png"))

        print(f"  Rendering msgenesis_sensitometry_offset (offset_log_exposure=True)...")
        img = cr.render_sensitometry_page(
            sensi_w, sensi_h, msg_transformed,
            stop_min=sensi_stop_min, stop_max=sensi_stop_max,
            offset_log_exposure=True,
            title="Sensitometry — MSGenesisE100",
        )
        Image.fromarray(to_uint8(img)).save(
            os.path.join(out_dir, "msgenesis_sensitometry_offset.png"))
    else:
        print(f"  (MSGenesisE100 LUT not found at {msg_lut_path}, skipping)")

    print(f"\nDone. Outputs in {out_dir}/")
    for f in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, f))
        print(f"  {f}  ({size // 1024} KB)")


if __name__ == "__main__":
    main()
