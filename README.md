# custom_dctls

Custom DCTLs for DaVinci Resolve — my personal collection of color and
film-emulation tools. New tools get added here over time.

## Contents

- [Installation](#installation)
- [HD Curve](#hd-curve) — live characteristic-curve (H&D) scope
- [Yedlin Grain](#yedlin-grain) — film grain plugin
- [Reference material](#reference-material)

## Installation

Copy the `.dctl` files into the Resolve LUT folder and refresh the LUT list
(right-click the node's LUT menu → Refresh, or restart Resolve):

- **macOS:** `/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT/`
- **Windows:** `%PROGRAMDATA%\Blackmagic Design\DaVinci Resolve\Support\LUT\`

Apply a DCTL by adding the **DCTL** ResolveFX plugin to a node and selecting the
file, or via the node's LUT dropdown. Tooltips need Resolve 19.1+.

---

## HD Curve

![HD Curve Display](docs/hd-curve.png)

*HD Curve measuring an ARRI ALEXA → 500T negative → 2383 print chain, plotted
as output % vs. camera stops / log exposure.*

A two-node scope that plots the characteristic curve (H&D curve) of everything
between two points of a node tree — LUTs, CSTs, curves, full film-emulation
chains — live, per channel, against a photometrically meaningful axis system.

**Current pairing: `HD Curve Probe_v2.3.dctl` ↔ `HD Curve Display_v2.23.dctl`.**
The two versions belong together, and the Probe Encoding setting must match on
both nodes.

### How it works

The probe renders a thin synthetic grayscale strip along the bottom of the
frame, sweeping exposure in stops around 18% mid gray and encoding it with the
selected camera transfer function. Everything downstream processes that strip
like real footage. The display node samples the strip's bottom row and plots
input exposure (x) against processed output (y).

```
[01 Probe] -> CST -> grade -> FilmOut LUT -> ... -> [last node: Display]
```

### Probe

- **Probe Encoding**: ARRI LogC3 (default), ARRI LogC4, DaVinci Intermediate.
  Use whatever your node tree expects as input footage.
- **Greyscale Height %**: strip height, default 0.5% (the display reads only
  the bottom pixel row).
- The strip sweeps each encoding's full usable range, symmetric around mid
  gray, bounded by the clip at code 1.0 and capped at ±10 stops:
  LogC3 ±8.2574 · LogC4 ±10 (clips at +11.35) · DaVinci Intermediate ±9.1178.

### Display

- **X Axis**: Stops or Log Exposure (default), both relative to 18% mid gray
  (the yellow reference line). In Log Exposure mode the plot shows log exposure
  on top and camera stops along the bottom, sensitometry-style.
- **Y Axis**: Output % (signal) or Density (−log₁₀ of output).
- **Output Signal**: for photometrically true density values, tell the display
  what the chain outputs — *None / Linear* (default, reads code values as-is),
  *Video Gamma 2.4*, or *sRGB Monitor*. Density measures light, so a
  gamma-encoded signal must be linearized or density and gamma read low by
  exactly the display gamma.
- **Clamp to Active Range** (default on): zooms the X axis to where the curve
  actually changes, trimming flat clipped tails. Detection runs in density
  space with separate shadow/highlight thresholds (`ACTIVE_RANGE_*` defines).
- **Offset Exposure to 0**: relabels the X axis to end at 0 and count down, so
  the leftmost number reads the total dynamic range directly. Pure relabeling.
- **Chart Size**: Fullscreen, or Right Overlay — a fixed square chart on the
  right with the whole frame scaled down beside it.
- **Plot Shape**: Fill Width · Square · **Datasheet 1:1** (one density unit =
  one log-exposure decade in pixels, so a 45° slope is exactly gamma 1.0, like
  a Kodak datasheet plot; needs the Density Y axis).
- **Show Grid / Grid Thickness / Transparent Background**: cosmetics. The RGB
  curves have a fixed thickness; the slider moves only grid and border.
- Density Y axis auto-scales to the curve's black point (next 0.5 increment).
- UI scales automatically with timeline resolution and aspect ratio; margins
  hug the labels.

### Math and verification

The transfer functions are implemented from the manufacturer specifications and
verified numerically (constants, mid-gray anchors, segment continuity,
round-trips against the official inverses, clip points, and an end-to-end
pipeline simulation with < 1e-6 code error):

| Encoding | 18% mid gray → code | Clip at code 1.0 |
|---|---|---|
| ARRI LogC3 (EI 800) | 0.391 | +8.26 stops |
| ARRI LogC4 | 0.278 | +11.35 stops |
| DaVinci Intermediate | 0.336 | +9.12 stops (linear 100) |

Caveats to keep in mind: the chart measures the *system* curve of everything
between probe and display (compare against reversal-stock datasheets directly,
or cascaded neg+print curves); spatial nodes (halation, grain, blur) contaminate
the synthetic strip — bypass them for exact reads; and the strip must reach the
display node intact (no crops/resizes of the bottom rows in between).

The `HD Curve/` folder also contains `Sensitometry Curve Display/Probe.dctl`
(the predecessor these were built from) and Python tooling (`hd_math.py`,
`chart_renderer.py`, `cube_lut.py`, `preview.py`, `test_hd_math.py`) for
previewing and testing the math outside Resolve.

---

## Yedlin Grain

`grain_plugin_v3.dctl` — a film grain plugin combining point-based stochastic
noise with subtle domain warping for a particulate, non-digital character.
Controls: Grain Amount, Grain Size, Grit, Chroma Separation, Highlight
Protection, Shadow Response, Seed.

---

## Reference material

- `thatcher-freeman/` — utilities by
  [Thatcher Freeman](https://github.com/thatcherfreeman/utility-dctls)
  (Exposure Chart, Exposure Strip, Film Curve), kept for reference. HD Curve's
  bitmap-font text renderer is based on the approach in these tools.
- `LUTs/` — reference `.cube` LUTs used for testing (film-match and print
  emulations).

## Conventions

Each functional change to a tool ships as a new `_vX.Y` file; superseded
versions are deleted (history lives in git). Paired tools (like HD Curve's probe
and display) are versioned together — check the header comment of each file for
its changelog.
