"""
cube_lut.py — Parse and apply .cube LUTs (1D and 3D).
Mimics Resolve's behaviour: tetrahedral interpolation for 3D LUTs (matching
Resolve's default for grading LUTs), no clamping unless DOMAIN_MIN/MAX.
"""

import numpy as np
import re


class CubeLUT:
    def __init__(self, size, lut_type, data, domain_min, domain_max):
        self.size = size                # int (1D) or int (3D, cube side length)
        self.lut_type = lut_type        # "1D" or "3D"
        self.data = data                # for 3D: shape (size, size, size, 3)
                                        # for 1D: shape (size, 3)
        self.domain_min = domain_min    # (3,) array
        self.domain_max = domain_max    # (3,) array

    @classmethod
    def from_file(cls, path):
        size = None
        lut_type = None
        domain_min = np.array([0.0, 0.0, 0.0])
        domain_max = np.array([1.0, 1.0, 1.0])
        entries = []

        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                # Header lines
                if s.upper().startswith("TITLE"):
                    continue
                if s.upper().startswith("LUT_3D_SIZE"):
                    size = int(s.split()[-1])
                    lut_type = "3D"
                    continue
                if s.upper().startswith("LUT_1D_SIZE"):
                    size = int(s.split()[-1])
                    lut_type = "1D"
                    continue
                if s.upper().startswith("DOMAIN_MIN"):
                    domain_min = np.array([float(x) for x in s.split()[1:4]])
                    continue
                if s.upper().startswith("DOMAIN_MAX"):
                    domain_max = np.array([float(x) for x in s.split()[1:4]])
                    continue
                # Data line
                parts = s.split()
                if len(parts) >= 3:
                    try:
                        entries.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue

        if lut_type is None or size is None:
            raise ValueError(f"Could not determine LUT type/size from {path}")

        data = np.array(entries, dtype=np.float64)
        if lut_type == "3D":
            expected = size ** 3
            if data.shape[0] != expected:
                raise ValueError(f"Expected {expected} entries, got {data.shape[0]}")
            # Cube file order: B varies fastest, then G, then R (per Adobe spec)
            # i.e. for r in 0..N-1: for g in 0..N-1: for b in 0..N-1: write entry
            data = data.reshape(size, size, size, 3)  # [r, g, b, channel]
        else:  # 1D
            if data.shape[0] != size:
                raise ValueError(f"Expected {size} entries, got {data.shape[0]}")

        return cls(size, lut_type, data, domain_min, domain_max)

    def apply(self, rgb):
        """
        Apply LUT to an RGB array of shape (..., 3).
        Returns array of same shape, float64.
        """
        rgb = np.asarray(rgb, dtype=np.float64)
        # Normalise input from [domain_min, domain_max] to [0, size-1]
        norm = (rgb - self.domain_min) / (self.domain_max - self.domain_min)
        idx = norm * (self.size - 1)
        # Clamp to valid index range (mimics Resolve's edge behaviour for OOR inputs)
        idx = np.clip(idx, 0.0, self.size - 1)

        if self.lut_type == "3D":
            return self._apply_3d(idx)
        else:
            return self._apply_1d(idx)

    def _apply_3d(self, idx):
        """
        Tetrahedral interpolation on a 3D cube — splits each unit cell into 6
        tetrahedra by sorting (fr, fg, fb), then blends 4 corners per sample.
        Matches Resolve's default 3D LUT interpolation.
        """
        i0 = np.floor(idx).astype(np.int64)
        i1 = np.minimum(i0 + 1, self.size - 1)
        f = idx - i0

        r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
        r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
        fr = f[..., 0:1]
        fg = f[..., 1:2]
        fb = f[..., 2:3]

        c000 = self.data[r0, g0, b0]
        c001 = self.data[r0, g0, b1]
        c010 = self.data[r0, g1, b0]
        c011 = self.data[r0, g1, b1]
        c100 = self.data[r1, g0, b0]
        c101 = self.data[r1, g0, b1]
        c110 = self.data[r1, g1, b0]
        c111 = self.data[r1, g1, b1]

        # Six tetrahedra dispatched on the sort order of (fr, fg, fb).
        m1 = (fr >= fg) & (fg >= fb)
        m2 = (fr >= fb) & (fb >= fg) & ~m1
        m3 = (fb >= fr) & (fr >= fg) & ~m1 & ~m2
        m4 = (fb >= fg) & (fg >= fr) & ~m1 & ~m2 & ~m3
        m5 = (fg >= fb) & (fb >= fr) & ~m1 & ~m2 & ~m3 & ~m4
        # m6: remaining case (fg >= fr >= fb)

        out1 = c000 + fr * (c100 - c000) + fg * (c110 - c100) + fb * (c111 - c110)
        out2 = c000 + fr * (c100 - c000) + fb * (c101 - c100) + fg * (c111 - c101)
        out3 = c000 + fb * (c001 - c000) + fr * (c101 - c001) + fg * (c111 - c101)
        out4 = c000 + fb * (c001 - c000) + fg * (c011 - c001) + fr * (c111 - c011)
        out5 = c000 + fg * (c010 - c000) + fb * (c011 - c010) + fr * (c111 - c011)
        out6 = c000 + fg * (c010 - c000) + fr * (c110 - c010) + fb * (c111 - c110)

        out = np.where(m1, out1,
              np.where(m2, out2,
              np.where(m3, out3,
              np.where(m4, out4,
              np.where(m5, out5, out6)))))
        return out

    def _apply_1d(self, idx):
        """Linear interpolation per-channel on a 1D LUT."""
        i0 = np.floor(idx).astype(np.int64)
        i1 = np.minimum(i0 + 1, self.size - 1)
        f  = idx - i0
        # Per-channel lookup
        out = np.empty_like(idx, dtype=np.float64)
        for ch in range(3):
            v0 = self.data[i0[..., ch], ch]
            v1 = self.data[i1[..., ch], ch]
            out[..., ch] = v0 * (1 - f[..., ch]) + v1 * f[..., ch]
        return out


def make_synthetic_test_lut(path):
    """
    Create a small 17-cube synthetic LUT that applies a known sigmoid
    to each channel independently. Used as a known-good reference for
    validating the inspector.

    Input: ARRI LogC3 (0..1 range typical)
    Output: a smooth S-curve in [0, 1]
    """
    size = 17
    lines = ["TITLE \"Synthetic Sigmoid Test LUT\"", f"LUT_3D_SIZE {size}", ""]
    # Generate cube. Order: r outermost, b innermost (cube file convention is
    # actually B fastest, but we write the body matching that):
    for r in range(size):
        for g in range(size):
            for b in range(size):
                rin = r / (size - 1)
                gin = g / (size - 1)
                bin = b / (size - 1)
                # Per-channel sigmoid mimicking film: R hottest, B coolest
                def sigmoid(x, center, slope=8.0):
                    return 1.0 / (1.0 + np.exp(-slope * (x - center)))
                # Slight per-channel offsets like the Kodak 2383 shape:
                # R curve is leftward (lower density at given exposure),
                # B curve is rightward (higher density / cooler).
                rout = sigmoid(rin, center=0.375, slope=7.5)
                gout = sigmoid(gin, center=0.391, slope=8.0)
                bout = sigmoid(bin, center=0.410, slope=7.5)
                lines.append(f"{rout:.6f} {gout:.6f} {bout:.6f}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    # Sanity check
    import os
    test_path = "/tmp/test_synthetic.cube"
    make_synthetic_test_lut(test_path)
    lut = CubeLUT.from_file(test_path)
    print(f"Loaded LUT: type={lut.lut_type}, size={lut.size}")
    print(f"  domain min={lut.domain_min}, max={lut.domain_max}")

    # Test: identity input at midgrey should pass through sigmoid centre
    test_rgb = np.array([[0.391, 0.391, 0.391]])
    out = lut.apply(test_rgb)
    print(f"  Input (0.391,0.391,0.391) -> {out[0]}")
    print(f"  Expected ~0.5 (sigmoid centre)")

    # Round trip an array
    ramp = np.linspace(0, 1, 11)
    rgb_in = np.stack([ramp, ramp, ramp], axis=-1)
    rgb_out = lut.apply(rgb_in)
    print(f"\n  Identity ramp through LUT:")
    for i in range(len(ramp)):
        print(f"    {rgb_in[i, 0]:.3f} -> {rgb_out[i, 0]:.4f}")
