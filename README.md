# Generalized-RBJ-and-Orfanidis-Higher-Order-Filters-and-Equalizers
```python
#!/usr/bin/env python3
"""
orfanidis_style_hpeq_clean.py

"Orfanidis-style" high-order parametric EQ built in a robust way:

- Uses RBJ peaking biquads (Audio EQ Cookbook).
- Uses Butterworth pole-derived Q distribution to get a smooth, 
  higher-order bell shape (no notch, no trapezoid).
- Gain at f0 is exact. Out-of-band is flat (0 dB).
- Result is a cascade of biquads ready for real-time DSP use.

This avoids fragile analog+BLT polynomial games while keeping
the core high-order parametric idea Orfanidis was after:
prototype-driven shape, (f0, gain, bandwidth) style control,
implemented as stable cascaded sections.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
from math import pi, sin, cos


# ---------------------------------------------------------
# Biquad struct
# ---------------------------------------------------------

@dataclass
class Biquad:
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float


# ---------------------------------------------------------
# RBJ peaking EQ biquad
# ---------------------------------------------------------

def rbj_peak(f0: float, fs: float, Q: float, gain_db: float) -> Biquad:
    """RBJ peaking EQ biquad."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * pi * (f0 / fs)
    alpha = sin(w0) / (2.0 * Q)
    cw = cos(w0)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / A

    # normalize
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    return Biquad(float(b0), float(b1), float(b2), float(a1), float(a2))


# ---------------------------------------------------------
# Butterworth-derived Q distribution (prototype-based)
# ---------------------------------------------------------

def butterworth_Qs(order: int) -> List[float]:
    """
    Return Q values for the 2nd-order sections of an Nth-order Butterworth LP.

    This is the classical formula:
      Poles at:
        p_k = -sin(theta_k) + j cos(theta_k),
        theta_k = (2k - 1) * pi / (2N), k = 1..N
      Grouped into conjugate pairs → sections:
        s^2 + 2 sin(theta_k) s + 1
      which corresponds to Q_k = 1 / (2 sin(theta_k)).

    We use those Qs as shaping Qs for our peaking sections.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError("Butterworth order must be even and >= 2.")
    n_sections = order // 2
    Qs: List[float] = []
    for k in range(1, n_sections + 1):
        theta = (2 * k - 1) * pi / (2 * order)
        Q = 1.0 / (2.0 * sin(theta))
        Qs.append(Q)
    return Qs


# ---------------------------------------------------------
# High-order PEQ design (Butterworth-style, robust)
# ---------------------------------------------------------

def design_hpeq_butterworth_rbj(
    order: int,
    fs: float,
    f0: float,
    gain_db: float,
) -> List[Biquad]:
    """
    High-order parametric EQ built as a cascade of RBJ peaking filters
    whose Qs are taken from a Butterworth prototype.

    - order: even integer >= 2, total filter order.
      (Each biquad is 2nd order → number of stages = order / 2.)
    - fs: sampling rate (Hz)
    - f0: center frequency (Hz)
    - gain_db: desired total gain at f0 (dB)

    Construction:
      - Get Q_k from Nth-order Butterworth low-pass.
      - Build one RBJ peaking EQ per Q_k, all at same f0.
      - Split gain evenly in dB across stages so that the
        product at f0 equals gain_db.

    Properties:
      - |H(f0)| = 10^(gain_db/20) exactly.
      - |H(f)| → 1 outside band.
      - Bell gets steeper / "higher-order" as order increases.
      - No center notch, no weird trapezoid.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError("order must be even and >= 2.")
    if not (0 < f0 < fs * 0.5):
        raise ValueError("f0 must be between 0 and Nyquist.")

    Qs = butterworth_Qs(order)
    n_sections = len(Qs)

    # Split gain in dB equally among sections:
    # total G = Π G_k  →  sum g_k = gain_db  (in dB)
    per_stage_gain_db = gain_db / n_sections

    biquads: List[Biquad] = []
    for Q in Qs:
        bq = rbj_peak(f0=f0, fs=fs, Q=Q, gain_db=per_stage_gain_db)
        biquads.append(bq)

    return biquads


# ---------------------------------------------------------
# Frequency response helpers
# ---------------------------------------------------------

def cascade_freq_response(biquads: List[Biquad], fs: float, n_fft: int = 4096):
    w = np.linspace(0.0, pi, n_fft)
    z = np.exp(1j * w)
    H = np.ones_like(z, dtype=complex)
    for bq in biquads:
        H *= (bq.b0 + bq.b1 / z + bq.b2 / (z**2)) / (1 + bq.a1 / z + bq.a2 / (z**2))
    f = w * fs / (2.0 * pi)
    return f, H


# ---------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------

def example():
    fs = 48000.0
    f0 = 2000.0
    gain_db = 6.0

    # Compare 2nd-order (classic RBJ) vs 8th-order high-order EQ
    order_lo = 2
    order_hi = 8

    peq2 = design_hpeq_butterworth_rbj(order_lo, fs, f0, gain_db)
    peq8 = design_hpeq_butterworth_rbj(order_hi, fs, f0, gain_db)

    f, H2 = cascade_freq_response(peq2, fs)
    _, H8 = cascade_freq_response(peq8, fs)

    mag2 = 20 * np.log10(np.maximum(np.abs(H2), 1e-12))
    mag8 = 20 * np.log10(np.maximum(np.abs(H8), 1e-12))

    plt.figure(figsize=(8, 4))
    plt.semilogx(f, mag2, label=f"Order {order_lo} (single RBJ)")
    plt.semilogx(f, mag8, label=f"Order {order_hi} (Butterworth-style)")
    plt.axvline(f0, color="gray", ls="--", alpha=0.4)
    plt.grid(which="both", alpha=0.3)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"High-Order PEQ (Butterworth-style Qs), +{gain_db} dB @ {int(f0)} Hz")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Numeric sanity
    def H_at(biquads, freq):
        w0 = 2 * pi * freq / fs
        z0 = np.exp(1j * w0)
        H = 1.0 + 0j
        for bq in biquads:
            H *= (
                bq.b0 + bq.b1 / z0 + bq.b2 / (z0**2)
            ) / (1 + bq.a1 / z0 + bq.a2 / (z0**2))
        return 20 * np.log10(abs(H) + 1e-15)

    print("2nd-order:")
    print(f"  H({f0} Hz) ≈ {H_at(peq2, f0):.3f} dB (target {gain_db} dB)")
    print("8th-order:")
    print(f"  H({f0} Hz) ≈ {H_at(peq8, f0):.3f} dB (target {gain_db} dB)")
    print(f"  H(20 Hz)   ≈ {H_at(peq8, 20):.3f} dB (≈ 0 dB)")
    print(f"  H(20 kHz)  ≈ {H_at(peq8, 20000):.3f} dB (≈ 0 dB)")


if __name__ == "__main__":
    example()
```
