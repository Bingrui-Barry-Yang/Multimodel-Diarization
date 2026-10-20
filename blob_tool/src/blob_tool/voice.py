"""Voice anonymisation.

McAdams-coefficient formant shifting, as in the VoicePrivacy Challenge secondary
baseline (Patino et al., 2021): each frame is LPC-analysed, the angle of every
complex pole is raised to the power ``alpha`` and the residual is re-filtered
through the warped all-pole filter. With a *fixed* alpha for a whole recording,
all speakers are moved by the same transform, so they stay distinguishable from
each other (useful for diarization) while no longer sounding like themselves.

An optional pitch shift (resample, then overlap-add time stretch back to the
original length) is applied afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter, resample


@dataclass
class VoiceParams:
    mcadams_alpha: float = 0.8  # 1.0 = unchanged; VPC baseline samples 0.5–0.9
    pitch_semitones: float = 0.0
    lpc_order: int = 20
    frame_ms: float = 20.0


def _lpc(frame: np.ndarray, order: int) -> np.ndarray:
    r = np.correlate(frame, frame, mode="full")[len(frame) - 1 : len(frame) + order]
    if r[0] <= 1e-10:
        return np.r_[1.0, np.zeros(order)]
    r = r.copy()
    r[0] *= 1.0 + 1e-6  # white-noise correction keeps the system well conditioned
    a = solve_toeplitz(r[:-1], -r[1:])
    return np.r_[1.0, a]


def mcadams(signal: np.ndarray, sample_rate: int, alpha: float, order: int = 20, frame_ms: float = 20.0) -> np.ndarray:
    if abs(alpha - 1.0) < 1e-6:
        return signal.astype(np.float32)
    x = signal.astype(np.float64)
    win_len = int(sample_rate * frame_ms / 1000)
    win_len += win_len % 2
    hop = win_len // 2
    window = np.hanning(win_len + 1)[:-1]  # periodic Hann sums to 1 at 50 % overlap
    n_frames = max(1, 1 + int(np.ceil((len(x) - win_len) / hop)))
    padded = np.zeros((n_frames - 1) * hop + win_len)
    padded[: len(x)] = x
    out = np.zeros_like(padded)

    for i in range(n_frames):
        start = i * hop
        frame = padded[start : start + win_len] * window
        if not np.any(frame):
            continue
        a = _lpc(frame, order)
        residual = lfilter(a, [1.0], frame)

        poles = np.roots(a)
        ang = np.angle(poles)
        warp = np.abs(np.imag(poles)) > 1e-12  # real poles stay put
        ang[warp] = np.sign(ang[warp]) * np.abs(ang[warp]) ** alpha
        new_poles = np.abs(poles) * np.exp(1j * ang)
        a_new = np.real(np.poly(new_poles))

        out[start : start + win_len] += lfilter([1.0], a_new, residual)

    out = out[: len(x)]
    peak_in, peak_out = np.max(np.abs(x)), np.max(np.abs(out))
    if peak_out > 0:
        out *= peak_in / peak_out
    return out.astype(np.float32)


def pitch_shift(signal: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    """Pitch shift by resampling, then time-stretch back with overlap-add so length is preserved."""
    if abs(semitones) < 1e-3:
        return signal.astype(np.float32)
    factor = 2.0 ** (semitones / 12.0)
    # 1) resample so that playback at sample_rate is `factor` higher (and shorter)
    shifted = resample(signal, max(1, int(round(len(signal) / factor))))
    # 2) stretch back to original length with overlapping Hann grains
    grain = int(0.04 * sample_rate)
    hop_out = grain // 2
    window = np.hanning(grain + 1)[:-1]
    out = np.zeros(len(signal) + grain)
    norm = np.zeros_like(out)
    pos_out = 0
    while pos_out < len(signal):
        pos_in = int(pos_out * len(shifted) / len(signal))
        chunk = shifted[pos_in : pos_in + grain]
        if len(chunk) < grain:
            chunk = np.pad(chunk, (0, grain - len(chunk)))
        out[pos_out : pos_out + grain] += chunk * window
        norm[pos_out : pos_out + grain] += window
        pos_out += hop_out
    norm[norm < 1e-6] = 1.0
    return (out / norm)[: len(signal)].astype(np.float32)


def anonymize(signal: np.ndarray, sample_rate: int, params: VoiceParams) -> np.ndarray:
    y = mcadams(signal, sample_rate, params.mcadams_alpha, params.lpc_order, params.frame_ms)
    y = pitch_shift(y, sample_rate, params.pitch_semitones)
    return np.clip(y, -1.0, 1.0)
