import numpy as np

import pytest

from blob_tool.voice import ALPHA_RANGE, VoiceParams, anonymize, draw, mcadams, pitch_shift

SR = 16000


def vowel(f0=140.0, formants=(700, 1200, 2600), seconds=1.0):
    """Crude synthetic vowel: impulse train through formant resonators."""
    from scipy.signal import lfilter

    n = int(SR * seconds)
    x = np.zeros(n)
    x[:: int(SR / f0)] = 1.0
    for f in formants:
        r, th = 0.97, 2 * np.pi * f / SR
        x = lfilter([1.0], [1.0, -2 * r * np.cos(th), r * r], x)
    return (x / np.max(np.abs(x)) * 0.5).astype(np.float32)


def spectral_centroid(x):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    return float((spec * freqs).sum() / spec.sum())


def test_alpha_one_is_identity():
    x = vowel()
    np.testing.assert_allclose(mcadams(x, SR, 1.0), x, atol=1e-6)


def test_mcadams_keeps_length_and_changes_spectrum():
    x = vowel()
    y = mcadams(x, SR, 0.8)
    assert y.shape == x.shape
    assert np.isfinite(y).all()
    # alpha < 1 raises pole angles below pi → formants move up
    assert spectral_centroid(y) > spectral_centroid(x) * 1.05


def test_same_alpha_keeps_two_speakers_apart():
    a, b = vowel(f0=110, formants=(650, 1100, 2500)), vowel(f0=220, formants=(850, 1500, 2900))
    p = VoiceParams(randomize=False, mcadams_alpha=0.8)
    ca, cb = spectral_centroid(anonymize(a, SR, p)), spectral_centroid(anonymize(b, SR, p))
    assert abs(ca - cb) > 0.5 * abs(spectral_centroid(a) - spectral_centroid(b))


def test_pitch_shift_length_and_direction():
    x = vowel(f0=150)
    y = pitch_shift(x, SR, 4.0)
    assert y.shape == x.shape

    assert f0(y) > f0(x) * 1.15


def f0(sig):
    ac = np.correlate(sig, sig, "full")[len(sig) - 1 :]
    lo, hi = SR // 400, SR // 60
    return SR / (lo + np.argmax(ac[lo:hi]))


def test_seed_fixes_alpha_and_keeps_pitch():
    x = vowel(f0=150)
    p = VoiceParams()
    np.testing.assert_array_equal(anonymize(x, SR, p, seed=7), anonymize(x, SR, p, seed=7))
    assert not np.allclose(anonymize(x, SR, p, seed=7), anonymize(x, SR, p, seed=8))
    alphas = [draw(s) for s in range(50)]
    assert len(set(alphas)) == 50 and all(ALPHA_RANGE[0] <= a <= ALPHA_RANGE[1] for a in alphas)
    assert abs(f0(anonymize(x, SR, p, seed=7)) / f0(x) - 1) < 0.05  # no pitch shift
    # seeded output = McAdams with the drawn alpha + a faint noise floor
    clean = mcadams(x, SR, draw(7))
    diff = anonymize(x, SR, p, seed=7) - clean
    assert 0 < np.sqrt(np.mean(diff**2)) < 0.01 * np.abs(clean).max()  # about -45 dB


def test_randomised_voice_needs_a_seed_and_fixed_values_ignore_it():
    x = vowel()
    with pytest.raises(ValueError):
        anonymize(x, SR, VoiceParams())
    fixed = VoiceParams(randomize=False, mcadams_alpha=0.8, pitch_semitones=2.0)
    np.testing.assert_array_equal(anonymize(x, SR, fixed, seed=1), anonymize(x, SR, fixed, seed=2))
    expected = np.clip(pitch_shift(mcadams(x, SR, 0.8), SR, 2.0), -1, 1)
    np.testing.assert_array_equal(anonymize(x, SR, fixed), expected)
