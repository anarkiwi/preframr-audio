"""Unit tests for preframr_audio.features."""

from __future__ import annotations

import numpy as np
import pytest

from preframr_audio.features import (
    DEFAULT_N_BANDS,
    DEFAULT_N_MELS,
    FEATURE_FNS,
    band_power_features,
    mel_features,
    raw_pcm_features,
    spectral_features,
)

SAMPLE_RATE = 22050
RNG_SEED = 7


def _sine(freq_hz: float, n: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(n) / sr
    return (np.sin(2 * np.pi * freq_hz * t) * 8000).astype(np.int16)


def _noise(n: int, seed: int = RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(scale=2000, size=n)).clip(-32767, 32767).astype(np.int16)


class TestMelFeatures:
    def test_output_shape_is_2n_mels(self):
        samples = _sine(440.0, SAMPLE_RATE)
        feat = mel_features(samples, SAMPLE_RATE)
        assert feat.shape == (2 * DEFAULT_N_MELS,)

    def test_custom_n_mels(self):
        samples = _sine(440.0, SAMPLE_RATE)
        feat = mel_features(samples, SAMPLE_RATE, n_mels=32)
        assert feat.shape == (64,)

    def test_deterministic(self):
        samples = _sine(440.0, SAMPLE_RATE)
        a = mel_features(samples, SAMPLE_RATE)
        b = mel_features(samples, SAMPLE_RATE)
        assert np.allclose(a, b)

    def test_empty_input_returns_zeros(self):
        feat = mel_features(np.zeros(0, dtype=np.int16), SAMPLE_RATE)
        assert feat.shape == (2 * DEFAULT_N_MELS,)
        assert np.all(feat == 0.0)

    def test_distinguishes_sine_from_noise(self):
        sine = mel_features(_sine(440.0, SAMPLE_RATE), SAMPLE_RATE)
        noise = mel_features(_noise(SAMPLE_RATE), SAMPLE_RATE)
        dist = np.linalg.norm(sine - noise)
        assert dist > 1.0, f"sine and noise too similar in mel space: dist={dist}"

    def test_distinguishes_different_frequencies(self):
        low = mel_features(_sine(110.0, SAMPLE_RATE), SAMPLE_RATE)
        high = mel_features(_sine(4400.0, SAMPLE_RATE), SAMPLE_RATE)
        dist = np.linalg.norm(low - high)
        assert dist > 1.0, f"110Hz and 4400Hz sines too similar: dist={dist}"

    def test_silence_returns_finite(self):
        feat = mel_features(np.zeros(SAMPLE_RATE, dtype=np.int16), SAMPLE_RATE)
        assert np.isfinite(feat).all()

    def test_short_input_pads(self):
        short = _sine(440.0, 100)
        feat = mel_features(short, SAMPLE_RATE)
        assert feat.shape == (2 * DEFAULT_N_MELS,)
        assert np.isfinite(feat).all()

    def test_explicit_fmax_changes_binning(self):
        s = _sine(4400.0, SAMPLE_RATE)
        default_fmax = mel_features(s, SAMPLE_RATE)
        narrow = mel_features(s, SAMPLE_RATE, fmax=2000.0)
        assert narrow.shape == default_fmax.shape
        assert np.isfinite(narrow).all()
        assert not np.allclose(narrow, default_fmax)


class TestSpectralFeatures:
    def test_output_shape_is_5(self):
        feat = spectral_features(_sine(440.0, SAMPLE_RATE), SAMPLE_RATE)
        assert feat.shape == (5,)

    def test_centroid_tracks_frequency(self):
        low = spectral_features(_sine(220.0, SAMPLE_RATE), SAMPLE_RATE)
        high = spectral_features(_sine(2200.0, SAMPLE_RATE), SAMPLE_RATE)
        assert (
            low[0] < high[0]
        ), f"centroid should rise with sine freq; got low={low[0]} high={high[0]}"

    def test_zcr_higher_for_high_freq(self):
        low = spectral_features(_sine(110.0, SAMPLE_RATE), SAMPLE_RATE)
        high = spectral_features(_sine(4400.0, SAMPLE_RATE), SAMPLE_RATE)
        assert low[3] < high[3], f"zcr low={low[3]} high={high[3]}"

    def test_empty_input(self):
        feat = spectral_features(np.zeros(0, dtype=np.int16), SAMPLE_RATE)
        assert feat.shape == (5,)
        assert np.all(feat == 0.0)


class TestBandPowerFeatures:
    def test_output_shape_is_n_bands(self):
        feat = band_power_features(_sine(440.0, SAMPLE_RATE), SAMPLE_RATE)
        assert feat.shape == (DEFAULT_N_BANDS,)

    def test_custom_n_bands(self):
        feat = band_power_features(_sine(440.0, SAMPLE_RATE), SAMPLE_RATE, n_bands=16)
        assert feat.shape == (16,)

    def test_empty_input_returns_zeros(self):
        feat = band_power_features(np.zeros(0, dtype=np.int16), SAMPLE_RATE)
        assert feat.shape == (DEFAULT_N_BANDS,)
        assert np.all(feat == 0.0)

    def test_silence_is_finite_and_uniform(self):
        feat = band_power_features(np.zeros(SAMPLE_RATE, dtype=np.int16), SAMPLE_RATE)
        assert np.isfinite(feat).all()
        # all-zero power -> uniform log-floor across bands
        assert np.allclose(feat, feat[0])

    def test_deterministic(self):
        s = _sine(440.0, SAMPLE_RATE)
        assert np.allclose(
            band_power_features(s, SAMPLE_RATE), band_power_features(s, SAMPLE_RATE)
        )

    def test_distinguishes_different_frequencies(self):
        low = band_power_features(_sine(220.0, SAMPLE_RATE), SAMPLE_RATE)
        high = band_power_features(_sine(3000.0, SAMPLE_RATE), SAMPLE_RATE)
        dist = np.linalg.norm(low - high)
        assert dist > 1.0, f"220Hz and 3000Hz too similar in band space: {dist}"

    def test_tolerant_to_small_pitch_shift(self):
        # a sub-band detune leaves the band sums (and so the distance) tiny,
        # while the large-shift distance above is big -- the whole point.
        ref = band_power_features(_sine(440.0, SAMPLE_RATE), SAMPLE_RATE)
        nudged = band_power_features(_sine(446.0, SAMPLE_RATE), SAMPLE_RATE)
        big = band_power_features(_sine(3000.0, SAMPLE_RATE), SAMPLE_RATE)
        assert np.linalg.norm(ref - nudged) < 0.25 * np.linalg.norm(ref - big)

    def test_custom_fmax_changes_band_edges(self):
        s = _sine(3000.0, SAMPLE_RATE)
        default_bands = band_power_features(s, SAMPLE_RATE)
        narrow = band_power_features(s, SAMPLE_RATE, fmax=2000.0)
        assert narrow.shape == default_bands.shape
        assert np.isfinite(narrow).all()
        assert not np.allclose(narrow, default_bands)


class TestRawPcmFeatures:
    def test_passes_through_normalised(self):
        samples = _sine(440.0, 1000)
        feat = raw_pcm_features(samples, SAMPLE_RATE)
        assert feat.shape == (1000,)
        assert np.abs(feat).max() == pytest.approx(1.0, abs=1e-9)

    def test_empty(self):
        feat = raw_pcm_features(np.zeros(0, dtype=np.int16), SAMPLE_RATE)
        assert feat.shape == (0,)


class TestFeatureFnsRegistry:
    def test_all_registered(self):
        assert "mel" in FEATURE_FNS
        assert "spectral" in FEATURE_FNS
        assert "band_power" in FEATURE_FNS
        assert "raw_pcm" in FEATURE_FNS

    def test_callables_match_signature(self):
        samples = _sine(440.0, 1000)
        for name, fn in FEATURE_FNS.items():
            feat = fn(samples, SAMPLE_RATE)
            assert isinstance(
                feat, np.ndarray
            ), f"{name} returned {type(feat)}, not ndarray"
