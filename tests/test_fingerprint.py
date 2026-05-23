"""Unit tests for preframr_audio.fingerprint."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preframr_audio._sid_constants import FRAME_REG
from preframr_audio.features import DEFAULT_N_MELS
from preframr_audio.fingerprint import (
    DEFAULT_IRQ_PAL,
    _normalise_writes,
    _resolve_feature_fn,
    canonical_scaffold,
    fingerprint_batch,
    fingerprint_writes,
)

pyresidfp = pytest.importorskip("pyresidfp")

NOISE_FLOOR_NORM = 1.5


class TestCanonicalScaffold:
    def test_returns_dataframe_with_expected_columns(self):
        df = canonical_scaffold([(0, 4, 0x21)])
        assert list(df.columns) == ["reg", "val", "delay"]

    def test_irq_attr_set(self):
        df = canonical_scaffold([(0, 4, 0x21)], irq=12345)
        assert df.attrs["irq"] == 12345

    def test_test_writes_appear_in_order(self):
        df = canonical_scaffold([(0, 22, 0xAA), (0, 23, 0xBB)])
        test_rows = df[(df["reg"] == 22) | (df["reg"] == 23)]
        assert list(test_rows["val"]) == [0xAA, 0xBB]

    def test_frame_markers_have_irq_delay(self):
        df = canonical_scaffold([(0, 4, 0x21)], irq=DEFAULT_IRQ_PAL)
        frame_rows = df[df["reg"] == FRAME_REG]
        assert (frame_rows["delay"] > 0).all(), "frame markers must clock the SID"

    def test_pre_post_frame_counts(self):
        df = canonical_scaffold([(0, 4, 0x21)], pre_frames=3, post_frames=4)
        n_frames = (df["reg"] == FRAME_REG).sum()
        assert n_frames == 1 + 3 + 1 + 4

    def test_custom_scaffold_writes(self):
        scaffold = [(0, 24, 10)]
        df = canonical_scaffold([(0, 4, 0x21)], scaffold_writes=scaffold)
        assert (df["reg"] == 24).sum() == 1
        assert df[df["reg"] == 24]["val"].iloc[0] == 10


class TestNormaliseWrites:
    def test_accepts_list_of_tuples(self):
        out = _normalise_writes([(100, 4, 0x21), (200, 5, 0x99)])
        assert out == [(100, 4, 0x21), (200, 5, 0x99)]

    def test_accepts_dataframe(self):
        df = pd.DataFrame({"clock": [100, 200], "reg": [4, 5], "val": [0x21, 0x99]})
        out = _normalise_writes(df)
        assert out == [(100, 4, 0x21), (200, 5, 0x99)]

    def test_dataframe_no_clock_column_defaults_zero(self):
        df = pd.DataFrame({"reg": [4], "val": [0x21]})
        out = _normalise_writes(df)
        assert out == [(0, 4, 0x21)]


class TestResolveFeatureFn:
    def test_string_resolves(self):
        fn = _resolve_feature_fn("mel")
        assert callable(fn)

    def test_callable_passes_through(self):
        custom = lambda s, sr: np.zeros(3)
        fn = _resolve_feature_fn(custom)
        assert fn is custom

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _resolve_feature_fn("not_a_real_feature")


class TestFingerprintWrites:
    def test_returns_fixed_length_mel_vector(self):
        feat = fingerprint_writes([(0, 4, 0x21)], feature="mel")
        assert feat.shape == (2 * DEFAULT_N_MELS,)
        assert feat.dtype == np.float64
        assert np.isfinite(feat).all()

    def test_spectral_feature(self):
        feat = fingerprint_writes([(0, 4, 0x21)], feature="spectral")
        assert feat.shape == (5,)
        assert np.isfinite(feat).all()

    def test_custom_callable_feature(self):
        def my_feat(samples, sr):
            return np.array([float(samples.size), float(sr)], dtype=np.float64)

        feat = fingerprint_writes([(0, 4, 0x21)], feature=my_feat)
        assert feat.shape == (2,)
        assert feat[0] > 0

    def test_distinguishes_different_waveforms(self):
        triangle = fingerprint_writes([(0, 4, 0x11)], feature="mel")
        sawtooth = fingerprint_writes([(0, 4, 0x21)], feature="mel")
        dist = np.linalg.norm(triangle - sawtooth)
        assert (
            dist > NOISE_FLOOR_NORM
        ), f"triangle vs sawtooth distance {dist} <= noise floor"

    def test_distinguishes_different_frequencies(self):
        low = fingerprint_writes([(0, 0, 0x10), (0, 1, 0x04)], feature="mel")
        high = fingerprint_writes([(0, 0, 0x10), (0, 1, 0x40)], feature="mel")
        dist = np.linalg.norm(low - high)
        assert (
            dist > NOISE_FLOOR_NORM
        ), f"low vs high freq distance {dist} <= noise floor"

    def test_reproducibility_within_noise_floor(self):
        a = fingerprint_writes([(0, 0, 0x10), (0, 1, 0x10)], feature="mel")
        b = fingerprint_writes([(0, 0, 0x10), (0, 1, 0x10)], feature="mel")
        dist = np.linalg.norm(a - b)
        assert (
            dist < NOISE_FLOOR_NORM
        ), f"reproduction noise {dist} >= signal floor {NOISE_FLOOR_NORM}"


class TestFingerprintBatch:
    def test_empty_returns_zero_shape(self):
        out = fingerprint_batch([], n_workers=1, feature="mel")
        assert out.shape == (0, 0)

    def test_sequential_n_workers_1(self):
        seqs = [[(0, 4, 0x11)], [(0, 4, 0x21)], [(0, 4, 0x41)]]
        out = fingerprint_batch(seqs, n_workers=1, feature="mel")
        assert out.shape == (3, 2 * DEFAULT_N_MELS)
        assert np.isfinite(out).all()

    def test_parallel_matches_sequential_within_noise(self):
        seqs = [[(0, 4, 0x11)], [(0, 4, 0x21)], [(0, 4, 0x41)]]
        seq = fingerprint_batch(seqs, n_workers=1, feature="mel")
        par = fingerprint_batch(seqs, n_workers=2, feature="mel")
        for i, (a, b) in enumerate(zip(seq, par)):
            dist = np.linalg.norm(a - b)
            assert (
                dist < NOISE_FLOOR_NORM
            ), f"seq vs parallel mismatch at i={i}: dist={dist}"

    def test_batch_matches_per_item_within_noise(self):
        seqs = [[(0, 4, 0x11)], [(0, 4, 0x21)]]
        batch = fingerprint_batch(seqs, n_workers=1, feature="mel")
        per_item = np.stack([fingerprint_writes(s, feature="mel") for s in seqs])
        for i, (a, b) in enumerate(zip(batch, per_item)):
            dist = np.linalg.norm(a - b)
            assert (
                dist < NOISE_FLOOR_NORM
            ), f"batch vs per-item mismatch at i={i}: dist={dist}"

    def test_batch_with_spectral_feature(self):
        seqs = [[(0, 4, 0x11)], [(0, 4, 0x21)]]
        out = fingerprint_batch(seqs, n_workers=1, feature="spectral")
        assert out.shape == (2, 5)
