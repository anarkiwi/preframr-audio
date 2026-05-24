"""In-memory dfs_render_equivalent wiring: render both dfs and compare with no
WAV/scipy round trip. The render is monkeypatched (no pyresidfp needed); the
compare logic itself is the already-tested compare_renders."""

from __future__ import annotations

import numpy as np

from preframr_audio import fidelity

SR = 44100


def _tone(n_frames: int = 200) -> np.ndarray:
    n = int(round(SR / fidelity.PAL_FRAME_HZ)) * n_frames
    return (np.sin(np.arange(n) / 8.0) * 4000).astype(np.int16)


def test_identical_renders_pass(monkeypatch):
    tone = _tone()
    monkeypatch.setattr(fidelity, "_render_df_samples", lambda df, **kw: (tone, SR))
    result = fidelity.dfs_render_equivalent(object(), object())
    assert result.passed


def test_sample_rate_mismatch_fails(monkeypatch):
    streams = iter(
        [(np.zeros(4096, np.int16), 44100), (np.zeros(4096, np.int16), 22050)]
    )
    monkeypatch.setattr(fidelity, "_render_df_samples", lambda df, **kw: next(streams))
    result = fidelity.dfs_render_equivalent(object(), object())
    assert not result.passed
    assert result.shape == "SAMPLE_RATE_MISMATCH"


def test_audible_difference_fails(monkeypatch):
    tone = _tone()
    silence = np.zeros_like(tone)
    streams = iter([(tone, SR), (silence, SR)])
    monkeypatch.setattr(fidelity, "_render_df_samples", lambda df, **kw: next(streams))
    result = fidelity.dfs_render_equivalent(object(), object())
    assert not result.passed


def test_duration_mismatch_fails(monkeypatch):
    streams = iter([(_tone(200), SR), (_tone(150), SR)])
    monkeypatch.setattr(fidelity, "_render_df_samples", lambda df, **kw: next(streams))
    result = fidelity.dfs_render_equivalent(object(), object())
    assert not result.passed
    assert result.shape == "DURATION_MISMATCH"
