"""In-memory dfs_render_equivalent wiring: render both dfs and compare with no
WAV/scipy round trip. The render is monkeypatched (no pyresidfp needed); the
compare logic itself is the already-tested compare_renders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_irq_from_df_reads_frame_diff():
    """_irq_from_df reads the frame period from the first positive FRAME-row diff
    via the LOCAL _read_initial_irq (no preframr_tokens import -- audio<->tokens
    would otherwise be a cycle; the trivial helper is replicated)."""
    df = pd.DataFrame({"reg": [fidelity._FRAME_REG], "diff": [19656]})
    assert fidelity._irq_from_df(df) == 19656


def test_irq_from_df_raises_without_frame_rows():
    df = pd.DataFrame({"reg": [0, 1], "diff": [0, 0]})
    with pytest.raises(ValueError):
        fidelity._irq_from_df(df)
