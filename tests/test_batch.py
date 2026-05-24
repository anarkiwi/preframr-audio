"""Batch render / equivalence orchestration: parallelism, ordering, result
mapping. The render itself is monkeypatched, so these run without pyresidfp;
the real render path is exercised by test_fingerprint / test_audio_driver."""

from __future__ import annotations

from preframr_audio import batch
from preframr_audio.fidelity import AudioFidelityResult


def test_render_batch_empty():
    assert batch.render_batch([]) == []


def test_verify_equivalent_batch_empty():
    assert batch.verify_equivalent_batch([]) == []


def test_render_batch_preserves_order(monkeypatch):
    monkeypatch.setattr(batch, "_render_df_samples", lambda df, **kw: (df, 44100))
    out = batch.render_batch(["a", "b", "c"], n_workers=1)
    assert [samples for samples, _ in out] == ["a", "b", "c"]
    assert all(sr == 44100 for _, sr in out)


def test_render_batch_passes_kwargs(monkeypatch):
    seen = {}

    def fake(df, *, cents, chip_model):
        seen["cents"] = cents
        seen["chip_model"] = chip_model
        return (df, 44100)

    monkeypatch.setattr(batch, "_render_df_samples", fake)
    batch.render_batch(["x"], n_workers=1, cents=37, chip_model="MOS6581")
    assert seen == {"cents": 37, "chip_model": "MOS6581"}


def test_verify_equivalent_batch_maps_and_orders(monkeypatch):
    def fake(df_a, df_b, **_kw):
        return AudioFidelityResult(passed=(df_a == df_b), shape="X", diagnostic="")

    monkeypatch.setattr(batch, "dfs_render_equivalent", fake)
    pairs = [("x", "x"), ("x", "y"), ("z", "z")]
    results = batch.verify_equivalent_batch(pairs, n_workers=1)
    assert [r.passed for r in results] == [True, False, True]


def test_verify_equivalent_batch_forwards_tolerance(monkeypatch):
    seen = {}

    def fake(_df_a, _df_b, *, cents, chip_model, tolerance, max_frame_drift):
        seen.update(
            cents=cents,
            chip_model=chip_model,
            tolerance=tolerance,
            max_frame_drift=max_frame_drift,
        )
        return AudioFidelityResult(passed=True, shape="PASS", diagnostic="")

    monkeypatch.setattr(batch, "dfs_render_equivalent", fake)
    batch.verify_equivalent_batch(
        [("a", "b")], n_workers=1, cents=25, tolerance=0.02, max_frame_drift=2
    )
    assert seen == {
        "cents": 25,
        "chip_model": "MOS8580",
        "tolerance": 0.02,
        "max_frame_drift": 2,
    }
