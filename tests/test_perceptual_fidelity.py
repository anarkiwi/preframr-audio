"""Perceptual render-equivalence gate: calibration, metric properties, and the
public API surface (mirrors the strict ``test_fidelity`` / ``test_dfs_equivalent``
style).

The calibration set is built synthetically in-repo from
``fingerprint.canonical_scaffold`` plus deterministic df edits (see
``fidelity._calibration_pairs``): INERT pairs (reorder / duplicate / <=50-cent
nudge) must read perceptually equivalent, BAD pairs (octave / waveform / cutoff /
envelope / dropped-voice) must read audibly different, with a comfortable
separation margin around ``PERCEPTUAL_DISTANCE_THRESHOLD``.

The calibration dfs use the ``canonical_scaffold`` (reg/val/delay) schema, which
``render_to_samples`` consumes directly; so ``fidelity._render_df_samples`` (the
raw-df ``prepare_df_for_audio`` path the gate shares with ``dfs_render_equivalent``)
is monkeypatched to that in-memory scaffold renderer -- exactly as
``test_dfs_equivalent`` patches the render to exercise the wiring.
"""

from __future__ import annotations

import numpy as np
import pytest

from preframr_audio import batch, fidelity

pytest.importorskip("pyresidfp")

SR = 48000


def _render_scaffold(df, *, cents: int = 50, chip_model: str = "MOS8580"):
    from preframr_audio.audio_driver import render_to_samples

    return render_to_samples(
        df,
        reg_widths={},
        irq=int(df.attrs["irq"]),
        cents=cents,
        chip_model=chip_model,
    )


@pytest.fixture(scope="module")
def calibration():
    """Drive every calibration pair through the real ``dfs_perceptually_equivalent``
    with the scaffold renderer patched in. Each unique df is rendered once.
    Returns ``[(label, kind, PerceptualFidelityResult)]``."""
    pairs = fidelity._calibration_pairs()
    cache: dict = {}

    def cached_render(df, *, cents: int = 50, chip_model: str = "MOS8580"):
        key = id(df)
        if key not in cache:
            cache[key] = _render_scaffold(df, cents=cents, chip_model=chip_model)
        return cache[key]

    original = fidelity._render_df_samples
    fidelity._render_df_samples = cached_render
    try:
        out = []
        for label, kind, df_a, df_b in pairs:
            result = fidelity.dfs_perceptually_equivalent(df_a, df_b)
            out.append((label, kind, result))
    finally:
        fidelity._render_df_samples = original
    return out


def test_inert_pairs_pass_bad_pairs_fail(calibration):
    for label, kind, result in calibration:
        if kind == "INERT":
            assert result.passed is True, (
                f"INERT pair {label!r} should be perceptually equivalent but "
                f"failed: distance {result.distance:.4f} > threshold "
                f"{result.threshold:.4f}"
            )
        else:
            assert result.passed is False, (
                f"BAD pair {label!r} should be audibly different but passed: "
                f"distance {result.distance:.4f} <= threshold "
                f"{result.threshold:.4f}"
            )


def test_separation_margin(calibration):
    inert = [r.distance for _, kind, r in calibration if kind == "INERT"]
    bad = [r.distance for _, kind, r in calibration if kind == "BAD"]
    inert_ceiling = max(inert)
    bad_floor = min(bad)
    threshold = fidelity.PERCEPTUAL_DISTANCE_THRESHOLD
    # threshold sits strictly in the gap ...
    assert inert_ceiling < threshold < bad_floor, (
        f"threshold {threshold} not in the gap "
        f"[{inert_ceiling:.4f}, {bad_floor:.4f}]"
    )
    # ... with a comfortable (>= 2x) margin between the classes.
    assert bad_floor > 2.0 * inert_ceiling, (
        f"INERT/BAD classes too close: ceiling {inert_ceiling:.4f}, floor "
        f"{bad_floor:.4f} (margin {bad_floor / inert_ceiling:.2f}x < 2x)"
    )


def test_each_required_bad_category_present(calibration):
    # every BAD category named in the spec is represented and fails
    labels = {label for label, kind, _ in calibration if kind == "BAD"}
    for needle in ("octave", "waveform", "filter_cutoff", "envelope", "drop"):
        matches = [label for label in labels if needle in label]
        assert matches, f"no BAD calibration pair for category {needle!r}"


def test_calibrate_helper_reproduces_threshold():
    report = fidelity.calibrate(verbose=False)
    assert report["margin"] >= 2.0
    assert report["inert_ceiling"] < report["bad_floor"]
    # the hardcoded constant sits in the reproduced gap
    assert (
        report["inert_ceiling"]
        < fidelity.PERCEPTUAL_DISTANCE_THRESHOLD
        < report["bad_floor"]
    )


# --- API surface ----------------------------------------------------------- #


def test_result_fields(calibration):
    _, _, result = calibration[0]
    assert isinstance(result, fidelity.PerceptualFidelityResult)
    for field in (
        "passed",
        "distance",
        "threshold",
        "worst_window_s",
        "n_windows",
        "shape",
        "diagnostic",
    ):
        assert hasattr(result, field)
    assert isinstance(result.passed, bool)
    assert result.n_windows >= 1
    assert result.threshold == fidelity.PERCEPTUAL_DISTANCE_THRESHOLD


def test_assert_helper_raises_on_bad_pair_and_returns_on_inert():
    pairs = fidelity._calibration_pairs()
    inert = next(p for p in pairs if p[1] == "INERT")
    bad = next(p for p in pairs if p[1] == "BAD")
    cache: dict = {}

    def cached_render(df, *, cents: int = 50, chip_model: str = "MOS8580"):
        key = id(df)
        if key not in cache:
            cache[key] = _render_scaffold(df, cents=cents, chip_model=chip_model)
        return cache[key]

    original = fidelity._render_df_samples
    fidelity._render_df_samples = cached_render
    try:
        ok = fidelity.assert_dfs_perceptually_equivalent(inert[2], inert[3])
        assert ok.passed is True
        with pytest.raises(AssertionError):
            fidelity.assert_dfs_perceptually_equivalent(bad[2], bad[3])
    finally:
        fidelity._render_df_samples = original


def test_verify_batch_preserves_order_and_count():
    pairs = fidelity._calibration_pairs()
    inert = next(p for p in pairs if p[1] == "INERT")
    bad = next(p for p in pairs if p[1] == "BAD")
    # interleave so an order swap would be caught
    df_pairs = [
        (inert[2], inert[3]),
        (bad[2], bad[3]),
        (inert[2], inert[3]),
    ]
    cache: dict = {}

    def cached_render(df, *, cents: int = 50, chip_model: str = "MOS8580"):
        key = id(df)
        if key not in cache:
            cache[key] = _render_scaffold(df, cents=cents, chip_model=chip_model)
        return cache[key]

    original = fidelity._render_df_samples
    fidelity._render_df_samples = cached_render
    try:
        results = batch.verify_perceptually_equivalent_batch(df_pairs, n_workers=1)
    finally:
        fidelity._render_df_samples = original
    assert len(results) == len(df_pairs)
    assert [r.passed for r in results] == [True, False, True]
    assert all(isinstance(r, fidelity.PerceptualFidelityResult) for r in results)


def test_verify_batch_empty():
    assert batch.verify_perceptually_equivalent_batch([], n_workers=1) == []


# --- perceptual_distance windowed (worst-window) properties ---------------- #


def _tone(freq, n, sr=SR):
    t = np.arange(n) / sr
    return (np.sin(2 * np.pi * freq * t) * 8000).astype(np.int16)


def test_identical_streams_zero_distance():
    a = _tone(440, 3 * SR)
    worst, per = fidelity.perceptual_distance(a, a.copy(), SR)
    assert worst < 1e-9
    assert per.shape == (3,)
    assert np.allclose(per, 0.0, atol=1e-9)


def test_localized_difference_is_not_diluted():
    # three 1s windows; only the middle window differs (a wrong "note").
    a = np.concatenate([_tone(440, SR), _tone(440, SR), _tone(440, SR)])
    b = np.concatenate([_tone(440, SR), _tone(3000, SR), _tone(440, SR)])
    worst, per = fidelity.perceptual_distance(a, b, SR)
    assert per.shape == (3,)
    assert int(np.argmax(per)) == 1, "worst window should be the localized change"
    assert per[1] > 10 * max(per[0], per[2]), "localized change was diluted"
    assert worst == per[1]


def test_distance_truncates_to_shorter():
    a = _tone(440, 3 * SR)
    b = _tone(440, 2 * SR)
    worst, per = fidelity.perceptual_distance(a, b, SR)
    assert per.shape == (2,)
    assert worst < 1e-6


# --- result-shape edge cases (mirror compare_renders) ---------------------- #


def test_too_short_input_is_no_info_pass():
    a = _tone(440, SR // 4)  # shorter than one 1s window
    result = fidelity._perceptual_result(
        a, a.copy(), SR, threshold=fidelity.PERCEPTUAL_DISTANCE_THRESHOLD, win_s=1.0
    )
    assert result.passed is True
    assert result.shape == "NO_INFO"
    assert result.n_windows == 0


def test_gross_duration_mismatch_fails():
    a = _tone(440, 3 * SR)
    b = _tone(440, 1 * SR)  # differs by >= one window
    result = fidelity._perceptual_result(
        a, b, SR, threshold=fidelity.PERCEPTUAL_DISTANCE_THRESHOLD, win_s=1.0
    )
    assert result.passed is False
    assert result.shape == "DURATION_MISMATCH"


def test_sample_rate_mismatch_fails(monkeypatch):
    streams = iter([(_tone(440, 2 * SR), SR), (_tone(440, 2 * SR), 22050)])
    monkeypatch.setattr(fidelity, "_render_df_samples", lambda df, **kw: next(streams))
    result = fidelity.dfs_perceptually_equivalent(object(), object())
    assert result.passed is False
    assert result.shape == "SAMPLE_RATE_MISMATCH"
