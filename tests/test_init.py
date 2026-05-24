"""Public-API facade: importing the package is cheap + resid-free; symbols
resolve lazily from their submodules."""

from __future__ import annotations

import pytest


def test_import_is_safe_without_resid():
    """``import preframr_audio`` must not pull in pyresidfp/scipy (the
    render-free path depends on this)."""
    import preframr_audio

    assert preframr_audio.__version__


def test_non_resid_symbols_resolve():
    import preframr_audio
    from preframr_audio import _EXPORTS

    for name, module in _EXPORTS.items():
        if module == "sidwav":  # hard-imports pyresidfp; covered below
            continue
        assert getattr(preframr_audio, name) is not None


def test_sidwav_symbols_resolve():
    pytest.importorskip("pyresidfp")
    import preframr_audio

    assert callable(preframr_audio.sidq)
    assert callable(preframr_audio.write_reg)


def test_unknown_attribute_raises():
    import preframr_audio

    with pytest.raises(AttributeError):
        _ = preframr_audio.does_not_exist


def test_dir_lists_exports():
    import preframr_audio

    listed = dir(preframr_audio)
    assert "compare_renders" in listed
    assert "render_batch" in listed
