"""SID utility helpers shared by the audio driver and tests."""

from pyresidfp import SoundInterfaceDevice
from pyresidfp.sound_interface_device import ChipModel

from preframr_audio._sid_constants import FC_LO_REG


def default_sid():
    return SoundInterfaceDevice(model=ChipModel.MOS8580)


def sidq(sid=None):
    """Seconds per SID clock cycle for this chip's real clock (PAL 985248 Hz,
    NTSC 1022730 Hz). ``reset_diffs`` multiplies cycle deltas by this and the
    renderer multiplies back by ``clock_frequency``, so it must be ``1/clock``;
    the old ``clock/1e6/1e6`` only held at exactly 1 MHz and ran PAL ~3% fast."""
    if sid is None:
        sid = default_sid()
    return 1.0 / sid.clock_frequency


def write_reg(sid, reg, val, reg_widths):
    """Apply one preframr-logical register write to ``sid`` (or any
    object exposing ``write_register(reg, val)`` and ``freq_mapper``).
    """
    width = reg_widths.get(reg, 1)
    lobits = 8
    if reg in (0, 7, 14):
        width = 2
        try:
            val = sid.freq_mapper.if_map[val]
        except KeyError:
            if val < 0:
                val = 0
            else:
                val = max(sid.freq_mapper.if_map.keys())
            val = sid.freq_mapper.if_map[val]
    elif reg in (2, 9, 16):
        width = 2
    elif reg == FC_LO_REG:
        width = 2
    for i in range(width):
        regval = val & (2**lobits - 1)
        sid.write_register(reg + i, regval)
        val >>= lobits
