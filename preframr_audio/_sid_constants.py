"""SID register constants and helpers vendored from the preframr codebase."""

from __future__ import annotations

FRAME_REG = -128
DELAY_REG = -127

FC_LO_REG = 21
MAX_REG = 24
MODE_VOL_REG = 24

VOICES = 3
VOICE_REG_SIZE = 7
VOICE_CTRL_REG = {v: v * VOICE_REG_SIZE + 4 for v in range(VOICES)}


def voice_of_reg(reg):
    """Reg index -> voice (0..VOICES-1) or None for non-voice regs."""
    if reg < 0 or reg >= VOICES * VOICE_REG_SIZE:
        return None
    return reg // VOICE_REG_SIZE


PAL_CLOCK = 17734475
TUNING_REF_HZ = 440
MIDI_N_TO_F = {n: (2 ** ((n - 69) / 12)) * TUNING_REF_HZ for n in range(128)}
