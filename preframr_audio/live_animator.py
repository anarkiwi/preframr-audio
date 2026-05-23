"""Live VT100 runtime screen for SID playback."""

import select
import sys
import time
from collections import deque
from typing import Optional, Set, TextIO

from preframr_audio._sid_constants import (
    VOICE_CTRL_REG,
    VOICE_REG_SIZE,
    VOICES,
    voice_of_reg,
)

__all__ = ("LiveAnimator", "VoiceState", "VOICE_CTRL_REG", "voice_of_reg")

_FREQ_LO_OFFSET = 0
_FREQ_HI_OFFSET = 1
_CTRL_OFFSET = 4

_SPARK_GLYPHS = " .,-_'\"`*+^#"

_WAVEFORM_NAME = {
    0x0: "---",
    0x1: "tri",
    0x2: "saw",
    0x4: "pul",
    0x8: "nse",
}


class _Vt100:
    ESC = "\x1b"
    CLEAR = f"{ESC}[2J{ESC}[H"
    HIDE = f"{ESC}[?25l"
    SHOW = f"{ESC}[?25h"
    REVERSE = f"{ESC}[7m"
    DIM = f"{ESC}[2m"
    RESET = f"{ESC}[0m"

    @staticmethod
    def goto(row: int, col: int) -> str:
        return f"{_Vt100.ESC}[{row};{col}H"


class VoiceState:
    """Per-voice state tracked across streamed frames."""

    def __init__(self, history_len: int = 30):
        self.ctrl = 0
        self.freq_lo = 0
        self.freq_hi = 0
        self.history = deque([0] * history_len, maxlen=history_len)
        self.prev_gate = False
        self.prev_waveform_nibble = 0
        self.prev_freq = 0

    @property
    def gate(self) -> bool:
        return bool(self.ctrl & 0x01)

    @property
    def waveform_nibble(self) -> int:
        return (self.ctrl >> 4) & 0x0F

    def waveform_name(self) -> str:
        return _WAVEFORM_NAME.get(self.waveform_nibble, "mix")

    def freq(self) -> int:
        return (self.freq_hi << 8) | self.freq_lo

    def absorb_op(self, op_reg: int, op_val: int, voice_index: int) -> bool:
        """Apply one ``FrameOp`` to this voice if the reg is in this
        voice's block; return True if it was relevant."""
        base = voice_index * VOICE_REG_SIZE
        if op_reg == base + _FREQ_LO_OFFSET:
            self.freq_lo = op_val & 0xFF
        elif op_reg == base + _FREQ_HI_OFFSET:
            self.freq_hi = op_val & 0xFF
        elif op_reg == base + _CTRL_OFFSET:
            self.ctrl = op_val & 0xFF
        else:
            return False
        return True

    def snapshot_prev(self) -> None:
        self.prev_gate = self.gate
        self.prev_waveform_nibble = self.waveform_nibble
        self.prev_freq = self.freq()


class _MacroBurst:
    """Per-voice transient label that flashes for a few frames on
    interesting events derived from voice-state deltas:
    """

    LABEL_WIDTH = 6
    FLASH_FRAMES = 3
    LARGE_FREQ_JUMP = 0x800

    def __init__(self):
        self.label = ""
        self.flash_remaining = 0

    def fire(self, label: str) -> None:
        self.label = label[: self.LABEL_WIDTH]
        self.flash_remaining = self.FLASH_FRAMES

    def update(self, vs: VoiceState) -> None:
        if vs.gate and not vs.prev_gate:
            self.fire("GATE+")
        elif not vs.gate and vs.prev_gate:
            self.fire("GATE-")
        elif vs.waveform_nibble != vs.prev_waveform_nibble:
            self.fire(vs.waveform_name().upper())
        elif vs.gate and abs(vs.freq() - vs.prev_freq) >= self.LARGE_FREQ_JUMP:
            self.fire("JUMP")
        elif self.flash_remaining > 0:
            self.flash_remaining -= 1

    def render(self) -> str:
        if self.flash_remaining > 0:
            return f"{_Vt100.REVERSE}{self.label.ljust(self.LABEL_WIDTH)}{_Vt100.RESET}"
        return " " * self.LABEL_WIDTH


class _BpsMeter:
    """Rolling 1-second window of bytes written. ``cap`` is the 8N1
    bytes/sec ceiling derived from the configured baud rate.
    """

    def __init__(self, baud: int = 115200):
        self.baud = baud
        self.cap = baud // 10
        self.window: deque = deque()

    def add(self, n: int) -> None:
        now = time.time()
        self.window.append((now, n))
        cutoff = now - 1.0
        while self.window and self.window[0][0] < cutoff:
            self.window.popleft()

    def now(self) -> int:
        return sum(n for _, n in self.window)


class _KeyReader:
    """Single-keystroke non-blocking reader. cbreak'd on open; restored
    on close. No-op if stdin isn't a TTY (e.g. unit tests, redirected
    input).
    """

    def __init__(self):
        self._fd: Optional[int] = None
        self._old_attr = None

    def open(self) -> None:
        if not sys.stdin.isatty():
            return
        import termios  # noqa: WPS433
        import tty  # noqa: WPS433

        self._fd = sys.stdin.fileno()
        self._old_attr = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def close(self) -> None:
        if self._fd is None or self._old_attr is None:
            return
        import termios  # noqa: WPS433

        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attr)
        self._fd = None
        self._old_attr = None

    def poll(self) -> Optional[str]:
        if self._fd is None:
            return None
        ready, _, _ = select.select([self._fd], [], [], 0)
        if not ready:
            return None
        return sys.stdin.read(1)


class LiveAnimator:
    """Per-frame VT100 runtime screen."""

    HEADER_ROW = 1
    VOICE_START_ROW = 3
    STATUS_ROW = 22
    HELP_ROW = 23
    SCREEN_WIDTH = 80

    def __init__(
        self,
        out: Optional[TextIO] = None,
        voices: int = VOICES,
        title: Optional[str] = None,
        baud: int = 115200,
        poll_keys: bool = False,
    ):
        self.out = out if out is not None else sys.stdout
        self.voices = voices
        self.state = [VoiceState() for _ in range(voices)]
        self.bursts = [_MacroBurst() for _ in range(voices)]
        self.frame_count = 0
        self.title = title or "preframr live render"
        self.bps = _BpsMeter(baud=baud)
        self.muted_voices: Set[int] = set()
        self.quit_requested = False
        self._opened = False
        self._key_reader: Optional[_KeyReader] = _KeyReader() if poll_keys else None

    def open(self) -> None:
        if self._key_reader is not None:
            self._key_reader.open()
        self._write(_Vt100.CLEAR + _Vt100.HIDE)
        self._write("preframr live render\n")
        self._draw_header()
        for v in range(self.voices):
            self._draw_voice_row(v)
        self._draw_status()
        self._draw_help()
        self._flush()
        self._opened = True

    def close(self) -> None:
        if not self._opened:
            return
        self._write(_Vt100.RESET + _Vt100.SHOW)
        self._goto(self.HELP_ROW + 1, 1)
        self._write("\n")
        self._flush()
        if self._key_reader is not None:
            self._key_reader.close()
        self._opened = False

    def on_frame(self, packet) -> None:
        for op in packet.ops:
            if op.reg < 0:
                continue
            for v, vs in enumerate(self.state):
                if vs.absorb_op(op.reg, op.val, v):
                    break
        for vs, burst in zip(self.state, self.bursts, strict=False):
            vs.history.append(vs.freq())
            burst.update(vs)
            vs.snapshot_prev()
        self.frame_count += 1
        if self._key_reader is not None:
            self._handle_keys()
        self._draw()

    def _handle_keys(self) -> None:  # pragma: no cover
        assert self._key_reader is not None
        while True:
            ch = self._key_reader.poll()
            if ch is None:
                return
            if ch == "q":
                self.quit_requested = True
            elif ch in ("1", "2", "3"):
                v = int(ch) - 1
                if v < self.voices:
                    if v in self.muted_voices:
                        self.muted_voices.discard(v)
                    else:
                        self.muted_voices.add(v)

    def is_quit(self) -> bool:
        return self.quit_requested

    def _draw(self) -> None:
        self._draw_header()
        for v in range(self.voices):
            self._draw_voice_row(v)
        self._draw_status()
        self._flush()

    def _draw_header(self) -> None:
        self._goto(self.HEADER_ROW, 1)
        bps_now = self.bps.now()
        bps_str = f"bps:{bps_now:>5}/{self.bps.cap}"
        line = f"{self.title}  frame {self.frame_count:>6}  {bps_str}"
        self._write(line.ljust(self.SCREEN_WIDTH))

    def _draw_voice_row(self, v: int) -> None:
        vs = self.state[v]
        burst = self.bursts[v]
        self._goto(self.VOICE_START_ROW + v, 1)
        gate_char = "G" if vs.gate else "."
        spark = self._sparkline(list(vs.history))
        body = (
            f"V{v + 1} [{gate_char}] {vs.waveform_name()}  "
            f"freq={vs.freq():>5}  {spark}  "
        )
        body = body.ljust(70)
        if v in self.muted_voices:
            self._write(_Vt100.DIM + body + _Vt100.RESET)
        else:
            self._write(body)
        self._write(burst.render())

    def _draw_status(self) -> None:
        self._goto(self.STATUS_ROW, 1)
        muted_str = "".join(
            "-" if v in self.muted_voices else str(v + 1) for v in range(self.voices)
        )
        self._write(f"mute: {muted_str}".ljust(self.SCREEN_WIDTH))

    def _draw_help(self) -> None:
        self._goto(self.HELP_ROW, 1)
        self._write("[1/2/3] mute  [q] quit".ljust(self.SCREEN_WIDTH))

    @staticmethod
    def _sparkline(history) -> str:
        history = list(history)
        if not history:
            return ""
        lo, hi = min(history), max(history)
        if hi == lo:
            return _SPARK_GLYPHS[len(_SPARK_GLYPHS) // 2] * len(history)
        out = []
        scale = hi - lo
        for v in history:
            idx = int((v - lo) / scale * (len(_SPARK_GLYPHS) - 1))
            out.append(_SPARK_GLYPHS[idx])
        return "".join(out)

    def _goto(self, row: int, col: int) -> None:
        self._write(_Vt100.goto(row, col))

    def _write(self, s: str) -> None:
        self.out.write(s)
        self.bps.add(len(s))

    def _flush(self) -> None:
        try:
            self.out.flush()
        except (ValueError, OSError):
            pass
