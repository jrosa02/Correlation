from typing import Literal

_GIGA = 1e9
_MEGA = 1e6
_KILO = 1e3
_MILLI = 1e-3
_MICRO = 1e-6


class Quantity:
    """A physical value stored internally as Hz."""

    def __init__(self, hz: float, display: Literal["freq", "time"] = "freq") -> None:
        self._hz = hz
        self._display = display

    def to_hz(self) -> float:
        return self._hz

    def to_s(self) -> float:
        return 1.0 / self._hz

    def __float__(self) -> float:
        return self._hz

    def __index__(self) -> int:
        return int(self._hz)

    def _fmt_freq(self) -> str:
        hz = self._hz
        if hz >= _GIGA:
            return f"{hz / _GIGA:.6g} GHz"
        if hz >= _MEGA:
            return f"{hz / _MEGA:.6g} MHz"
        if hz >= _KILO:
            return f"{hz / _KILO:.6g} kHz"
        return f"{hz:.6g} Hz"

    def _fmt_time(self) -> str:
        s = 1.0 / self._hz
        if s >= 1.0:
            return f"{s:.6g} s"
        if s >= _MILLI:
            return f"{s * 1e3:.6g} ms"
        if s >= _MICRO:
            return f"{s * 1e6:.6g} us"
        return f"{s * 1e9:.6g} ns"

    def set_repr(self, display: Literal["freq", "time"]) -> "Quantity":
        self._display = display
        return self

    def __repr__(self) -> str:
        inner = self._fmt_time() if self._display == "time" else self._fmt_freq()
        return f"Quantity({inner})"

    def __str__(self) -> str:
        return self._fmt_time() if self._display == "time" else self._fmt_freq()


class _FreqUnit:
    """Frequency unit. 50 * MHz → Quantity(50e6 Hz)."""

    def __init__(self, scale: float) -> None:
        self._scale = scale

    def __rmul__(self, value: float) -> Quantity:
        return Quantity(value * self._scale, display="freq")


class _PeriodUnit:
    """Period unit. 5 * ns → Quantity(1 / 5e-9 Hz)."""

    def __init__(self, scale_s: float) -> None:
        self._scale = scale_s

    def __rmul__(self, value: float) -> Quantity:
        return Quantity(1.0 / (value * self._scale), display="time")


# Frequency units
MHz = _FreqUnit(1e6)
kHz = _FreqUnit(1e3)
Hz = _FreqUnit(1.0)

# Period units (converted to frequency via 1/T)
s = _PeriodUnit(1.0)
ms = _PeriodUnit(1e-3)
us = _PeriodUnit(1e-6)
ns = _PeriodUnit(1e-9)
