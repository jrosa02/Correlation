class Quantity:
    """A physical value stored internally as Hz."""
    def __init__(self, hz: float) -> None:
        self._hz = hz

    def to_hz(self) -> float:
        return self._hz

    def to_s(self) -> float:
        return 1.0 / self._hz

    def __float__(self) -> float:
        return self._hz

    def __index__(self) -> int:
        return int(self._hz)


class _FreqUnit:
    """Frequency unit. 50 * MHz → Quantity(50e6 Hz)."""
    def __init__(self, scale: float) -> None:
        self._scale = scale

    def __rmul__(self, value: float) -> Quantity:
        return Quantity(value * self._scale)


class _PeriodUnit:
    """Period unit. 5 * ns → Quantity(1 / 5e-9 Hz)."""
    def __init__(self, scale_s: float) -> None:
        self._scale = scale_s

    def __rmul__(self, value: float) -> Quantity:
        return Quantity(1.0 / (value * self._scale))


# Frequency units
MHz = _FreqUnit(1e6)
kHz = _FreqUnit(1e3)
Hz  = _FreqUnit(1.0)

# Period units (converted to frequency via 1/T)
s  = _PeriodUnit(1.0)
ms = _PeriodUnit(1e-3)
us = _PeriodUnit(1e-6)
ns = _PeriodUnit(1e-9)
