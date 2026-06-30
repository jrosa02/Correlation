from typing import ClassVar

import numpy as np
import pytest

from dzida_phy import (
    BandpassPipe_Simple,
    BandpassPipe_Timed,
    BestFitPipe_Simple,
    BestFitPipe_Timed,
    CorrPipe_Simple,
    CorrPipe_Timed,
    DecodeSink_Simple,
    DecodeSink_Timed,
    RectCorrModule_Timed,
    UpSampler_Simple,
    UpSampler_Timed,
)
from dzida_phy.correlator_pipe import rect_reference
from dzida_phy.physical_units import Hz, MHz, Quantity, kHz, ms, ns, s, us

# --- physical_units ---


class TestQuantity:
    def test_freq_mhz(self):
        assert (200 * MHz).to_hz() == pytest.approx(200e6)

    def test_freq_khz(self):
        assert (500 * kHz).to_hz() == pytest.approx(500e3)

    def test_freq_hz(self):
        assert (1000 * Hz).to_hz() == pytest.approx(1000.0)

    def test_period_ns(self):
        # 5 ns period → 1/5e-9 = 200 MHz
        assert (5 * ns).to_hz() == pytest.approx(200e6)

    def test_period_us(self):
        assert (1 * us).to_hz() == pytest.approx(1e6)

    def test_period_ms(self):
        assert (1 * ms).to_hz() == pytest.approx(1e3)

    def test_period_s(self):
        assert (2 * s).to_hz() == pytest.approx(0.5)

    def test_period_ns_equals_freq_mhz(self):
        assert (5 * ns).to_hz() == pytest.approx((200 * MHz).to_hz())

    def test_returns_quantity(self):
        assert isinstance(5 * ns, Quantity)
        assert isinstance(200 * MHz, Quantity)

    def test_to_s_from_freq(self):
        assert (200 * MHz).to_s() == pytest.approx(5e-9)

    def test_to_s_from_period(self):
        assert (5 * ns).to_s() == pytest.approx(5e-9)

    def test_to_s_to_hz_roundtrip(self):
        q = 6.25 * MHz
        assert 1.0 / q.to_s() == pytest.approx(q.to_hz())


# --- BandpassPipe_Timed ---


class TestBandpassPipeTimed:
    def test_inheritance(self):
        assert issubclass(BandpassPipe_Timed, BandpassPipe_Simple)

    def test_low_cutoff_normalized(self):
        p = BandpassPipe_Timed(1 * MHz, 90 * MHz, 200 * MHz)
        # Nyquist = 100 MHz; low_norm = 1/100
        assert p.low == pytest.approx(1e6 / 100e6)

    def test_high_cutoff_normalized(self):
        p = BandpassPipe_Timed(1 * MHz, 90 * MHz, 200 * MHz)
        assert p.high == pytest.approx(90e6 / 100e6)

    def test_period_sample_rate(self):
        # 5 ns = 200 MHz → same normalized cutoffs
        p_period = BandpassPipe_Timed(1 * MHz, 90 * MHz, 5 * ns)
        p_freq = BandpassPipe_Timed(1 * MHz, 90 * MHz, 200 * MHz)
        assert p_period.low == pytest.approx(p_freq.low)
        assert p_period.high == pytest.approx(p_freq.high)

    def test_order_passthrough(self):
        p = BandpassPipe_Timed(1 * MHz, 90 * MHz, 200 * MHz, order=6)
        assert p.order == 6


# --- UpSampler_Timed ---


class TestUpSamplerTimed:
    def test_inheritance(self):
        assert issubclass(UpSampler_Timed, UpSampler_Simple)

    def test_rate_from_hz(self):
        u = UpSampler_Timed(200 * MHz, 6.25 * MHz)
        assert u.rate == 32

    def test_rate_from_period(self):
        u = UpSampler_Timed(5 * ns, 6.25 * MHz)
        assert u.rate == 32

    def test_rate_rounding(self):
        # 200 MHz / 7 MHz ≈ 28.57 → rounds to 29
        u = UpSampler_Timed(200 * MHz, 7 * MHz)
        assert u.rate == round(200e6 / 7e6)

    def test_method_passthrough(self):
        u = UpSampler_Timed(200 * MHz, 6.25 * MHz, method="repeat")
        assert u.method == "repeat"


# --- CorrPipe_Timed ---


class TestCorrPipeTimed:
    def test_inheritance(self):
        assert issubclass(CorrPipe_Timed, CorrPipe_Simple)

    def test_pulse_width_from_hz(self):
        c = CorrPipe_Timed(200 * MHz, 6.25 * MHz)
        assert c.pulse_width == 32

    def test_pulse_width_from_period(self):
        c = CorrPipe_Timed(5 * ns, 6.25 * MHz)
        assert c.pulse_width == 32

    def test_ref_type_passthrough(self):
        c = CorrPipe_Timed(200 * MHz, 6.25 * MHz, ref_type="triangle")
        assert c.ref_type == "triangle"


# --- rect_reference (recovered from the compiled corr_ext kernel) ---


class TestRectReference:
    PULSE_WIDTHS: ClassVar = [4, 8, 16, 32, 64, 128, 256]

    @pytest.mark.parametrize("pw", PULSE_WIDTHS)
    def test_length(self, pw):
        assert len(rect_reference(pw)) == 2 * pw

    @pytest.mark.parametrize("pw", PULSE_WIDTHS)
    def test_zero_mean(self, pw):
        # bipolar template: -1...-1 | +1...+1 | -1...-1, integral ~= 0
        assert rect_reference(pw).sum() == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("pw", PULSE_WIDTHS)
    def test_symmetric_palindrome(self, pw):
        ref = rect_reference(pw)
        assert np.allclose(ref, ref[::-1])

    def test_exact_values_pw4(self):
        # smallest table (RECT_4 in corr_ext.cpp), hand-checkable
        assert np.allclose(rect_reference(4), [-1, -2, 2, 1, 1, 2, -2, -1])

    def test_unsupported_pulse_width_raises(self):
        with pytest.raises(ValueError):
            rect_reference(7)


# --- RectCorrModule_Timed ---


class TestRectCorrModuleTimed:
    def test_constructs_without_plotting(self):
        m = RectCorrModule_Timed(200 * MHz, 6.25 * MHz)
        assert m.pipes[0].pulse_width == 32

    def test_runs_signal_through(self):
        m = RectCorrModule_Timed(200 * MHz, 6.25 * MHz)
        sig = np.zeros((1, 256))
        sig[0, 128] = 1.0
        out = m.pipes[0].process(sig)
        assert out.shape == sig.shape


# --- DecodeSink_Timed ---


class TestDecodeSinkTimed:
    def test_inheritance(self):
        assert issubclass(DecodeSink_Timed, DecodeSink_Simple)

    def test_sampling_rate_from_hz(self):
        d = DecodeSink_Timed(100, 10, 1024, 200 * MHz, 6.25 * MHz)
        assert d.sampling_rate == pytest.approx(32.0)

    def test_sampling_rate_from_period(self):
        d = DecodeSink_Timed(100, 10, 1024, 5 * ns, 6.25 * MHz)
        assert d.sampling_rate == pytest.approx(32.0)

    def test_chunk_size_passthrough(self):
        d = DecodeSink_Timed(100, 10, 1024, 200 * MHz, 6.25 * MHz)
        assert d.chunk_size == 10

    def test_n_slots_passthrough(self):
        d = DecodeSink_Timed(100, 10, 1024, 200 * MHz, 6.25 * MHz)
        assert d.n_slots == 1024


# --- BestFitPipe_Timed ---


class TestBestFitPipeTimed:
    def test_inheritance(self):
        assert issubclass(BestFitPipe_Timed, BestFitPipe_Simple)

    def test_rate_from_hz(self):
        b = BestFitPipe_Timed(200 * MHz, 6.25 * MHz)
        assert b.rate == 32

    def test_rate_from_period(self):
        b = BestFitPipe_Timed(5 * ns, 6.25 * MHz)
        assert b.rate == 32
