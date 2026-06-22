import numpy as np
import pytest

from dzida_phy.native_optimized import corr_ext

PULSE_WIDTHS = [4, 8, 16, 32, 64]
RNG = np.random.default_rng(0)


# --- reference implementations (match original correlator_pipe.py logic) ---

def _rect_ref(pw: int) -> np.ndarray:
    return np.concatenate((
        np.repeat([-1], pw // 2 - 1),
        [-2, 2],
        np.repeat([1], pw - 2),
        [2, -2],
        np.repeat([-1], pw // 2 - 1),
    ), dtype=np.float64)


def _tri_ref(pw: int) -> np.ndarray:
    arr = np.concatenate(
        ([i for i in range(pw)], [pw - i for i in range(pw)]),
        dtype=np.float64,
    )
    return (arr - arr.mean()) / arr.std()


def _numpy_correlate(signal: np.ndarray, ref: np.ndarray) -> np.ndarray:
    rows = signal.reshape(-1, signal.shape[-1]) if signal.ndim > 1 else signal[np.newaxis]
    result = np.stack([np.correlate(row.astype(np.float64), ref, "same") for row in rows])
    result /= len(ref)
    return result if signal.ndim > 1 else result[0]


def _c_correlate(signal: np.ndarray, ref_type: int, pw: int) -> np.ndarray:
    rows = signal.reshape(-1, signal.shape[-1]) if signal.ndim > 1 else signal[np.newaxis]
    rows = np.ascontiguousarray(rows, dtype=np.float64)
    out = np.empty_like(rows)
    corr_ext.correlate(rows, ref_type, pw, out)
    return out if signal.ndim > 1 else out[0]


# --- output shape ---

class TestOutputShape:
    def test_1d_input_returns_1d(self):
        sig = RNG.standard_normal(128)
        out = _c_correlate(sig, 0, 4)
        assert out.ndim == 1
        assert out.shape == sig.shape

    def test_2d_input_returns_2d(self):
        sig = RNG.standard_normal((3, 128))
        out = _c_correlate(sig, 0, 4)
        assert out.ndim == 2
        assert out.shape == sig.shape

    @pytest.mark.parametrize("pw", PULSE_WIDTHS)
    def test_output_length_equals_signal_length(self, pw):
        sig = RNG.standard_normal(256)
        out = _c_correlate(sig, 0, pw)
        assert len(out) == len(sig)


# --- numerical agreement with numpy reference ---

class TestNumericalMatch:
    TOL = 1e-12

    @pytest.mark.parametrize("pw", PULSE_WIDTHS)
    def test_rect_matches_numpy(self, pw):
        sig = RNG.standard_normal((4, 256))
        ref = _rect_ref(pw)
        expected = _numpy_correlate(sig, ref)
        got = _c_correlate(sig, 0, pw)
        assert np.max(np.abs(expected - got)) < self.TOL

    @pytest.mark.parametrize("pw", PULSE_WIDTHS)
    def test_tri_matches_numpy(self, pw):
        sig = RNG.standard_normal((4, 256))
        ref = _tri_ref(pw)
        expected = _numpy_correlate(sig, ref)
        got = _c_correlate(sig, 0, pw)  # rect — reuse shape check
        got = _c_correlate(sig, 1, pw)
        assert np.max(np.abs(expected - got)) < self.TOL

    def test_zero_signal_gives_zero_output(self):
        sig = np.zeros((2, 128), dtype=np.float64)
        out = _c_correlate(sig, 0, 8)
        assert np.all(out == 0.0)

    def test_constant_signal_rect_near_zero(self):
        # rect ref has zero mean by construction; corr of constant should be ~0
        sig = np.ones((1, 256), dtype=np.float64)
        out = _c_correlate(sig, 0, 8)
        # interior values (away from edges) must be exactly 0
        ref_len = 16  # 2*pw
        interior = out[0, ref_len:-ref_len]
        assert np.allclose(interior, 0.0, atol=1e-14)


# --- edge / boundary handling ---

class TestEdgePadding:
    """C code zero-pads (matching numpy 'same') at signal boundaries.
    Signal must be longer than the reference (2*pw), so use 256 samples."""

    @pytest.mark.parametrize("pw", PULSE_WIDTHS)
    def test_first_sample_not_nan(self, pw):
        sig = RNG.standard_normal(256)
        out = _c_correlate(sig, 0, pw)
        assert np.isfinite(out[0])

    @pytest.mark.parametrize("pw", PULSE_WIDTHS)
    def test_last_sample_not_nan(self, pw):
        sig = RNG.standard_normal(256)
        out = _c_correlate(sig, 0, pw)
        assert np.isfinite(out[-1])


# --- normalization ---

class TestNormalization:
    def test_rect_normalized_by_ref_len(self):
        # signal of all ones, away from edges: rect interior = sum(ref)/ref_len = 0
        # but we can check normalization via impulse response
        pw = 4
        ref = _rect_ref(pw)  # len=8
        # impulse at center
        sig = np.zeros(64)
        sig[32] = 1.0
        out = _c_correlate(sig, 0, pw)
        # at position 32, each ref value flipped contributes sig[32]*ref[half-k]
        # normalized result should equal reversed_ref/ref_len at that point
        expected = _numpy_correlate(sig, ref)
        assert np.allclose(out, expected, atol=1e-14)


# --- error handling ---

class TestErrors:
    def test_invalid_ref_type_raises(self):
        sig = np.ones((1, 64), dtype=np.float64)
        out = np.empty_like(sig)
        with pytest.raises(ValueError):
            corr_ext.correlate(sig, 2, 4, out)

    def test_unsupported_pulse_width_raises(self):
        sig = np.ones((1, 64), dtype=np.float64)
        out = np.empty_like(sig)
        with pytest.raises(ValueError):
            corr_ext.correlate(sig, 0, 7, out)

    def test_negative_ref_type_raises(self):
        sig = np.ones((1, 64), dtype=np.float64)
        out = np.empty_like(sig)
        with pytest.raises(ValueError):
            corr_ext.correlate(sig, -1, 4, out)
