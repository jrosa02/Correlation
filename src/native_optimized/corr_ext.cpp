#include <Python.h>
#include <immintrin.h>

/* rect(pw): concat(repeat(-1,pw//2-1), [-2,2], repeat(1,pw-2), [2,-2],
 *                  repeat(-1,pw//2-1))  len = 2*pw
 * tri(pw):  (arr - mean) / std  where arr=[0..pw-1, pw, pw-1..1]
 *           values computed via numpy to full double precision
 *
 * alignas(32): required for _mm256_load_pd on ref arrays
 */

alignas(32) static const double RECT_4[8] = {
    -1.0, -2.0, 2.0, 1.0, 1.0, 2.0, -2.0, -1.0
};
alignas(32) static const double RECT_8[16] = {
    -1.0, -1.0, -1.0, -2.0, 2.0, 1.0, 1.0, 1.0,
     1.0,  1.0,  1.0,  2.0,-2.0,-1.0,-1.0,-1.0
};
alignas(32) static const double RECT_16[32] = {
    -1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-2.0,
     2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0,
    -2.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0
};
alignas(32) static const double RECT_32[64] = {
    -1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-2.0,
     2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0,
    -2.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0
};
alignas(32) static const double RECT_64[128] = {
    -1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
    -1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-2.0,
     2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0,
    -2.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
    -1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0
};

alignas(32) static const double TRI_4[8] = {
    -1.6329931618554523, -0.81649658092772615,  0.0,  0.81649658092772615,
     1.6329931618554523,  0.81649658092772615,  0.0, -0.81649658092772615
};
alignas(32) static const double TRI_8[16] = {
    -1.7056057308448833, -1.2792042981336627, -0.85280286542244166, -0.42640143271122083,
     0.0,                0.42640143271122083,  0.85280286542244166,  1.2792042981336627,
     1.7056057308448833,  1.2792042981336627,  0.85280286542244166,  0.42640143271122083,
     0.0,               -0.42640143271122083, -0.85280286542244166, -1.2792042981336627
};
alignas(32) static const double TRI_16[32] = {
    -1.7253243712550146, -1.5096588248481377, -1.2939932784412609, -1.078327732034384,
    -0.86266218562750729,-0.64699663922063044,-0.43133109281375365,-0.21566554640687682,
     0.0,                0.21566554640687682,  0.43133109281375365,  0.64699663922063044,
     0.86266218562750729, 1.078327732034384,   1.2939932784412609,  1.5096588248481377,
     1.7253243712550146,  1.5096588248481377,  1.2939932784412609,  1.078327732034384,
     0.86266218562750729, 0.64699663922063044, 0.43133109281375365, 0.21566554640687682,
     0.0,               -0.21566554640687682, -0.43133109281375365,-0.64699663922063044,
    -0.86266218562750729,-1.078327732034384,  -1.2939932784412609, -1.5096588248481377
};
alignas(32) static const double TRI_32[64] = {
    -1.7303618253948005, -1.6222142113076254, -1.5140665972204503, -1.4059189831332755,
    -1.2977713690461004, -1.1896237549589253, -1.0814761408717504, -0.97332852678457527,
    -0.86518091269740027,-0.75703329861022517,-0.64888568452305018,-0.54073807043587518,
    -0.43259045634870014,-0.32444284226152509,-0.21629522817435007,-0.10814761408717503,
     0.0,                0.10814761408717503,  0.21629522817435007,  0.32444284226152509,
     0.43259045634870014, 0.54073807043587518,  0.64888568452305018,  0.75703329861022517,
     0.86518091269740027, 0.97332852678457527,  1.0814761408717504,  1.1896237549589253,
     1.2977713690461004,  1.4059189831332755,   1.5140665972204503,  1.6222142113076254,
     1.7303618253948005,  1.6222142113076254,   1.5140665972204503,  1.4059189831332755,
     1.2977713690461004,  1.1896237549589253,   1.0814761408717504,  0.97332852678457527,
     0.86518091269740027, 0.75703329861022517,  0.64888568452305018,  0.54073807043587518,
     0.43259045634870014, 0.32444284226152509,  0.21629522817435007,  0.10814761408717503,
     0.0,               -0.10814761408717503, -0.21629522817435007, -0.32444284226152509,
    -0.43259045634870014,-0.54073807043587518, -0.64888568452305018,-0.75703329861022517,
    -0.86518091269740027,-0.97332852678457527, -1.0814761408717504, -1.1896237549589253,
    -1.2977713690461004, -1.4059189831332755,  -1.5140665972204503, -1.6222142113076254
};
alignas(32) static const double TRI_64[128] = {
    -1.7316280983966108, -1.6775147203217167, -1.6234013422468225, -1.5692879641719284,
    -1.5151745860970345, -1.4610612080221403, -1.4069478299472462, -1.3528344518723521,
    -1.2987210737974582, -1.244607695722564,  -1.1904943176476699, -1.1363809395727758,
    -1.0822675614978816, -1.0281541834229877, -0.97404080534809356,-0.91992742727319943,
    -0.8658140491983054, -0.81170067112341127,-0.75758729304851724,-0.7034739149736231,
    -0.64936053689872908,-0.59524715882383494,-0.54113378074894081,-0.48702040267404678,
    -0.4329070245991527, -0.37879364652425862,-0.32468026844936454,-0.2705668903744704,
    -0.21645351229957635,-0.16234013422468227,-0.10822675614978818,-0.054113378074894088,
     0.0,                0.054113378074894088, 0.10822675614978818, 0.16234013422468227,
     0.21645351229957635, 0.2705668903744704,  0.32468026844936454, 0.37879364652425862,
     0.4329070245991527,  0.48702040267404678, 0.54113378074894081, 0.59524715882383494,
     0.64936053689872908, 0.7034739149736231,  0.75758729304851724, 0.81170067112341127,
     0.8658140491983054,  0.91992742727319943, 0.97404080534809356, 1.0281541834229877,
     1.0822675614978816,  1.1363809395727758,  1.1904943176476699,  1.244607695722564,
     1.2987210737974582,  1.3528344518723521,  1.4069478299472462,  1.4610612080221403,
     1.5151745860970345,  1.5692879641719284,  1.6234013422468225,  1.6775147203217167,
     1.7316280983966108,  1.6775147203217167,  1.6234013422468225,  1.5692879641719284,
     1.5151745860970345,  1.4610612080221403,  1.4069478299472462,  1.3528344518723521,
     1.2987210737974582,  1.244607695722564,   1.1904943176476699,  1.1363809395727758,
     1.0822675614978816,  1.0281541834229877,  0.97404080534809356, 0.91992742727319943,
     0.8658140491983054,  0.81170067112341127, 0.75758729304851724, 0.7034739149736231,
     0.64936053689872908, 0.59524715882383494, 0.54113378074894081, 0.48702040267404678,
     0.4329070245991527,  0.37879364652425862, 0.32468026844936454, 0.2705668903744704,
     0.21645351229957635, 0.16234013422468227, 0.10822675614978818, 0.054113378074894088,
     0.0,               -0.054113378074894088,-0.10822675614978818,-0.16234013422468227,
    -0.21645351229957635,-0.2705668903744704, -0.32468026844936454,-0.37879364652425862,
    -0.4329070245991527, -0.48702040267404678,-0.54113378074894081,-0.59524715882383494,
    -0.64936053689872908,-0.7034739149736231, -0.75758729304851724,-0.81170067112341127,
    -0.8658140491983054, -0.91992742727319943,-0.97404080534809356,-1.0281541834229877,
    -1.0822675614978816, -1.1363809395727758, -1.1904943176476699, -1.244607695722564,
    -1.2987210737974582, -1.3528344518723521, -1.4069478299472462, -1.4610612080221403,
    -1.5151745860970345, -1.5692879641719284, -1.6234013422468225, -1.6775147203217167
};

/* -----------------------------------------------------------------------
 * inner_dot<RL>
 *   AVX-2 FMA dot product of sig_win[0..RL] with ref[0..RL], divided by RL.
 *   RL is always a multiple of 4, so the loop fills ymm registers exactly.
 *   sig_win may be unaligned; ref is alignas(32) → _mm256_load_pd is safe.
 * ----------------------------------------------------------------------- */
template<int RL>
[[gnu::target("avx2,fma")]]
static inline double inner_dot(
    const double * __restrict__ sig_win,
    const double * __restrict__ ref) noexcept
{
    __m256d acc = _mm256_setzero_pd();
    for (int k = 0; k < RL; k += 4) {
        __m256d s = _mm256_loadu_pd(sig_win + k);
        __m256d r = _mm256_load_pd(ref + k);
        acc = _mm256_fmadd_pd(s, r, acc);
    }
    __m128d lo  = _mm256_castpd256_pd128(acc);
    __m128d hi  = _mm256_extractf128_pd(acc, 1);
    __m128d sum = _mm_add_pd(lo, hi);
            sum = _mm_hadd_pd(sum, sum);
    return _mm_cvtsd_f64(sum) / RL;
}

/* -----------------------------------------------------------------------
 * corr_left_edge<RL>   [0, half)  — scalar, guards j >= 0
 * corr_interior<RL>    [half, interior_end) — branch-free, calls inner_dot
 * corr_right_edge<RL>  [interior_end, row_len) — scalar, guards j < row_len
 * ----------------------------------------------------------------------- */
template<int RL>
static void corr_left_edge(
    const double * __restrict__ row,
    const double * __restrict__ ref,
    double       * __restrict__ dst) noexcept
{
    constexpr int half = RL / 2;
    for (int i = 0; i < half; i++) {
        double sum = 0.0;
        for (int k = 0; k < RL; k++) {
            int j = i + k - half;
            if (j >= 0) sum += row[j] * ref[k];
        }
        dst[i] = sum / RL;
    }
}

template<int RL>
[[gnu::target("avx2,fma")]]
static void corr_interior(
    const double * __restrict__ row, int half, int interior_end,
    const double * __restrict__ ref,
    double       * __restrict__ dst) noexcept
{
    for (int i = half; i < interior_end; i++)
        dst[i] = inner_dot<RL>(row + i - half, ref);
}

template<int RL>
static void corr_right_edge(
    const double * __restrict__ row, int row_len, int interior_end, int half,
    const double * __restrict__ ref,
    double       * __restrict__ dst) noexcept
{
    for (int i = interior_end; i < row_len; i++) {
        double sum = 0.0;
        for (int k = 0; k < RL; k++) {
            int j = i + k - half;
            if (j < row_len) sum += row[j] * ref[k];
        }
        dst[i] = sum / RL;
    }
}

/* -----------------------------------------------------------------------
 * batch_corr<RL>  — outer row loop only; delegates to the three region fns
 * ----------------------------------------------------------------------- */
template<int RL>
static void batch_corr(
    const double * __restrict__ ref,
    const double * __restrict__ rows, int n_rows, int row_len,
    double       * __restrict__ out) noexcept
{
    constexpr int half         = RL / 2;
    const     int interior_end = row_len - (RL - half);
    for (int r = 0; r < n_rows; r++) {
        const double *row = rows + r * row_len;
        double       *dst = out  + r * row_len;
        corr_left_edge<RL> (row,                          ref, dst);
        corr_interior<RL>  (row, half, interior_end,      ref, dst);
        corr_right_edge<RL>(row, row_len, interior_end, half, ref, dst);
    }
}

/* Dispatch tables */
using corr_fn = void(*)(const double*, const double*, int, int, double*);

static const double * const REF_TABLE[2][5] = {
    {RECT_4, RECT_8, RECT_16, RECT_32, RECT_64},
    {TRI_4,  TRI_8,  TRI_16,  TRI_32,  TRI_64},
};
static const corr_fn DISPATCH[2][5] = {
    {batch_corr<8>, batch_corr<16>, batch_corr<32>, batch_corr<64>, batch_corr<128>},
    {batch_corr<8>, batch_corr<16>, batch_corr<32>, batch_corr<64>, batch_corr<128>},
};

static const int PW_TABLE[5] = {4, 8, 16, 32, 64};

static int pw_to_idx(int pw) {
    for (int i = 0; i < 5; i++)
        if (PW_TABLE[i] == pw) return i;
    return -1;
}

/* Python-facing wrapper */
extern "C" {

static PyObject *py_correlate(PyObject *self, PyObject *args)
{
    PyObject *sig_obj, *out_obj;
    int ref_type, pw;
    if (!PyArg_ParseTuple(args, "OiiO", &sig_obj, &ref_type, &pw, &out_obj))
        return nullptr;

    if (ref_type < 0 || ref_type > 1) {
        PyErr_SetString(PyExc_ValueError, "ref_type must be 0 (rect) or 1 (tri)");
        return nullptr;
    }
    int idx = pw_to_idx(pw);
    if (idx < 0) {
        PyErr_Format(PyExc_ValueError, "unsupported pulse_width %d; must be 4,8,16,32,64", pw);
        return nullptr;
    }

    Py_buffer sig_buf, out_buf;
    if (PyObject_GetBuffer(sig_obj, &sig_buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return nullptr;
    if (PyObject_GetBuffer(out_obj, &out_buf,
                           PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE | PyBUF_FORMAT) < 0) {
        PyBuffer_Release(&sig_buf);
        return nullptr;
    }

    int n_rows, row_len;
    if (sig_buf.ndim == 2) {
        n_rows  = static_cast<int>(sig_buf.shape[0]);
        row_len = static_cast<int>(sig_buf.shape[1]);
    } else {
        n_rows  = 1;
        row_len = static_cast<int>(sig_buf.len / sizeof(double));
    }

    DISPATCH[ref_type][idx](
        REF_TABLE[ref_type][idx],
        static_cast<const double *>(sig_buf.buf), n_rows, row_len,
        static_cast<double *>(out_buf.buf));

    PyBuffer_Release(&sig_buf);
    PyBuffer_Release(&out_buf);
    Py_RETURN_NONE;
}

static PyMethodDef corr_methods[] = {
    {"correlate", py_correlate, METH_VARARGS,
     "correlate(signal, ref_type, pw, out) -> None\n"
     "signal: 2D float64 C-contiguous array\n"
     "ref_type: 0=rect 1=triangle\n"
     "pw: pulse width in {4,8,16,32,64}\n"
     "out: pre-allocated float64 array same shape as signal"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef corr_module = {
    PyModuleDef_HEAD_INIT, "corr_ext", nullptr, -1, corr_methods
};

PyMODINIT_FUNC PyInit_corr_ext(void) {
    return PyModule_Create(&corr_module);
}

} // extern "C"
