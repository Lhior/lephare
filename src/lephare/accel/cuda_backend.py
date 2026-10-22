"""Custom CUDA / CuPy backend for hybrid LePHARE estimation."""

from __future__ import annotations

import numpy as np

from .gpu_util import choose_tiling_axis, suggest_chi2_chunk_size
from .kernels_ref import (
    HIGH_CHI2,
    BayZgIndex,
    bay_zg_pdf,
    pdf_zmode_zmean,
    prepare_bay_zg_index,
    rm_discrepant,
)
from .maglib import MagLibTensors


def _chi2_batch_kernel_src(real: str) -> str:
    """Grid-stride over N_obj * N_t; each thread owns one (object, template)."""
    return f"""
extern "C" __global__
void chi2_batch_kernel(
    const {real} *flux,          // [N_t, N_b]
    const {real} *ab,            // [N_obj, N_b]
    const {real} *invsab,        // [N_obj, N_b]
    const {real} *invsabSq,      // [N_obj, N_b]
    const {real} *abinvsabSq,    // [N_obj, N_b]
    {real} *chi2_out,            // [N_obj, N_t]
    {real} *dm_out,              // [N_obj, N_t]
    int N_obj, int N_t, int N_b
) {{
    long long ntot = (long long)N_obj * (long long)N_t;
    long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long idx = tid; idx < ntot; idx += stride) {{
        int o = (int)(idx / N_t);
        int i = (int)(idx % N_t);
        {real} avmago = 0;
        {real} avmagt = 0;
        for (int k = 0; k < N_b; ++k) {{
            {real} f = flux[i * N_b + k];
            avmago += f * abinvsabSq[o * N_b + k];
            avmagt += f * f * invsabSq[o * N_b + k];
        }}
        {real} dm = (avmagt > 0) ? (avmago / avmagt) : ({real})(-999.0);
        dm_out[o * N_t + i] = dm;
        {real} chi2 = 0;
        for (int k = 0; k < N_b; ++k) {{
            {real} f = flux[i * N_b + k];
            {real} inter = ab[o * N_b + k] * invsab[o * N_b + k]
                         - dm * f * invsab[o * N_b + k];
            chi2 += inter * inter;
        }}
        chi2_out[o * N_t + i] = chi2;
    }}
}}
"""


_CHI2_BATCH_F64 = _chi2_batch_kernel_src("double")
_CHI2_BATCH_F32 = _chi2_batch_kernel_src("float")
_KERNEL_CACHE: dict = {}
_FLUX_CACHE: dict = {}
_INDEX_CACHE: dict = {}


def _require_cupy():
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "backend requires CuPy. Install with: pip install 'lephare[cuda]'"
        ) from exc
    return cp


def _get_batch_kernel(cp, dtype):
    key = "f64" if dtype == np.float64 else "f32"
    if key not in _KERNEL_CACHE:
        src = _CHI2_BATCH_F64 if dtype == np.float64 else _CHI2_BATCH_F32
        module = cp.RawModule(code=src)
        _KERNEL_CACHE[key] = module.get_function("chi2_batch_kernel")
    return _KERNEL_CACHE[key]


def _device_flux(cp, flux: np.ndarray, dtype):
    from .maglib import cached_by_identity

    return cached_by_identity(
        _FLUX_CACHE, flux, (np.dtype(dtype).str,),
        lambda: cp.asarray(np.ascontiguousarray(flux.astype(dtype))),
    )


def _device_bay_zg(cp, tensors: MagLibTensors) -> tuple[BayZgIndex, dict]:
    """Host BayZgIndex plus cached device arrays for PDF / type mins."""
    from .maglib import cached_by_identity

    def build():
        index = prepare_bay_zg_index(tensors)
        qso_idx = np.flatnonzero(np.asarray(tensors.types) == 1)
        qso_bins = _z_bins(np.asarray(tensors.z)[qso_idx], index.gridz)
        # global template columns per z bin (MIN_ZG / MIN_ZQ on the object path)
        zbin_cols = {
            "gal": [cp.asarray(index.gal_idx[c]) for c in index.bin_cols],
            "qso": [cp.asarray(qso_idx[qso_bins == b]) for b in range(index.n_z)],
        }
        return (
            index,
            {
                "gal_idx": cp.asarray(index.gal_idx),
                "bin_idx": cp.asarray(index.bin_idx),
                "dz": cp.asarray(index.dz, dtype=np.float64),
                "bin_cols": [cp.asarray(c) for c in index.bin_cols],
                "types": cp.asarray(tensors.types),
                "mag0": cp.asarray(tensors.mag0, dtype=np.float64),
                "z_lib": cp.asarray(tensors.z, dtype=np.float64),
                "prior": cp.asarray(tensors.prior_lib, dtype=np.float64),
                "zbin_cols": zbin_cols,
            },
        )

    return cached_by_identity(_INDEX_CACHE, tensors, (), build)


def _z_bins(z: np.ndarray, gridz: np.ndarray) -> np.ndarray:
    """Nearest ``gridz`` bin (same rule as ``prepare_bay_zg_index``)."""
    if gridz.size <= 1:
        return np.zeros(z.size, dtype=np.int64)
    inv = 1.0 / (gridz[1] - gridz[0])
    return np.clip(np.rint((z - gridz[0]) * inv).astype(np.int64), 0, gridz.size - 1)


def _fit_winners_device(cp, d_chi2, index: BayZgIndex, d_aux: dict):
    """Per-type ``(indmin, chimin)`` and MIN_ZG / MIN_ZQ curves from device χ² (host out)."""
    n_obj = d_chi2.shape[0]
    rows = cp.arange(n_obj)
    indmin = cp.full((n_obj, 3), -99, dtype=cp.int64)
    chimin = cp.full((n_obj, 3), HIGH_CHI2, dtype=cp.float64)
    for t in range(3):
        vals = cp.where(d_aux["types"][None, :] == t, d_chi2, HIGH_CHI2)
        idx = cp.argmin(vals, axis=1)
        cmin = vals[rows, idx].astype(cp.float64)
        ok = cmin < HIGH_CHI2
        indmin[:, t] = cp.where(ok, idx, -99)
        chimin[:, t] = cp.where(ok, cmin, HIGH_CHI2)
    mins = []
    for kind in ("gal", "qso"):
        out = cp.full((n_obj, index.n_z), HIGH_CHI2, dtype=cp.float64)
        for b, cols in enumerate(d_aux["zbin_cols"][kind]):
            if cols.size:
                out[:, b] = d_chi2[:, cols].min(axis=1)
        mins.append(cp.asnumpy(out))
    return cp.asnumpy(indmin), cp.asnumpy(chimin), mins[0], mins[1]


def _apply_mabs_priors(
    chi2: np.ndarray,
    dm: np.ndarray,
    types: np.ndarray,
    mag0: np.ndarray,
    z_lib: np.ndarray,
    prior: np.ndarray,
    funz0: float,
) -> np.ndarray:
    """Apply Mabs priors; ``chi2``/``dm`` may be ``[N_t]`` or ``[N_obj, N_t]``."""
    chi2 = chi2.copy()
    if prior[0] < 0 and prior[1] < 0:
        abs_mag = mag0 - 2.5 * np.log10(np.clip(dm, 1e-300, None))
        abs_mag = np.where(z_lib < 1e-10, abs_mag - funz0, abs_mag)
        reject = (types == 0) & ((abs_mag <= prior[0]) | (abs_mag >= prior[1]))
        chi2 = np.where(reject, HIGH_CHI2, chi2)
    if prior[2] < 0 and prior[3] < 0:
        abs_mag = mag0 - 2.5 * np.log10(np.clip(dm, 1e-300, None))
        abs_mag = np.where(z_lib < 1e-10, abs_mag - funz0, abs_mag)
        reject = (types == 1) & ((abs_mag <= prior[2]) | (abs_mag >= prior[3]))
        chi2 = np.where(reject, HIGH_CHI2, chi2)
    return chi2


def _apply_mabs_priors_device(cp, d_chi2, d_dm, d_types, d_mag0, d_z, prior, funz0):
    """In-place Mabs priors on device arrays."""
    prior = np.asarray(prior, dtype=np.float64)
    if prior[0] < 0 and prior[1] < 0:
        abs_mag = d_mag0 - 2.5 * cp.log10(cp.clip(d_dm, 1e-300, None))
        abs_mag = cp.where(d_z < 1e-10, abs_mag - funz0, abs_mag)
        reject = (d_types == 0) & ((abs_mag <= prior[0]) | (abs_mag >= prior[1]))
        d_chi2 = cp.where(reject, HIGH_CHI2, d_chi2)
    if prior[2] < 0 and prior[3] < 0:
        abs_mag = d_mag0 - 2.5 * cp.log10(cp.clip(d_dm, 1e-300, None))
        abs_mag = cp.where(d_z < 1e-10, abs_mag - funz0, abs_mag)
        reject = (d_types == 1) & ((abs_mag <= prior[2]) | (abs_mag >= prior[3]))
        d_chi2 = cp.where(reject, HIGH_CHI2, d_chi2)
    return d_chi2


def _improved_device(cp, d_chi2, d_types):
    """Per-object min χ² across SED types."""
    n_obj = d_chi2.shape[0]
    improved = cp.full(n_obj, HIGH_CHI2, dtype=d_chi2.dtype)
    for t in range(3):
        mask = d_types == t
        if bool(mask.any()):
            vals = cp.where(mask[None, :], d_chi2, HIGH_CHI2)
            improved = cp.minimum(improved, vals.min(axis=1))
    return improved


def _bay_zg_pdf_device(cp, d_chi2, index: BayZgIndex, d_gal_idx, d_bin_cols, d_dz):
    """Device BAY_ZG: ``d_chi2[N_obj, N_t]`` → ``pdfs[N_obj, N_z]`` (device)."""
    n_obj = d_chi2.shape[0]
    n_z = index.n_z
    pdf = cp.zeros((n_obj, n_z), dtype=cp.float64)
    if d_gal_idx.size == 0 or n_z == 0:
        return pdf
    chi2g = d_chi2[:, d_gal_idx].astype(cp.float64, copy=False)
    chimin0 = chi2g.min(axis=1, keepdims=True)
    ok = chi2g < HIGH_CHI2
    prob = cp.where(ok, cp.exp(-0.5 * (chi2g - chimin0)), 0.0)
    for b, cols in enumerate(d_bin_cols):
        if cols.size:
            pdf[:, b] = prob[:, cols].sum(axis=1)
    if n_z > 1 and d_dz.size > 0:
        somme = cp.sum((pdf[:, :-1] + pdf[:, 1:]) * 0.5 * d_dz[None, :], axis=1)
        good = somme > 0
        pdf = cp.where(good[:, None], pdf / cp.where(good[:, None], somme[:, None], 1.0), pdf)
    return pdf


def chi2_cuda(
    flux: np.ndarray,
    ab: np.ndarray,
    sab: np.ndarray,
    busnorma: np.ndarray,
    types: np.ndarray,
    mag0: np.ndarray,
    z_lib: np.ndarray,
    prior: np.ndarray,
    funz0: float,
    dtype=np.float64,
    d_flux=None,
) -> tuple[np.ndarray, np.ndarray]:
    """χ² for one object (``ab`` shape ``[N_b]``) via batched kernel with N_obj=1."""
    chi2, dm = chi2_cuda_batch(
        flux,
        ab[None, :],
        sab[None, :],
        busnorma[None, :],
        types,
        mag0,
        z_lib,
        prior,
        funz0,
        dtype=dtype,
        d_flux=d_flux,
    )
    return chi2[0], dm[0]


def _chi2_cuda_batch_device(
    flux: np.ndarray,
    ab: np.ndarray,
    sab: np.ndarray,
    busnorma: np.ndarray,
    types: np.ndarray,
    mag0: np.ndarray,
    z_lib: np.ndarray,
    prior: np.ndarray,
    funz0: float,
    dtype=np.float64,
    d_flux=None,
    d_aux=None,
):
    """Run χ² + Mabs priors on device; return ``(d_chi2, d_dm)`` without D2H."""
    cp = _require_cupy()
    n_obj, n_b = ab.shape
    n_t = flux.shape[0]
    kernel = _get_batch_kernel(cp, dtype)
    if d_flux is None:
        d_flux = _device_flux(cp, flux, dtype)

    d_ab = cp.asarray(np.ascontiguousarray(ab.astype(dtype)))
    d_sab = cp.asarray(np.ascontiguousarray(sab.astype(dtype)))
    d_bus = cp.asarray(np.ascontiguousarray(busnorma.astype(dtype)))
    # invsab* on device (was host NumPy + 3 uploads)
    d_inv = cp.where(d_bus > 0, d_bus / d_sab, 0.0)
    d_inv2 = d_inv * d_inv
    d_abinv = d_ab * d_inv2
    d_chi2 = cp.empty((n_obj, n_t), dtype=dtype)
    d_dm = cp.empty((n_obj, n_t), dtype=dtype)

    threads = 256
    ntot = n_obj * n_t
    blocks = min(2048, (ntot + threads - 1) // threads)
    kernel(
        (blocks,),
        (threads,),
        (
            d_flux,
            d_ab,
            d_inv,
            d_inv2,
            d_abinv,
            d_chi2,
            d_dm,
            np.int32(n_obj),
            np.int32(n_t),
            np.int32(n_b),
        ),
    )
    if d_aux is None:
        d_types = cp.asarray(types)
        d_mag0 = cp.asarray(mag0, dtype=np.float64)
        d_z = cp.asarray(z_lib, dtype=np.float64)
    else:
        d_types = d_aux["types"]
        d_mag0 = d_aux["mag0"]
        d_z = d_aux["z_lib"]
    d_chi2 = _apply_mabs_priors_device(cp, d_chi2, d_dm, d_types, d_mag0, d_z, prior, funz0)
    return d_chi2, d_dm


def chi2_cuda_batch(
    flux: np.ndarray,
    ab: np.ndarray,
    sab: np.ndarray,
    busnorma: np.ndarray,
    types: np.ndarray,
    mag0: np.ndarray,
    z_lib: np.ndarray,
    prior: np.ndarray,
    funz0: float,
    dtype=np.float64,
    d_flux=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched custom CUDA χ²: ``ab`` shape ``[N_obj, N_b]`` → ``chi2[N_obj, N_t]``."""
    cp = _require_cupy()
    d_chi2, d_dm = _chi2_cuda_batch_device(
        flux,
        ab,
        sab,
        busnorma,
        types,
        mag0,
        z_lib,
        prior,
        funz0,
        dtype=dtype,
        d_flux=d_flux,
    )
    return cp.asnumpy(d_chi2), cp.asnumpy(d_dm)


def pdf_batch_cuda(chi2: np.ndarray, tensors: MagLibTensors) -> np.ndarray:
    """Device BAY_ZG PDF from host χ² (for parity tests)."""
    cp = _require_cupy()
    index, d_aux = _device_bay_zg(cp, tensors)
    d_chi2 = cp.asarray(np.asarray(chi2, dtype=np.float64))
    if d_chi2.ndim == 1:
        d_chi2 = d_chi2[None, :]
    d_pdf = _bay_zg_pdf_device(
        cp, d_chi2, index, d_aux["gal_idx"], d_aux["bin_cols"], d_aux["dz"]
    )
    return cp.asnumpy(d_pdf)


def _type_mins(chi2: np.ndarray, types: np.ndarray) -> float:
    vals = []
    for t in range(3):
        mask = types == t
        if np.any(mask):
            vals.append(float(np.min(chi2[mask])))
        else:
            vals.append(HIGH_CHI2)
    return float(np.min(vals))


def prepare_photometry_cuda_device(
    mags: np.ndarray,
    magerrs: np.ndarray,
    tensors: MagLibTensors,
    context: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    dtype=np.float64,
):
    """CuPy photometry prep; returns device arrays ``ab,sab,busnorma,busul,nbul``."""
    cp = _require_cupy()
    d_mags = cp.asarray(np.ascontiguousarray(mags, dtype=dtype))
    d_err = cp.asarray(np.ascontiguousarray(magerrs, dtype=dtype))
    n_obj, n_b = int(d_mags.shape[0]), int(d_mags.shape[1])

    ab = cp.where(
        cp.isfinite(d_mags) & (d_mags < 90),
        cp.power(10.0, -0.4 * (d_mags + 48.6)),
        -9999.0,
    )
    sab = cp.where(ab > 0, ab * d_err / 1.086, -9999.0)

    err_scale = np.asarray(tensors.err_scale, dtype=dtype)
    err_factor = np.asarray(tensors.err_factor, dtype=dtype)
    sab_cols = []
    for k in range(n_b):
        scale = float(err_scale[k] if err_scale.size > 1 else err_scale[0])
        fac = float(err_factor[k] if err_factor.size > 1 else err_factor[0])
        good = (sab[:, k] >= 0) & (ab[:, k] != 0.0)
        sab_mag = cp.where(good, 1.086 * sab[:, k] / ab[:, k], 0.0)
        sab_mag = cp.sqrt(sab_mag * sab_mag + scale * scale)
        sk = cp.where(good, cp.abs(ab[:, k]) * sab_mag / 1.086 * fac, sab[:, k])
        sab_cols.append(sk)
    sab = cp.stack(sab_cols, axis=1)

    if offsets is not None:
        corr = cp.power(10.0, 0.4 * cp.asarray(offsets, dtype=dtype))
        use = (ab > 0) | (sab > 0)
        ab = cp.where(use, ab * corr[None, :], ab)

    if context is None:
        ctx = cp.zeros(n_obj, dtype=cp.int64)
        for k in range(n_b):
            mask = (d_mags[:, k] > 0) & cp.isfinite(d_mags[:, k])
            mask = mask & (d_err[:, k] > 0) & cp.isfinite(d_err[:, k])
            ctx = ctx + mask.astype(cp.int64) * cp.int64(1 << k)
    else:
        ctx = cp.asarray(context, dtype=cp.int64)

    gbcont = int(tensors.gbcont)
    contforb = int(tensors.contforb)
    if gbcont >= 0:
        cont = cp.full(n_obj, gbcont, dtype=cp.int64)
    else:
        cont = ctx

    ab_cols = []
    sab_cols2 = []
    busnorma_cols = []
    busul_cols = []
    for k in range(n_b):
        bit = cp.int64(1 << k)
        if gbcont < 0:
            # per-object context bits
            used = cp.where(cont <= 0, cp.ones(n_obj, dtype=cp.int64), ((cont & bit) != 0).astype(cp.int64))
        else:
            # global context: band used if bit set (or all if cont<=0)
            used_flag = 1 if (gbcont <= 0 or (gbcont & (1 << k))) else 0
            used = cp.full(n_obj, used_flag, dtype=cp.int64)
        if contforb > 0 and (contforb & (1 << k)):
            used = cp.zeros(n_obj, dtype=cp.int64)
        used = cp.where((sab[:, k] < 0) & (ab[:, k] < 0), cp.int64(0), used)
        zero_err = sab[:, k] == 0
        used = cp.where(zero_err, cp.int64(0), used)
        ab_k = cp.where(zero_err, cp.asarray(-99.9, dtype=ab.dtype), ab[:, k])
        sab_k = cp.where(zero_err, cp.asarray(-99.9, dtype=sab.dtype), sab[:, k])
        bn = cp.where(sab_k < 0, cp.int64(0), used)
        busul_k = ((used == 1) & (sab_k < 0)).astype(cp.float64)
        ab_cols.append(ab_k)
        sab_cols2.append(sab_k)
        busnorma_cols.append(bn.astype(ab.dtype))
        busul_cols.append(busul_k)
    ab = cp.stack(ab_cols, axis=1)
    sab = cp.stack(sab_cols2, axis=1)
    busnorma = cp.stack(busnorma_cols, axis=1)
    busul = cp.stack(busul_cols, axis=1)
    nbul = cp.sum(busul, axis=1).astype(cp.int32)
    ab = ab.astype(dtype, copy=False)
    sab = sab.astype(dtype, copy=False)
    busnorma = busnorma.astype(dtype, copy=False)
    return ab, sab, busnorma, busul, nbul


def prepare_photometry_cuda(
    mags: np.ndarray,
    magerrs: np.ndarray,
    tensors: MagLibTensors,
    context: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    dtype=np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CuPy device photometry prep; returns host NumPy (parity / hard-path)."""
    cp = _require_cupy()
    ab, sab, busnorma, busul, nbul = prepare_photometry_cuda_device(
        mags, magerrs, tensors, context=context, offsets=offsets, dtype=dtype
    )
    return (
        cp.asnumpy(ab),
        cp.asnumpy(sab),
        cp.asnumpy(busnorma),
        cp.asnumpy(busul),
        cp.asnumpy(nbul),
    )


def _zancil_device(cp, d_pdf, d_zgrid):
    """Device zmode (argmax) + zmean."""
    idx = cp.argmax(d_pdf, axis=1)
    zmode = d_zgrid[idx]
    denom = d_pdf.sum(axis=1)
    zmean = cp.where(denom > 0, (d_pdf * d_zgrid[None, :]).sum(axis=1) / denom, cp.nan)
    return zmode, zmean


def process_batch_cuda(
    mags: np.ndarray,
    magerrs: np.ndarray,
    tensors: MagLibTensors,
    context: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    dtype=np.float64,
    return_ancil: bool = True,
) -> tuple:
    """CUDA χ²+PDF; template-fused path when ``chi2[N,Nt]`` does not fit."""
    n_obj = int(np.asarray(mags).shape[0])
    n_t = int(tensors.flux.shape[0])
    axis, size = choose_tiling_axis(n_obj, n_t, dtype=dtype)
    # FP32 always takes the fused path: only it applies the FP32 amplitude scale
    # (BC03 fluxes ~1e-40 underflow otherwise) and it never materializes chi2[N, Nt].
    if axis == "template" or np.dtype(dtype) == np.float32:
        from .cuda_template import process_template_tiles_cuda

        return process_template_tiles_cuda(
            mags,
            magerrs,
            tensors,
            context=context,
            offsets=offsets,
            dtype=dtype,
            return_ancil=return_ancil,
            tile_t=size,
        )

    pdfs, zgrid, zmode, zmean, ancil = _process_batch_cuda_core(
        mags, magerrs, tensors, context=context, offsets=offsets, dtype=dtype
    )
    if not return_ancil:
        return pdfs, zgrid
    return pdfs, zgrid, zmode, zmean, ancil


def _process_batch_cuda_core(
    mags: np.ndarray,
    magerrs: np.ndarray,
    tensors: MagLibTensors,
    context: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    dtype=np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Object-batched CUDA χ²+PDF; selective host RM_DISCREPANT for C++ parity."""
    from .bestfit import (
        bestfit_ancil_from_chi2,
        bestfit_ancil_from_type_mins,
        empty_ancil,
        merge_ancil,
    )

    cp = _require_cupy()
    ab, sab, busnorma, busul, nbul = prepare_photometry_cuda(
        mags, magerrs, tensors, context=context, offsets=offsets, dtype=dtype
    )
    types = tensors.types
    threshold = tensors.rm_discrepant
    n_obj, n_b = ab.shape
    n_z = len(tensors.gridz)
    pdfs = np.zeros((n_obj, n_z), dtype=np.float64)
    zmode = np.full(n_obj, np.nan, dtype=np.float64)
    zmean = np.full(n_obj, np.nan, dtype=np.float64)
    ancil = empty_ancil(n_obj, n_b)
    chunk = suggest_chi2_chunk_size(tensors.flux.shape[0], dtype=dtype)
    d_flux = _device_flux(cp, tensors.flux, dtype)
    index, d_aux = _device_bay_zg(cp, tensors)
    d_zgrid = cp.asarray(tensors.gridz, dtype=np.float64)

    for start in range(0, n_obj, chunk):
        end = min(start + chunk, n_obj)
        ul_idx = [i for i in range(start, end) if int(nbul[i]) > 0]
        batch_idx = [i for i in range(start, end) if int(nbul[i]) == 0]

        for i in ul_idx:
            chi2, _, _, bus = rm_discrepant(
                ab[i], sab[i], busnorma[i], busul[i], int(nbul[i]), tensors
            )
            merge_ancil(ancil, bestfit_ancil_from_chi2(chi2, tensors, busnorma=bus), np.asarray([i]))
            pdfs[i] = bay_zg_pdf(chi2, tensors)
            zm, zmn = pdf_zmode_zmean(pdfs[i : i + 1], tensors.gridz)
            zmode[i] = zm[0]
            zmean[i] = zmn[0]

        if not batch_idx:
            continue

        sel = np.asarray(batch_idx, dtype=np.int64)
        d_chi2, _d_dm = _chi2_cuda_batch_device(
            tensors.flux,
            ab[sel],
            sab[sel],
            busnorma[sel],
            types,
            tensors.mag0,
            tensors.z,
            tensors.prior_lib,
            tensors.funz0,
            dtype=dtype,
            d_flux=d_flux,
            d_aux=d_aux,
        )
        del _d_dm

        improved = cp.asnumpy(_improved_device(cp, d_chi2, d_aux["types"]))
        easy_j = []
        easy_i = []
        hard = []
        for j, i in enumerate(batch_idx):
            if float(improved[j]) > threshold:
                hard.append((j, i))
            else:
                easy_j.append(j)
                easy_i.append(i)

        if easy_j:
            easy_j_np = np.asarray(easy_j, dtype=np.int64)
            d_pdf = _bay_zg_pdf_device(
                cp,
                d_chi2[easy_j_np],
                index,
                d_aux["gal_idx"],
                d_aux["bin_cols"],
                d_aux["dz"],
            )
            d_zm, d_zmn = _zancil_device(cp, d_pdf, d_zgrid)
            easy_i_np = np.asarray(easy_i)
            pdfs[easy_i_np] = cp.asnumpy(d_pdf)
            zmode[easy_i_np] = cp.asnumpy(d_zm)
            zmean[easy_i_np] = cp.asnumpy(d_zmn)
            indmin, chimin, min_zg, min_zq = _fit_winners_device(
                cp, d_chi2[easy_j_np], index, d_aux
            )
            merge_ancil(
                ancil,
                bestfit_ancil_from_type_mins(
                    indmin,
                    chimin,
                    tensors,
                    min_zg=min_zg,
                    min_zq=min_zq,
                    busnorma=busnorma[easy_i_np],
                ),
                easy_i_np,
            )

        if hard:
            hard_j = np.asarray([j for j, _ in hard], dtype=np.int64)
            chi2_hard = cp.asnumpy(d_chi2[hard_j])
            for row, (j, i) in enumerate(hard):
                chi2 = chi2_hard[row]
                improved_i = _type_mins(chi2, types)
                bus = busnorma[i].copy()
                nb_removed = 0

                def _fit(bus_np, _i=i):
                    return chi2_cuda(
                        tensors.flux,
                        ab[_i],
                        sab[_i],
                        bus_np,
                        types,
                        tensors.mag0,
                        tensors.z,
                        tensors.prior_lib,
                        tensors.funz0,
                        dtype=dtype,
                        d_flux=d_flux,
                    )

                while improved_i > threshold and nb_removed < 2:
                    fl_dis = -1
                    best = improved_i
                    for k in range(n_b):
                        if bus[k] == 1:
                            bus[k] = 0
                            c2, _ = _fit(bus)
                            newmin = _type_mins(c2, types)
                            if best > newmin:
                                best = newmin
                                fl_dis = k
                            bus[k] = 1
                    if fl_dis >= 0:
                        bus[fl_dis] = 0
                        chi2, _ = _fit(bus)
                        improved_i = _type_mins(chi2, types)
                        nb_removed += 1
                    else:
                        break
                merge_ancil(
                    ancil, bestfit_ancil_from_chi2(chi2, tensors, busnorma=bus), np.asarray([i])
                )
                pdfs[i] = bay_zg_pdf(chi2, tensors)
                zm, zmn = pdf_zmode_zmean(pdfs[i : i + 1], tensors.gridz)
                zmode[i] = zm[0]
                zmean[i] = zmn[0]

    return pdfs, tensors.gridz.astype(np.float64, copy=False), zmode, zmean, ancil


def cuda_available() -> bool:
    try:
        cp = _require_cupy()
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False
