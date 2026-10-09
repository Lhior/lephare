"""Fused FP64 CUDA template path: χ² in registers → reduce (no chi2[N,Nt] HBM).

Exact two-pass BAY_ZG (global GAL chimin) matching the NumPy / JAX reference.
"""

from __future__ import annotations

import os

import numpy as np

from .bestfit import bestfit_ancil_from_chi2, bestfit_ancil_from_type_mins, empty_ancil, merge_ancil
from .fp32_numerics import (
    FP32_FLUX_SCALE_CAP,
    FP32_TARGET_ABS,
    fp32_amp_scale,
    fp32_apply_scale,
    mag_abs_dm_bounds,
)
from .kernels_ref import HIGH_CHI2, bay_zg_pdf, pdf_zmode_zmean, rm_discrepant
from .maglib import MagLibTensors

_FUSE_KERNEL_CACHE: dict = {}
_LIB_CACHE: dict = {}

# Shared-memory z-bin scratch. Cardinal FAIR uses Nz=701; 1024 covers that.
_MAX_NZ_SHARED = 1024
_MAX_NB = 16
_THREADS = 256
_NB_UNROLL = (6, 9)
_FP32_FLUX_SCALE = FP32_FLUX_SCALE_CAP
_FP32_TARGET_ABS = FP32_TARGET_ABS
_SMEM_BUDGET = 96 * 1024
_DEFAULT_TILE_T = 256


def _fp32_amp_scale(flux: np.ndarray) -> float:
    return fp32_amp_scale(flux)


def _require_cupy():
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "backend requires CuPy. Install with: pip install 'lephare[cuda]'"
        ) from exc
    return cp


def _z_bin_array(z: np.ndarray, gridz: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    gridz = np.asarray(gridz, dtype=np.float64)
    n_z = int(gridz.size)
    if n_z <= 1:
        return np.zeros(z.shape[0], dtype=np.int32)
    inv = 1.0 / (gridz[1] - gridz[0])
    return np.clip(np.rint((z - gridz[0]) * inv).astype(np.int32), 0, n_z - 1)


def _mag_abs_dm_bounds(
    mag0: np.ndarray,
    z_lib: np.ndarray,
    types: np.ndarray,
    prior: np.ndarray,
    funz0: float,
    dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """MAG_ABS window as ``dm_min < dm < dm_max`` (same reject as C++ ``log10``)."""
    return mag_abs_dm_bounds(mag0, z_lib, types, prior, funz0, dtype)


def _tile_smem_bytes(n_oc: int, n_z: int, n_b: int, tile_t: int, nbytes: int) -> tuple[int, int]:
    """Pass1 / pass2 dynamic SMEM for object-chunk × template-tile kernels."""
    flux = n_b * tile_t * nbytes
    meta = tile_t * (2 * nbytes + 8)  # dm_min, dm_max, types+z_bin ints
    mins = n_oc * n_z * 2 * nbytes
    chi_idx = n_oc * 3 * nbytes + n_oc * 3 * 8
    p1 = flux + meta + mins + chi_idx
    p2 = flux + meta + n_oc * n_z * nbytes
    return int(p1), int(p2)


def _choose_tiled_launch(n_z: int, n_b: int, nbytes: int) -> tuple[int, int, int, int] | None:
    """Return ``(n_obj_chunk, tile_t, smem_p1, smem_p2)`` or None for legacy path.

    ``LEPHARE_ACCEL_CUDA_TILE_OBJS``: unset/0 = legacy; 1 = auto; N>=2 = objects/block.
    """
    raw = os.environ.get("LEPHARE_ACCEL_CUDA_TILE_OBJS", "").strip()
    if not raw or raw == "0":
        return None
    tile_t = int(os.environ.get("LEPHARE_ACCEL_CUDA_TILE_T", str(_DEFAULT_TILE_T)))
    tile_t = max(32, min(tile_t, 1024))
    forced_oc = None if raw == "1" else max(1, int(raw))
    candidates = (forced_oc,) if forced_oc is not None else (16, 8, 4, 2)
    for n_oc in candidates:
        p1, p2 = _tile_smem_bytes(n_oc, n_z, n_b, tile_t, nbytes)
        if max(p1, p2) <= _SMEM_BUDGET:
            return n_oc, tile_t, p1, p2
    for tile_t in (128, 64):
        for n_oc in (4, 2, 1):
            p1, p2 = _tile_smem_bytes(n_oc, n_z, n_b, tile_t, nbytes)
            if max(p1, p2) <= _SMEM_BUDGET:
                return n_oc, tile_t, p1, p2
    return None


def _fuse_kernels_src(real: str = "double") -> str:
    """Block-per-object fused pass1 / pass2; flux is ``[N_b, N_t]``.

    ``real`` is ``double`` (default, accuracy path) or ``float`` (throughput
    proxy on consumer GPUs with 1:64 FP64).
    """
    if real not in ("double", "float"):
        raise ValueError(real)
    high = f"{HIGH_CHI2:.1f}"
    if real == "float":
        log10, exp, fmax, tiny, lit0 = "log10f", "expf", "fmaxf", "1e-30f", "0.0f"
        dchi_max = "80.0f"
        atomic_min = """
__device__ __forceinline__ float atomicMinReal(float *addr, float val) {
    int *addr_i = (int *)addr;
    int old = *addr_i, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) break;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}
"""
    else:
        log10, exp, fmax, tiny, lit0 = "log10", "exp", "fmax", "1e-300", "0.0"
        dchi_max = "700.0"
        atomic_min = """
__device__ __forceinline__ double atomicMinReal(double *addr, double val) {
    unsigned long long *addr_ull = (unsigned long long *)addr;
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val) break;
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}
"""
    return f"""
#define HIGH_CHI2 ({high})
#define MAX_NZ {_MAX_NZ_SHARED}

#ifndef USE_PRIORS
#define USE_PRIORS 0
#endif

{atomic_min}

__device__ __forceinline__ void min_with_idx(
    {real} &c, long long &idx, {real} oc, long long oidx
) {{
    if (oc < c || (oc == c && oidx >= 0 && (idx < 0 || oidx < idx))) {{
        c = oc;
        idx = oidx;
    }}
}}

extern "C" __global__
void fuse_pass1_kernel(
    const {real}* __restrict__ flux_bt,
    const {real}* __restrict__ invsabSq,
    const {real}* __restrict__ abinvsabSq,
    const {real}* __restrict__ base,
    const int* __restrict__ types,
    const {real}* __restrict__ dm_min,
    const {real}* __restrict__ dm_max,
    const int* __restrict__ z_bin,
    {real}* __restrict__ chimin,
    long long* __restrict__ indmin,
    {real}* __restrict__ min_zg,
    {real}* __restrict__ min_zq,
    int N_obj, int N_t, int N_b, int N_z
) {{
    int o = (int)blockIdx.x;
    if (o >= N_obj) return;

    const int nwarps = (int)blockDim.x / 32;
    const int warp = (int)threadIdx.x / 32;
    const int lane = (int)threadIdx.x & 31;

    extern __shared__ unsigned char raw[];
    {real} *s_zg = ({real} *)raw;
    {real} *s_zq = s_zg + nwarps * N_z;
    {real} *s_chi = s_zq + nwarps * N_z;
    long long *s_idx = (long long *)(s_chi + nwarps * 3);

#ifdef N_B
    const int nb = N_B;
#else
    const int nb = N_b;
#endif
    {real} r_inv2[16];
    {real} r_abinv[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {{
        if (k < nb) {{
            int idx = o * nb + k;
            r_inv2[k] = invsabSq[idx];
            r_abinv[k] = abinvsabSq[idx];
        }}
    }}
    {real} r_base = base[o];

    for (int b = (int)threadIdx.x; b < nwarps * N_z; b += (int)blockDim.x) {{
        s_zg[b] = HIGH_CHI2;
        s_zq[b] = HIGH_CHI2;
    }}
    if ((int)threadIdx.x < nwarps * 3) {{
        s_chi[threadIdx.x] = HIGH_CHI2;
        s_idx[threadIdx.x] = (long long)(-99);
    }}
    __syncthreads();

    {real} p_chimin[3] = {{HIGH_CHI2, HIGH_CHI2, HIGH_CHI2}};
    long long p_indmin[3] = {{-99, -99, -99}};

    for (int i = (int)threadIdx.x; i < N_t; i += (int)blockDim.x) {{
        int typ = types[i];
        if (typ < 0) continue;
        {real} avmago = {lit0};
        {real} avmagt = {lit0};
#ifdef N_B
#pragma unroll
        for (int k = 0; k < N_B; ++k) {{
            {real} f = flux_bt[k * N_t + i];
            avmago += f * r_abinv[k];
            avmagt += f * f * r_inv2[k];
        }}
#else
        for (int k = 0; k < nb; ++k) {{
            {real} f = flux_bt[k * N_t + i];
            avmago += f * r_abinv[k];
            avmagt += f * f * r_inv2[k];
        }}
#endif
        {real} chi2 = r_base;
        if (avmagt > {lit0})
            chi2 = r_base - (avmago * avmago) / avmagt;
#if USE_PRIORS
        {real} dm = (avmagt > {lit0}) ? (avmago / avmagt) : ({real})(-999.0);
        if (typ == 0 || typ == 1) {{
            if (dm > {lit0} && (dm <= dm_min[i] || dm >= dm_max[i]))
                chi2 = HIGH_CHI2;
        }}
#endif
        if (typ >= 0 && typ < 3)
            min_with_idx(p_chimin[typ], p_indmin[typ], chi2, (long long)i);
        int zb = z_bin[i];
        if (zb >= 0 && zb < N_z) {{
            if (typ == 0) atomicMinReal(&s_zg[warp * N_z + zb], chi2);
            else if (typ == 1) atomicMinReal(&s_zq[warp * N_z + zb], chi2);
        }}
    }}

#pragma unroll
    for (int t = 0; t < 3; ++t) {{
        for (int off = 16; off > 0; off >>= 1) {{
            {real} oc = __shfl_xor_sync(0xffffffff, p_chimin[t], off);
            long long oi = __shfl_xor_sync(0xffffffff, p_indmin[t], off);
            min_with_idx(p_chimin[t], p_indmin[t], oc, oi);
        }}
        if (lane == 0) {{
            s_chi[warp * 3 + t] = p_chimin[t];
            s_idx[warp * 3 + t] = p_indmin[t];
        }}
    }}
    __syncthreads();

    if (threadIdx.x == 0) {{
        {real} bchi[3] = {{HIGH_CHI2, HIGH_CHI2, HIGH_CHI2}};
        long long bidx[3] = {{-99, -99, -99}};
        for (int w = 0; w < nwarps; ++w) {{
#pragma unroll
            for (int t = 0; t < 3; ++t)
                min_with_idx(bchi[t], bidx[t], s_chi[w * 3 + t], s_idx[w * 3 + t]);
        }}
#pragma unroll
        for (int t = 0; t < 3; ++t) {{
            chimin[o * 3 + t] = bchi[t];
            indmin[o * 3 + t] = bidx[t];
        }}
    }}
    for (int b = (int)threadIdx.x; b < N_z; b += (int)blockDim.x) {{
        {real} zg = s_zg[b];
        {real} zq = s_zq[b];
        for (int w = 1; w < nwarps; ++w) {{
            {real} zg2 = s_zg[w * N_z + b];
            {real} zq2 = s_zq[w * N_z + b];
            if (zg2 < zg) zg = zg2;
            if (zq2 < zq) zq = zq2;
        }}
        min_zg[o * N_z + b] = zg;
        min_zq[o * N_z + b] = zq;
    }}
}}

extern "C" __global__
void fuse_pass2_kernel(
    const {real}* __restrict__ flux_bt,
    const {real}* __restrict__ invsabSq,
    const {real}* __restrict__ abinvsabSq,
    const {real}* __restrict__ base,
    const {real}* __restrict__ dm_min,
    const {real}* __restrict__ dm_max,
    const int* __restrict__ z_bin,
    const {real}* __restrict__ chimin_gal,
    {real}* __restrict__ pdf,
    int N_obj, int N_t, int N_b, int N_z
) {{
    int o = (int)blockIdx.x;
    if (o >= N_obj) return;

    const int nwarps = (int)blockDim.x / 32;
    const int warp = (int)threadIdx.x / 32;

    extern __shared__ unsigned char raw[];
    {real} *s_pdf = ({real} *)raw;

#ifdef N_B
    const int nb = N_B;
#else
    const int nb = N_b;
#endif
    {real} r_inv2[16];
    {real} r_abinv[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {{
        if (k < nb) {{
            int idx = o * nb + k;
            r_inv2[k] = invsabSq[idx];
            r_abinv[k] = abinvsabSq[idx];
        }}
    }}
    {real} r_base = base[o];
    {real} cmin = chimin_gal[o];

    for (int b = (int)threadIdx.x; b < nwarps * N_z; b += (int)blockDim.x)
        s_pdf[b] = 0.0;
    __syncthreads();

    for (int i = (int)threadIdx.x; i < N_t; i += (int)blockDim.x) {{
        {real} avmago = {lit0};
        {real} avmagt = {lit0};
#ifdef N_B
#pragma unroll
        for (int k = 0; k < N_B; ++k) {{
            {real} f = flux_bt[k * N_t + i];
            avmago += f * r_abinv[k];
            avmagt += f * f * r_inv2[k];
        }}
#else
        for (int k = 0; k < nb; ++k) {{
            {real} f = flux_bt[k * N_t + i];
            avmago += f * r_abinv[k];
            avmagt += f * f * r_inv2[k];
        }}
#endif
        {real} chi2 = r_base;
        if (avmagt > {lit0})
            chi2 = r_base - (avmago * avmago) / avmagt;
#if USE_PRIORS
        {real} dm = (avmagt > {lit0}) ? (avmago / avmagt) : ({real})(-999.0);
        if (dm > {lit0} && (dm <= dm_min[i] || dm >= dm_max[i]))
            chi2 = HIGH_CHI2;
#endif
        if (chi2 >= HIGH_CHI2) continue;
        {real} dchi = chi2 - cmin;
        if (dchi < {lit0}) dchi = {lit0};
        if (dchi > {dchi_max}) continue;
        {real} prob = {exp}(({real})(-0.5) * dchi);
        int zb = z_bin[i];
        if (zb >= 0 && zb < N_z)
            atomicAdd(&s_pdf[warp * N_z + zb], prob);
    }}
    __syncthreads();

    for (int b = (int)threadIdx.x; b < N_z; b += (int)blockDim.x) {{
        {real} acc = {lit0};
        for (int w = 0; w < nwarps; ++w)
            acc += s_pdf[w * N_z + b];
        pdf[o * N_z + b] = acc;
    }}
}}
"""


def _get_fuse_kernels(cp, *, use_priors: bool, n_b: int, real: str = "double"):
    unroll = int(n_b) if int(n_b) in _NB_UNROLL else 0
    prec = "f64" if real == "double" else "f32"
    key = (prec, int(use_priors), unroll, "native_fma_exp")
    if key not in _FUSE_KERNEL_CACHE:
        opts = ["-std=c++11", f"-DUSE_PRIORS={1 if use_priors else 0}"]
        if unroll:
            opts.append(f"-DN_B={unroll}")
        mod = cp.RawModule(code=_fuse_kernels_src(real), options=tuple(opts))
        _FUSE_KERNEL_CACHE[key] = (
            mod.get_function("fuse_pass1_kernel"),
            mod.get_function("fuse_pass2_kernel"),
        )
    return _FUSE_KERNEL_CACHE[key]


def _fuse_tiled_kernels_src(real: str = "double") -> str:
    """Object-chunk × template-tile fused kernels (SMEM maglib reuse)."""
    if real not in ("double", "float"):
        raise ValueError(real)
    high = f"{HIGH_CHI2:.1f}"
    if real == "float":
        exp, lit0, dchi_max = "expf", "0.0f", "80.0f"
        atomic_min = """
__device__ __forceinline__ float atomicMinReal(float *addr, float val) {
    int *addr_i = (int *)addr;
    int old = *addr_i, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) break;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}
"""
    else:
        exp, lit0, dchi_max = "exp", "0.0", "700.0"
        atomic_min = """
__device__ __forceinline__ double atomicMinReal(double *addr, double val) {
    unsigned long long *addr_ull = (unsigned long long *)addr;
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val) break;
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}
"""
    return f"""
#define HIGH_CHI2 ({high})

{atomic_min}

__device__ __forceinline__ void min_with_idx(
    {real} &c, long long &idx, {real} oc, long long oidx
) {{
    if (oc < c || (oc == c && oidx >= 0 && (idx < 0 || oidx < idx))) {{
        c = oc;
        idx = oidx;
    }}
}}

extern "C" __global__
void fuse_tile_pass1_kernel(
    const {real}* __restrict__ flux_bt,
    const {real}* __restrict__ invsabSq,
    const {real}* __restrict__ abinvsabSq,
    const {real}* __restrict__ base,
    const int* __restrict__ types,
    const {real}* __restrict__ dm_min,
    const {real}* __restrict__ dm_max,
    const int* __restrict__ z_bin,
    {real}* __restrict__ chimin,
    long long* __restrict__ indmin,
    {real}* __restrict__ min_zg,
    {real}* __restrict__ min_zq,
    int N_obj, int N_t, int N_b, int N_z,
    int n_obj_chunk, int tile_t
) {{
    int o0 = (int)blockIdx.x * n_obj_chunk;
    if (o0 >= N_obj) return;
    int n_oc = n_obj_chunk;
    if (o0 + n_oc > N_obj) n_oc = N_obj - o0;
    const int warp = (int)threadIdx.x / 32;
    const int lane = (int)threadIdx.x & 31;
    const int nwarps = (int)blockDim.x / 32;
#ifdef N_B
    const int nb = N_B;
#else
    const int nb = N_b;
#endif
    extern __shared__ unsigned char raw[];
    {real} *s_flux = ({real} *)raw;
    {real} *s_dmin = s_flux + (size_t)nb * (size_t)tile_t;
    {real} *s_dmax = s_dmin + tile_t;
    int *s_typ = (int *)(s_dmax + tile_t);
    int *s_zb = s_typ + tile_t;
    {real} *s_zg = ({real} *)(s_zb + tile_t);
    {real} *s_zq = s_zg + n_obj_chunk * N_z;
    {real} *s_chi = s_zq + n_obj_chunk * N_z;
    unsigned char *idx_raw = (unsigned char *)(s_chi + n_obj_chunk * 3);
    unsigned long long idx_addr = ((unsigned long long)idx_raw + 7ull) & ~7ull;
    long long *s_idx = (long long *)idx_addr;

    for (int b = (int)threadIdx.x; b < n_oc * N_z; b += (int)blockDim.x) {{
        s_zg[b] = HIGH_CHI2;
        s_zq[b] = HIGH_CHI2;
    }}
    if ((int)threadIdx.x < n_oc * 3) {{
        s_chi[threadIdx.x] = HIGH_CHI2;
        s_idx[threadIdx.x] = (long long)(-99);
    }}
    __syncthreads();

    for (int ts = 0; ts < N_t; ts += tile_t) {{
        int tlen = tile_t;
        if (ts + tlen > N_t) tlen = N_t - ts;
        for (int idx = (int)threadIdx.x; idx < nb * tlen; idx += (int)blockDim.x) {{
            int k = idx / tlen;
            int j = idx - k * tlen;
            s_flux[k * tile_t + j] = flux_bt[(size_t)k * (size_t)N_t + (size_t)(ts + j)];
        }}
        for (int j = (int)threadIdx.x; j < tlen; j += (int)blockDim.x) {{
            int gi = ts + j;
            s_dmin[j] = dm_min[gi];
            s_dmax[j] = dm_max[gi];
            s_typ[j] = types[gi];
            s_zb[j] = z_bin[gi];
        }}
        __syncthreads();

        for (int ow = warp; ow < n_oc; ow += nwarps) {{
            int o = o0 + ow;
            {real} r_inv2[16];
            {real} r_abinv[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) {{
                if (k < nb) {{
                    int ix = o * nb + k;
                    r_inv2[k] = invsabSq[ix];
                    r_abinv[k] = abinvsabSq[ix];
                }}
            }}
            {real} r_base = base[o];
            {real} p_chimin[3] = {{HIGH_CHI2, HIGH_CHI2, HIGH_CHI2}};
            long long p_indmin[3] = {{-99, -99, -99}};
            for (int j = lane; j < tlen; j += 32) {{
                int typ = s_typ[j];
                if (typ < 0) continue;
                {real} avmago = {lit0};
                {real} avmagt = {lit0};
#ifdef N_B
#pragma unroll
                for (int k = 0; k < N_B; ++k) {{
                    {real} f = s_flux[k * tile_t + j];
                    avmago += f * r_abinv[k];
                    avmagt += f * f * r_inv2[k];
                }}
#else
                for (int k = 0; k < nb; ++k) {{
                    {real} f = s_flux[k * tile_t + j];
                    avmago += f * r_abinv[k];
                    avmagt += f * f * r_inv2[k];
                }}
#endif
                {real} chi2 = r_base;
                if (avmagt > {lit0})
                    chi2 = r_base - (avmago * avmago) / avmagt;
#if USE_PRIORS
                {real} dm = (avmagt > {lit0}) ? (avmago / avmagt) : ({real})(-999.0);
                if ((typ == 0 || typ == 1) && dm > {lit0} &&
                    (dm <= s_dmin[j] || dm >= s_dmax[j]))
                    chi2 = HIGH_CHI2;
#endif
                long long gi = (long long)(ts + j);
                if (typ >= 0 && typ < 3)
                    min_with_idx(p_chimin[typ], p_indmin[typ], chi2, gi);
                int zb = s_zb[j];
                if (zb >= 0 && zb < N_z) {{
                    if (typ == 0) atomicMinReal(&s_zg[ow * N_z + zb], chi2);
                    else if (typ == 1) atomicMinReal(&s_zq[ow * N_z + zb], chi2);
                }}
            }}
#pragma unroll
            for (int t = 0; t < 3; ++t) {{
                for (int off = 16; off > 0; off >>= 1) {{
                    {real} oc = __shfl_xor_sync(0xffffffff, p_chimin[t], off);
                    long long oi = __shfl_xor_sync(0xffffffff, p_indmin[t], off);
                    min_with_idx(p_chimin[t], p_indmin[t], oc, oi);
                }}
                if (lane == 0)
                    min_with_idx(s_chi[ow * 3 + t], s_idx[ow * 3 + t], p_chimin[t], p_indmin[t]);
            }}
        }}
        __syncthreads();
    }}

    for (int ow = (int)threadIdx.x; ow < n_oc; ow += (int)blockDim.x) {{
        int o = o0 + ow;
#pragma unroll
        for (int t = 0; t < 3; ++t) {{
            chimin[o * 3 + t] = s_chi[ow * 3 + t];
            indmin[o * 3 + t] = s_idx[ow * 3 + t];
        }}
    }}
    for (int b = (int)threadIdx.x; b < n_oc * N_z; b += (int)blockDim.x) {{
        int ow = b / N_z;
        int zb = b - ow * N_z;
        int o = o0 + ow;
        min_zg[o * N_z + zb] = s_zg[b];
        min_zq[o * N_z + zb] = s_zq[b];
    }}
}}

extern "C" __global__
void fuse_tile_pass2_kernel(
    const {real}* __restrict__ flux_bt,
    const {real}* __restrict__ invsabSq,
    const {real}* __restrict__ abinvsabSq,
    const {real}* __restrict__ base,
    const {real}* __restrict__ dm_min,
    const {real}* __restrict__ dm_max,
    const int* __restrict__ z_bin,
    const {real}* __restrict__ chimin_gal,
    {real}* __restrict__ pdf,
    int N_obj, int N_t, int N_b, int N_z,
    int n_obj_chunk, int tile_t
) {{
    int o0 = (int)blockIdx.x * n_obj_chunk;
    if (o0 >= N_obj) return;
    int n_oc = n_obj_chunk;
    if (o0 + n_oc > N_obj) n_oc = N_obj - o0;
    const int warp = (int)threadIdx.x / 32;
    const int lane = (int)threadIdx.x & 31;
    const int nwarps = (int)blockDim.x / 32;
#ifdef N_B
    const int nb = N_B;
#else
    const int nb = N_b;
#endif
    extern __shared__ unsigned char raw[];
    {real} *s_flux = ({real} *)raw;
    {real} *s_dmin = s_flux + (size_t)nb * (size_t)tile_t;
    {real} *s_dmax = s_dmin + tile_t;
    int *s_zb = (int *)(s_dmax + tile_t);
    unsigned char *pdf_raw = (unsigned char *)(s_zb + tile_t);
    unsigned long long pdf_addr = ((unsigned long long)pdf_raw + 7ull) & ~7ull;
    {real} *s_pdf = ({real} *)pdf_addr;

    for (int b = (int)threadIdx.x; b < n_oc * N_z; b += (int)blockDim.x)
        s_pdf[b] = 0.0;
    __syncthreads();

    for (int ts = 0; ts < N_t; ts += tile_t) {{
        int tlen = tile_t;
        if (ts + tlen > N_t) tlen = N_t - ts;
        for (int idx = (int)threadIdx.x; idx < nb * tlen; idx += (int)blockDim.x) {{
            int k = idx / tlen;
            int j = idx - k * tlen;
            s_flux[k * tile_t + j] = flux_bt[(size_t)k * (size_t)N_t + (size_t)(ts + j)];
        }}
        for (int j = (int)threadIdx.x; j < tlen; j += (int)blockDim.x) {{
            int gi = ts + j;
            s_dmin[j] = dm_min[gi];
            s_dmax[j] = dm_max[gi];
            s_zb[j] = z_bin[gi];
        }}
        __syncthreads();

        for (int ow = warp; ow < n_oc; ow += nwarps) {{
            int o = o0 + ow;
            {real} r_inv2[16];
            {real} r_abinv[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) {{
                if (k < nb) {{
                    int ix = o * nb + k;
                    r_inv2[k] = invsabSq[ix];
                    r_abinv[k] = abinvsabSq[ix];
                }}
            }}
            {real} r_base = base[o];
            {real} cmin = chimin_gal[o];
            for (int j = lane; j < tlen; j += 32) {{
                {real} avmago = {lit0};
                {real} avmagt = {lit0};
#ifdef N_B
#pragma unroll
                for (int k = 0; k < N_B; ++k) {{
                    {real} f = s_flux[k * tile_t + j];
                    avmago += f * r_abinv[k];
                    avmagt += f * f * r_inv2[k];
                }}
#else
                for (int k = 0; k < nb; ++k) {{
                    {real} f = s_flux[k * tile_t + j];
                    avmago += f * r_abinv[k];
                    avmagt += f * f * r_inv2[k];
                }}
#endif
                {real} chi2 = r_base;
                if (avmagt > {lit0})
                    chi2 = r_base - (avmago * avmago) / avmagt;
#if USE_PRIORS
                {real} dm = (avmagt > {lit0}) ? (avmago / avmagt) : ({real})(-999.0);
                if (dm > {lit0} && (dm <= s_dmin[j] || dm >= s_dmax[j]))
                    chi2 = HIGH_CHI2;
#endif
                if (chi2 >= HIGH_CHI2) continue;
                {real} dchi = chi2 - cmin;
                if (dchi < {lit0}) dchi = {lit0};
                if (dchi > {dchi_max}) continue;
                {real} prob = {exp}(({real})(-0.5) * dchi);
                int zb = s_zb[j];
                if (zb >= 0 && zb < N_z)
                    atomicAdd(&s_pdf[ow * N_z + zb], prob);
            }}
        }}
        __syncthreads();
    }}

    for (int b = (int)threadIdx.x; b < n_oc * N_z; b += (int)blockDim.x) {{
        int ow = b / N_z;
        int zb = b - ow * N_z;
        pdf[(o0 + ow) * N_z + zb] = s_pdf[b];
    }}
}}
"""


def _get_tiled_kernels(cp, *, use_priors: bool, n_b: int, real: str = "double"):
    unroll = int(n_b) if int(n_b) in _NB_UNROLL else 0
    prec = "f64" if real == "double" else "f32"
    key = ("tile", prec, int(use_priors), unroll, "native_fma_exp")
    if key not in _FUSE_KERNEL_CACHE:
        opts = ["-std=c++11", f"-DUSE_PRIORS={1 if use_priors else 0}"]
        if unroll:
            opts.append(f"-DN_B={unroll}")
        mod = cp.RawModule(code=_fuse_tiled_kernels_src(real), options=tuple(opts))
        _FUSE_KERNEL_CACHE[key] = (
            mod.get_function("fuse_tile_pass1_kernel"),
            mod.get_function("fuse_tile_pass2_kernel"),
        )
    return _FUSE_KERNEL_CACHE[key]


def _set_max_dynamic_smem(kernel, nbytes: int) -> None:
    """Allow >48 KB dynamic shared memory (Cardinal Nz=701 needs ~88 KB)."""
    setter = getattr(kernel, "max_dynamic_shared_size_bytes", None)
    if setter is not None:
        try:
            kernel.max_dynamic_shared_size_bytes = int(nbytes)
            return
        except Exception:
            pass
    try:
        import cupy as cp

        cp.cuda.runtime.funcSetAttribute(
            int(kernel.ptr),
            cp.cuda.runtime.cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(nbytes),
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to raise dynamic shared memory to {nbytes} bytes"
        ) from exc


def _device_lib(cp, tensors: MagLibTensors, dtype=np.float64):
    dtype = np.dtype(dtype).type
    key = (tensors.flux.__array_interface__["data"][0], tensors.flux.shape, dtype)
    if key not in _LIB_CACHE:
        flux = np.ascontiguousarray(tensors.flux, dtype=np.float64)
        fp32_scale = 1.0
        if dtype == np.float32:
            fp32_scale = _fp32_amp_scale(flux)
            flux = fp32_apply_scale(flux, fp32_scale)
        flux = np.ascontiguousarray(flux, dtype=dtype)  # [Nt, Nb]
        types = np.asarray(tensors.types, dtype=np.int32)
        mag0 = np.asarray(tensors.mag0, dtype=dtype)
        z_lib = np.asarray(tensors.z, dtype=dtype)
        z_bin = _z_bin_array(tensors.z, tensors.gridz)
        prior = np.asarray(tensors.prior_lib, dtype=dtype)
        flux_bt = np.ascontiguousarray(flux.T)  # [Nb, Nt]
        gal = types == 0
        flux_gal = np.ascontiguousarray(flux_bt[:, gal])
        mag0_gal = mag0[gal]
        z_gal = z_lib[gal]
        z_bin_gal = z_bin[gal]
        dm_min, dm_max = _mag_abs_dm_bounds(
            mag0, z_lib, types, prior, tensors.funz0, dtype
        )
        _LIB_CACHE[key] = {
            "flux_bt": cp.asarray(flux_bt),
            "types": cp.asarray(types),
            "mag0": cp.asarray(mag0),
            "z_lib": cp.asarray(z_lib),
            "z_bin": cp.asarray(z_bin),
            "dm_min": cp.asarray(dm_min),
            "dm_max": cp.asarray(dm_max),
            "flux_gal": cp.asarray(flux_gal),
            "mag0_gal": cp.asarray(mag0_gal),
            "z_gal": cp.asarray(z_gal),
            "z_bin_gal": cp.asarray(z_bin_gal),
            "dm_min_gal": cp.asarray(dm_min[gal]),
            "dm_max_gal": cp.asarray(dm_max[gal]),
            "n_gal": int(gal.sum()),
            "prior": prior,
            "funz0": dtype(tensors.funz0),
            "gal_on": bool(prior[0] < 0 and prior[1] < 0),
            "qso_on": bool(prior[2] < 0 and prior[3] < 0),
            "n_t": int(flux.shape[0]),
            "n_b": int(flux.shape[1]),
            "n_z": int(np.asarray(tensors.gridz).size),
            "dz": np.asarray(
                np.diff(np.asarray(tensors.gridz, dtype=np.float64)), dtype=np.float64
            ),
            "gridz": np.asarray(tensors.gridz, dtype=np.float64),
            "dtype": dtype,
            "fp32_scale": fp32_scale,
        }
    return _LIB_CACHE[key]


def _normalize_pdf_host(pdf: np.ndarray, dz: np.ndarray) -> np.ndarray:
    pdf = np.asarray(pdf, dtype=np.float64)
    if pdf.shape[1] > 1 and dz.size > 0:
        somme = np.sum((pdf[:, :-1] + pdf[:, 1:]) * 0.5 * dz[None, :], axis=1)
        scale = np.ones_like(somme)
        np.divide(1.0, somme, out=scale, where=somme > 0)
        pdf = pdf * scale[:, None]
    return pdf


def process_template_tiles_cuda(
    mags: np.ndarray,
    magerrs: np.ndarray,
    tensors: MagLibTensors,
    context: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    dtype=np.float64,
    return_ancil: bool = True,
    tile_t: int | None = None,
) -> tuple:
    """Two-pass fused CUDA template streaming (BAY_ZG).

    Default launch is one block per object. Set ``LEPHARE_ACCEL_CUDA_TILE_OBJS``
    to enable object-chunk × template-tile SMEM reuse (1=auto, N>=2=chunk size).
    ``tile_t`` is the legacy chi²-materializing tile size and is ignored;
    template tile length is ``LEPHARE_ACCEL_CUDA_TILE_T`` (default 256).
    Default ``dtype=float64`` matches C++. ``float32`` is a throughput proxy.
    """
    del tile_t  # fused path does not materialize chi2[N,T]
    dtype = np.dtype(dtype).type
    if dtype not in (np.float64, np.float32):
        raise ValueError(f"fused CUDA template path requires float64 or float32, got {dtype}")
    cp = _require_cupy()
    lib = _device_lib(cp, tensors, dtype=dtype)
    real = "double" if dtype == np.float64 else "float"
    cp_t = cp.float64 if dtype == np.float64 else cp.float32
    nbytes = 8 if dtype == np.float64 else 4

    n_obj, n_b = mags.shape
    n_t = lib["n_t"]
    n_z = lib["n_z"]
    n_gal = lib["n_gal"]
    if n_b > _MAX_NB:
        raise ValueError(f"fused CUDA kernel supports N_b<={_MAX_NB}, got {n_b}")
    if n_z > _MAX_NZ_SHARED:
        raise ValueError(f"fused CUDA kernel supports N_z<={_MAX_NZ_SHARED}, got {n_z}")

    use_priors = bool(lib["gal_on"] or lib["qso_on"])
    tiled = _choose_tiled_launch(n_z, n_b, nbytes)
    if tiled is not None:
        pass1_k, pass2_k = _get_tiled_kernels(
            cp, use_priors=use_priors, n_b=n_b, real=real
        )
    else:
        pass1_k, pass2_k = _get_fuse_kernels(
            cp, use_priors=use_priors, n_b=n_b, real=real
        )

    from .cuda_backend import prepare_photometry_cuda_device

    d_ab, d_sab, d_bus, d_busul, d_nbul = prepare_photometry_cuda_device(
        mags, magerrs, tensors, context=context, offsets=offsets, dtype=dtype
    )
    if dtype == np.float32:
        scale = cp.asarray(lib["fp32_scale"], dtype=cp_t)
        d_ab = d_ab * scale
        d_sab = d_sab * scale
    d_inv = cp.where(d_bus > 0, d_bus / d_sab, cp.asarray(0, dtype=cp_t))
    d_inv2 = d_inv * d_inv
    d_abinv = d_ab * d_inv2
    d_base = cp.sum((d_ab * d_inv) ** 2, axis=1)
    nbul = cp.asnumpy(d_nbul)

    d_chimin = cp.empty((n_obj, 3), dtype=cp_t)
    d_indmin = cp.empty((n_obj, 3), dtype=cp.int64)
    d_min_zg = cp.empty((n_obj, n_z), dtype=cp_t)
    d_min_zq = cp.empty((n_obj, n_z), dtype=cp_t)
    d_pdf = cp.empty((n_obj, n_z), dtype=cp_t)

    threads = _THREADS
    if tiled is not None:
        n_oc, ttile, smem_p1, smem_p2 = tiled
        grid = (int((n_obj + n_oc - 1) // n_oc),)
        _set_max_dynamic_smem(pass1_k, smem_p1)
        _set_max_dynamic_smem(pass2_k, smem_p2)
        pass1_k(
            grid,
            (threads,),
            (
                lib["flux_bt"],
                d_inv2,
                d_abinv,
                d_base,
                lib["types"],
                lib["dm_min"],
                lib["dm_max"],
                lib["z_bin"],
                d_chimin,
                d_indmin,
                d_min_zg,
                d_min_zq,
                np.int32(n_obj),
                np.int32(n_t),
                np.int32(n_b),
                np.int32(n_z),
                np.int32(n_oc),
                np.int32(ttile),
            ),
            shared_mem=smem_p1,
        )
        d_chimin_gal = cp.ascontiguousarray(d_chimin[:, 0])
        n_t2 = n_gal
        pass2_k(
            grid,
            (threads,),
            (
                lib["flux_gal"],
                d_inv2,
                d_abinv,
                d_base,
                lib["dm_min_gal"],
                lib["dm_max_gal"],
                lib["z_bin_gal"],
                d_chimin_gal,
                d_pdf,
                np.int32(n_obj),
                np.int32(n_t2),
                np.int32(n_b),
                np.int32(n_z),
                np.int32(n_oc),
                np.int32(ttile),
            ),
            shared_mem=smem_p2,
        )
    else:
        nwarps = threads // 32
        smem_p1 = nwarps * n_z * 2 * nbytes + nwarps * 3 * nbytes + nwarps * 3 * 8
        smem_p2 = nwarps * n_z * nbytes
        _set_max_dynamic_smem(pass1_k, smem_p1)
        _set_max_dynamic_smem(pass2_k, smem_p2)
        pass1_k(
            (n_obj,),
            (threads,),
            (
                lib["flux_bt"],
                d_inv2,
                d_abinv,
                d_base,
                lib["types"],
                lib["dm_min"],
                lib["dm_max"],
                lib["z_bin"],
                d_chimin,
                d_indmin,
                d_min_zg,
                d_min_zq,
                np.int32(n_obj),
                np.int32(n_t),
                np.int32(n_b),
                np.int32(n_z),
            ),
            shared_mem=smem_p1,
        )
        d_chimin_gal = cp.ascontiguousarray(d_chimin[:, 0])
        n_t2 = n_gal
        pass2_k(
            (n_obj,),
            (threads,),
            (
                lib["flux_gal"],
                d_inv2,
                d_abinv,
                d_base,
                lib["dm_min_gal"],
                lib["dm_max_gal"],
                lib["z_bin_gal"],
                d_chimin_gal,
                d_pdf,
                np.int32(n_obj),
                np.int32(n_t2),
                np.int32(n_b),
                np.int32(n_z),
            ),
            shared_mem=smem_p2,
        )

    pdfs = _normalize_pdf_host(np.asarray(cp.asnumpy(d_pdf), dtype=np.float64), lib["dz"])
    chimin_np = np.asarray(cp.asnumpy(d_chimin), dtype=np.float64)
    indmin_np = cp.asnumpy(d_indmin)
    min_zg_np = np.asarray(cp.asnumpy(d_min_zg), dtype=np.float64)
    min_zq_np = np.asarray(cp.asnumpy(d_min_zq), dtype=np.float64)

    threshold = float(tensors.rm_discrepant)
    improved = np.min(chimin_np, axis=1)
    easy_mask = (nbul == 0) & (improved <= threshold)
    if dtype == np.float32:
        # Throughput proxy only: skip per-object RM_DISCREPANT (would dominate).
        easy_mask = np.ones(n_obj, dtype=bool)
    hard_idxs = np.flatnonzero(~easy_mask)

    ancil = empty_ancil(n_obj)
    easy_idxs = np.flatnonzero(easy_mask)
    if easy_idxs.size:
        merge_ancil(
            ancil,
            bestfit_ancil_from_type_mins(
                indmin_np[easy_idxs],
                chimin_np[easy_idxs],
                tensors,
                min_zg=min_zg_np[easy_idxs],
                min_zq=min_zq_np[easy_idxs],
            ),
            easy_idxs,
        )

    if hard_idxs.size:
        from .cuda_backend import _device_flux, _type_mins, chi2_cuda

        ab = cp.asnumpy(d_ab)
        sab = cp.asnumpy(d_sab)
        busnorma = cp.asnumpy(d_bus)
        busul = cp.asnumpy(d_busul)
        d_flux = _device_flux(cp, tensors.flux, dtype)
        types = tensors.types
        for i in hard_idxs:
            i = int(i)
            if int(nbul[i]) > 0:
                c2, _, _, _ = rm_discrepant(
                    ab[i], sab[i], busnorma[i], busul[i], int(nbul[i]), tensors
                )
            else:
                bus = busnorma[i].copy()
                c2, _ = chi2_cuda(
                    tensors.flux,
                    ab[i],
                    sab[i],
                    bus,
                    types,
                    tensors.mag0,
                    tensors.z,
                    tensors.prior_lib,
                    tensors.funz0,
                    dtype=dtype,
                    d_flux=d_flux,
                )
                improved_i = _type_mins(c2, types)
                nb_removed = 0
                while improved_i > threshold and nb_removed < 2:
                    fl_dis = -1
                    best = improved_i
                    best_c2 = c2
                    for k in range(n_b):
                        if bus[k] == 1:
                            bus[k] = 0
                            c2k, _ = chi2_cuda(
                                tensors.flux,
                                ab[i],
                                sab[i],
                                bus,
                                types,
                                tensors.mag0,
                                tensors.z,
                                tensors.prior_lib,
                                tensors.funz0,
                                dtype=dtype,
                                d_flux=d_flux,
                            )
                            newmin = _type_mins(c2k, types)
                            if best > newmin:
                                best = newmin
                                fl_dis = k
                                best_c2 = c2k
                            bus[k] = 1
                    if fl_dis >= 0:
                        bus[fl_dis] = 0
                        c2 = best_c2
                        improved_i = best
                        nb_removed += 1
                    else:
                        break
            pdfs[i] = bay_zg_pdf(c2, tensors)
            merge_ancil(ancil, bestfit_ancil_from_chi2(c2, tensors), np.asarray([i]))

    zgrid = lib["gridz"]
    zmode, zmean = pdf_zmode_zmean(pdfs, zgrid)
    if not return_ancil:
        return pdfs, zgrid
    return pdfs, zgrid, zmode, zmean, ancil
