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
        raise ImportError("backend requires CuPy. Install with: pip install 'lephare[cuda]'") from exc
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


# Float min via native integer atomics: IEEE bit patterns order like signed ints for
# val >= 0 and reversed (as unsigned) for val < 0. A CAS loop here made pass1 ~12x
# slower on large BC03 libraries even though only a few percent of calls update.
_ATOMIC_MIN_SRC = {
    "float": """
__device__ __forceinline__ float atomicMinReal(float *addr, float val) {
    return (val >= 0.0f)
        ? __int_as_float(atomicMin((int *)addr, __float_as_int(val)))
        : __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(val)));
}
""",
    "double": """
__device__ __forceinline__ double atomicMinReal(double *addr, double val) {
    return (val >= 0.0)
        ? __longlong_as_double(atomicMin((long long *)addr, __double_as_longlong(val)))
        : __longlong_as_double((long long)atomicMax(
              (unsigned long long *)addr, (unsigned long long)__double_as_longlong(val)));
}
""",
}


def _fuse_kernels_src(real: str = "double") -> str:
    """Block-per-object fused pass1 / pass2; flux is ``[N_b, N_t]``.

    ``real`` is ``double`` (default, accuracy path) or ``float`` (throughput
    proxy on consumer GPUs with 1:64 FP64).
    """
    if real not in ("double", "float"):
        raise ValueError(real)
    high = f"{HIGH_CHI2:.1f}"
    if real == "float":
        exp, lit0 = "expf", "0.0f"
        dchi_max = "80.0f"
        atomic_min = _ATOMIC_MIN_SRC["float"]
    else:
        exp, lit0 = "exp", "0.0"
        dchi_max = "700.0"
        atomic_min = _ATOMIC_MIN_SRC["double"]
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
        atomic_min = _ATOMIC_MIN_SRC["float"]
    else:
        exp, lit0, dchi_max = "exp", "0.0", "700.0"
        atomic_min = _ATOMIC_MIN_SRC["double"]
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
        raise RuntimeError(f"failed to raise dynamic shared memory to {nbytes} bytes") from exc


def _device_lib(cp, tensors: MagLibTensors, dtype=np.float64):
    dtype = np.dtype(dtype).type
    from .maglib import cached_by_identity

    def build():
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
        dm_min, dm_max = _mag_abs_dm_bounds(mag0, z_lib, types, prior, tensors.funz0, dtype)
        return {
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
            "dz": np.asarray(np.diff(np.asarray(tensors.gridz, dtype=np.float64)), dtype=np.float64),
            "gridz": np.asarray(tensors.gridz, dtype=np.float64),
            "dtype": dtype,
            "fp32_scale": fp32_scale,
        }

    return cached_by_identity(_LIB_CACHE, tensors, (np.dtype(dtype).str,), build)


def _normalize_pdf_host(pdf: np.ndarray, dz: np.ndarray) -> np.ndarray:
    pdf = np.asarray(pdf, dtype=np.float64)
    if pdf.shape[1] > 1 and dz.size > 0:
        somme = np.sum((pdf[:, :-1] + pdf[:, 1:]) * 0.5 * dz[None, :], axis=1)
        scale = np.ones_like(somme)
        np.divide(1.0, somme, out=scale, where=somme > 0)
        pdf = pdf * scale[:, None]
    return pdf


def _fused_obj_bytes(n_b: int, n_z: int, nbytes: int) -> int:
    """Device bytes per object on the fused path: photometry prep (~16 band vectors),
    per-type minima / indices, MIN_ZG, MIN_ZQ and the PDF, with 1.5x slack."""
    return int(1.5 * (3 * n_z * nbytes + 16 * n_b * 8 + 3 * (nbytes + 8)))


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
    """:func:`_process_template_tiles_once` over VRAM-sized object chunks.

    The chunk is the largest that fits ``LEPHARE_ACCEL_CHI2_MEM_FRACTION`` of free VRAM
    (one launch for the whole batch when it fits); an out-of-memory error halves it.
    """
    from .gpu_util import run_with_oom_backoff, suggest_obj_chunk

    mags = np.asarray(mags)
    magerrs = np.asarray(magerrs)
    n_obj, n_b = mags.shape
    if n_obj == 0:
        return _process_template_tiles_once(
            mags,
            magerrs,
            tensors,
            context=context,
            offsets=offsets,
            dtype=dtype,
            return_ancil=return_ancil,
        )
    n_z = len(tensors.gridz)
    nbytes = 8 if np.dtype(dtype) == np.float64 else 4
    from .physical import fused_extra_obj_bytes, fused_request

    per_obj = _fused_obj_bytes(n_b, n_z, nbytes)
    if fused_request() is not None:
        per_obj += fused_extra_obj_bytes(dtype)
    chunk = suggest_obj_chunk(per_obj)
    if n_obj <= chunk:
        chunk = max(n_obj, 1)
    ctx = None if context is None else np.asarray(context)
    per_obj_ctx = ctx is not None and ctx.ndim > 0 and ctx.shape[0] == n_obj
    parts: dict = {}

    def run(s, e):
        c = ctx[s:e] if per_obj_ctx else ctx
        res = _process_template_tiles_once(
            mags[s:e],
            magerrs[s:e],
            tensors,
            context=c,
            offsets=offsets,
            dtype=dtype,
            return_ancil=True,
        )
        if (s, e) == (0, n_obj):
            parts["whole"] = res
            return
        pdfs, zgrid, zmode, zmean, ancil = res
        if "pdfs" not in parts:
            parts["pdfs"] = np.empty((n_obj, pdfs.shape[1]), dtype=pdfs.dtype)
            parts["zmode"] = np.empty(n_obj)
            parts["zmean"] = np.empty(n_obj)
            parts["ancil"] = {
                k: np.empty((n_obj,) + np.shape(v)[1:], dtype=np.float64) for k, v in ancil.items()
            }
            parts["zgrid"] = zgrid
        parts["pdfs"][s:e] = pdfs
        parts["zmode"][s:e] = zmode
        parts["zmean"][s:e] = zmean
        for k, v in ancil.items():
            parts["ancil"][k][s:e] = v

    run_with_oom_backoff(run, n_obj, chunk, min_chunk=256, tag="fused")
    if "whole" in parts:
        pdfs, zgrid, zmode, zmean, ancil = parts["whole"]
    else:
        pdfs, zgrid = parts["pdfs"], parts["zgrid"]
        zmode, zmean, ancil = parts["zmode"], parts["zmean"], parts["ancil"]
    if not return_ancil:
        return pdfs, zgrid
    return pdfs, zgrid, zmode, zmean, ancil


def _process_template_tiles_once(
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
    from .physical import fused_request

    # the fused histograms run in the one-block-per-object pass2 (after either pass1)
    phys_req = fused_request()
    if phys_req is not None and phys_req["tensors"] is not tensors:
        phys_req = None
    phys_props = None
    if tiled is not None:
        pass1_k, pass2_k = _get_tiled_kernels(cp, use_priors=use_priors, n_b=n_b, real=real)
    else:
        pass1_k, pass2_k = _get_fuse_kernels(cp, use_priors=use_priors, n_b=n_b, real=real)

    from .cuda_backend import prepare_photometry_cuda_device

    d_ab, d_sab, d_bus, d_busul, d_nbul = prepare_photometry_cuda_device(
        mags, magerrs, tensors, context=context, offsets=offsets, dtype=dtype
    )
    if dtype == np.float32:
        scale = cp.asarray(lib["fp32_scale"], dtype=cp_t)
        d_ab = d_ab * scale
        d_sab = d_sab * scale
    nbul = cp.asnumpy(d_nbul)
    threads = _THREADS

    def launch(rows, d_bus_v, want_pdf: bool) -> dict:
        """pass1 (+ pass2, or pass2 with the fused galaxy-property histograms) for object
        ``rows`` (device indices; None = all) fitted with the band mask ``d_bus_v``."""
        ab_v = d_ab if rows is None else d_ab[rows]
        sab_v = d_sab if rows is None else d_sab[rows]
        inv = cp.where(d_bus_v > 0, d_bus_v / sab_v, cp.asarray(0, dtype=cp_t))
        inv2 = cp.ascontiguousarray(inv * inv)
        abinv = cp.ascontiguousarray(ab_v * inv2)
        base = cp.ascontiguousarray(cp.sum((ab_v * inv) ** 2, axis=1))
        m = int(base.shape[0])
        out = {
            "chimin": cp.empty((m, 3), dtype=cp_t),
            "indmin": cp.empty((m, 3), dtype=cp.int64),
            "min_zg": cp.empty((m, n_z), dtype=cp_t),
            "min_zq": cp.empty((m, n_z), dtype=cp_t),
            "pdf": None,
            "props": None,
        }
        p1_args = (
            lib["flux_bt"],
            inv2,
            abinv,
            base,
            lib["types"],
            lib["dm_min"],
            lib["dm_max"],
            lib["z_bin"],
            out["chimin"],
            out["indmin"],
            out["min_zg"],
            out["min_zq"],
            np.int32(m),
            np.int32(n_t),
            np.int32(n_b),
            np.int32(n_z),
        )
        if tiled is not None:
            n_oc, ttile, smem_p1, smem_p2 = tiled
            grid = (int((m + n_oc - 1) // n_oc),)
            _set_max_dynamic_smem(pass1_k, smem_p1)
            pass1_k(grid, (threads,), p1_args + (np.int32(n_oc), np.int32(ttile)), shared_mem=smem_p1)
        else:
            nwarps = threads // 32
            smem_p1 = nwarps * n_z * 2 * nbytes + nwarps * 3 * nbytes + nwarps * 3 * 8
            smem_p2 = nwarps * n_z * nbytes
            _set_max_dynamic_smem(pass1_k, smem_p1)
            pass1_k((m,), (threads,), p1_args, shared_mem=smem_p1)
        if not want_pdf:
            return out
        out["pdf"] = cp.empty((m, n_z), dtype=cp_t)
        d_cmin = cp.ascontiguousarray(out["chimin"][:, 0])
        p2_args = (
            lib["flux_gal"],
            inv2,
            abinv,
            base,
            lib["dm_min_gal"],
            lib["dm_max_gal"],
            lib["z_bin_gal"],
            d_cmin,
            out["pdf"],
            np.int32(m),
            np.int32(n_gal),
            np.int32(n_b),
            np.int32(n_z),
        )
        if phys_req is not None:
            from .physical import fused_pass2_phys

            unroll = int(n_b) if int(n_b) in _NB_UNROLL else 0
            out["props"] = fused_pass2_phys(
                cp,
                phys_req,
                lib,
                inv2,
                abinv,
                base,
                d_cmin,
                out["pdf"],
                n_b,
                n_z,
                dtype,
                use_priors,
                threads,
                unroll,
            )
        elif tiled is not None:
            _set_max_dynamic_smem(pass2_k, smem_p2)
            pass2_k(grid, (threads,), p2_args + (np.int32(n_oc), np.int32(ttile)), shared_mem=smem_p2)
        else:
            _set_max_dynamic_smem(pass2_k, smem_p2)
            pass2_k((m,), (threads,), p2_args, shared_mem=smem_p2)
        return out

    res = launch(None, d_bus, True)
    phys_props = res["props"]
    pdfs = _normalize_pdf_host(np.asarray(cp.asnumpy(res["pdf"]), dtype=np.float64), lib["dz"])
    chimin_np = np.asarray(cp.asnumpy(res["chimin"]), dtype=np.float64)
    indmin_np = cp.asnumpy(res["indmin"])
    min_zg_np = np.asarray(cp.asnumpy(res["min_zg"]), dtype=np.float64)
    min_zq_np = np.asarray(cp.asnumpy(res["min_zq"]), dtype=np.float64)
    del res

    threshold = float(tensors.rm_discrepant)
    improved = np.min(chimin_np, axis=1)
    easy_mask = (nbul == 0) & (improved <= threshold)
    if dtype == np.float32:
        # Throughput proxy only: skip per-object RM_DISCREPANT (would dominate).
        easy_mask = np.ones(n_obj, dtype=bool)
    hard_idxs = np.flatnonzero(~easy_mask)
    from .gpu_util import STATS

    st = STATS.setdefault("rm_discrepant", {"n_obj": 0, "n_hard": 0, "n_upper_limit": 0, "t_host_s": 0.0})
    st["n_obj"] += int(n_obj)
    st["n_hard"] += int(hard_idxs.size)
    st["n_upper_limit"] += int(np.count_nonzero(nbul[hard_idxs] > 0)) if hard_idxs.size else 0

    ancil = empty_ancil(n_obj, n_b)
    busnorma = np.asarray(cp.asnumpy(d_bus), dtype=np.float64)
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
                busnorma=busnorma[easy_idxs],
            ),
            easy_idxs,
        )

    ul_idxs = hard_idxs[nbul[hard_idxs] > 0] if hard_idxs.size else hard_idxs
    disc_idxs = hard_idxs[nbul[hard_idxs] == 0] if hard_idxs.size else hard_idxs
    if disc_idxs.size:
        # RM_DISCREPANT, batched: each round fits every (object, one band removed) trial
        # in one pass1 launch; an object drops the band giving the lowest type-min chi2
        # (first such band, strict improvement), at most twice, while above threshold.
        bus = busnorma[disc_idxs].copy()
        best = improved[disc_idxs].copy()
        n_removed = np.zeros(disc_idxs.size, dtype=np.int64)
        active = np.ones(disc_idxs.size, dtype=bool)
        while True:
            act = np.flatnonzero(active & (best > threshold) & (n_removed < 2))
            if act.size == 0:
                break
            obj, band = np.nonzero(bus[act] == 1)  # row-major: bands ascending per object
            if obj.size == 0:
                break
            trial = bus[act[obj]].copy()
            trial[np.arange(obj.size), band] = 0.0
            r = launch(cp.asarray(disc_idxs[act[obj]]), cp.asarray(trial, dtype=cp_t), False)
            newmin = np.min(np.asarray(cp.asnumpy(r["chimin"]), dtype=np.float64), axis=1)
            del r
            starts = np.searchsorted(obj, np.arange(act.size))
            ends = np.searchsorted(obj, np.arange(act.size), side="right")
            for j in range(act.size):
                if ends[j] == starts[j]:
                    active[act[j]] = False
                    continue
                seg = newmin[starts[j] : ends[j]]
                k = int(np.argmin(seg))
                if seg[k] < best[act[j]]:
                    bus[act[j], band[starts[j] + k]] = 0.0
                    best[act[j]] = seg[k]
                    n_removed[act[j]] += 1
                else:
                    active[act[j]] = False
        # final fit of these objects with their remaining bands (p(z), winners, histograms)
        r = launch(cp.asarray(disc_idxs), cp.asarray(bus, dtype=cp_t), True)
        pdfs[disc_idxs] = _normalize_pdf_host(np.asarray(cp.asnumpy(r["pdf"]), dtype=np.float64), lib["dz"])
        merge_ancil(
            ancil,
            bestfit_ancil_from_type_mins(
                cp.asnumpy(r["indmin"]),
                np.asarray(cp.asnumpy(r["chimin"]), dtype=np.float64),
                tensors,
                min_zg=np.asarray(cp.asnumpy(r["min_zg"]), dtype=np.float64),
                min_zq=np.asarray(cp.asnumpy(r["min_zq"]), dtype=np.float64),
                busnorma=bus,
            ),
            disc_idxs,
        )
        if phys_props is not None and r["props"] is not None:
            for k, v in r["props"].items():
                phys_props[k][disc_idxs] = v
        st["n_band_removed"] = st.get("n_band_removed", 0) + int(n_removed.sum())
        del r

    if ul_idxs.size:
        # upper-limit objects keep the host RM_DISCREPANT (rare)
        ab = cp.asnumpy(d_ab)
        sab = cp.asnumpy(d_sab)
        busul = cp.asnumpy(d_busul)
        for i in ul_idxs:
            i = int(i)
            c2, _, _, bus_i = rm_discrepant(ab[i], sab[i], busnorma[i], busul[i], int(nbul[i]), tensors)
            pdfs[i] = bay_zg_pdf(c2, tensors)
            merge_ancil(ancil, bestfit_ancil_from_chi2(c2, tensors, busnorma=bus_i), np.asarray([i]))

    if phys_props is not None:
        # rows refit by RM_DISCREPANT above have different bands: galaxy_properties redoes them
        done = np.ones(n_obj, dtype=np.float64)
        done[ul_idxs] = 0.0
        ancil["_PHYSDONE"] = done
        for k, v in phys_props.items():
            ancil[f"_PHYS:{k}"] = np.asarray(v, dtype=np.float64)
    zgrid = lib["gridz"]
    zmode, zmean = pdf_zmode_zmean(pdfs, zgrid)
    if not return_ancil:
        return pdfs, zgrid
    return pdfs, zgrid, zmode, zmean, ancil
