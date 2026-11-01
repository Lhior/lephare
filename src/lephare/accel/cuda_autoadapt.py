"""CUDA Z-fixed χ² for AUTO_ADAPT (CSR z-slice, block per object)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .autoadapt import (
    build_zslice_csr,
    calculate_offsets,
    indexz,
    zfix_type_mins,
)
from .kernels_ref import HIGH_CHI2
from .maglib import MagLibTensors

_THREADS = 256
_NB_UNROLL = (6, 9)
_KERNEL_CACHE: dict = {}


def _require_cupy():
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CUDA AUTO_ADAPT requires CuPy. Install with: pip install 'lephare[cuda]'") from exc
    return cp


def _kernel_src() -> str:
    high = f"{HIGH_CHI2:.1f}"
    return f"""
#define HIGH_CHI2 ({high})

__device__ __forceinline__ void min_with_idx_dm(
    double &c, long long &idx, double &dm,
    double oc, long long oidx, double odm
) {{
    if (oc < c || (oc == c && oidx >= 0 && (idx < 0 || oidx < idx))) {{
        c = oc;
        idx = oidx;
        dm = odm;
    }}
}}

extern "C" __global__
void zfix_chi2_kernel(
    const double* __restrict__ flux_bt,
    const int* __restrict__ types,
    const double* __restrict__ mag0,
    const double* __restrict__ z_lib,
    const int* __restrict__ csr_off,
    const int* __restrict__ csr_idx,
    const int* __restrict__ z_bin,
    const double* __restrict__ invsab,
    const double* __restrict__ invsabSq,
    const double* __restrict__ s2n,
    const double* __restrict__ abinvsabSq,
    const double* __restrict__ ab,
    const double* __restrict__ busul,
    const int* __restrict__ nbul,
    double prior0, double prior1, double prior2, double prior3,
    int mabs_gal, int mabs_agn, double funz0,
    long long* __restrict__ indmin,
    double* __restrict__ dm_gal,
    int N_obj, int N_t, int N_b
) {{
    int o = (int)blockIdx.x;
    if (o >= N_obj) return;

    const int nwarps = (int)blockDim.x / 32;
    const int warp = (int)threadIdx.x / 32;
    const int lane = (int)threadIdx.x & 31;

    extern __shared__ double smem[];
    double *s_chi = smem;
    double *s_dm = s_chi + nwarps * 3;
    long long *s_idx = (long long *)(s_dm + nwarps * 3);

#ifdef N_B
    const int nb = N_B;
#else
    const int nb = N_b;
#endif
    double r_inv2[16];
    double r_abinv[16];
    double r_inv[16];
    double r_s2n[16];
    double r_ab[16];
    double r_ul[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {{
        if (k < nb) {{
            int idx = o * nb + k;
            r_inv2[k] = invsabSq[idx];
            r_abinv[k] = abinvsabSq[idx];
            r_inv[k] = invsab[idx];
            r_s2n[k] = s2n[idx];
            r_ab[k] = ab[idx];
            r_ul[k] = busul[idx];
        }}
    }}
    int nb_ul = nbul[o];
    int zb = z_bin[o];
    int start = csr_off[zb];
    int end = csr_off[zb + 1];
    int nslice = end - start;

    if ((int)threadIdx.x < nwarps * 3) {{
        s_chi[threadIdx.x] = HIGH_CHI2;
        s_dm[threadIdx.x] = -999.0;
        s_idx[threadIdx.x] = (long long)(-99);
    }}
    __syncthreads();

    double p_chimin[3] = {{HIGH_CHI2, HIGH_CHI2, HIGH_CHI2}};
    long long p_indmin[3] = {{-99, -99, -99}};
    double p_dm[3] = {{-999.0, -999.0, -999.0}};

    for (int j = (int)threadIdx.x; j < nslice; j += (int)blockDim.x) {{
        int i = csr_idx[start + j];
        int typ = types[i];
        if (typ < 0 || typ > 2) continue;
        double avmago = 0.0;
        double avmagt = 0.0;
#ifdef N_B
#pragma unroll
        for (int k = 0; k < N_B; ++k) {{
            double f = flux_bt[k * N_t + i];
            avmago += f * r_abinv[k];
            avmagt += f * f * r_inv2[k];
        }}
#else
        for (int k = 0; k < nb; ++k) {{
            double f = flux_bt[k * N_t + i];
            avmago += f * r_abinv[k];
            avmagt += f * f * r_inv2[k];
        }}
#endif
        double dmloc = (avmagt > 0.0) ? (avmago / avmagt) : -999.0;
        double chi2 = 0.0;
#ifdef N_B
#pragma unroll
        for (int k = 0; k < N_B; ++k) {{
            double f = flux_bt[k * N_t + i];
            double inter = r_s2n[k] - dmloc * f * r_inv[k];
            chi2 += inter * inter;
        }}
#else
        for (int k = 0; k < nb; ++k) {{
            double f = flux_bt[k * N_t + i];
            double inter = r_s2n[k] - dmloc * f * r_inv[k];
            chi2 += inter * inter;
        }}
#endif
        if (nb_ul > 0) {{
            for (int k = 0; k < nb; ++k) {{
                if ((dmloc * r_ul[k] * flux_bt[k * N_t + i]) > r_ab[k] && r_inv[k] != 0.0)
                    chi2 = HIGH_CHI2;
            }}
        }}
        if ((mabs_gal && typ == 0) || (mabs_agn && typ == 1)) {{
            double abs_mag = mag0[i] - 2.5 * log10(fmax(dmloc, 1e-300));
            if (z_lib[i] < 1e-10) abs_mag -= funz0;
            if (typ == 0 && (abs_mag <= prior0 || abs_mag >= prior1))
                chi2 = HIGH_CHI2;
            if (typ == 1 && (abs_mag <= prior2 || abs_mag >= prior3))
                chi2 = HIGH_CHI2;
        }}
        if (chi2 < HIGH_CHI2)
            min_with_idx_dm(p_chimin[typ], p_indmin[typ], p_dm[typ], chi2, (long long)i, dmloc);
    }}

#pragma unroll
    for (int t = 0; t < 3; ++t) {{
        for (int off = 16; off > 0; off >>= 1) {{
            double oc = __shfl_xor_sync(0xffffffff, p_chimin[t], off);
            long long oi = __shfl_xor_sync(0xffffffff, p_indmin[t], off);
            double od = __shfl_xor_sync(0xffffffff, p_dm[t], off);
            min_with_idx_dm(p_chimin[t], p_indmin[t], p_dm[t], oc, oi, od);
        }}
        if (lane == 0) {{
            s_chi[warp * 3 + t] = p_chimin[t];
            s_idx[warp * 3 + t] = p_indmin[t];
            s_dm[warp * 3 + t] = p_dm[t];
        }}
    }}
    __syncthreads();

    if (threadIdx.x == 0) {{
        double bchi[3] = {{HIGH_CHI2, HIGH_CHI2, HIGH_CHI2}};
        long long bidx[3] = {{-99, -99, -99}};
        double bdm[3] = {{-999.0, -999.0, -999.0}};
        for (int w = 0; w < nwarps; ++w) {{
#pragma unroll
            for (int t = 0; t < 3; ++t)
                min_with_idx_dm(
                    bchi[t], bidx[t], bdm[t],
                    s_chi[w * 3 + t], s_idx[w * 3 + t], s_dm[w * 3 + t]
                );
        }}
#pragma unroll
        for (int t = 0; t < 3; ++t)
            indmin[o * 3 + t] = bidx[t];
        dm_gal[o] = bdm[0];
    }}
}}
"""


def _get_kernel(cp, n_b: int):
    unroll = int(n_b) if int(n_b) in _NB_UNROLL else 0
    key = ("zfix", unroll)
    if key not in _KERNEL_CACHE:
        opts = ["-std=c++11"]
        if unroll:
            opts.append(f"-DN_B={unroll}")
        mod = cp.RawModule(code=_kernel_src(), options=tuple(opts))
        _KERNEL_CACHE[key] = mod.get_function("zfix_chi2_kernel")
    return _KERNEL_CACHE[key]


class _ZFixCuda:
    def __init__(self, tensors: MagLibTensors):
        self.cp = _require_cupy()
        cp = self.cp
        self.tensors = tensors
        flux = np.asarray(tensors.flux, dtype=np.float64)
        self.n_t, self.n_b = int(flux.shape[0]), int(flux.shape[1])
        flux_bt = np.ascontiguousarray(flux.T)
        self.d_flux = cp.asarray(flux_bt)
        self.d_types = cp.asarray(np.asarray(tensors.types, dtype=np.int32))
        self.d_mag0 = cp.asarray(np.asarray(tensors.mag0, dtype=np.float64))
        self.d_z = cp.asarray(np.asarray(tensors.z, dtype=np.float64))
        csr_off, csr_idx = build_zslice_csr(tensors)
        self.csr_off = csr_off
        self.csr_idx = csr_idx
        self.d_off = cp.asarray(csr_off)
        self.d_idx = cp.asarray(csr_idx)
        prior = np.asarray(tensors.prior_lib, dtype=np.float64)
        self.prior = prior
        self.mabs_gal = int(prior[0] < 0 and prior[1] < 0)
        self.mabs_agn = int(prior[2] < 0 and prior[3] < 0)
        self.funz0 = float(tensors.funz0)
        self.kernel = _get_kernel(cp, self.n_b)
        nwarps = _THREADS // 32
        self.smem = nwarps * 3 * 8 + nwarps * 3 * 8 + nwarps * 3 * 8

    def __call__(self, ab, sab, busnorma, busul, nbul, zs):
        nbul_np = np.asarray(nbul, dtype=np.int32)
        host = np.flatnonzero(nbul_np > 0)
        # Upper-limit objects: host RM_DISCREPANT on the z-slice.
        indmin, dm_gal = self._launch(ab, sab, busnorma, busul, nbul_np, zs)
        if host.size:
            h_ind, h_dm = zfix_type_mins(
                ab[host],
                sab[host],
                busnorma[host],
                busul[host],
                nbul_np[host],
                self.tensors,
                np.asarray(zs, dtype=np.float64)[host],
                self.csr_off,
                self.csr_idx,
            )
            indmin[host] = h_ind
            dm_gal[host] = h_dm
        return indmin, dm_gal

    def _launch(self, ab, sab, busnorma, busul, nbul, zs):
        cp = self.cp
        n_obj = int(ab.shape[0])
        n_b = self.n_b
        invsab = np.where(busnorma > 0, busnorma / sab, 0.0)
        invsab_sq = invsab * invsab
        s2n = ab * invsab
        abinv = ab * invsab_sq
        zb = indexz(np.asarray(zs, dtype=np.float64), self.tensors.gridz).astype(np.int32)

        d_inv = cp.asarray(np.ascontiguousarray(invsab))
        d_inv2 = cp.asarray(np.ascontiguousarray(invsab_sq))
        d_s2n = cp.asarray(np.ascontiguousarray(s2n))
        d_abinv = cp.asarray(np.ascontiguousarray(abinv))
        d_ab = cp.asarray(np.ascontiguousarray(ab))
        d_ul = cp.asarray(np.ascontiguousarray(busul))
        d_nbul = cp.asarray(nbul)
        d_zb = cp.asarray(zb)
        d_ind = cp.empty((n_obj, 3), dtype=cp.int64)
        d_dm = cp.empty(n_obj, dtype=cp.float64)

        self.kernel(
            (n_obj,),
            (_THREADS,),
            (
                self.d_flux,
                self.d_types,
                self.d_mag0,
                self.d_z,
                self.d_off,
                self.d_idx,
                d_zb,
                d_inv,
                d_inv2,
                d_s2n,
                d_abinv,
                d_ab,
                d_ul,
                d_nbul,
                float(self.prior[0]),
                float(self.prior[1]),
                float(self.prior[2]),
                float(self.prior[3]),
                np.int32(self.mabs_gal),
                np.int32(self.mabs_agn),
                float(self.funz0),
                d_ind,
                d_dm,
                np.int32(n_obj),
                np.int32(self.n_t),
                np.int32(n_b),
            ),
            shared_mem=self.smem,
        )
        return cp.asnumpy(d_ind), cp.asnumpy(d_dm)


def calculate_offsets_cuda(
    mags: np.ndarray,
    magerrs: np.ndarray,
    zs: np.ndarray,
    tensors: MagLibTensors,
    config: dict[str, Any],
    context: np.ndarray | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """AUTO_ADAPT on GPU (Z-fixed χ²); host ``interp_lib`` + median."""
    zfix = _ZFixCuda(tensors)
    return calculate_offsets(
        mags,
        magerrs,
        zs,
        tensors,
        config,
        context=context,
        verbose=verbose,
        zfix_fn=zfix,
    )
