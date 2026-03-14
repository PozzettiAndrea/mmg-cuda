/**
 * \file mmgs/cuda/quality_s.cu
 * \brief GPU-accelerated triangle quality computation for mmgs.
 *
 * CUDA kernels for anisotropic and isotropic quality of all surface triangles.
 * Direct port of MMG5_caltri33_ani (common/quality.c) and MMG5_surftri33_ani
 * (common/anisosiz.c).
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#ifndef MMGS_ALPHAD
#define MMGS_ALPHAD    3.464101615137755  /* 6.0 / sqrt(3.0) */
#endif
#ifndef MMG5_EPSD2
#define MMG5_EPSD2     1.0e-200
#endif
#ifndef MMG5_EPSD
#define MMG5_EPSD      1.0e-30
#endif
#ifndef MMG5_ATHIRD
#define MMG5_ATHIRD    0.333333333333333333
#endif
#ifndef MMG5_NULKAL
#define MMG5_NULKAL    1.0e-30
#endif
#ifndef MMG5_EPSOK
#define MMG5_EPSOK     1.0e-15
#endif

/* ================================================================
 * Device kernels
 * ================================================================ */

/**
 * Anisotropic surface area: sqrt(det(t^J * M * J))
 * Direct port of MMG5_surftri33_ani (common/anisosiz.c:171-217).
 */
__device__ double device_surftri33_ani(
    const double *coords, const double *met_m,
    int ia, int ib, int ic)
{
  const double *a = &coords[3*ia];
  const double *b = &coords[3*ib];
  const double *c = &coords[3*ic];

  double abx = b[0]-a[0], aby = b[1]-a[1], abz = b[2]-a[2];
  double acx = c[0]-a[0], acy = c[1]-a[1], acz = c[2]-a[2];

  /* Average metric */
  const double *ma = &met_m[6*ia];
  const double *mb = &met_m[6*ib];
  const double *mc = &met_m[6*ic];
  double mm[6];
  for (int i = 0; i < 6; i++)
    mm[i] = MMG5_ATHIRD * (ma[i] + mb[i] + mc[i]);

  /* Compute t^J * M * J (2x2 matrix stored as dens[0], dens[1], dens[2]) */
  double dens0 = (abx*abx*mm[0] + abx*aby*mm[1] + abx*abz*mm[2])
               + (aby*abx*mm[1] + aby*aby*mm[3] + aby*abz*mm[4])
               + (abz*abx*mm[2] + abz*aby*mm[4] + abz*abz*mm[5]);

  double dens1 = (abx*acx*mm[0] + abx*acy*mm[1] + abx*acz*mm[2])
               + (aby*acx*mm[1] + aby*acy*mm[3] + aby*acz*mm[4])
               + (abz*acx*mm[2] + abz*acy*mm[4] + abz*acz*mm[5]);

  double dens2 = (acx*acx*mm[0] + acx*acy*mm[1] + acx*acz*mm[2])
               + (acy*acx*mm[1] + acy*acy*mm[3] + acy*acz*mm[4])
               + (acz*acx*mm[2] + acz*acy*mm[4] + acz*acz*mm[5]);

  double surf = dens0*dens2 - dens1*dens1;
  if (surf < MMG5_EPSD) return 0.0;

  return sqrt(surf);
}

/**
 * Anisotropic triangle quality: Q = anisurf / (l0^2 + l1^2 + l2^2)
 * Direct port of MMG5_caltri33_ani (common/quality.c:47-101).
 */
__device__ double device_caltri33_ani(
    const double *coords, const double *met_m,
    int ia, int ib, int ic)
{
  const double *ma = &met_m[6*ia];
  const double *mb = &met_m[6*ib];
  const double *mc = &met_m[6*ic];

  /* Anisotropic area */
  double anisurf = device_surftri33_ani(coords, met_m, ia, ib, ic);
  if (anisurf <= MMG5_EPSD2) return 0.0;

  /* Average metric for edge lengths */
  double m[6];
  for (int i = 0; i < 6; i++)
    m[i] = MMG5_ATHIRD * (ma[i] + mb[i] + mc[i]);

  const double *a = &coords[3*ia];
  const double *b = &coords[3*ib];
  const double *c = &coords[3*ic];

  double abx = b[0]-a[0], aby = b[1]-a[1], abz = b[2]-a[2];
  double acx = c[0]-a[0], acy = c[1]-a[1], acz = c[2]-a[2];
  double bcx = c[0]-b[0], bcy = c[1]-b[1], bcz = c[2]-b[2];

  /* Edge lengths in metric */
  double l0 = m[0]*abx*abx + m[3]*aby*aby + m[5]*abz*abz
    + 2.0*(m[1]*abx*aby + m[2]*abx*abz + m[4]*aby*abz);

  double l1 = m[0]*acx*acx + m[3]*acy*acy + m[5]*acz*acz
    + 2.0*(m[1]*acx*acy + m[2]*acx*acz + m[4]*acy*acz);

  double l2 = m[0]*bcx*bcx + m[3]*bcy*bcy + m[5]*bcz*bcz
    + 2.0*(m[1]*bcx*bcy + m[2]*bcx*bcz + m[4]*bcy*bcz);

  double rap = l0 + l1 + l2;
  if (rap > MMG5_EPSD2) {
    return anisurf / rap;
  }
  return 0.0;
}

/**
 * Isotropic triangle quality: Q = 4*sqrt(3)*area / (l0^2 + l1^2 + l2^2)
 * Simplified from common/quality.c — no metric.
 */
__device__ double device_caltri_iso(
    const double *coords, int ia, int ib, int ic)
{
  const double *a = &coords[3*ia];
  const double *b = &coords[3*ib];
  const double *c = &coords[3*ic];

  double abx = b[0]-a[0], aby = b[1]-a[1], abz = b[2]-a[2];
  double acx = c[0]-a[0], acy = c[1]-a[1], acz = c[2]-a[2];
  double bcx = c[0]-b[0], bcy = c[1]-b[1], bcz = c[2]-b[2];

  /* Cross product for area */
  double nx = aby*acz - abz*acy;
  double ny = abz*acx - abx*acz;
  double nz = abx*acy - aby*acx;
  double area = sqrt(nx*nx + ny*ny + nz*nz);  /* 2*area */
  if (area < MMG5_EPSD2) return 0.0;

  /* Sum of squared edge lengths */
  double rap = (abx*abx+aby*aby+abz*abz)
             + (acx*acx+acy*acy+acz*acz)
             + (bcx*bcx+bcy*bcy+bcz*bcz);
  if (rap < MMG5_EPSD2) return 0.0;

  return area / rap;
}

/**
 * Main kernel: compute quality of all triangles.
 * One thread per triangle (1-based indexing).
 *
 * mode: 0 = anisotropic (met->size==6), 1 = isotropic (no metric)
 */
__global__ void kernel_triQual(
    const double *coords,     /* vertex coordinates, 1-based */
    const double *met_m,      /* metric array, 1-based (only for mode==0) */
    const int    *tri_v,      /* tri connectivity: 3 ints per tri, 1-based */
    const int    *tri_valid,  /* 1 if MG_EOK, 0 otherwise */
    double       *qual_out,   /* output quality per tri, 1-based */
    int           nt,         /* number of triangles */
    int           mode)       /* 0=anisotropic, 1=isotropic */
{
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;  /* 1-based */
  if (k > nt) return;

  if (!tri_valid[k]) {
    qual_out[k] = 0.0;
    return;
  }

  int ia = tri_v[3*k + 0];
  int ib = tri_v[3*k + 1];
  int ic = tri_v[3*k + 2];

  double q;
  if (mode == 0) {
    q = device_caltri33_ani(coords, met_m, ia, ib, ic);
  } else {
    q = device_caltri_iso(coords, ia, ib, ic);
  }
  qual_out[k] = q;
}

/**
 * Min/max/sum reduction kernel for quality statistics.
 */
__global__ void kernel_qualStats(
    const double *qual,       /* 1-based quality array */
    const int    *tri_valid,
    int           nt,
    double       *d_minqual,
    double       *d_maxqual,
    double       *d_sumqual,
    int          *d_minidx,
    int          *d_count)
{
  extern __shared__ char smem[];
  double *s_min = (double*)smem;
  double *s_max = s_min + blockDim.x;
  double *s_sum = s_max + blockDim.x;
  int    *s_idx = (int*)(s_sum + blockDim.x);
  int    *s_cnt = s_idx + blockDim.x;

  int tid = threadIdx.x;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;

  s_min[tid] = 1.0e30;
  s_max[tid] = 0.0;
  s_sum[tid] = 0.0;
  s_idx[tid] = 1;
  s_cnt[tid] = 0;

  if (k <= nt && tri_valid[k]) {
    double q = qual[k];
    s_min[tid] = q;
    s_max[tid] = q;
    s_sum[tid] = q;
    s_idx[tid] = k;
    s_cnt[tid] = 1;
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (s_min[tid + s] < s_min[tid]) {
        s_min[tid] = s_min[tid + s];
        s_idx[tid] = s_idx[tid + s];
      }
      if (s_max[tid + s] > s_max[tid]) {
        s_max[tid] = s_max[tid + s];
      }
      s_sum[tid] += s_sum[tid + s];
      s_cnt[tid] += s_cnt[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    /* Atomic min via CAS */
    unsigned long long *addr = (unsigned long long *)d_minqual;
    unsigned long long assumed, old = *addr;
    double val = s_min[0];
    do {
      assumed = old;
      if (val >= __longlong_as_double(assumed)) break;
      old = atomicCAS(addr, assumed, __double_as_longlong(val));
    } while (assumed != old);

    /* Atomic max */
    addr = (unsigned long long *)d_maxqual;
    old = *addr;
    val = s_max[0];
    do {
      assumed = old;
      if (val <= __longlong_as_double(assumed)) break;
      old = atomicCAS(addr, assumed, __double_as_longlong(val));
    } while (assumed != old);

    /* Atomic sum and count */
    atomicAdd((unsigned long long*)d_sumqual,
              __double_as_longlong(s_sum[0]));  /* not exact but close enough for avg */
    atomicAdd(d_count, s_cnt[0]);

    /* Best-effort min index */
    double cur_min = __longlong_as_double(*(unsigned long long*)d_minqual);
    if (s_min[0] <= cur_min) {
      atomicExch(d_minidx, s_idx[0]);
    }
  }
}

/* ================================================================
 * Host wrapper
 * ================================================================ */

extern "C" {

#include "mmgcommon_private.h"
#include "libmmgs_private.h"
#include "mmgs_cuda.h"

int MMGS_triQual_cuda(MMG5_pMesh mesh, MMG5_pSol met) {
  int nt = (int)mesh->nt;
  int np = (int)mesh->np;
  int npmax = (int)mesh->npmax;
  int ntmax = (int)mesh->ntmax;

  /* Determine mode */
  int mode;
  if (met && met->m && met->size == 6) {
    mode = 0;  /* anisotropic */
  } else {
    mode = 1;  /* isotropic / no metric */
  }

  fprintf(stdout, "[CUDA-QUAL-S] Computing quality for %d tris (%s mode)\n",
          nt, mode == 0 ? "anisotropic" : "isotropic");

  /* ---- Timing ---- */
  cudaEvent_t t_start, t_end;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start);

  /* ---- Extract AOS → SOA ---- */
  size_t coords_size = (size_t)3 * (npmax + 1) * sizeof(double);
  double *h_coords = (double*)calloc((size_t)3 * (npmax + 1), sizeof(double));
  for (int k = 1; k <= np; k++) {
    h_coords[3*k+0] = mesh->point[k].c[0];
    h_coords[3*k+1] = mesh->point[k].c[1];
    h_coords[3*k+2] = mesh->point[k].c[2];
  }

  size_t tri_v_size = (size_t)3 * (ntmax + 1) * sizeof(int);
  int *h_tri_v = (int*)calloc((size_t)3 * (ntmax + 1), sizeof(int));
  int *h_tri_valid = (int*)calloc((size_t)(ntmax + 1), sizeof(int));
  for (int k = 1; k <= nt; k++) {
    MMG5_pTria ptt = &mesh->tria[k];
    h_tri_v[3*k+0] = (int)ptt->v[0];
    h_tri_v[3*k+1] = (int)ptt->v[1];
    h_tri_v[3*k+2] = (int)ptt->v[2];
    h_tri_valid[k] = (ptt->v[0] > 0) ? 1 : 0;
  }

  double *h_met = NULL;
  size_t met_size = 0;
  if (mode == 0) {
    met_size = (size_t)6 * (npmax + 1) * sizeof(double);
    h_met = (double*)calloc((size_t)6 * (npmax + 1), sizeof(double));
    memcpy(h_met, met->m, (size_t)met->size * (np + 1) * sizeof(double));
  }

  double *h_qual = (double*)calloc((size_t)(ntmax + 1), sizeof(double));

  /* ---- GPU alloc + upload ---- */
  double *d_coords, *d_met = NULL, *d_qual;
  double *d_minqual, *d_maxqual, *d_sumqual;
  int *d_tri_v, *d_tri_valid, *d_minidx, *d_count;

  cudaMalloc(&d_coords, coords_size);
  cudaMalloc(&d_tri_v, tri_v_size);
  cudaMalloc(&d_tri_valid, (size_t)(ntmax + 1) * sizeof(int));
  cudaMalloc(&d_qual, (size_t)(ntmax + 1) * sizeof(double));
  cudaMalloc(&d_minqual, sizeof(double));
  cudaMalloc(&d_maxqual, sizeof(double));
  cudaMalloc(&d_sumqual, sizeof(double));
  cudaMalloc(&d_minidx, sizeof(int));
  cudaMalloc(&d_count, sizeof(int));
  if (mode == 0) cudaMalloc(&d_met, met_size);

  cudaMemcpy(d_coords, h_coords, coords_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_tri_v, h_tri_v, tri_v_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_tri_valid, h_tri_valid, (size_t)(ntmax+1)*sizeof(int), cudaMemcpyHostToDevice);
  if (mode == 0) cudaMemcpy(d_met, h_met, met_size, cudaMemcpyHostToDevice);

  /* Init reduction vars */
  double init_min = 1.0e30, init_max = 0.0, init_sum = 0.0;
  int init_idx = 1, init_cnt = 0;
  cudaMemcpy(d_minqual, &init_min, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_maxqual, &init_max, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sumqual, &init_sum, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_minidx, &init_idx, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_count, &init_cnt, sizeof(int), cudaMemcpyHostToDevice);

  /* ---- Launch kernels ---- */
  int blockSize = 256;
  int numBlocks = (nt + blockSize - 1) / blockSize;

  kernel_triQual<<<numBlocks, blockSize>>>(
      d_coords, d_met, d_tri_v, d_tri_valid, d_qual, nt, mode);

  size_t smem = blockSize * (3 * sizeof(double) + 2 * sizeof(int));
  kernel_qualStats<<<numBlocks, blockSize, smem>>>(
      d_qual, d_tri_valid, nt, d_minqual, d_maxqual, d_sumqual,
      d_minidx, d_count);

  /* ---- Download ---- */
  cudaMemcpy(h_qual, d_qual, (size_t)(nt+1)*sizeof(double), cudaMemcpyDeviceToHost);

  double minqual, maxqual, sumqual;
  int minidx, count;
  cudaMemcpy(&minqual, d_minqual, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&maxqual, d_maxqual, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&sumqual, d_sumqual, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&minidx, d_minidx, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

  /* ---- Timing ---- */
  cudaEventRecord(t_end);
  cudaEventSynchronize(t_end);
  float gpu_ms;
  cudaEventElapsedTime(&gpu_ms, t_start, t_end);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA-QUAL-S] CUDA error: %s\n", cudaGetErrorString(err));
    free(h_coords); free(h_tri_v); free(h_tri_valid); free(h_qual);
    if (h_met) free(h_met);
    cudaFree(d_coords); cudaFree(d_tri_v); cudaFree(d_tri_valid);
    cudaFree(d_qual); cudaFree(d_minqual); cudaFree(d_maxqual);
    cudaFree(d_sumqual); cudaFree(d_minidx); cudaFree(d_count);
    if (d_met) cudaFree(d_met);
    cudaEventDestroy(t_start); cudaEventDestroy(t_end);
    return 0;
  }

  /* ---- Write back ---- */
  for (int k = 1; k <= nt; k++) {
    mesh->tria[k].qual = h_qual[k];
  }

  /* ---- Compute avg on CPU from quality array (atomic double sum is unreliable) ---- */
  {
    double sum = 0.0;
    int cnt = 0;
    for (int k = 1; k <= nt; k++) {
      if (h_qual[k] > 0.0) { sum += h_qual[k]; cnt++; }
    }
    double avgq = (cnt > 0) ? sum / cnt : 0.0;
    fprintf(stdout, "\n  -- MESH QUALITY   %d\n", cnt);
    fprintf(stdout, "     BEST   %8.6f  AVRG.   %8.6f  WRST.   %8.6f (%d)\n",
            MMGS_ALPHAD*maxqual, MMGS_ALPHAD*avgq, MMGS_ALPHAD*minqual, minidx);
    fprintf(stdout, "     [CUDA: %.3f ms]\n", gpu_ms);
  }

  /* ---- Cleanup ---- */
  free(h_coords); free(h_tri_v); free(h_tri_valid); free(h_qual);
  if (h_met) free(h_met);
  cudaFree(d_coords); cudaFree(d_tri_v); cudaFree(d_tri_valid);
  cudaFree(d_qual); cudaFree(d_minqual); cudaFree(d_maxqual);
  cudaFree(d_sumqual); cudaFree(d_minidx); cudaFree(d_count);
  if (d_met) cudaFree(d_met);
  cudaEventDestroy(t_start); cudaEventDestroy(t_end);

  /* Quality check */
  if (MMGS_ALPHAD * minqual < MMG5_NULKAL) {
    fprintf(stderr, "\n  ## Error: %s: too bad quality (elt %d -> %15e)\n",
            __func__, minidx, minqual);
    return 0;
  }

  return 1;
}

} /* extern "C" */
