/**
 * \file mmg3d/cuda/quality_3d.cu
 * \brief GPU-accelerated tetrahedron quality computation for mmg3d.
 *
 * Provides CUDA kernels for computing anisotropic and isotropic quality
 * of all tetrahedra in the mesh. Direct port of MMG5_caltet33_ani and
 * MMG5_caltet_iso_4pt from quality_3d.c / inlined_functions_3d_private.h.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

/* Constants from mmg — use #ifndef to avoid redefinition when mmg headers
   are included in the host wrapper section below */
#ifndef MMG3D_ALPHAD
#define MMG3D_ALPHAD   20.7846096908265  /* 12*sqrt(3) */
#endif
#ifndef MMG5_EPSD2
#define MMG5_EPSD2     1.0e-200
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
 * Anisotropic quality: Q = sqrt(det(M)) * vol / (sum_edge_lens)^(3/2)
 * where edge lengths are measured in the averaged metric M.
 * Direct port of MMG5_caltet33_ani (quality_3d.c:109-197).
 */
__device__ double device_caltet33_ani(
    const double *coords,  /* 3 * (npmax+1) doubles, 1-based: coords[3*ip+0..2] */
    const double *met_m,   /* 6 * (npmax+1) doubles, 1-based: met_m[6*ip+0..5]  */
    int ip0, int ip1, int ip2, int ip3)
{
  double mm[6];
  int iad0 = 6 * ip0, iad1 = 6 * ip1, iad2 = 6 * ip2, iad3 = 6 * ip3;

  /* average metric over 4 vertices */
  for (int k = 0; k < 6; k++)
    mm[k] = 0.25 * (met_m[iad0+k] + met_m[iad1+k] + met_m[iad2+k] + met_m[iad3+k]);

  const double *a = &coords[3*ip0];
  const double *b = &coords[3*ip1];
  const double *c = &coords[3*ip2];
  const double *d = &coords[3*ip3];

  /* edge vectors */
  double abx = b[0]-a[0], aby = b[1]-a[1], abz = b[2]-a[2];
  double acx = c[0]-a[0], acy = c[1]-a[1], acz = c[2]-a[2];
  double adx = d[0]-a[0], ady = d[1]-a[1], adz = d[2]-a[2];
  double bcx = c[0]-b[0], bcy = c[1]-b[1], bcz = c[2]-b[2];
  double bdx = d[0]-b[0], bdy = d[1]-b[1], bdz = d[2]-b[2];
  double cdx = d[0]-c[0], cdy = d[1]-c[1], cdz = d[2]-c[2];

  /* volume = (ac x ad) . ab */
  double v1 = acy*adz - acz*ady;
  double v2 = acz*adx - acx*adz;
  double v3 = acx*ady - acy*adx;
  double vol = abx*v1 + aby*v2 + abz*v3;
  if (vol <= 0.0) return 0.0;

  /* determinant of metric */
  double det = mm[0]*(mm[3]*mm[5] - mm[4]*mm[4])
             - mm[1]*(mm[1]*mm[5] - mm[2]*mm[4])
             + mm[2]*(mm[1]*mm[4] - mm[2]*mm[3]);
  if (det < MMG5_EPSD2) return 0.0;
  det = sqrt(det) * vol;

  /* edge lengths in metric */
  double h1 = mm[0]*abx*abx + mm[3]*aby*aby + mm[5]*abz*abz
    + 2.0*(mm[1]*abx*aby + mm[2]*abx*abz + mm[4]*aby*abz);
  double h2 = mm[0]*acx*acx + mm[3]*acy*acy + mm[5]*acz*acz
    + 2.0*(mm[1]*acx*acy + mm[2]*acx*acz + mm[4]*acy*acz);
  double h3 = mm[0]*adx*adx + mm[3]*ady*ady + mm[5]*adz*adz
    + 2.0*(mm[1]*adx*ady + mm[2]*adx*adz + mm[4]*ady*adz);
  double h4 = mm[0]*bcx*bcx + mm[3]*bcy*bcy + mm[5]*bcz*bcz
    + 2.0*(mm[1]*bcx*bcy + mm[2]*bcx*bcz + mm[4]*bcy*bcz);
  double h5 = mm[0]*bdx*bdx + mm[3]*bdy*bdy + mm[5]*bdz*bdz
    + 2.0*(mm[1]*bdx*bdy + mm[2]*bdx*bdz + mm[4]*bdy*bdz);
  double h6 = mm[0]*cdx*cdx + mm[3]*cdy*cdy + mm[5]*cdz*cdz
    + 2.0*(mm[1]*cdx*cdy + mm[2]*cdx*cdz + mm[4]*cdy*cdz);

  double rap = h1 + h2 + h3 + h4 + h5 + h6;
  double num = sqrt(rap) * rap;

  return det / num;
}

/**
 * Isotropic quality: Q = 6*vol / (sum_edge_lens^2)^(3/2)
 * Direct port of MMG5_caltet_iso_4pt (inlined_functions_3d_private.h:340-386).
 */
__device__ double device_caltet_iso_4pt(
    const double *coords, int ip0, int ip1, int ip2, int ip3)
{
  const double *a = &coords[3*ip0];
  const double *b = &coords[3*ip1];
  const double *c = &coords[3*ip2];
  const double *d = &coords[3*ip3];

  double abx = b[0]-a[0], aby = b[1]-a[1], abz = b[2]-a[2];
  double rap = abx*abx + aby*aby + abz*abz;

  double acx = c[0]-a[0], acy = c[1]-a[1], acz = c[2]-a[2];
  rap += acx*acx + acy*acy + acz*acz;

  double adx = d[0]-a[0], ady = d[1]-a[1], adz = d[2]-a[2];
  rap += adx*adx + ady*ady + adz*adz;

  double v1 = acy*adz - acz*ady;
  double v2 = acz*adx - acx*adz;
  double v3 = acx*ady - acy*adx;
  double vol = abx*v1 + aby*v2 + abz*v3;
  if (vol < MMG5_EPSD2) return 0.0;

  double bcx = c[0]-b[0], bcy = c[1]-b[1], bcz = c[2]-b[2];
  rap += bcx*bcx + bcy*bcy + bcz*bcz;

  double bdx = d[0]-b[0], bdy = d[1]-b[1], bdz = d[2]-b[2];
  rap += bdx*bdx + bdy*bdy + bdz*bdz;

  double cdx = d[0]-c[0], cdy = d[1]-c[1], cdz = d[2]-c[2];
  rap += cdx*cdx + cdy*cdy + cdz*cdz;
  if (rap < MMG5_EPSD2) return 0.0;

  rap = rap * sqrt(rap);
  return vol / rap;
}

/**
 * Main kernel: compute quality of all tetrahedra.
 * One thread per tetrahedron (1-based indexing).
 *
 * mode: 0 = anisotropic (met->size==6), 1 = isotropic (no metric)
 */
__global__ void kernel_tetraQual(
    const double *coords,     /* vertex coordinates, 1-based */
    const double *met_m,      /* metric array, 1-based (NULL-safe: only used if mode==0) */
    const int    *tet_v,      /* tet connectivity: 4 ints per tet, 1-based */
    const int    *tet_valid,  /* 1 if MG_EOK, 0 otherwise */
    double       *qual_out,   /* output quality per tet, 1-based */
    int           ne,         /* number of tetrahedra */
    int           mode)       /* 0=anisotropic, 1=isotropic */
{
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;  /* 1-based */
  if (k > ne) return;

  if (!tet_valid[k]) {
    qual_out[k] = 0.0;
    return;
  }

  int ip0 = tet_v[4*k + 0];
  int ip1 = tet_v[4*k + 1];
  int ip2 = tet_v[4*k + 2];
  int ip3 = tet_v[4*k + 3];

  double q;
  if (mode == 0) {
    q = device_caltet33_ani(coords, met_m, ip0, ip1, ip2, ip3);
  } else {
    q = device_caltet_iso_4pt(coords, ip0, ip1, ip2, ip3);
  }
  qual_out[k] = q;
}

/**
 * Min-reduction kernel: find minimum quality and its index.
 * Uses shared memory reduction within blocks, then atomicMin across blocks.
 */
__global__ void kernel_minQual(
    const double *qual,       /* 1-based quality array */
    const int    *tet_valid,  /* 1-based validity array */
    int           ne,
    double       *d_minqual,  /* single output: min quality */
    int          *d_minidx)   /* single output: index of min */
{
  extern __shared__ double sdata[];
  int *sidx = (int*)&sdata[blockDim.x];

  int tid = threadIdx.x;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;  /* 1-based */

  /* Initialize with large value */
  sdata[tid] = 1.0e30;
  sidx[tid] = 1;

  if (k <= ne && tet_valid[k]) {
    sdata[tid] = qual[k];
    sidx[tid] = k;
  }

  __syncthreads();

  /* Block-level reduction */
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid + s] < sdata[tid]) {
        sdata[tid] = sdata[tid + s];
        sidx[tid] = sidx[tid + s];
      }
    }
    __syncthreads();
  }

  /* Thread 0 writes block result via atomic compare-and-swap on double */
  if (tid == 0) {
    /* Use atomicCAS-based double atomicMin */
    double val = sdata[0];
    int idx = sidx[0];

    /* Spin-lock style atomic min on double */
    unsigned long long *addr = (unsigned long long *)d_minqual;
    unsigned long long assumed, old;
    old = *addr;
    do {
      assumed = old;
      double old_val = __longlong_as_double(assumed);
      if (val >= old_val) break;  /* our value is not smaller */
      unsigned long long desired = __double_as_longlong(val);
      old = atomicCAS(addr, assumed, desired);
    } while (assumed != old);

    /* If we wrote the min, also write the index.
       This is not perfectly atomic with the value, but for quality
       reporting purposes it's sufficient. */
    double current_min = __longlong_as_double(*addr);
    if (val <= current_min) {
      atomicExch(d_minidx, idx);
    }
  }
}

/* ================================================================
 * Host wrapper — called from C code
 * ================================================================ */

extern "C" {

/* Need the mmg types */
#include "mmgcommon_private.h"
#include "libmmg3d_private.h"
#include "mmg3d_cuda.h"

int MMG3D_tetraQual_cuda(MMG5_pMesh mesh, MMG5_pSol met, int8_t metRidTyp) {
  int ne = (int)mesh->ne;
  int np = (int)mesh->np;
  int npmax = (int)mesh->npmax;
  int nemax = (int)mesh->nemax;

  /* Determine mode: 0=anisotropic, 1=isotropic */
  int mode;
  if (!metRidTyp && met && met->m && met->size == 6) {
    mode = 0;  /* anisotropic */
  } else if (!(met && met->m)) {
    mode = 1;  /* no metric, isotropic */
  } else {
    /* orcal / function-pointer path — fall back to CPU */
    fprintf(stdout, "[CUDA-QUAL] Falling back to CPU (orcal path)\n");
    return MMG3D_tetraQual(mesh, met, metRidTyp);
  }

  fprintf(stdout, "[CUDA-QUAL] Computing quality for %d tets (%s mode)\n",
          ne, mode == 0 ? "anisotropic" : "isotropic");

  /* ---- Extract AOS data into SOA staging buffers ---- */

  /* Coordinates: 1-based, 3 doubles per point */
  size_t coords_size = (size_t)3 * (npmax + 1) * sizeof(double);
  double *h_coords = (double*)calloc((size_t)3 * (npmax + 1), sizeof(double));
  if (!h_coords) { fprintf(stderr, "[CUDA-QUAL] alloc failed\n"); return 0; }
  for (int k = 1; k <= np; k++) {
    h_coords[3*k+0] = mesh->point[k].c[0];
    h_coords[3*k+1] = mesh->point[k].c[1];
    h_coords[3*k+2] = mesh->point[k].c[2];
  }

  /* Tetrahedra connectivity: 1-based, 4 ints per tet */
  size_t tet_v_size = (size_t)4 * (nemax + 1) * sizeof(int);
  int *h_tet_v = (int*)calloc((size_t)4 * (nemax + 1), sizeof(int));
  int *h_tet_valid = (int*)calloc((size_t)(nemax + 1), sizeof(int));
  if (!h_tet_v || !h_tet_valid) { fprintf(stderr, "[CUDA-QUAL] alloc failed\n"); return 0; }
  for (int k = 1; k <= ne; k++) {
    MMG5_pTetra pt = &mesh->tetra[k];
    h_tet_v[4*k+0] = (int)pt->v[0];
    h_tet_v[4*k+1] = (int)pt->v[1];
    h_tet_v[4*k+2] = (int)pt->v[2];
    h_tet_v[4*k+3] = (int)pt->v[3];
    h_tet_valid[k] = (pt->v[0] > 0) ? 1 : 0;  /* MG_EOK */
  }

  /* Metric: 1-based, 6 doubles per point (anisotropic only) */
  double *h_met = NULL;
  size_t met_size = 0;
  if (mode == 0) {
    met_size = (size_t)6 * (npmax + 1) * sizeof(double);
    h_met = (double*)calloc((size_t)6 * (npmax + 1), sizeof(double));
    if (!h_met) { fprintf(stderr, "[CUDA-QUAL] alloc failed\n"); return 0; }
    /* met->m is indexed as met->m[met->size * ip + j] */
    memcpy(h_met, met->m, (size_t)met->size * (np + 1) * sizeof(double));
  }

  /* Output quality: 1-based */
  double *h_qual = (double*)calloc((size_t)(nemax + 1), sizeof(double));
  if (!h_qual) { fprintf(stderr, "[CUDA-QUAL] alloc failed\n"); return 0; }

  /* ---- Allocate GPU memory ---- */
  double *d_coords, *d_met = NULL, *d_qual, *d_minqual;
  int *d_tet_v, *d_tet_valid, *d_minidx;

  cudaMalloc(&d_coords, coords_size);
  cudaMalloc(&d_tet_v, tet_v_size);
  cudaMalloc(&d_tet_valid, (size_t)(nemax + 1) * sizeof(int));
  cudaMalloc(&d_qual, (size_t)(nemax + 1) * sizeof(double));
  cudaMalloc(&d_minqual, sizeof(double));
  cudaMalloc(&d_minidx, sizeof(int));
  if (mode == 0) {
    cudaMalloc(&d_met, met_size);
  }

  /* ---- Upload ---- */
  cudaMemcpy(d_coords, h_coords, coords_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_tet_v, h_tet_v, tet_v_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_tet_valid, h_tet_valid, (size_t)(nemax + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  if (mode == 0) {
    cudaMemcpy(d_met, h_met, met_size, cudaMemcpyHostToDevice);
  }

  /* Initialize min reduction */
  double init_minqual = 2.0 / MMG3D_ALPHAD;
  int init_minidx = 1;
  cudaMemcpy(d_minqual, &init_minqual, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_minidx, &init_minidx, sizeof(int), cudaMemcpyHostToDevice);

  /* ---- Launch quality kernel ---- */
  int blockSize = 256;
  int numBlocks = (ne + blockSize - 1) / blockSize;

  kernel_tetraQual<<<numBlocks, blockSize>>>(
      d_coords, d_met, d_tet_v, d_tet_valid, d_qual, ne, mode);

  /* ---- Launch min-reduction kernel ---- */
  size_t smem = blockSize * (sizeof(double) + sizeof(int));
  kernel_minQual<<<numBlocks, blockSize, smem>>>(
      d_qual, d_tet_valid, ne, d_minqual, d_minidx);

  /* ---- Download results ---- */
  cudaMemcpy(h_qual, d_qual, (size_t)(ne + 1) * sizeof(double),
             cudaMemcpyDeviceToHost);

  double minqual;
  int minidx;
  cudaMemcpy(&minqual, d_minqual, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&minidx, d_minidx, sizeof(int), cudaMemcpyDeviceToHost);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA-QUAL] CUDA error: %s\n", cudaGetErrorString(err));
    /* Fall back to CPU */
    free(h_coords); free(h_tet_v); free(h_tet_valid); free(h_qual);
    if (h_met) free(h_met);
    cudaFree(d_coords); cudaFree(d_tet_v); cudaFree(d_tet_valid);
    cudaFree(d_qual); cudaFree(d_minqual); cudaFree(d_minidx);
    if (d_met) cudaFree(d_met);
    return MMG3D_tetraQual(mesh, met, metRidTyp);
  }

  /* ---- Write results back to mesh ---- */
  for (int k = 1; k <= ne; k++) {
    mesh->tetra[k].qual = h_qual[k];
  }

  /* ---- Cleanup ---- */
  free(h_coords);
  free(h_tet_v);
  free(h_tet_valid);
  free(h_qual);
  if (h_met) free(h_met);

  cudaFree(d_coords);
  cudaFree(d_tet_v);
  cudaFree(d_tet_valid);
  cudaFree(d_qual);
  cudaFree(d_minqual);
  cudaFree(d_minidx);
  if (d_met) cudaFree(d_met);

  fprintf(stdout, "[CUDA-QUAL] Done: minqual=%e at tet %d\n", minqual, minidx);

  /* Quality check (same as CPU path) */
  double minqualAlpha = minqual * MMG3D_ALPHAD;
  if (minqualAlpha < MMG5_NULKAL) {
    fprintf(stderr, "\n  ## Error: %s: too bad quality for the worst element: "
            "(elt %d -> %15e)\n", __func__, minidx, minqual);
    return 0;
  }
  else if (minqualAlpha < MMG5_EPSOK) {
    fprintf(stderr, "\n  ## Warning: %s: very bad quality for the worst element: "
            "(elt %d -> %15e)\n", __func__, minidx, minqual);
  }

  return 1;
}

} /* extern "C" */
