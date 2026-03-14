/**
 * \file mmgs/cuda/anisosiz_s.cu
 * \brief GPU-accelerated metric operations for mmgs surface remeshing.
 *
 * Provides CUDA kernels for:
 * 1. Metric initialization at uninitialized points (MMG5_defUninitSize)
 * 2. Batch anisotropic edge length computation (MMG5_lenSurfEdg33_ani)
 * 3. Batch quality+length marking for split/collapse candidate identification
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#ifndef MMG5_EPSD2
#define MMG5_EPSD2   1.0e-200
#endif
#ifndef MMG5_EPS
#define MMG5_EPS     1.0e-06
#endif
#ifndef MMGS_LOPTL
#define MMGS_LOPTL   1.4
#endif
#ifndef MMGS_LOPTS
#define MMGS_LOPTS   0.71
#endif

/* Point tags — must match mmg definitions */
#define MG_BDY   (1 << 1)
#define MG_GEO   (1 << 2)
#define MG_NOM   (1 << 3)
#define MG_REF   (1 << 4)
#define MG_REQ   (1 << 5)
#define MG_CRN   (1 << 6)
#define MG_SIN_OR_NOM(tag) ((tag) & (MG_CRN | MG_REQ | MG_NOM))
#define MG_SIN(tag)        ((tag) & (MG_CRN | MG_REQ))

/* ================================================================
 * Device: rotation matrix (port of MMG5_rotmatrix)
 * ================================================================ */

__device__ void device_rotmatrix(const double n[3], double r[3][3]) {
  double aa = n[0]*n[0], bb = n[1]*n[1], ab = n[0]*n[1];
  double ll = aa + bb;
  double cosalpha = n[2];
  double sa = 1.0 - cosalpha*cosalpha;
  if (sa < 0.0) sa = 0.0;
  double sinalpha = sqrt(sa);

  if (ll < 1.0e-06) {
    if (n[2] > 0.0) {
      r[0][0]=1; r[0][1]=0; r[0][2]=0;
      r[1][0]=0; r[1][1]=1; r[1][2]=0;
      r[2][0]=0; r[2][1]=0; r[2][2]=1;
    } else {
      r[0][0]=-1; r[0][1]=0; r[0][2]=0;
      r[1][0]=0;  r[1][1]=1; r[1][2]=0;
      r[2][0]=0;  r[2][1]=0; r[2][2]=-1;
    }
  } else {
    double l = sqrt(ll);
    r[0][0] = (aa*cosalpha + bb)/ll;
    r[0][1] = ab*(cosalpha - 1.0)/ll;
    r[0][2] = -n[0]*sinalpha/l;
    r[1][0] = r[0][1];
    r[1][1] = (bb*cosalpha + aa)/ll;
    r[1][2] = -n[1]*sinalpha/l;
    r[2][0] = n[0]*sinalpha/l;
    r[2][1] = n[1]*sinalpha/l;
    r[2][2] = cosalpha;
  }
}

/* ================================================================
 * Kernel 1: Metric initialization at uninitialized points
 * Port of MMG5_defUninitSize (common/anisosiz.c:228-271)
 * ================================================================ */

__global__ void kernel_defUninitSize(
    const double *coords,      /* 3*(npmax+1) */
    const double *normals,     /* 3*(npmax+1), point normals (p->n) */
    const double *xpoint_n1,   /* 3*(xpmax+1), xpoint normals (NULL if no xpoints) */
    const int    *point_xp,    /* npmax+1, xpoint index per point */
    const uint16_t *point_tag, /* npmax+1 */
    const int    *point_flag,  /* npmax+1 */
    const int    *point_valid, /* npmax+1, MG_VOK */
    double       *met_m,       /* 6*(npmax+1), metric to write */
    int          *out_flag,    /* npmax+1, updated flag */
    int           np,
    int           ismet,
    double        isqhmax)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (k > np) return;
  if (!point_valid[k] || point_flag[k] > 0) return;

  uint16_t tag = point_tag[k];
  double *m = &met_m[6*k];

  if (ismet) {
    /* If metric provided: only set ridge points to isqhmax */
    if (!(MG_SIN_OR_NOM(tag)) && (tag & MG_GEO)) {
      m[0] = m[1] = m[2] = m[3] = m[4] = isqhmax;
      m[5] = 0.0;
    }
    out_flag[k] = 1;
    return;
  }

  /* No metric: set isotropic hmax-based metric */
  m[0] = m[1] = m[2] = m[3] = m[4] = m[5] = 0.0;

  if (MG_SIN(tag) || (tag & MG_NOM)) {
    m[0] = m[3] = m[5] = isqhmax;
  }
  else if (tag & MG_GEO) {
    m[0] = m[1] = m[2] = m[3] = m[4] = isqhmax;
  }
  else {
    /* Regular or REF point: need rotation matrix from normal */
    double n[3];
    if (tag & MG_REF) {
      int xpi = point_xp[k];
      if (xpi > 0 && xpoint_n1 != NULL) {
        n[0] = xpoint_n1[3*xpi+0];
        n[1] = xpoint_n1[3*xpi+1];
        n[2] = xpoint_n1[3*xpi+2];
      } else {
        n[0] = normals[3*k+0];
        n[1] = normals[3*k+1];
        n[2] = normals[3*k+2];
      }
    } else {
      n[0] = normals[3*k+0];
      n[1] = normals[3*k+1];
      n[2] = normals[3*k+2];
    }

    double r[3][3];
    device_rotmatrix(n, r);

    m[0] = isqhmax*(r[0][0]*r[0][0]+r[1][0]*r[1][0]+r[2][0]*r[2][0]);
    m[1] = isqhmax*(r[0][0]*r[0][1]+r[1][0]*r[1][1]+r[2][0]*r[2][1]);
    m[2] = isqhmax*(r[0][0]*r[0][2]+r[1][0]*r[1][2]+r[2][0]*r[2][2]);
    m[3] = isqhmax*(r[0][1]*r[0][1]+r[1][1]*r[1][1]+r[2][1]*r[2][1]);
    m[4] = isqhmax*(r[0][1]*r[0][2]+r[1][1]*r[1][2]+r[2][1]*r[2][2]);
    m[5] = isqhmax*(r[0][2]*r[0][2]+r[1][2]*r[1][2]+r[2][2]*r[2][2]);
  }
  out_flag[k] = 2;
}

/* ================================================================
 * Kernel 2: Batch anisotropic edge length computation
 * Port of MMG5_lenSurfEdg33_ani
 *
 * Computes length of each edge in anisotropic metric space.
 * One thread per triangle edge (3 edges per tri).
 * ================================================================ */

__device__ double device_lenSurfEdg33_ani(
    const double *coords, const double *met_m,
    int np, int nq)
{
  const double *ca = &coords[3*np];
  const double *cb = &coords[3*nq];
  const double *ma = &met_m[6*np];
  const double *mb = &met_m[6*nq];

  double dx = cb[0]-ca[0], dy = cb[1]-ca[1], dz = cb[2]-ca[2];

  /* Length at vertex np: sqrt(d^T * M_np * d) */
  double len_a = ma[0]*dx*dx + ma[3]*dy*dy + ma[5]*dz*dz
    + 2.0*(ma[1]*dx*dy + ma[2]*dx*dz + ma[4]*dy*dz);
  if (len_a <= 0.0) len_a = 0.0;
  len_a = sqrt(len_a);

  /* Length at vertex nq */
  double len_b = mb[0]*dx*dx + mb[3]*dy*dy + mb[5]*dz*dz
    + 2.0*(mb[1]*dx*dy + mb[2]*dx*dz + mb[4]*dy*dz);
  if (len_b <= 0.0) len_b = 0.0;
  len_b = sqrt(len_b);

  /* Average length (trapezoidal rule) */
  return 0.5 * (len_a + len_b);
}

/**
 * Isotropic edge length in metric space.
 * Port of MMG5_lenSurfEdg_iso (common/inlined_functions_private.h:293-308).
 *
 * met_m[ip] = h (target edge size at vertex ip), met_size=1.
 * len = |p-q| / h  when h1≈h2, or |p-q|/(h2-h1) * log(h2/h1) for varying h.
 */
__device__ double device_lenEdg_iso(
    const double *coords, const double *met_m,
    int np, int nq, int met_size)
{
  const double *ca = &coords[3*np];
  const double *cb = &coords[3*nq];
  double dx = cb[0]-ca[0], dy = cb[1]-ca[1], dz = cb[2]-ca[2];
  double l = sqrt(dx*dx + dy*dy + dz*dz);

  if (met_m && met_size == 1) {
    double h1 = met_m[np], h2 = met_m[nq];
    if (h1 <= 0.0) h1 = 1.0;
    if (h2 <= 0.0) h2 = 1.0;
    double r = h2 / h1 - 1.0;
    if (fabs(r) < 1.0e-06)
      return l / h1;
    else
      return l / (h2 - h1) * log1p(r);
  }
  return l;
}

/**
 * Compute edge lengths for all triangle edges.
 * Output: edge_len[3*k + i] = length of edge i in triangle k (1-based).
 *
 * Also marks split/collapse candidates:
 *   edge_mark[3*k+i] = +1 if length > LOPTL (split candidate)
 *                     = -1 if length < LOPTS (collapse candidate)
 *                     =  0 otherwise
 *
 * mode: 0=anisotropic (met_size=6), 1=isotropic (met_size=1 or no metric)
 */
__global__ void kernel_edgeLengths(
    const double *coords,
    const double *met_m,
    const int    *tri_v,       /* 3*(ntmax+1) ints, 1-based */
    const int    *tri_valid,   /* ntmax+1 */
    double       *edge_len,    /* 3*(ntmax+1) output */
    int          *edge_mark,   /* 3*(ntmax+1) output: +1=split, -1=collapse, 0=ok */
    int           nt,
    int           mode,        /* 0=aniso, 1=iso */
    int           met_size)    /* 1 or 6 */
{
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (k > nt) return;
  if (!tri_valid[k]) {
    edge_len[3*k+0] = edge_len[3*k+1] = edge_len[3*k+2] = 0.0;
    edge_mark[3*k+0] = edge_mark[3*k+1] = edge_mark[3*k+2] = 0;
    return;
  }

  int v0 = tri_v[3*k+0], v1 = tri_v[3*k+1], v2 = tri_v[3*k+2];

  double l0, l1, l2;
  /* Edge 0: v1-v2, Edge 1: v0-v2, Edge 2: v0-v1 (mmgs convention) */
  if (mode == 0) {
    l0 = device_lenSurfEdg33_ani(coords, met_m, v1, v2);
    l1 = device_lenSurfEdg33_ani(coords, met_m, v0, v2);
    l2 = device_lenSurfEdg33_ani(coords, met_m, v0, v1);
  } else {
    l0 = device_lenEdg_iso(coords, met_m, v1, v2, met_size);
    l1 = device_lenEdg_iso(coords, met_m, v0, v2, met_size);
    l2 = device_lenEdg_iso(coords, met_m, v0, v1, met_size);
  }

  edge_len[3*k+0] = l0;
  edge_len[3*k+1] = l1;
  edge_len[3*k+2] = l2;

  edge_mark[3*k+0] = (l0 > MMGS_LOPTL) ? 1 : (l0 < MMGS_LOPTS ? -1 : 0);
  edge_mark[3*k+1] = (l1 > MMGS_LOPTL) ? 1 : (l1 < MMGS_LOPTS ? -1 : 0);
  edge_mark[3*k+2] = (l2 > MMGS_LOPTL) ? 1 : (l2 < MMGS_LOPTS ? -1 : 0);
}

/**
 * Count split/collapse candidates via parallel reduction.
 */
__global__ void kernel_countMarks(
    const int *edge_mark,
    const int *tri_valid,
    int nt,
    int *d_nsplit,    /* count of edges > LOPTL */
    int *d_ncollapse) /* count of edges < LOPTS */
{
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (k > nt || !tri_valid[k]) return;

  for (int i = 0; i < 3; i++) {
    int m = edge_mark[3*k+i];
    if (m > 0) atomicAdd(d_nsplit, 1);
    else if (m < 0) atomicAdd(d_ncollapse, 1);
  }
}


/* ================================================================
 * Host wrappers
 * ================================================================ */

extern "C" {

#include "mmgcommon_private.h"
#include "libmmgs_private.h"
#include "mmgs_cuda.h"

/**
 * GPU-accelerated metric initialization at uninitialized points.
 * Replaces MMG5_defUninitSize for the bulk of uninitialized vertices.
 */
int MMGS_defUninitSize_cuda(MMG5_pMesh mesh, MMG5_pSol met, int8_t ismet) {
  int np = (int)mesh->np;
  int npmax = (int)mesh->npmax;
  double isqhmax = 1.0 / (mesh->info.hmax * mesh->info.hmax);

  fprintf(stdout, "[CUDA-MET-S] Initializing metrics for %d points (ismet=%d)\n",
          np, ismet);

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);

  /* ---- Extract AOS → SOA ---- */
  size_t np1 = (size_t)(npmax + 1);
  double *h_normals   = (double*)calloc(3*np1, sizeof(double));
  double *h_xpoint_n1 = NULL;
  int    *h_point_xp  = (int*)calloc(np1, sizeof(int));
  uint16_t *h_tag     = (uint16_t*)calloc(np1, sizeof(uint16_t));
  int    *h_flag      = (int*)calloc(np1, sizeof(int));
  int    *h_valid     = (int*)calloc(np1, sizeof(int));

  for (int k = 1; k <= np; k++) {
    MMG5_pPoint pp = &mesh->point[k];
    h_normals[3*k+0] = pp->n[0];
    h_normals[3*k+1] = pp->n[1];
    h_normals[3*k+2] = pp->n[2];
    h_point_xp[k] = (int)pp->xp;
    h_tag[k] = pp->tag;
    h_flag[k] = (int)pp->flag;
    h_valid[k] = (pp->c[0] != 0.0 || pp->c[1] != 0.0 || pp->c[2] != 0.0 ||
                  pp->ref != 0 || pp->tag != 0) ? 1 : 0;
  }

  /* xpoint normals if available */
  int xpmax = (int)mesh->xpmax;
  if (mesh->xpoint && mesh->xp > 0) {
    h_xpoint_n1 = (double*)calloc(3*(size_t)(xpmax+1), sizeof(double));
    for (int k = 1; k <= (int)mesh->xp; k++) {
      h_xpoint_n1[3*k+0] = mesh->xpoint[k].n1[0];
      h_xpoint_n1[3*k+1] = mesh->xpoint[k].n1[1];
      h_xpoint_n1[3*k+2] = mesh->xpoint[k].n1[2];
    }
  }

  /* ---- GPU alloc + upload ---- */
  double *d_normals, *d_xpoint_n1 = NULL, *d_met;
  int *d_point_xp, *d_flag, *d_valid, *d_out_flag;
  uint16_t *d_tag;

  cudaMalloc(&d_normals, 3*np1*sizeof(double));
  cudaMalloc(&d_point_xp, np1*sizeof(int));
  cudaMalloc(&d_tag, np1*sizeof(uint16_t));
  cudaMalloc(&d_flag, np1*sizeof(int));
  cudaMalloc(&d_valid, np1*sizeof(int));
  cudaMalloc(&d_out_flag, np1*sizeof(int));
  cudaMalloc(&d_met, 6*np1*sizeof(double));

  cudaMemcpy(d_normals, h_normals, 3*np1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_point_xp, h_point_xp, np1*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tag, h_tag, np1*sizeof(uint16_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_flag, h_flag, np1*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_valid, h_valid, np1*sizeof(int), cudaMemcpyHostToDevice);
  /* Upload existing metric (might have partial values from defsiz) */
  cudaMemcpy(d_met, met->m, 6*np1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(d_out_flag, 0, np1*sizeof(int));

  if (h_xpoint_n1) {
    cudaMalloc(&d_xpoint_n1, 3*(size_t)(xpmax+1)*sizeof(double));
    cudaMemcpy(d_xpoint_n1, h_xpoint_n1, 3*(size_t)(xpmax+1)*sizeof(double),
               cudaMemcpyHostToDevice);
  }

  /* ---- Launch ---- */
  int blockSize = 256;
  int numBlocks = (np + blockSize - 1) / blockSize;
  kernel_defUninitSize<<<numBlocks, blockSize>>>(
      NULL, /* coords not needed */
      d_normals, d_xpoint_n1, d_point_xp, d_tag, d_flag, d_valid,
      d_met, d_out_flag, np, ismet, isqhmax);

  /* ---- Download ---- */
  cudaMemcpy(met->m, d_met, 6*np1*sizeof(double), cudaMemcpyDeviceToHost);
  int *h_out_flag = (int*)calloc(np1, sizeof(int));
  cudaMemcpy(h_out_flag, d_out_flag, np1*sizeof(int), cudaMemcpyDeviceToHost);

  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float gpu_ms;
  cudaEventElapsedTime(&gpu_ms, t0, t1);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA-MET-S] CUDA error: %s — falling back to CPU\n",
            cudaGetErrorString(err));
    free(h_normals); free(h_point_xp); free(h_tag); free(h_flag);
    free(h_valid); free(h_out_flag);
    if (h_xpoint_n1) free(h_xpoint_n1);
    cudaFree(d_normals); cudaFree(d_point_xp); cudaFree(d_tag);
    cudaFree(d_flag); cudaFree(d_valid); cudaFree(d_out_flag);
    cudaFree(d_met);
    if (d_xpoint_n1) cudaFree(d_xpoint_n1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    /* Fallback */
    MMG5_defUninitSize(mesh, met, ismet);
    return 1;
  }

  /* Write back flags */
  int count = 0;
  for (int k = 1; k <= np; k++) {
    if (h_out_flag[k] > 0) {
      mesh->point[k].flag = h_out_flag[k];
      count++;
    }
  }

  fprintf(stdout, "[CUDA-MET-S] Initialized %d points [%.3f ms]\n", count, gpu_ms);

  /* Cleanup */
  free(h_normals); free(h_point_xp); free(h_tag); free(h_flag);
  free(h_valid); free(h_out_flag);
  if (h_xpoint_n1) free(h_xpoint_n1);
  cudaFree(d_normals); cudaFree(d_point_xp); cudaFree(d_tag);
  cudaFree(d_flag); cudaFree(d_valid); cudaFree(d_out_flag);
  cudaFree(d_met);
  if (d_xpoint_n1) cudaFree(d_xpoint_n1);
  cudaEventDestroy(t0); cudaEventDestroy(t1);

  return 1;
}

/**
 * GPU-accelerated batch edge length computation for all triangle edges.
 * Computes anisotropic edge lengths and marks split/collapse candidates.
 *
 * Results stored in caller-provided arrays (or allocated if NULL).
 */
int MMGS_edgeLengths_cuda(MMG5_pMesh mesh, MMG5_pSol met,
                          double **out_edge_len, int **out_edge_mark,
                          int *out_nsplit, int *out_ncollapse) {
  int nt = (int)mesh->nt;
  int np = (int)mesh->np;
  int npmax = (int)mesh->npmax;
  int ntmax = (int)mesh->ntmax;

  int mode;  /* 0=aniso, 1=iso */
  if (met && met->m && met->size == 6) {
    mode = 0;
  } else {
    mode = 1;  /* isotropic — use Euclidean lengths scaled by scalar metric */
  }

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);

  /* ---- Extract data ---- */
  size_t np1 = (size_t)(npmax+1), nt1 = (size_t)(ntmax+1);
  double *h_coords = (double*)calloc(3*np1, sizeof(double));
  int *h_tri_v = (int*)calloc(3*nt1, sizeof(int));
  int *h_tri_valid = (int*)calloc(nt1, sizeof(int));

  for (int k = 1; k <= np; k++) {
    h_coords[3*k+0] = mesh->point[k].c[0];
    h_coords[3*k+1] = mesh->point[k].c[1];
    h_coords[3*k+2] = mesh->point[k].c[2];
  }
  for (int k = 1; k <= nt; k++) {
    h_tri_v[3*k+0] = (int)mesh->tria[k].v[0];
    h_tri_v[3*k+1] = (int)mesh->tria[k].v[1];
    h_tri_v[3*k+2] = (int)mesh->tria[k].v[2];
    h_tri_valid[k] = (mesh->tria[k].v[0] > 0) ? 1 : 0;
  }

  /* ---- GPU alloc ---- */
  double *d_coords, *d_met, *d_edge_len;
  int *d_tri_v, *d_tri_valid, *d_edge_mark, *d_nsplit, *d_ncollapse;

  cudaMalloc(&d_coords, 3*np1*sizeof(double));
  cudaMalloc(&d_met, 6*np1*sizeof(double));
  cudaMalloc(&d_tri_v, 3*nt1*sizeof(int));
  cudaMalloc(&d_tri_valid, nt1*sizeof(int));
  cudaMalloc(&d_edge_len, 3*nt1*sizeof(double));
  cudaMalloc(&d_edge_mark, 3*nt1*sizeof(int));
  cudaMalloc(&d_nsplit, sizeof(int));
  cudaMalloc(&d_ncollapse, sizeof(int));

  cudaMemcpy(d_coords, h_coords, 3*np1*sizeof(double), cudaMemcpyHostToDevice);
  int met_size_val = (met && met->m) ? met->size : 0;
  if (met && met->m) {
    cudaMemcpy(d_met, met->m, (size_t)met_size_val*np1*sizeof(double), cudaMemcpyHostToDevice);
  } else {
    cudaMemset(d_met, 0, 6*np1*sizeof(double));
  }
  cudaMemcpy(d_tri_v, h_tri_v, 3*nt1*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tri_valid, h_tri_valid, nt1*sizeof(int), cudaMemcpyHostToDevice);
  int zero = 0;
  cudaMemcpy(d_nsplit, &zero, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ncollapse, &zero, sizeof(int), cudaMemcpyHostToDevice);

  /* ---- Launch ---- */
  int blockSize = 256;
  int numBlocks = (nt + blockSize - 1) / blockSize;

  kernel_edgeLengths<<<numBlocks, blockSize>>>(
      d_coords, d_met, d_tri_v, d_tri_valid,
      d_edge_len, d_edge_mark, nt, mode, met_size_val);

  kernel_countMarks<<<numBlocks, blockSize>>>(
      d_edge_mark, d_tri_valid, nt, d_nsplit, d_ncollapse);

  /* ---- Download ---- */
  double *h_edge_len = (double*)calloc(3*nt1, sizeof(double));
  int *h_edge_mark = (int*)calloc(3*nt1, sizeof(int));
  int nsplit, ncollapse;

  cudaMemcpy(h_edge_len, d_edge_len, 3*nt1*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_edge_mark, d_edge_mark, 3*nt1*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&nsplit, d_nsplit, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ncollapse, d_ncollapse, sizeof(int), cudaMemcpyDeviceToHost);

  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float gpu_ms;
  cudaEventElapsedTime(&gpu_ms, t0, t1);

  fprintf(stdout, "[CUDA-LEN-S] %d tris: %d split candidates, %d collapse candidates [%.3f ms]\n",
          nt, nsplit, ncollapse, gpu_ms);

  /* Return results */
  if (out_edge_len)    *out_edge_len = h_edge_len;    else free(h_edge_len);
  if (out_edge_mark)   *out_edge_mark = h_edge_mark;  else free(h_edge_mark);
  if (out_nsplit)      *out_nsplit = nsplit;
  if (out_ncollapse)   *out_ncollapse = ncollapse;

  /* Cleanup GPU */
  free(h_coords); free(h_tri_v); free(h_tri_valid);
  cudaFree(d_coords); cudaFree(d_met); cudaFree(d_tri_v);
  cudaFree(d_tri_valid); cudaFree(d_edge_len); cudaFree(d_edge_mark);
  cudaFree(d_nsplit); cudaFree(d_ncollapse);
  cudaEventDestroy(t0); cudaEventDestroy(t1);

  return 1;
}

} /* extern "C" */
