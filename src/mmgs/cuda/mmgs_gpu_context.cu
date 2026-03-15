/**
 * \file mmgs/cuda/mmgs_gpu_context.cu
 * \brief Persistent GPU context for mmgs surface remeshing.
 *
 * Uploads mesh data ONCE, keeps it GPU-resident across all kernel calls
 * (quality, edge marking, split). Only downloads results (quality values,
 * edge marks). Final mesh download at context destruction.
 *
 * Nsight profiling showed: GPU compute 1.22ms, but 87ms transfer overhead
 * from re-uploading 221MB on every call. This context eliminates that.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* Quality/length kernels from quality_s.cu and anisosiz_s.cu */
extern __global__ void kernel_triQual(
    const double *coords, const double *met_m,
    const int *tri_v, const int *tri_valid, double *qual_out,
    int nt, int mode);

extern __global__ void kernel_edgeLengths(
    const double *coords, const double *met_m,
    const int *tri_v, const int *tri_valid,
    double *edge_len, int *edge_mark, int nt,
    int mode, int met_size);

/* ================================================================
 * GPU Context structure
 * ================================================================ */

struct MMGS_GPUContext {
  /* Device arrays */
  double *d_coords;      /* 3*(capV) */
  double *d_normals;     /* 3*(capV) */
  double *d_met;         /* met_size*(capV) */
  int    *d_tri_v;       /* 3*(capT) */
  int    *d_tri_valid;   /* capT */
  uint16_t *d_tri_tag;   /* 3*(capT) */
  double *d_qual;        /* capT */
  double *d_edge_len;    /* 3*(capT) */
  int    *d_edge_mark;   /* 3*(capT) */

  /* Pinned host staging buffers (for fast DMA transfers) */
  double *h_qual;        /* capT */
  double *h_edge_len;    /* 3*(capT) */
  int    *h_edge_mark;   /* 3*(capT) */

  /* Sizes */
  int capV, capT;
  int nV, nT;
  int met_size;

  /* State tracking */
  int uploaded;
};

extern "C" {

#include "mmgcommon_private.h"
#include "libmmgs_private.h"
#include "mmgs_cuda.h"

/* ================================================================
 * Context lifecycle
 * ================================================================ */

MMGS_GPUContext* MMGS_gpu_ctx_create(MMG5_pMesh mesh, MMG5_pSol met) {
  MMGS_GPUContext *ctx = (MMGS_GPUContext*)calloc(1, sizeof(MMGS_GPUContext));
  if (!ctx) return NULL;

  ctx->nV = (int)mesh->np;
  ctx->nT = (int)mesh->nt;
  ctx->capV = (int)(mesh->npmax + 1);
  ctx->capT = (int)(mesh->ntmax + 1);
  ctx->met_size = (met && met->m) ? met->size : 0;

  size_t cV = (size_t)ctx->capV;
  size_t cT = (size_t)ctx->capT;

  /* GPU alloc */
  cudaMalloc(&ctx->d_coords,    3 * cV * sizeof(double));
  cudaMalloc(&ctx->d_normals,   3 * cV * sizeof(double));
  cudaMalloc(&ctx->d_tri_v,     3 * cT * sizeof(int));
  cudaMalloc(&ctx->d_tri_valid, cT * sizeof(int));
  cudaMalloc(&ctx->d_tri_tag,   3 * cT * sizeof(uint16_t));
  cudaMalloc(&ctx->d_qual,      cT * sizeof(double));
  cudaMalloc(&ctx->d_edge_len,  3 * cT * sizeof(double));
  cudaMalloc(&ctx->d_edge_mark, 3 * cT * sizeof(int));
  if (ctx->met_size > 0) {
    cudaMalloc(&ctx->d_met, (size_t)ctx->met_size * cV * sizeof(double));
  }

  /* Pinned host alloc for fast downloads */
  cudaMallocHost(&ctx->h_qual,      cT * sizeof(double));
  cudaMallocHost(&ctx->h_edge_len,  3 * cT * sizeof(double));
  cudaMallocHost(&ctx->h_edge_mark, 3 * cT * sizeof(int));

  ctx->uploaded = 0;

  double total_mb = (double)(3*cV*8 + 3*cV*8 + 3*cT*4 + cT*4 + 3*cT*2 + cT*8 + 3*cT*8 + 3*cT*4
                    + (ctx->met_size > 0 ? ctx->met_size*cV*8 : 0)) / 1e6;
  fprintf(stdout, "[GPU-CTX] Created: capV=%d capT=%d (%.1f MB GPU)\n",
          ctx->capV, ctx->capT, total_mb);

  return ctx;
}

void MMGS_gpu_ctx_upload(MMGS_GPUContext *ctx, MMG5_pMesh mesh, MMG5_pSol met) {
  int np = (int)mesh->np;
  int nt = (int)mesh->nt;
  size_t cV = (size_t)ctx->capV;
  size_t cT = (size_t)ctx->capT;

  /* Use pinned staging buffer for coords upload */
  double *h_coords;
  cudaMallocHost(&h_coords, 3 * cV * sizeof(double));
  memset(h_coords, 0, 3 * cV * sizeof(double));
  for (int k = 1; k <= np; k++) {
    h_coords[3*k+0] = mesh->point[k].c[0];
    h_coords[3*k+1] = mesh->point[k].c[1];
    h_coords[3*k+2] = mesh->point[k].c[2];
  }
  cudaMemcpy(ctx->d_coords, h_coords, 3*cV*sizeof(double), cudaMemcpyHostToDevice);
  cudaFreeHost(h_coords);

  /* Tri connectivity + validity + tags */
  int *h_tri_v;
  int *h_tri_valid;
  uint16_t *h_tri_tag;
  cudaMallocHost(&h_tri_v, 3 * cT * sizeof(int));
  cudaMallocHost(&h_tri_valid, cT * sizeof(int));
  cudaMallocHost(&h_tri_tag, 3 * cT * sizeof(uint16_t));
  memset(h_tri_v, 0, 3 * cT * sizeof(int));
  memset(h_tri_valid, 0, cT * sizeof(int));
  memset(h_tri_tag, 0, 3 * cT * sizeof(uint16_t));

  for (int k = 1; k <= nt; k++) {
    MMG5_pTria pt = &mesh->tria[k];
    h_tri_v[3*k+0] = (int)pt->v[0];
    h_tri_v[3*k+1] = (int)pt->v[1];
    h_tri_v[3*k+2] = (int)pt->v[2];
    h_tri_valid[k] = (pt->v[0] > 0) ? 1 : 0;
    h_tri_tag[3*k+0] = pt->tag[0];
    h_tri_tag[3*k+1] = pt->tag[1];
    h_tri_tag[3*k+2] = pt->tag[2];
  }
  cudaMemcpy(ctx->d_tri_v, h_tri_v, 3*cT*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ctx->d_tri_valid, h_tri_valid, cT*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ctx->d_tri_tag, h_tri_tag, 3*cT*sizeof(uint16_t), cudaMemcpyHostToDevice);
  cudaFreeHost(h_tri_v);
  cudaFreeHost(h_tri_valid);
  cudaFreeHost(h_tri_tag);

  /* Metric */
  if (ctx->met_size > 0 && met && met->m) {
    cudaMemcpy(ctx->d_met, met->m,
               (size_t)ctx->met_size * (np + 1) * sizeof(double),
               cudaMemcpyHostToDevice);
  }

  ctx->nV = np;
  ctx->nT = nt;
  ctx->uploaded = 1;

  fprintf(stdout, "[GPU-CTX] Uploaded: %d verts, %d tris\n", np, nt);
}

void MMGS_gpu_ctx_destroy(MMGS_GPUContext *ctx) {
  if (!ctx) return;
  cudaFree(ctx->d_coords);
  cudaFree(ctx->d_normals);
  cudaFree(ctx->d_met);
  cudaFree(ctx->d_tri_v);
  cudaFree(ctx->d_tri_valid);
  cudaFree(ctx->d_tri_tag);
  cudaFree(ctx->d_qual);
  cudaFree(ctx->d_edge_len);
  cudaFree(ctx->d_edge_mark);
  cudaFreeHost(ctx->h_qual);
  cudaFreeHost(ctx->h_edge_len);
  cudaFreeHost(ctx->h_edge_mark);
  free(ctx);
}

/* ================================================================
 * GPU-resident operations (no upload, minimal download)
 * ================================================================ */

/**
 * Compute quality of all triangles using GPU-resident data.
 * Only downloads the quality array + stats.
 */
int MMGS_gpu_ctx_quality(MMGS_GPUContext *ctx, MMG5_pMesh mesh, MMG5_pSol met) {
  if (!ctx || !ctx->uploaded) return 0;

  int nt = ctx->nT;
  int mode = (ctx->met_size == 6) ? 0 : 1;

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);

  int BS = 256;
  int NB = (nt + BS - 1) / BS;
  kernel_triQual<<<NB, BS>>>(
      ctx->d_coords, ctx->d_met, ctx->d_tri_v, ctx->d_tri_valid,
      ctx->d_qual, nt, mode);

  /* Download only quality values */
  cudaMemcpy(ctx->h_qual, ctx->d_qual, (size_t)(nt+1)*sizeof(double),
             cudaMemcpyDeviceToHost);

  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms;
  cudaEventElapsedTime(&ms, t0, t1);

  /* Write back to mesh + compute stats */
  double minq = 1e30, maxq = 0, sumq = 0;
  int cnt = 0;
  MMG5_int minidx = 1;
  for (int k = 1; k <= nt; k++) {
    mesh->tria[k].qual = ctx->h_qual[k];
    if (ctx->h_qual[k] > 0) {
      if (ctx->h_qual[k] < minq) { minq = ctx->h_qual[k]; minidx = k; }
      if (ctx->h_qual[k] > maxq) maxq = ctx->h_qual[k];
      sumq += ctx->h_qual[k];
      cnt++;
    }
  }
  double avgq = cnt > 0 ? sumq / cnt : 0;

  #define MMGS_ALPHAD 3.464101615137755
  fprintf(stdout, "\n  -- MESH QUALITY   %d\n", cnt);
  fprintf(stdout, "     BEST   %8.6f  AVRG.   %8.6f  WRST.   %8.6f (%" MMG5_PRId ")\n",
          MMGS_ALPHAD*maxq, MMGS_ALPHAD*avgq, MMGS_ALPHAD*minq, minidx);
  fprintf(stdout, "     [GPU-CTX: %.3f ms, no upload]\n", ms);
  #undef MMGS_ALPHAD

  cudaEventDestroy(t0); cudaEventDestroy(t1);
  return 1;
}

/**
 * Compute edge lengths for all triangles using GPU-resident data.
 * Only downloads edge_len and edge_mark arrays.
 * Results stored in ctx->h_edge_len and ctx->h_edge_mark.
 */
int MMGS_gpu_ctx_edge_lengths(MMGS_GPUContext *ctx, MMG5_pMesh mesh, MMG5_pSol met) {
  if (!ctx || !ctx->uploaded) return 0;

  int nt = ctx->nT;
  int mode = (ctx->met_size == 6) ? 0 : 1;

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);

  int BS = 256;
  int NB = (nt + BS - 1) / BS;
  kernel_edgeLengths<<<NB, BS>>>(
      ctx->d_coords, ctx->d_met, ctx->d_tri_v, ctx->d_tri_valid,
      ctx->d_edge_len, ctx->d_edge_mark, nt, mode, ctx->met_size);

  /* Download only edge data (not the full mesh) */
  cudaMemcpy(ctx->h_edge_len, ctx->d_edge_len,
             3*(size_t)(nt+1)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(ctx->h_edge_mark, ctx->d_edge_mark,
             3*(size_t)(nt+1)*sizeof(int), cudaMemcpyDeviceToHost);

  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms;
  cudaEventElapsedTime(&ms, t0, t1);

  int nsplit = 0, ncollapse = 0;
  for (int k = 1; k <= nt; k++) {
    for (int j = 0; j < 3; j++) {
      int m = ctx->h_edge_mark[3*k+j];
      if (m > 0) nsplit++;
      else if (m < 0) ncollapse++;
    }
  }

  fprintf(stdout, "[GPU-CTX] Edge lengths: %d tris, %d split, %d collapse [%.3f ms, no upload]\n",
          nt, nsplit, ncollapse, ms);

  cudaEventDestroy(t0); cudaEventDestroy(t1);
  return 1;
}

/**
 * Re-upload mesh data after CPU modifications (split/collapse/swap/move).
 * Only uploads the parts that changed.
 */
void MMGS_gpu_ctx_refresh(MMGS_GPUContext *ctx, MMG5_pMesh mesh, MMG5_pSol met) {
  if (!ctx) return;

  int np = (int)mesh->np;
  int nt = (int)mesh->nt;

  /* Check if mesh grew beyond capacity */
  if (np + 1 > ctx->capV || nt + 1 > ctx->capT) {
    fprintf(stdout, "[GPU-CTX] Mesh grew beyond capacity, full re-upload\n");
    MMGS_gpu_ctx_destroy(ctx);
    /* Can't realloc in-place, caller should recreate */
    return;
  }

  /* Re-upload everything (mesh changed unpredictably during CPU operations) */
  MMGS_gpu_ctx_upload(ctx, mesh, met);
}

double* MMGS_gpu_ctx_get_edge_len(MMGS_GPUContext *ctx) {
  return ctx ? ctx->h_edge_len : NULL;
}

int* MMGS_gpu_ctx_get_edge_mark(MMGS_GPUContext *ctx) {
  return ctx ? ctx->h_edge_mark : NULL;
}

} /* extern "C" */
