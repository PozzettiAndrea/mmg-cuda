/**
 * \file mmgs/cuda/mmgs_cuda.h
 * \brief CUDA acceleration, checkpointing, and strategy dispatch for mmgs.
 *
 * Pipeline stage definitions, checkpoint save/load API,
 * and GPU kernel dispatch declarations for surface remeshing.
 */

#ifndef MMGS_CUDA_H
#define MMGS_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Use the real mmg types if already included, otherwise forward-declare. */
#ifndef _LIBMMGTYPES_H
  struct MMG5_Mesh_s;
  struct MMG5_Sol_s;
  typedef struct MMG5_Mesh_s MMG5_Mesh;
  typedef struct MMG5_Sol_s  MMG5_Sol;
  typedef MMG5_Mesh * MMG5_pMesh;
  typedef MMG5_Sol  * MMG5_pSol;
#endif

/* ================================================================
 * Pipeline stages for checkpointing
 * ================================================================ */

typedef enum {
  MMGS_STAGE_NONE        = -1,
  MMGS_STAGE_POST_LOAD   =  0,  /* after loading + input validation       */
  MMGS_STAGE_POST_SCALE  =  1,  /* after MMG5_scaleMesh + setfunc         */
  MMGS_STAGE_POST_ANALYS =  2,  /* after MMGS_analys                      */
  MMGS_STAGE_POST_INQUA  =  3,  /* after MMGS_inqua (initial quality)     */
  MMGS_STAGE_POST_GEOM   =  4,  /* after geometric mesh (anatri stage 1)  */
  MMGS_STAGE_POST_DEFSIZ =  5,  /* after MMGS_defsiz                      */
  MMGS_STAGE_POST_GRADSIZ=  6,  /* after MMGS_gradsiz                     */
  MMGS_STAGE_POST_ADAPT  =  7,  /* after full adaptation (anatri+adptri)  */
  MMGS_STAGE_POST_UNSCALE=  8,  /* after MMG5_unscaleMesh                 */
  MMGS_STAGE_COUNT        =  9
} MMGS_PipelineStage;

#ifdef __cplusplus
extern "C" {
#endif

/** Convert stage name string to enum value. Returns MMGS_STAGE_NONE on error. */
MMGS_PipelineStage MMGS_cuda_stage_from_name(const char *name);

/** Return the printable name for a stage. */
const char* MMGS_cuda_stage_name(MMGS_PipelineStage s);

/** Print all stage names to stdout. */
void MMGS_cuda_list_stages(void);

/* ================================================================
 * Checkpoint save / load
 * ================================================================ */

/** 576-byte binary header written at the start of every .msc file. */
typedef struct {
  char      magic[4];             /* "MSC\0"                          */
  int       version;              /* format version = 1               */
  char      stage[64];            /* stage name string                */
  /* strategy flags */
  int       quality_strategy;
  int       metvol_strategy;
  int       gradation_strategy;
  /* mmg info params needed for resume */
  double    dhd, hmin, hmax, hsiz, hgrad, hgradreq, hausd;
  int       target_faces;         /* reserved                         */
  char      input_mesh[256];      /* mesh->namein                     */
  long long timestamp;
  char      reserved[148];        /* padding to 576 bytes             */
} MMGS_CheckpointHeader;

/**
 * Save full mesh+metric state at the given pipeline stage.
 * \return 1 on success, 0 on failure.
 */
int MMGS_save_checkpoint(MMG5_pMesh mesh, MMG5_pSol met,
                         MMGS_PipelineStage stage, const char *dir);

/**
 * Load mesh+metric state from a checkpoint file.
 * \return the loaded stage on success, MMGS_STAGE_NONE on failure.
 */
MMGS_PipelineStage MMGS_load_checkpoint(MMG5_pMesh mesh, MMG5_pSol met,
                                        const char *dir,
                                        MMGS_PipelineStage stage);

/** Check whether a checkpoint file exists for the given stage. */
int MMGS_checkpoint_exists(const char *dir, MMGS_PipelineStage stage);

/* ================================================================
 * GPU kernel dispatch declarations
 * ================================================================ */

#ifdef WITH_CUDA

/**
 * GPU-accelerated quality computation for all triangles.
 * Computes quality using MMG5_caltri33_ani (anisotropic) or
 * MMG5_caltri_iso (isotropic) and stores in mesh->tria[k].qual.
 * Also computes and reports min/max/avg quality.
 *
 * \return 1 if success, 0 if worst quality is below threshold.
 */
int MMGS_triQual_cuda(MMG5_pMesh mesh, MMG5_pSol met);

/**
 * GPU-accelerated metric initialization at uninitialized points.
 * Sets isqhmax-based isotropic metric at all unflagged interior vertices.
 * Replaces MMG5_defUninitSize for batch processing.
 *
 * \return 1 on success, 0 on failure.
 */
int MMGS_defUninitSize_cuda(MMG5_pMesh mesh, MMG5_pSol met, int8_t ismet);

/**
 * GPU-accelerated batch edge length computation.
 * Computes anisotropic edge lengths for all triangle edges and marks
 * split/collapse candidates.
 *
 * \param out_edge_len  If non-NULL, receives allocated array of 3*ntmax+1 edge lengths
 * \param out_edge_mark If non-NULL, receives allocated array of marks (+1=split, -1=collapse)
 * \param out_nsplit    If non-NULL, receives count of split candidates
 * \param out_ncollapse If non-NULL, receives count of collapse candidates
 * \return 1 on success, 0 on failure.
 */
int MMGS_edgeLengths_cuda(MMG5_pMesh mesh, MMG5_pSol met,
                          double **out_edge_len, int **out_edge_mark,
                          int *out_nsplit, int *out_ncollapse);

#endif /* WITH_CUDA */

#ifdef __cplusplus
}
#endif

#endif /* MMGS_CUDA_H */
