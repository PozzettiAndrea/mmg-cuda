/**
 * \file mmg3d/cuda/mmg3d_cuda.h
 * \brief CUDA acceleration, checkpointing, and strategy dispatch for mmg3d.
 *
 * Provides pipeline stage definitions, checkpoint save/load API,
 * and GPU kernel dispatch declarations for anisotropic remeshing.
 */

#ifndef MMG3D_CUDA_H
#define MMG3D_CUDA_H

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
  MMG3D_STAGE_NONE        = -1,
  MMG3D_STAGE_POST_LOAD   =  0,  /* after loading + input validation       */
  MMG3D_STAGE_POST_SCALE  =  1,  /* after MMG5_scaleMesh + setfunc         */
  MMG3D_STAGE_POST_QUAL1  =  2,  /* after first MMG3D_tetraQual            */
  MMG3D_STAGE_POST_ANALYS =  3,  /* after MMG3D_analys                     */
  MMG3D_STAGE_POST_DEFSIZ =  4,  /* after MMG3D_defsiz_ani                 */
  MMG3D_STAGE_POST_GRADSIZ=  5,  /* after MMG3D_gradsiz_ani                */
  MMG3D_STAGE_POST_QUAL2  =  6,  /* after second MMG3D_tetraQual           */
  MMG3D_STAGE_POST_ADAPT  =  7,  /* after adaptation loop (adptet/adpdel)  */
  MMG3D_STAGE_POST_UNSCALE=  8,  /* after MMG5_unscaleMesh                 */
  MMG3D_STAGE_COUNT        =  9
} MMG3D_PipelineStage;

#ifdef __cplusplus
extern "C" {
#endif

/** Convert stage name string to enum value. Returns MMG3D_STAGE_NONE on error. */
MMG3D_PipelineStage MMG3D_stage_from_name(const char *name);

/** Return the printable name for a stage. */
const char* MMG3D_stage_name(MMG3D_PipelineStage s);

/** Print all stage names to stdout. */
void MMG3D_list_stages(void);

/* ================================================================
 * Checkpoint save / load
 * ================================================================ */

/** 576-byte binary header written at the start of every .m3c file. */
typedef struct {
  char      magic[4];             /* "M3C\0"                          */
  int       version;              /* format version = 1               */
  char      stage[64];            /* stage name string                */
  /* strategy flags */
  int       quality_strategy;
  int       metvol_strategy;
  int       gradation_strategy;
  /* mmg info params needed for resume */
  double    dhd, hmin, hmax, hsiz, hgrad, hgradreq, hausd;
  int       target_faces;         /* unused for mmg (reserved)        */
  char      input_mesh[256];      /* mesh->namein                     */
  long long timestamp;
  char      reserved[148];        /* padding to 576 bytes             */
} MMG3D_CheckpointHeader;

/**
 * Save full mesh+metric state at the given pipeline stage.
 * \return 1 on success, 0 on failure.
 */
int MMG3D_save_checkpoint(MMG5_pMesh mesh, MMG5_pSol met,
                          MMG3D_PipelineStage stage, const char *dir);

/**
 * Load mesh+metric state from a checkpoint file.
 * \return the loaded stage on success, MMG3D_STAGE_NONE on failure.
 */
MMG3D_PipelineStage MMG3D_load_checkpoint(MMG5_pMesh mesh, MMG5_pSol met,
                                          const char *dir,
                                          MMG3D_PipelineStage stage);

/** Check whether a checkpoint file exists for the given stage. */
int MMG3D_checkpoint_exists(const char *dir, MMG3D_PipelineStage stage);

/* ================================================================
 * GPU kernel dispatch declarations (Sprint 3+)
 * ================================================================ */

#ifdef WITH_CUDA

/** GPU-accelerated quality computation for all tetrahedra. */
int MMG3D_tetraQual_cuda(MMG5_pMesh mesh, MMG5_pSol met, int8_t metRidTyp);

/** GPU-accelerated metric definition at internal vertices. */
int MMG5_defmetvol_cuda(MMG5_pMesh mesh, MMG5_pSol met, int8_t ismet);

/** GPU-accelerated iterative metric gradation. */
int MMG3D_gradsiz_ani_cuda(MMG5_pMesh mesh, MMG5_pSol met);

#endif /* WITH_CUDA */

#ifdef __cplusplus
}
#endif

#endif /* MMG3D_CUDA_H */
