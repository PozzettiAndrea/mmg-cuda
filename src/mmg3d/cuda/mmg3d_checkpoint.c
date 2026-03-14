/**
 * \file mmg3d/cuda/mmg3d_checkpoint.c
 * \brief Binary checkpoint save/load for mmg3d pipeline.
 *
 * Serializes full MMG5_Mesh + MMG5_Sol state at pipeline stage boundaries.
 * File format: .m3c (mmg3d checkpoint) — 576-byte header + field-by-field data.
 */

/* Include mmg headers first so types are fully defined before mmg3d_cuda.h */
#include "mmgcommon_private.h"
#include "libmmg3d_private.h"
#include "mmg3d_cuda.h"

#include <sys/stat.h>
#include <time.h>

/* ================================================================
 * Stage name mapping
 * ================================================================ */

static const char* stage_names[MMG3D_STAGE_COUNT] = {
  "post-load",
  "post-scale",
  "post-quality1",
  "post-analys",
  "post-defsiz",
  "post-gradsiz",
  "post-quality2",
  "post-adapt",
  "post-unscale"
};

MMG3D_PipelineStage MMG3D_stage_from_name(const char *name) {
  int i;
  for (i = 0; i < MMG3D_STAGE_COUNT; ++i) {
    if (strcmp(name, stage_names[i]) == 0) return (MMG3D_PipelineStage)i;
  }
  return MMG3D_STAGE_NONE;
}

const char* MMG3D_stage_name(MMG3D_PipelineStage s) {
  if (s >= 0 && s < MMG3D_STAGE_COUNT) return stage_names[s];
  return "unknown";
}

void MMG3D_list_stages(void) {
  int i;
  fprintf(stdout, "Pipeline stages:\n");
  for (i = 0; i < MMG3D_STAGE_COUNT; ++i) {
    fprintf(stdout, "  %2d. %s\n", i, stage_names[i]);
  }
}

/* ================================================================
 * Helpers
 * ================================================================ */

static char* checkpoint_path(const char *dir, MMG3D_PipelineStage stage) {
  static char buf[512];
  snprintf(buf, sizeof(buf), "%s/%s.m3c", dir, stage_names[stage]);
  return buf;
}

int MMG3D_checkpoint_exists(const char *dir, MMG3D_PipelineStage stage) {
  struct stat st;
  return (stat(checkpoint_path(dir, stage), &st) == 0);
}

/* Field-by-field I/O helpers */
static int write_int(FILE *fp, int v)      { return fwrite(&v, sizeof(int), 1, fp) == 1; }
static int write_lint(FILE *fp, MMG5_int v){ return fwrite(&v, sizeof(MMG5_int), 1, fp) == 1; }
static int write_dbl(FILE *fp, double v)   { return fwrite(&v, sizeof(double), 1, fp) == 1; }

static int read_int(FILE *fp, int *v)      { return fread(v, sizeof(int), 1, fp) == 1; }
static int read_lint(FILE *fp, MMG5_int *v){ return fread(v, sizeof(MMG5_int), 1, fp) == 1; }
static int read_dbl(FILE *fp, double *v)   { return fread(v, sizeof(double), 1, fp) == 1; }

/* ================================================================
 * Save checkpoint
 * ================================================================ */

int MMG3D_save_checkpoint(MMG5_pMesh mesh, MMG5_pSol met,
                          MMG3D_PipelineStage stage, const char *dir) {
  FILE *fp;
  char *path;
  MMG3D_CheckpointHeader hdr;
  long file_size;
  struct stat st;

  /* Ensure directory exists */
#ifndef _WIN32
  mkdir(dir, 0755);
#else
  _mkdir(dir);
#endif

  path = checkpoint_path(dir, stage);
  fp = fopen(path, "wb");
  if (!fp) {
    fprintf(stderr, "[CHECKPOINT] ERROR: Cannot open %s for writing\n", path);
    return 0;
  }

  /* ---- Header ---- */
  memset(&hdr, 0, sizeof(hdr));
  memcpy(hdr.magic, "M3C", 4);
  hdr.version = 1;
  strncpy(hdr.stage, stage_names[stage], sizeof(hdr.stage) - 1);

#ifdef WITH_CUDA
  hdr.quality_strategy   = mesh->info.quality_strategy;
  hdr.metvol_strategy    = mesh->info.metvol_strategy;
  hdr.gradation_strategy = mesh->info.gradation_strategy;
#endif

  hdr.dhd      = mesh->info.dhd;
  hdr.hmin     = mesh->info.hmin;
  hdr.hmax     = mesh->info.hmax;
  hdr.hsiz     = mesh->info.hsiz;
  hdr.hgrad    = mesh->info.hgrad;
  hdr.hgradreq = mesh->info.hgradreq;
  hdr.hausd    = mesh->info.hausd;

  if (mesh->namein)
    strncpy(hdr.input_mesh, mesh->namein, sizeof(hdr.input_mesh) - 1);
  hdr.timestamp = (long long)time(NULL);

  fwrite(&hdr, sizeof(hdr), 1, fp);

  /* ---- Stage index ---- */
  write_int(fp, (int)stage);

  /* ---- Mesh counts ---- */
  write_lint(fp, mesh->np);
  write_lint(fp, mesh->ne);
  write_lint(fp, mesh->nt);
  write_lint(fp, mesh->na);
  write_lint(fp, mesh->xp);
  write_lint(fp, mesh->xt);
  write_lint(fp, mesh->npmax);
  write_lint(fp, mesh->nemax);
  write_lint(fp, mesh->ntmax);
  write_lint(fp, mesh->namax);
  write_lint(fp, mesh->xpmax);
  write_lint(fp, mesh->xtmax);
  write_lint(fp, mesh->nquad);
  write_lint(fp, mesh->nprism);
  write_int(fp, mesh->ver);
  write_int(fp, mesh->dim);
  write_int(fp, mesh->nsols);

  /* ---- Points (1-based) ---- */
  if (mesh->np > 0 && mesh->point) {
    /* Write field-by-field for portability */
    MMG5_int k;
    for (k = 1; k <= mesh->np; ++k) {
      MMG5_pPoint pp = &mesh->point[k];
      fwrite(pp->c, sizeof(double), 3, fp);
      fwrite(pp->n, sizeof(double), 3, fp);
      write_lint(fp, pp->ref);
      write_lint(fp, pp->xp);
      write_lint(fp, pp->tmp);
      write_lint(fp, pp->flag);
      write_lint(fp, pp->s);
      fwrite(&pp->tag, sizeof(uint16_t), 1, fp);
      fwrite(&pp->tagdel, sizeof(int8_t), 1, fp);
    }
  }

  /* ---- Tetrahedra (1-based) ---- */
  if (mesh->ne > 0 && mesh->tetra) {
    MMG5_int k;
    for (k = 1; k <= mesh->ne; ++k) {
      MMG5_pTetra pt = &mesh->tetra[k];
      write_dbl(fp, pt->qual);
      fwrite(pt->v, sizeof(MMG5_int), 4, fp);
      write_lint(fp, pt->ref);
      write_lint(fp, pt->base);
      write_lint(fp, pt->mark);
      write_lint(fp, pt->xt);
      write_lint(fp, pt->flag);
      fwrite(&pt->tag, sizeof(uint16_t), 1, fp);
    }
  }

  /* ---- xTetra (1-based) ---- */
  if (mesh->xt > 0 && mesh->xtetra) {
    MMG5_int k;
    for (k = 1; k <= mesh->xt; ++k) {
      MMG5_pxTetra pxt = &mesh->xtetra[k];
      fwrite(pxt->ref, sizeof(MMG5_int), 4, fp);
      fwrite(pxt->edg, sizeof(MMG5_int), 6, fp);
      fwrite(pxt->ftag, sizeof(uint16_t), 4, fp);
      fwrite(pxt->tag, sizeof(uint16_t), 6, fp);
      fwrite(&pxt->ori, sizeof(int8_t), 1, fp);
    }
  }

  /* ---- xPoint (1-based) ---- */
  if (mesh->xp > 0 && mesh->xpoint) {
    MMG5_int k;
    for (k = 1; k <= mesh->xp; ++k) {
      MMG5_pxPoint pxp = &mesh->xpoint[k];
      fwrite(pxp->n1, sizeof(double), 3, fp);
      fwrite(pxp->n2, sizeof(double), 3, fp);
      fwrite(&pxp->nnor, sizeof(int8_t), 1, fp);
    }
  }

  /* ---- Triangles (1-based) ---- */
  if (mesh->nt > 0 && mesh->tria) {
    MMG5_int k;
    for (k = 1; k <= mesh->nt; ++k) {
      MMG5_pTria ptt = &mesh->tria[k];
      write_dbl(fp, ptt->qual);
      fwrite(ptt->v, sizeof(MMG5_int), 3, fp);
      write_lint(fp, ptt->ref);
      write_lint(fp, ptt->base);
      write_lint(fp, ptt->cc);
      fwrite(ptt->edg, sizeof(MMG5_int), 3, fp);
      write_lint(fp, ptt->flag);
      fwrite(ptt->tag, sizeof(uint16_t), 3, fp);
    }
  }

  /* ---- Edges (1-based) ---- */
  if (mesh->na > 0 && mesh->edge) {
    MMG5_int k;
    for (k = 1; k <= mesh->na; ++k) {
      MMG5_pEdge pa = &mesh->edge[k];
      write_lint(fp, pa->a);
      write_lint(fp, pa->b);
      write_lint(fp, pa->ref);
      write_lint(fp, pa->base);
      fwrite(&pa->tag, sizeof(uint16_t), 1, fp);
    }
  }

  /* ---- Adjacency ---- */
  {
    int has_adja = (mesh->adja != NULL) ? 1 : 0;
    write_int(fp, has_adja);
    if (has_adja && mesh->ne > 0) {
      fwrite(mesh->adja + 1, sizeof(MMG5_int), 4 * mesh->ne, fp);
    }
  }

  /* ---- Info flags ---- */
  {
    fwrite(&mesh->info.ani, sizeof(uint8_t), 1, fp);
    fwrite(&mesh->info.optim, sizeof(uint8_t), 1, fp);
    fwrite(&mesh->info.optimLES, sizeof(uint8_t), 1, fp);
    fwrite(&mesh->info.noinsert, sizeof(uint8_t), 1, fp);
    fwrite(&mesh->info.noswap, sizeof(uint8_t), 1, fp);
    fwrite(&mesh->info.nomove, sizeof(uint8_t), 1, fp);
    fwrite(&mesh->info.nosurf, sizeof(uint8_t), 1, fp);
    fwrite(&mesh->info.nosizreq, sizeof(uint8_t), 1, fp);
    fwrite(&mesh->info.metRidTyp, sizeof(uint8_t), 1, fp);
    write_int(fp, mesh->info.renum);
    write_int(fp, mesh->info.PROctree);
    write_int(fp, mesh->info.imprim);
    fwrite(&mesh->info.fem, sizeof(int8_t), 1, fp);
    fwrite(&mesh->info.iso, sizeof(int8_t), 1, fp);
    fwrite(&mesh->info.lag, sizeof(int8_t), 1, fp);
    fwrite(&mesh->info.nreg, sizeof(int8_t), 1, fp);
    fwrite(&mesh->info.xreg, sizeof(int8_t), 1, fp);
  }

  /* ---- Metric / solution data ---- */
  {
    int has_met = (met && met->m && met->np > 0) ? 1 : 0;
    write_int(fp, has_met);
    if (has_met) {
      write_lint(fp, met->np);
      write_int(fp, met->size);
      write_int(fp, met->type);
      /* met->m is 0-based, length = met->size * (met->np + 1) */
      fwrite(met->m, sizeof(double), (size_t)met->size * (met->np + 1), fp);
    }
  }

  fclose(fp);

  /* Report */
  file_size = 0;
  if (stat(path, &st) == 0) file_size = st.st_size;
  fprintf(stdout, "[CHECKPOINT] Saved '%s' to %s (%.1f MB)\n",
          stage_names[stage], path, file_size / (1024.0 * 1024.0));

  return 1;
}

/* ================================================================
 * Load checkpoint
 * ================================================================ */

MMG3D_PipelineStage MMG3D_load_checkpoint(MMG5_pMesh mesh, MMG5_pSol met,
                                          const char *dir,
                                          MMG3D_PipelineStage stage) {
  FILE *fp;
  char *path;
  MMG3D_CheckpointHeader hdr;
  int stage_idx;
  MMG3D_PipelineStage saved_stage;

  path = checkpoint_path(dir, stage);
  fp = fopen(path, "rb");
  if (!fp) {
    fprintf(stderr, "[CHECKPOINT] ERROR: Cannot open %s for reading\n", path);
    return MMG3D_STAGE_NONE;
  }

  /* ---- Header ---- */
  if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
    fprintf(stderr, "[CHECKPOINT] ERROR: Failed to read header from %s\n", path);
    fclose(fp);
    return MMG3D_STAGE_NONE;
  }
  if (memcmp(hdr.magic, "M3C", 4) != 0) {
    fprintf(stderr, "[CHECKPOINT] ERROR: Invalid magic in %s\n", path);
    fclose(fp);
    return MMG3D_STAGE_NONE;
  }
  if (hdr.version != 1) {
    fprintf(stderr, "[CHECKPOINT] ERROR: Unsupported version %d in %s\n",
            hdr.version, path);
    fclose(fp);
    return MMG3D_STAGE_NONE;
  }

  read_int(fp, &stage_idx);
  saved_stage = (MMG3D_PipelineStage)stage_idx;

  fprintf(stdout, "[CHECKPOINT] Loading '%s' from %s\n",
          stage_names[saved_stage], path);
  fprintf(stdout, "[CHECKPOINT]   strategies: quality=%d metvol=%d gradation=%d\n",
          hdr.quality_strategy, hdr.metvol_strategy, hdr.gradation_strategy);
  fprintf(stdout, "[CHECKPOINT]   input mesh: %s\n", hdr.input_mesh);

  /* Restore info params from header */
  mesh->info.dhd      = hdr.dhd;
  mesh->info.hmin     = hdr.hmin;
  mesh->info.hmax     = hdr.hmax;
  mesh->info.hsiz     = hdr.hsiz;
  mesh->info.hgrad    = hdr.hgrad;
  mesh->info.hgradreq = hdr.hgradreq;
  mesh->info.hausd    = hdr.hausd;

#ifdef WITH_CUDA
  mesh->info.quality_strategy   = hdr.quality_strategy;
  mesh->info.metvol_strategy    = hdr.metvol_strategy;
  mesh->info.gradation_strategy = hdr.gradation_strategy;
#endif

  /* ---- Mesh counts ---- */
  {
    MMG5_int np, ne, nt, na, xp, xt, npmax, nemax, ntmax, namax, xpmax, xtmax;
    MMG5_int nquad, nprism;
    int ver, dim, nsols;

    read_lint(fp, &np);
    read_lint(fp, &ne);
    read_lint(fp, &nt);
    read_lint(fp, &na);
    read_lint(fp, &xp);
    read_lint(fp, &xt);
    read_lint(fp, &npmax);
    read_lint(fp, &nemax);
    read_lint(fp, &ntmax);
    read_lint(fp, &namax);
    read_lint(fp, &xpmax);
    read_lint(fp, &xtmax);
    read_lint(fp, &nquad);
    read_lint(fp, &nprism);
    read_int(fp, &ver);
    read_int(fp, &dim);
    read_int(fp, &nsols);

    mesh->np     = np;
    mesh->ne     = ne;
    mesh->nt     = nt;
    mesh->na     = na;
    mesh->xp     = xp;
    mesh->xt     = xt;
    mesh->npmax  = npmax;
    mesh->nemax  = nemax;
    mesh->ntmax  = ntmax;
    mesh->namax  = namax;
    mesh->xpmax  = xpmax;
    mesh->xtmax  = xtmax;
    mesh->nquad  = nquad;
    mesh->nprism = nprism;
    mesh->ver    = ver;
    mesh->dim    = dim;
    mesh->nsols  = nsols;
  }

  /* ---- Points ---- */
  if (mesh->np > 0) {
    MMG5_int k;
    MMG5_SAFE_CALLOC(mesh->point, mesh->npmax + 1, MMG5_Point,
                     fclose(fp); return MMG3D_STAGE_NONE);
    for (k = 1; k <= mesh->np; ++k) {
      MMG5_pPoint pp = &mesh->point[k];
      fread(pp->c, sizeof(double), 3, fp);
      fread(pp->n, sizeof(double), 3, fp);
      read_lint(fp, &pp->ref);
      read_lint(fp, &pp->xp);
      read_lint(fp, &pp->tmp);
      read_lint(fp, &pp->flag);
      read_lint(fp, &pp->s);
      fread(&pp->tag, sizeof(uint16_t), 1, fp);
      fread(&pp->tagdel, sizeof(int8_t), 1, fp);
    }
  }

  /* ---- Tetrahedra ---- */
  if (mesh->ne > 0) {
    MMG5_int k;
    MMG5_SAFE_CALLOC(mesh->tetra, mesh->nemax + 1, MMG5_Tetra,
                     fclose(fp); return MMG3D_STAGE_NONE);
    for (k = 1; k <= mesh->ne; ++k) {
      MMG5_pTetra pt = &mesh->tetra[k];
      read_dbl(fp, &pt->qual);
      fread(pt->v, sizeof(MMG5_int), 4, fp);
      read_lint(fp, &pt->ref);
      read_lint(fp, &pt->base);
      read_lint(fp, &pt->mark);
      read_lint(fp, &pt->xt);
      read_lint(fp, &pt->flag);
      fread(&pt->tag, sizeof(uint16_t), 1, fp);
    }
  }

  /* ---- xTetra ---- */
  if (mesh->xt > 0) {
    MMG5_int k;
    MMG5_SAFE_CALLOC(mesh->xtetra, mesh->xtmax + 1, MMG5_xTetra,
                     fclose(fp); return MMG3D_STAGE_NONE);
    for (k = 1; k <= mesh->xt; ++k) {
      MMG5_pxTetra pxt = &mesh->xtetra[k];
      fread(pxt->ref, sizeof(MMG5_int), 4, fp);
      fread(pxt->edg, sizeof(MMG5_int), 6, fp);
      fread(pxt->ftag, sizeof(uint16_t), 4, fp);
      fread(pxt->tag, sizeof(uint16_t), 6, fp);
      fread(&pxt->ori, sizeof(int8_t), 1, fp);
    }
  }

  /* ---- xPoint ---- */
  if (mesh->xp > 0) {
    MMG5_int k;
    MMG5_SAFE_CALLOC(mesh->xpoint, mesh->xpmax + 1, MMG5_xPoint,
                     fclose(fp); return MMG3D_STAGE_NONE);
    for (k = 1; k <= mesh->xp; ++k) {
      MMG5_pxPoint pxp = &mesh->xpoint[k];
      fread(pxp->n1, sizeof(double), 3, fp);
      fread(pxp->n2, sizeof(double), 3, fp);
      fread(&pxp->nnor, sizeof(int8_t), 1, fp);
    }
  }

  /* ---- Triangles ---- */
  if (mesh->nt > 0) {
    MMG5_int k;
    MMG5_SAFE_CALLOC(mesh->tria, mesh->ntmax + 1, MMG5_Tria,
                     fclose(fp); return MMG3D_STAGE_NONE);
    for (k = 1; k <= mesh->nt; ++k) {
      MMG5_pTria ptt = &mesh->tria[k];
      read_dbl(fp, &ptt->qual);
      fread(ptt->v, sizeof(MMG5_int), 3, fp);
      read_lint(fp, &ptt->ref);
      read_lint(fp, &ptt->base);
      read_lint(fp, &ptt->cc);
      fread(ptt->edg, sizeof(MMG5_int), 3, fp);
      read_lint(fp, &ptt->flag);
      fread(ptt->tag, sizeof(uint16_t), 3, fp);
    }
  }

  /* ---- Edges ---- */
  if (mesh->na > 0) {
    MMG5_int k;
    MMG5_SAFE_CALLOC(mesh->edge, mesh->namax + 1, MMG5_Edge,
                     fclose(fp); return MMG3D_STAGE_NONE);
    for (k = 1; k <= mesh->na; ++k) {
      MMG5_pEdge pa = &mesh->edge[k];
      read_lint(fp, &pa->a);
      read_lint(fp, &pa->b);
      read_lint(fp, &pa->ref);
      read_lint(fp, &pa->base);
      fread(&pa->tag, sizeof(uint16_t), 1, fp);
    }
  }

  /* ---- Adjacency ---- */
  {
    int has_adja;
    read_int(fp, &has_adja);
    if (has_adja && mesh->ne > 0) {
      MMG5_SAFE_CALLOC(mesh->adja, 4 * mesh->nemax + 5, MMG5_int,
                       fclose(fp); return MMG3D_STAGE_NONE);
      fread(mesh->adja + 1, sizeof(MMG5_int), 4 * mesh->ne, fp);
    }
  }

  /* ---- Info flags ---- */
  {
    fread(&mesh->info.ani, sizeof(uint8_t), 1, fp);
    fread(&mesh->info.optim, sizeof(uint8_t), 1, fp);
    fread(&mesh->info.optimLES, sizeof(uint8_t), 1, fp);
    fread(&mesh->info.noinsert, sizeof(uint8_t), 1, fp);
    fread(&mesh->info.noswap, sizeof(uint8_t), 1, fp);
    fread(&mesh->info.nomove, sizeof(uint8_t), 1, fp);
    fread(&mesh->info.nosurf, sizeof(uint8_t), 1, fp);
    fread(&mesh->info.nosizreq, sizeof(uint8_t), 1, fp);
    fread(&mesh->info.metRidTyp, sizeof(uint8_t), 1, fp);
    read_int(fp, &mesh->info.renum);
    read_int(fp, &mesh->info.PROctree);
    read_int(fp, &mesh->info.imprim);
    fread(&mesh->info.fem, sizeof(int8_t), 1, fp);
    fread(&mesh->info.iso, sizeof(int8_t), 1, fp);
    fread(&mesh->info.lag, sizeof(int8_t), 1, fp);
    fread(&mesh->info.nreg, sizeof(int8_t), 1, fp);
    fread(&mesh->info.xreg, sizeof(int8_t), 1, fp);
  }

  /* ---- Metric / solution data ---- */
  {
    int has_met;
    read_int(fp, &has_met);
    if (has_met && met) {
      read_lint(fp, &met->np);
      read_int(fp, &met->size);
      read_int(fp, &met->type);
      met->npmax = mesh->npmax;
      met->dim   = mesh->dim;
      met->ver   = mesh->ver;
      MMG5_SAFE_CALLOC(met->m, (size_t)met->size * (met->npmax + 1), double,
                       fclose(fp); return MMG3D_STAGE_NONE);
      fread(met->m, sizeof(double), (size_t)met->size * (met->np + 1), fp);
    }
  }

  fclose(fp);

  fprintf(stdout, "[CHECKPOINT] Load complete: stage '%s', np=%" MMG5_PRId
          " ne=%" MMG5_PRId "\n",
          stage_names[saved_stage], mesh->np, mesh->ne);

  return saved_stage;
}
