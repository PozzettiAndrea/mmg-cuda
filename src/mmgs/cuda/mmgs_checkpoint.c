/**
 * \file mmgs/cuda/mmgs_checkpoint.c
 * \brief Binary checkpoint save/load for mmgs surface remeshing pipeline.
 *
 * Serializes full MMG5_Mesh + MMG5_Sol state at pipeline stage boundaries.
 * File format: .msc (mmgs checkpoint) — 576-byte header + field-by-field data.
 */

#include "mmgcommon_private.h"
#include "libmmgs_private.h"
#include "mmgs_cuda.h"

#include <sys/stat.h>
#include <time.h>

/* ================================================================
 * Stage name mapping
 * ================================================================ */

static const char* stage_names[MMGS_STAGE_COUNT] = {
  "post-load",
  "post-scale",
  "post-analys",
  "post-inqua",
  "post-geom",
  "post-defsiz",
  "post-gradsiz",
  "post-adapt",
  "post-unscale"
};

MMGS_PipelineStage MMGS_cuda_stage_from_name(const char *name) {
  int i;
  for (i = 0; i < MMGS_STAGE_COUNT; ++i) {
    if (strcmp(name, stage_names[i]) == 0) return (MMGS_PipelineStage)i;
  }
  return MMGS_STAGE_NONE;
}

const char* MMGS_cuda_stage_name(MMGS_PipelineStage s) {
  if (s >= 0 && s < MMGS_STAGE_COUNT) return stage_names[s];
  return "unknown";
}

void MMGS_cuda_list_stages(void) {
  int i;
  fprintf(stdout, "Pipeline stages (mmgs):\n");
  for (i = 0; i < MMGS_STAGE_COUNT; ++i) {
    fprintf(stdout, "  %2d. %s\n", i, stage_names[i]);
  }
}

/* ================================================================
 * Helpers
 * ================================================================ */

static char* checkpoint_path(const char *dir, MMGS_PipelineStage stage) {
  static char buf[512];
  snprintf(buf, sizeof(buf), "%s/%s.msc", dir, stage_names[stage]);
  return buf;
}

int MMGS_checkpoint_exists(const char *dir, MMGS_PipelineStage stage) {
  struct stat st;
  return (stat(checkpoint_path(dir, stage), &st) == 0);
}

static int write_int(FILE *fp, int v)       { return fwrite(&v, sizeof(int), 1, fp) == 1; }
static int write_lint(FILE *fp, MMG5_int v) { return fwrite(&v, sizeof(MMG5_int), 1, fp) == 1; }
static int write_dbl(FILE *fp, double v)    { return fwrite(&v, sizeof(double), 1, fp) == 1; }

static int read_int(FILE *fp, int *v)       { return fread(v, sizeof(int), 1, fp) == 1; }
static int read_lint(FILE *fp, MMG5_int *v) { return fread(v, sizeof(MMG5_int), 1, fp) == 1; }
static int read_dbl(FILE *fp, double *v)    { return fread(v, sizeof(double), 1, fp) == 1; }

/* ================================================================
 * Save checkpoint
 * ================================================================ */

int MMGS_save_checkpoint(MMG5_pMesh mesh, MMG5_pSol met,
                         MMGS_PipelineStage stage, const char *dir) {
  FILE *fp;
  char *path;
  MMGS_CheckpointHeader hdr;
  long file_size;
  struct stat st;

#ifndef _WIN32
  mkdir(dir, 0755);
#else
  _mkdir(dir);
#endif

  path = checkpoint_path(dir, stage);
  fp = fopen(path, "wb");
  if (!fp) {
    fprintf(stderr, "[CHECKPOINT-S] ERROR: Cannot open %s for writing\n", path);
    return 0;
  }

  /* ---- Header ---- */
  memset(&hdr, 0, sizeof(hdr));
  memcpy(hdr.magic, "MSC", 4);
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
  write_lint(fp, mesh->nt);
  write_lint(fp, mesh->na);
  write_lint(fp, mesh->xp);
  write_lint(fp, mesh->npmax);
  write_lint(fp, mesh->ntmax);
  write_lint(fp, mesh->namax);
  write_lint(fp, mesh->xpmax);
  write_int(fp, mesh->ver);
  write_int(fp, mesh->dim);

  /* ---- Points (1-based) ---- */
  if (mesh->np > 0 && mesh->point) {
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
    if (has_adja && mesh->nt > 0) {
      fwrite(mesh->adja + 1, sizeof(MMG5_int), 3 * mesh->nt, fp);  /* 3 per tri */
    }
  }

  /* ---- Info flags ---- */
  {
    fwrite(&mesh->info.ani, sizeof(uint8_t), 1, fp);
    fwrite(&mesh->info.optim, sizeof(uint8_t), 1, fp);
    fwrite(&mesh->info.noinsert, sizeof(uint8_t), 1, fp);
    fwrite(&mesh->info.noswap, sizeof(uint8_t), 1, fp);
    fwrite(&mesh->info.nomove, sizeof(uint8_t), 1, fp);
    fwrite(&mesh->info.nosurf, sizeof(uint8_t), 1, fp);
    fwrite(&mesh->info.nosizreq, sizeof(uint8_t), 1, fp);
    fwrite(&mesh->info.metRidTyp, sizeof(uint8_t), 1, fp);
    write_int(fp, mesh->info.renum);
    write_int(fp, mesh->info.imprim);
    fwrite(&mesh->info.iso, sizeof(int8_t), 1, fp);
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
      fwrite(met->m, sizeof(double), (size_t)met->size * (met->np + 1), fp);
    }
  }

  fclose(fp);

  file_size = 0;
  if (stat(path, &st) == 0) file_size = st.st_size;
  fprintf(stdout, "[CHECKPOINT-S] Saved '%s' to %s (%.1f MB)\n",
          stage_names[stage], path, file_size / (1024.0 * 1024.0));

  return 1;
}

/* ================================================================
 * Load checkpoint
 * ================================================================ */

MMGS_PipelineStage MMGS_load_checkpoint(MMG5_pMesh mesh, MMG5_pSol met,
                                        const char *dir,
                                        MMGS_PipelineStage stage) {
  FILE *fp;
  char *path;
  MMGS_CheckpointHeader hdr;
  int stage_idx;
  MMGS_PipelineStage saved_stage;

  path = checkpoint_path(dir, stage);
  fp = fopen(path, "rb");
  if (!fp) {
    fprintf(stderr, "[CHECKPOINT-S] ERROR: Cannot open %s for reading\n", path);
    return MMGS_STAGE_NONE;
  }

  if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
    fprintf(stderr, "[CHECKPOINT-S] ERROR: Failed to read header from %s\n", path);
    fclose(fp);
    return MMGS_STAGE_NONE;
  }
  if (memcmp(hdr.magic, "MSC", 4) != 0) {
    fprintf(stderr, "[CHECKPOINT-S] ERROR: Invalid magic in %s\n", path);
    fclose(fp);
    return MMGS_STAGE_NONE;
  }
  if (hdr.version != 1) {
    fprintf(stderr, "[CHECKPOINT-S] ERROR: Unsupported version %d\n", hdr.version);
    fclose(fp);
    return MMGS_STAGE_NONE;
  }

  read_int(fp, &stage_idx);
  saved_stage = (MMGS_PipelineStage)stage_idx;

  fprintf(stdout, "[CHECKPOINT-S] Loading '%s' from %s\n",
          stage_names[saved_stage], path);
  fprintf(stdout, "[CHECKPOINT-S]   strategies: quality=%d metvol=%d gradation=%d\n",
          hdr.quality_strategy, hdr.metvol_strategy, hdr.gradation_strategy);

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
    MMG5_int np, nt, na, xp, npmax, ntmax, namax, xpmax;
    int ver, dim;

    read_lint(fp, &np);   read_lint(fp, &nt);
    read_lint(fp, &na);   read_lint(fp, &xp);
    read_lint(fp, &npmax); read_lint(fp, &ntmax);
    read_lint(fp, &namax); read_lint(fp, &xpmax);
    read_int(fp, &ver);   read_int(fp, &dim);

    mesh->np = np;  mesh->nt = nt;
    mesh->na = na;  mesh->xp = xp;
    mesh->npmax = npmax; mesh->ntmax = ntmax;
    mesh->namax = namax; mesh->xpmax = xpmax;
    mesh->ver = ver; mesh->dim = dim;
  }

  /* ---- Points ---- */
  if (mesh->np > 0) {
    MMG5_int k;
    MMG5_SAFE_CALLOC(mesh->point, mesh->npmax + 1, MMG5_Point,
                     fclose(fp); return MMGS_STAGE_NONE);
    for (k = 1; k <= mesh->np; ++k) {
      MMG5_pPoint pp = &mesh->point[k];
      fread(pp->c, sizeof(double), 3, fp);
      fread(pp->n, sizeof(double), 3, fp);
      read_lint(fp, &pp->ref);  read_lint(fp, &pp->xp);
      read_lint(fp, &pp->tmp);  read_lint(fp, &pp->flag);
      read_lint(fp, &pp->s);
      fread(&pp->tag, sizeof(uint16_t), 1, fp);
      fread(&pp->tagdel, sizeof(int8_t), 1, fp);
    }
  }

  /* ---- Triangles ---- */
  if (mesh->nt > 0) {
    MMG5_int k;
    MMG5_SAFE_CALLOC(mesh->tria, mesh->ntmax + 1, MMG5_Tria,
                     fclose(fp); return MMGS_STAGE_NONE);
    for (k = 1; k <= mesh->nt; ++k) {
      MMG5_pTria ptt = &mesh->tria[k];
      read_dbl(fp, &ptt->qual);
      fread(ptt->v, sizeof(MMG5_int), 3, fp);
      read_lint(fp, &ptt->ref);  read_lint(fp, &ptt->base);
      read_lint(fp, &ptt->cc);
      fread(ptt->edg, sizeof(MMG5_int), 3, fp);
      read_lint(fp, &ptt->flag);
      fread(ptt->tag, sizeof(uint16_t), 3, fp);
    }
  }

  /* ---- xPoint ---- */
  if (mesh->xp > 0) {
    MMG5_int k;
    MMG5_SAFE_CALLOC(mesh->xpoint, mesh->xpmax + 1, MMG5_xPoint,
                     fclose(fp); return MMGS_STAGE_NONE);
    for (k = 1; k <= mesh->xp; ++k) {
      MMG5_pxPoint pxp = &mesh->xpoint[k];
      fread(pxp->n1, sizeof(double), 3, fp);
      fread(pxp->n2, sizeof(double), 3, fp);
      fread(&pxp->nnor, sizeof(int8_t), 1, fp);
    }
  }

  /* ---- Edges ---- */
  if (mesh->na > 0) {
    MMG5_int k;
    MMG5_SAFE_CALLOC(mesh->edge, mesh->namax + 1, MMG5_Edge,
                     fclose(fp); return MMGS_STAGE_NONE);
    for (k = 1; k <= mesh->na; ++k) {
      MMG5_pEdge pa = &mesh->edge[k];
      read_lint(fp, &pa->a);  read_lint(fp, &pa->b);
      read_lint(fp, &pa->ref); read_lint(fp, &pa->base);
      fread(&pa->tag, sizeof(uint16_t), 1, fp);
    }
  }

  /* ---- Adjacency ---- */
  {
    int has_adja;
    read_int(fp, &has_adja);
    if (has_adja && mesh->nt > 0) {
      MMG5_SAFE_CALLOC(mesh->adja, 3 * mesh->ntmax + 4, MMG5_int,
                       fclose(fp); return MMGS_STAGE_NONE);
      fread(mesh->adja + 1, sizeof(MMG5_int), 3 * mesh->nt, fp);
    }
  }

  /* ---- Info flags ---- */
  {
    fread(&mesh->info.ani, sizeof(uint8_t), 1, fp);
    fread(&mesh->info.optim, sizeof(uint8_t), 1, fp);
    fread(&mesh->info.noinsert, sizeof(uint8_t), 1, fp);
    fread(&mesh->info.noswap, sizeof(uint8_t), 1, fp);
    fread(&mesh->info.nomove, sizeof(uint8_t), 1, fp);
    fread(&mesh->info.nosurf, sizeof(uint8_t), 1, fp);
    fread(&mesh->info.nosizreq, sizeof(uint8_t), 1, fp);
    fread(&mesh->info.metRidTyp, sizeof(uint8_t), 1, fp);
    read_int(fp, &mesh->info.renum);
    read_int(fp, &mesh->info.imprim);
    fread(&mesh->info.iso, sizeof(int8_t), 1, fp);
    fread(&mesh->info.nreg, sizeof(int8_t), 1, fp);
    fread(&mesh->info.xreg, sizeof(int8_t), 1, fp);
  }

  /* ---- Metric ---- */
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
                       fclose(fp); return MMGS_STAGE_NONE);
      fread(met->m, sizeof(double), (size_t)met->size * (met->np + 1), fp);
    }
  }

  fclose(fp);

  fprintf(stdout, "[CHECKPOINT-S] Load complete: stage '%s', np=%" MMG5_PRId
          " nt=%" MMG5_PRId "\n",
          stage_names[saved_stage], mesh->np, mesh->nt);

  return saved_stage;
}
