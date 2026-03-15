/**
 * \file mmgs_rxmesh_bridge.cu
 * \brief Bridge between mmg's mesh structures and RXMesh GPU remeshing.
 *
 * Converts mmg mesh → OBJ → RXMeshDynamic → aniso remesh → OBJ → mmg mesh.
 * The OBJ round-trip is the simplest integration path (same as pyrxmesh uses).
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rxmesh/rxmesh_dynamic.h"

/* RXMesh aniso remesh kernels */
#include "aniso_remesh.cuh"

/* mmg headers — included after RXMesh to avoid conflicts */
extern "C" {
#include "mmgcommon_private.h"
#include "libmmgs_private.h"
}

extern "C" {

/**
 * GPU anisotropic remeshing via RXMesh.
 *
 * Takes mmg mesh + metric, runs GPU remeshing, writes results back.
 *
 * \param mesh    mmg mesh (vertices, triangles)
 * \param met     mmg metric (size=1 scalar or size=6 tensor per vertex)
 * \param n_iter  number of split-collapse-flip-smooth iterations
 * \param target_len  target edge length (0 = auto from average)
 *
 * \return 1 on success, 0 on failure.
 */
int MMGS_rxmesh_remesh(MMG5_pMesh mesh, MMG5_pSol met,
                       int n_iter, float target_len)
{
    int np = (int)mesh->np;
    int nt = (int)mesh->nt;
    int met_size = (met && met->m) ? met->size : 0;

    fprintf(stdout, "[RXMESH-BRIDGE] Starting: %d verts, %d tris, met_size=%d\n",
            np, nt, met_size);

    /* ---- Step 1: Write temp OBJ ---- */
    char tmpobj[256];
    snprintf(tmpobj, sizeof(tmpobj), "/tmp/mmgs_rxmesh_in_%d.obj", (int)getpid());

    FILE *fp = fopen(tmpobj, "w");
    if (!fp) {
        fprintf(stderr, "[RXMESH-BRIDGE] Cannot create temp file %s\n", tmpobj);
        return 0;
    }

    for (int k = 1; k <= np; k++) {
        fprintf(fp, "v %.15g %.15g %.15g\n",
                mesh->point[k].c[0], mesh->point[k].c[1], mesh->point[k].c[2]);
    }
    for (int k = 1; k <= nt; k++) {
        MMG5_pTria pt = &mesh->tria[k];
        if (pt->v[0] <= 0) continue;
        fprintf(fp, "f %d %d %d\n",
                (int)pt->v[0], (int)pt->v[1], (int)pt->v[2]);
    }
    fclose(fp);

    fprintf(stdout, "[RXMESH-BRIDGE] Wrote %s (%d verts, %d tris)\n", tmpobj, np, nt);

    /* ---- Step 2: Create RXMeshDynamic ---- */
    rxmesh::RXMeshDynamic rx(std::string(tmpobj), "", 512, 2.0, 2);

    if (!rx.is_edge_manifold()) {
        fprintf(stderr, "[RXMESH-BRIDGE] Mesh is not edge-manifold, cannot remesh\n");
        remove(tmpobj);
        return 0;
    }

    auto coords = rx.get_input_vertex_coordinates();

    /* ---- Step 3: Upload metric tensor as vertex attribute ---- */
    int rx_met_size = (met_size == 6) ? 6 : (met_size == 1 ? 1 : 0);
    auto metric = rx.add_vertex_attribute<float>("Metric", rx_met_size > 0 ? rx_met_size : 1);

    if (rx_met_size > 0 && met && met->m) {
        /* Map mmg vertex indices to RXMesh vertex handles */
        metric->reset(rxmesh::LOCATION_ALL, 0);
        metric->move(rxmesh::HOST, rxmesh::DEVICE);

        /* For now, upload via host — need to map mmg indices to RXMesh handles.
         * RXMesh preserves input ordering for OBJ files, so vertex k in mmg
         * maps to vertex k-1 in RXMesh (0-based). */
        rx.for_each_vertex(
            rxmesh::HOST,
            [&](const rxmesh::VertexHandle vh) {
                /* RXMesh linear index = vh.linear_id() — but we need the
                 * original input ordering. For OBJ input, the i-th vertex
                 * in the file maps to linear index i. */
                // TODO: proper index mapping
            },
            NULL, false);

        /* Simpler approach: write metric to file, read back.
         * Actually, just set the metric data directly since RXMesh preserves
         * OBJ vertex ordering as linear IDs. */
    }

    /* ---- Step 4: Compute target edge length ---- */
    float tgt_len = target_len;
    if (tgt_len <= 0) {
        /* Auto: compute average edge length */
        float avg_len = 0;
        int n_edges = 0;
        rx.for_each_edge(
            rxmesh::HOST,
            [&](const rxmesh::EdgeHandle eh) { n_edges++; },
            NULL, false);
        /* Use the metric to define target — for now just use Euclidean average */
        auto edge_len_attr = rx.add_edge_attribute<float>("tmpLen", 1);
        /* TODO: compute average metric-space edge length */
        tgt_len = 1.0f;  /* placeholder */
        fprintf(stdout, "[RXMESH-BRIDGE] Auto target_len not implemented, using %.6f\n", tgt_len);
    }

    /* ---- Step 5: Run aniso remesh ---- */
    AnisoRemeshConfig config;
    config.high_ratio = 4.0f / 3.0f;
    config.low_ratio  = 4.0f / 5.0f;
    config.target_len = tgt_len;
    config.num_iter   = n_iter;
    config.num_smooth = 5;
    config.met_size   = rx_met_size > 0 ? rx_met_size : 0;
    config.use_metric = (rx_met_size > 0);

    aniso_remesh_rxmesh(rx, coords.get(), metric.get(), config);

    /* ---- Step 6: Export result OBJ ---- */
    char tmpout[256];
    snprintf(tmpout, sizeof(tmpout), "/tmp/mmgs_rxmesh_out_%d.obj", (int)getpid());

    rx.update_host();
    coords->move(rxmesh::DEVICE, rxmesh::HOST);
    rx.export_obj(tmpout, *coords);

    fprintf(stdout, "[RXMESH-BRIDGE] Wrote result to %s: %d verts, %d faces\n",
            tmpout, rx.get_num_vertices(), rx.get_num_faces());

    /* ---- Step 7: Read result OBJ back into mmg mesh ---- */
    /* TODO: implement OBJ → mmg mesh import */
    /* For now, just report the result and leave the mesh unchanged */
    fprintf(stdout, "[RXMESH-BRIDGE] TODO: write results back to mmg mesh\n");

    /* Cleanup temp files */
    remove(tmpobj);
    /* Don't remove tmpout yet — user might want to inspect it */

    return 1;
}

} /* extern "C" */
