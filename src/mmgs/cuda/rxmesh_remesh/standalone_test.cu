/**
 * \file standalone_test.cu
 * \brief Standalone test for RXMesh-based GPU anisotropic remeshing.
 *
 * Usage: rxmesh_aniso_remesh input.obj [target_len_ratio] [num_iter]
 *
 * Loads an OBJ mesh, runs GPU remeshing, saves result.
 * No mmg dependency — uses RXMesh directly.
 */

#include <cstdio>
#include <cstdlib>
#include <string>

#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/util/util.h"

/* Include the remesh kernels */
#include "aniso_metric.cuh"

/* We need EdgeStatus from the split/collapse headers */
using EdgeStatus = int8_t;
enum : EdgeStatus { UNSEEN = 0, SKIP = 1, UPDATE = 2, ADDED = 3 };

/* Include the actual remesh app from RXMesh — for now use isotropic as baseline */
/* We'll swap in our aniso kernels once the build works */

int main(int argc, char** argv)
{
    using namespace rxmesh;

    fprintf(stderr, "[RXMESH-TEST] main() entered\n");
    fflush(stderr);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s input.obj [target_len_ratio] [num_iter]\n", argv[0]);
        return 1;
    }

    std::string input_obj = argv[1];
    float target_ratio = (argc > 2) ? atof(argv[2]) : 0.5f;
    int num_iter = (argc > 3) ? atoi(argv[3]) : 3;

    fprintf(stderr, "[RXMESH-TEST] About to load OBJ...\n");
    fflush(stderr);
    fprintf(stdout, "[RXMESH-TEST] Loading %s\n", input_obj.c_str());
    fflush(stdout);

    /* Create RXMeshDynamic */
    RXMeshDynamic rx(input_obj, "", 512, 2.0, 2);

    fprintf(stdout, "[RXMESH-TEST] Loaded: %d verts, %d edges, %d faces, %d patches\n",
            rx.get_num_vertices(), rx.get_num_edges(),
            rx.get_num_faces(), rx.get_num_patches());

    if (!rx.is_edge_manifold()) {
        fprintf(stderr, "[RXMESH-TEST] ERROR: mesh is not edge-manifold\n");
        return 1;
    }

    auto coords = rx.get_input_vertex_coordinates();

    fprintf(stdout, "[RXMESH-TEST] Target ratio = %.4f, num_iter = %d\n",
            target_ratio, num_iter);

    /* TODO: run aniso_remesh_rxmesh here */
    fprintf(stdout, "[RXMESH-TEST] GPU remesh not yet wired — just testing build\n");

    /* Export result */
    std::string output = "/tmp/rxmesh_test_out.obj";
    rx.export_obj(output, *coords);
    fprintf(stdout, "[RXMESH-TEST] Saved %s\n", output.c_str());

    return 0;
}
