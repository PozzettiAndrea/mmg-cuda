/* Mirrors pyrxmesh/src/op_remesh.cu pattern exactly */
#include <cstdio>
#include <string>
#include <vector>
#include <filesystem>

#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

#include "glm_compat.h"

using namespace rxmesh;

static char* s_argv[] = {(char*)"test", nullptr};
static struct arg {
    std::string obj_file_name;
    std::string output_folder = "/tmp";
    uint32_t    nx = 66;
    uint32_t    ny = 66;
    float       relative_len = 1.0f;
    int         num_smooth_iters = 5;
    uint32_t    num_iter = 3;
    uint32_t    device_id = 0;
    char**      argv = s_argv;
    int         argc = 1;
} Arg;

#include "Remesh/remesh_rxmesh.cuh"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s input.obj [relative_len] [iterations]\n", argv[0]);
        return 1;
    }

    std::string input = argv[1];
    Arg.obj_file_name = input;
    if (argc > 2) Arg.relative_len = atof(argv[2]);
    if (argc > 3) Arg.num_iter = atoi(argv[3]);

    fprintf(stderr, "[TEST] Loading %s, rel_len=%.2f, iters=%d\n",
            input.c_str(), Arg.relative_len, Arg.num_iter);

    RXMeshDynamic rx(input, "", 512, 2.0f, 2);

    fprintf(stderr, "[TEST] Loaded: V=%d E=%d F=%d P=%d\n",
            rx.get_num_vertices(), rx.get_num_edges(),
            rx.get_num_faces(), rx.get_num_patches());

    if (!rx.is_edge_manifold()) {
        fprintf(stderr, "[TEST] Not edge-manifold!\n");
        return 1;
    }

    remesh_rxmesh(rx);

    auto coords = rx.get_input_vertex_coordinates();
    std::string outpath = "/tmp/rxmesh_remesh_out.obj";
    rx.export_obj(outpath, *coords);

    fprintf(stderr, "[TEST] Output: V=%d F=%d → %s\n",
            rx.get_num_vertices(), rx.get_num_faces(), outpath.c_str());
    return 0;
}
