/* Minimal RXMesh test — just create a mesh and report stats.
 * No remesh headers, no __device__ lambdas, just RXMeshStatic. */
#include <cstdio>
#include <vector>
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/rxmesh.h"

int main(int argc, char** argv) {
    fprintf(stderr, "[TEST] main()\n"); fflush(stderr);
    rxmesh::rx_init(0);
    fprintf(stderr, "[TEST] rx_init done\n"); fflush(stderr);

    std::vector<std::vector<uint32_t>> fv = {
        {0,1,2}, {0,2,3}, {0,1,4}, {1,2,4}, {2,3,4}, {3,0,4}
    };

    fprintf(stderr, "[TEST] creating RXMeshStatic from %zu faces...\n", fv.size());
    fflush(stderr);

    rxmesh::RXMeshStatic rx(fv);

    fprintf(stderr, "[TEST] OK: V=%d E=%d F=%d P=%d\n",
            rx.get_num_vertices(), rx.get_num_edges(),
            rx.get_num_faces(), rx.get_num_patches());

    return 0;
}
