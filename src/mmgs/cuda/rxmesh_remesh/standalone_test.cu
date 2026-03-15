/* RXMesh test — load OBJ or use built-in tetrahedron. */
#include <cstdio>
#include <string>
#include <vector>
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/rxmesh.h"

int main(int argc, char** argv) {
    fprintf(stderr, "[TEST] main()\n"); fflush(stderr);
    rxmesh::rx_init(0);
    fprintf(stderr, "[TEST] rx_init done\n"); fflush(stderr);

    if (argc > 1) {
        /* Load OBJ file */
        std::string input = argv[1];
        fprintf(stderr, "[TEST] loading %s...\n", input.c_str()); fflush(stderr);
        rxmesh::RXMeshStatic rx(input);
        fprintf(stderr, "[TEST] OK: V=%d E=%d F=%d P=%d\n",
                rx.get_num_vertices(), rx.get_num_edges(),
                rx.get_num_faces(), rx.get_num_patches());
    } else {
        /* Built-in tetrahedron */
        std::vector<std::vector<uint32_t>> fv = {
            {0,1,2}, {0,2,3}, {0,1,4}, {1,2,4}, {2,3,4}, {3,0,4}
        };
        fprintf(stderr, "[TEST] creating from %zu faces...\n", fv.size()); fflush(stderr);
        rxmesh::RXMeshStatic rx(fv);
        fprintf(stderr, "[TEST] OK: V=%d E=%d F=%d P=%d\n",
                rx.get_num_vertices(), rx.get_num_edges(),
                rx.get_num_faces(), rx.get_num_patches());
    }

    return 0;
}
