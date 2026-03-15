#pragma once
/**
 * \file aniso_collapse.cuh
 * \brief GPU anisotropic edge collapse using RXMesh cavity operator.
 *
 * Modified from RXMesh/apps/Remesh/collapse.cuh to use metric tensor
 * edge lengths instead of Euclidean distance.
 */

#include <cuda_profiler_api.h>
#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/rxmesh_dynamic.h"
#include "aniso_metric.cuh"

/* link_condition.cuh from RXMesh */
#include "link_condition.cuh"

template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    aniso_edge_collapse(
        rxmesh::Context                   context,
        const rxmesh::VertexAttribute<T>  coords,
        rxmesh::VertexAttribute<T>        metric,
        rxmesh::EdgeAttribute<EdgeStatus> edge_status,
        const T                           low_len_sq,
        const T                           high_len_sq,
        int                               met_size)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::EV> cavity(
        block, context, shrd_alloc, true);

    const uint32_t pid = cavity.patch_id();
    if (pid == INVALID32) return;

    Bitmask edge_mask(cavity.patch_info().edges_capacity, shrd_alloc);
    edge_mask.reset(block);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    Bitmask v0_mask(cavity.patch_info().num_vertices[0], shrd_alloc);
    Bitmask v1_mask(cavity.patch_info().num_vertices[0], shrd_alloc);

    Query<blockThreads> query(context, pid);
    query.prologue<Op::EVDiamond>(block, shrd_alloc);
    block.sync();

    // 1. Mark short edges for collapse
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        if (edge_status(eh) != UNSEEN) return;

        const VertexIterator iter =
            query.template get_iterator<VertexIterator>(eh.local_id());
        assert(iter.size() == 4);

        const VertexHandle v0 = iter[0];
        const VertexHandle v1 = iter[2];
        const VertexHandle v2 = iter[1];
        const VertexHandle v3 = iter[3];

        if (!v2.is_valid() || !v3.is_valid()) return;
        if (v0 == v1 || v0 == v2 || v0 == v3 || v1 == v2 || v1 == v3 || v2 == v3)
            return;

        // Anisotropic edge length
        T edge_len;
        if (met_size == 6) {
            edge_len = aniso_edge_len_sq(coords, metric, v0, v1);
        } else {
            edge_len = iso_edge_len_sq(coords, v0, v1);
        }

        if (edge_len < low_len_sq) {
            edge_mask.set(eh.local_id(), true);
        }
    });
    block.sync();

    // 2. Link condition check (from RXMesh)
    link_condition(
        block, cavity.patch_info(), query, edge_mask, v0_mask, v1_mask, 0, 2);
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    // 3. Create cavities for valid collapses
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        if (edge_mask(eh.local_id())) {
            // Check that collapse won't create too-long edges
            const VertexIterator iter =
                query.template get_iterator<VertexIterator>(eh.local_id());
            const VertexHandle v0 = iter[0];
            const VertexHandle v1 = iter[2];

            // Midpoint of collapsed edge
            T mx = (coords(v0, 0) + coords(v1, 0)) * T(0.5);
            T my = (coords(v0, 1) + coords(v1, 1)) * T(0.5);
            T mz = (coords(v0, 2) + coords(v1, 2)) * T(0.5);

            cavity.create(eh);
            edge_status(eh) = SKIP;
        }
    });
    block.sync();

    // 4. Prologue + apply collapses
    if (cavity.prologue(block, shrd_alloc, coords, edge_status, metric)) {
        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            const EdgeHandle src =
                cavity.template get_creator<EdgeHandle>(c);

            const VertexHandle v0 = cavity.get_cavity_vertex(c, 0);
            const VertexHandle v1 = cavity.get_cavity_vertex(c, 2);

            // Keep v0, remove v1 → move v0 to midpoint
            coords(v0, 0) = (coords(v0, 0) + coords(v1, 0)) * T(0.5);
            coords(v0, 1) = (coords(v0, 1) + coords(v1, 1)) * T(0.5);
            coords(v0, 2) = (coords(v0, 2) + coords(v1, 2)) * T(0.5);

            // Average metric at the surviving vertex
            for (int i = 0; i < met_size; i++) {
                metric(v0, i) = (metric(v0, i) + metric(v1, i)) * T(0.5);
            }

            // Rebuild faces around v0, skipping the two removed faces
            for (uint16_t i = 0; i < size; ++i) {
                const VertexHandle vi = cavity.get_cavity_vertex(c, i);
                const VertexHandle vn =
                    cavity.get_cavity_vertex(c, (i + 1) % size);

                if (vi == v1) continue;  // skip removed vertex

                DEdgeHandle e0, e1;
                if (vn == v1) {
                    // This face connects to the removed vertex — replace with v0
                    const VertexHandle vnn =
                        cavity.get_cavity_vertex(c, (i + 2) % size);
                    e0 = cavity.add_edge(v0, vnn);
                    e1 = cavity.add_edge(vnn, vi);
                    DEdgeHandle e2 = cavity.add_edge(vi, v0);
                    cavity.add_face(e2, e0, e1);
                } else if (vi != v0) {
                    e0 = cavity.add_edge(v0, vi);
                    e1 = cavity.add_edge(vi, vn);
                    DEdgeHandle e2 = cavity.add_edge(vn, v0);
                    cavity.add_face(e0, e1, e2);
                }
            }
        });
        cavity.epilogue(block);
        block.sync();
    }
}
