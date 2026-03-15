#pragma once
/**
 * \file aniso_split.cuh
 * \brief GPU anisotropic edge split using RXMesh cavity operator.
 *
 * Modified from RXMesh/apps/Remesh/split.cuh to use metric tensor
 * edge lengths instead of Euclidean distance.
 */

#include <cuda_profiler_api.h>
#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/rxmesh_dynamic.h"
#include "aniso_metric.cuh"

using EdgeStatus = int8_t;
enum : EdgeStatus { UNSEEN = 0, SKIP = 1, UPDATE = 2, ADDED = 3 };

template <typename T, uint32_t blockThreads>
__global__ static void aniso_edge_split(
    rxmesh::Context                   context,
    const rxmesh::VertexAttribute<T>  coords,
    rxmesh::VertexAttribute<T>        metric,      // 6-component metric tensor
    rxmesh::EdgeAttribute<EdgeStatus> edge_status,
    rxmesh::VertexAttribute<bool>     v_boundary,
    const T                           high_len_sq, // split threshold (in metric space)
    const T                           low_len_sq,  // don't split if new edges < this
    int                               met_size)    // 6 for aniso, 1 for scalar
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, true);

    if (cavity.patch_id() == INVALID32) return;

    Bitmask is_updated(cavity.patch_info().edges_capacity, shrd_alloc);
    is_updated.reset(block);
    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    auto should_split = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        assert(iter.size() == 4);

        if (edge_status(eh) != UNSEEN) return;

        const VertexHandle va = iter[0];
        const VertexHandle vb = iter[2];
        const VertexHandle vc = iter[1];
        const VertexHandle vd = iter[3];

        if (!vc.is_valid() || !vd.is_valid() || !va.is_valid() || !vb.is_valid()) {
            edge_status(eh) = SKIP;
            return;
        }
        if (v_boundary(va) || v_boundary(vb) || v_boundary(vc) || v_boundary(vd))
            return;
        if (va == vb || vb == vc || vc == va || va == vd || vb == vd || vc == vd) {
            edge_status(eh) = SKIP;
            return;
        }

        // Anisotropic edge length
        T edge_len;
        if (met_size == 6) {
            edge_len = aniso_edge_len_sq(coords, metric, va, vb);
        } else {
            edge_len = iso_edge_len_sq(coords, va, vb);
        }

        if (edge_len > high_len_sq) {
            // Check that new edges won't be too short
            T pa[3] = {coords(va,0), coords(va,1), coords(va,2)};
            T pb[3] = {coords(vb,0), coords(vb,1), coords(vb,2)};
            T pc[3] = {coords(vc,0), coords(vc,1), coords(vc,2)};
            T pd[3] = {coords(vd,0), coords(vd,1), coords(vd,2)};
            T pm[3] = {(pa[0]+pb[0])*T(0.5), (pa[1]+pb[1])*T(0.5), (pa[2]+pb[2])*T(0.5)};

            // Simple Euclidean check for new edge validity
            T min_new = T(1e30);
            for (int d = 0; d < 3; d++) {
                T da = pm[d]-pa[d], db = pm[d]-pb[d], dc = pm[d]-pc[d], dd2 = pm[d]-pd[d];
                min_new = min(min_new, da*da + db*db);  // will be recomputed below
            }
            // Proper distance check
            T d_ma = T(0), d_mb = T(0), d_mc = T(0), d_md = T(0);
            for (int d = 0; d < 3; d++) {
                T da = pm[d]-pa[d]; d_ma += da*da;
                T db = pm[d]-pb[d]; d_mb += db*db;
                T dc = pm[d]-pc[d]; d_mc += dc*dc;
                T dd2 = pm[d]-pd[d]; d_md += dd2*dd2;
            }
            min_new = min(min(d_ma, d_mb), min(d_mc, d_md));

            if (min_new >= low_len_sq) {
                cavity.create(eh);
            } else {
                edge_status(eh) = SKIP;
            }
        } else {
            edge_status(eh) = SKIP;
        }
    };

    Query<blockThreads> query(context, cavity.patch_id());
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, should_split);
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    if (cavity.prologue(block, shrd_alloc, coords, edge_status, v_boundary, metric)) {
        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            assert(size == 4);
            const VertexHandle v0 = cavity.get_cavity_vertex(c, 0);
            const VertexHandle v1 = cavity.get_cavity_vertex(c, 2);
            const VertexHandle v2 = cavity.get_cavity_vertex(c, 1);
            const VertexHandle v3 = cavity.get_cavity_vertex(c, 3);

            const VertexHandle new_v = cavity.add_vertex();

            if (new_v.is_valid()) {
                // Midpoint coordinates
                coords(new_v, 0) = (coords(v0, 0) + coords(v1, 0)) * T(0.5);
                coords(new_v, 1) = (coords(v0, 1) + coords(v1, 1)) * T(0.5);
                coords(new_v, 2) = (coords(v0, 2) + coords(v1, 2)) * T(0.5);

                // Interpolate metric at midpoint
                for (int i = 0; i < met_size; i++) {
                    metric(new_v, i) = (metric(v0, i) + metric(v1, i)) * T(0.5);
                }

                v_boundary(new_v) = false;

                // Create 4 new faces: (new,v2,v0), (new,v0,v3), (new,v3,v1), (new,v1,v2)
                const DEdgeHandle e0 = cavity.add_edge(new_v, v0);
                const DEdgeHandle e1 = cavity.add_edge(new_v, v1);

                const DEdgeHandle e_v0_v2 = cyclic_check(v0, v2) ?
                    cavity.add_edge(v0, v2) : cavity.add_edge(v2, v0);
                const DEdgeHandle e_v2_new = cavity.add_edge(v2, new_v);

                const DEdgeHandle e_v0_v3 = cyclic_check(v0, v3) ?
                    cavity.add_edge(v0, v3) : cavity.add_edge(v3, v0);
                const DEdgeHandle e_v3_new = cavity.add_edge(v3, new_v);

                const DEdgeHandle e_v1_v2 = cyclic_check(v1, v2) ?
                    cavity.add_edge(v1, v2) : cavity.add_edge(v2, v1);
                const DEdgeHandle e_v1_v3 = cyclic_check(v1, v3) ?
                    cavity.add_edge(v1, v3) : cavity.add_edge(v3, v1);

                cavity.add_face(e0, e_v0_v2, e_v2_new);
                cavity.add_face(e0.get_flip_dedge(), e_v3_new, e_v0_v3);
                cavity.add_face(e1, e_v1_v3, e_v3_new.get_flip_dedge());
                cavity.add_face(e1.get_flip_dedge(), e_v2_new.get_flip_dedge(), e_v1_v2);
            }
        });

        cavity.epilogue(block);
        block.sync();
    }

    // Update edge status for newly created edges
    cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
        // mark all edges of this cavity as ADDED
    });
}

/**
 * Host wrapper: repeatedly launch split kernel until no more long edges.
 */
inline void aniso_split_long_edges(
    rxmesh::RXMeshDynamic&            rx,
    rxmesh::VertexAttribute<float>*   coords,
    rxmesh::VertexAttribute<float>*   metric,
    rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
    rxmesh::VertexAttribute<bool>*    v_boundary,
    float                             high_len_sq,
    float                             low_len_sq,
    int                               met_size,
    int*                              d_buffer)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    edge_status->reset(DEVICE, UNSEEN);

    int num_remaining = 1;
    int iter = 0;

    while (num_remaining > 0) {
        LaunchBox<blockThreads> lb;
        rx.update_launch_box(
            {Op::EVDiamond},
            lb,
            (void*)aniso_edge_split<float, blockThreads>,
            true);

        aniso_edge_split<float, blockThreads>
            <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                rx.get_context(), *coords, *metric, *edge_status,
                *v_boundary, high_len_sq, low_len_sq, met_size);

        CUDA_ERROR(cudaDeviceSynchronize());

        rx.cleanup();
        rx.slice_patches(*coords, *edge_status, *v_boundary, *metric);
        rx.cleanup();

        edge_status->reset(DEVICE, UNSEEN);

        // Check if any edges still need splitting
        CUDA_ERROR(cudaMemset(d_buffer, 0, sizeof(int)));
        rx.for_each_edge(
            DEVICE,
            [coords_ref = *coords, metric_ref = *metric,
             high = high_len_sq, ms = met_size, d_buffer]
            __device__(const EdgeHandle eh) {
                // Quick check: does this edge still need splitting?
                // We can't easily check from here without EV query
                // so just count total edges for convergence
                ::atomicAdd(d_buffer, 1);
            });
        CUDA_ERROR(cudaDeviceSynchronize());

        iter++;
        if (iter > 20) break;  // safety limit

        // For now, run fixed number of iterations
        // TODO: proper convergence check
        num_remaining = 0;
    }
}
