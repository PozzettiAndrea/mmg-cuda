#pragma once
/**
 * \file aniso_metric.cuh
 * \brief Device functions for anisotropic metric operations.
 *
 * Ported from mmg's metric math to CUDA __device__ functions.
 * Works with RXMesh's VertexAttribute<float> for the metric tensor.
 *
 * Metric tensor storage: 6 floats per vertex in symmetric Voigt order:
 *   m[0]=M00, m[1]=M01, m[2]=M02, m[3]=M11, m[4]=M12, m[5]=M22
 */

#include "rxmesh/attribute.h"

/**
 * Compute anisotropic edge length squared: d^T * M * d
 * where M is the averaged metric tensor at the two endpoints.
 */
template <typename T>
__device__ __forceinline__ T
aniso_edge_len_sq(const rxmesh::VertexAttribute<T>& coords,
                  const rxmesh::VertexAttribute<T>& metric,
                  const rxmesh::VertexHandle         va,
                  const rxmesh::VertexHandle         vb)
{
    T dx = coords(vb, 0) - coords(va, 0);
    T dy = coords(vb, 1) - coords(va, 1);
    T dz = coords(vb, 2) - coords(va, 2);

    // Average metric at the two endpoints
    T m0 = T(0.5) * (metric(va, 0) + metric(vb, 0));
    T m1 = T(0.5) * (metric(va, 1) + metric(vb, 1));
    T m2 = T(0.5) * (metric(va, 2) + metric(vb, 2));
    T m3 = T(0.5) * (metric(va, 3) + metric(vb, 3));
    T m4 = T(0.5) * (metric(va, 4) + metric(vb, 4));
    T m5 = T(0.5) * (metric(va, 5) + metric(vb, 5));

    // Quadratic form: d^T * M * d
    T len_sq = m0*dx*dx + m3*dy*dy + m5*dz*dz
             + T(2.0)*(m1*dx*dy + m2*dx*dz + m4*dy*dz);

    return len_sq > T(0) ? len_sq : T(0);
}

/**
 * Compute anisotropic edge length (not squared).
 */
template <typename T>
__device__ __forceinline__ T
aniso_edge_len(const rxmesh::VertexAttribute<T>& coords,
               const rxmesh::VertexAttribute<T>& metric,
               const rxmesh::VertexHandle         va,
               const rxmesh::VertexHandle         vb)
{
    return sqrt(aniso_edge_len_sq(coords, metric, va, vb));
}

/**
 * Compute isotropic edge length squared (Euclidean).
 * Fallback when no metric is provided.
 */
template <typename T>
__device__ __forceinline__ T
iso_edge_len_sq(const rxmesh::VertexAttribute<T>& coords,
                const rxmesh::VertexHandle         va,
                const rxmesh::VertexHandle         vb)
{
    T dx = coords(vb, 0) - coords(va, 0);
    T dy = coords(vb, 1) - coords(va, 1);
    T dz = coords(vb, 2) - coords(va, 2);
    return dx*dx + dy*dy + dz*dz;
}

/**
 * Compute scalar-metric edge length (mmg isotropic mode).
 * metric has 1 component per vertex: h (target edge size).
 * Returns l/h with log-scale interpolation.
 */
template <typename T>
__device__ __forceinline__ T
scalar_metric_edge_len(const rxmesh::VertexAttribute<T>& coords,
                       const rxmesh::VertexAttribute<T>& metric,
                       const rxmesh::VertexHandle         va,
                       const rxmesh::VertexHandle         vb)
{
    T dx = coords(vb, 0) - coords(va, 0);
    T dy = coords(vb, 1) - coords(va, 1);
    T dz = coords(vb, 2) - coords(va, 2);
    T l = sqrt(dx*dx + dy*dy + dz*dz);

    T h0 = metric(va, 0);
    T h1 = metric(vb, 0);
    if (h0 <= T(0)) h0 = T(1);
    if (h1 <= T(0)) h1 = T(1);

    T r = h1 / h0 - T(1);
    if (abs(r) < T(1e-6))
        return l / h0;
    else
        return l / (h1 - h0) * log1p(r);
}

/**
 * Interpolate metric tensor at midpoint of edge (va, vb).
 * Simple averaging — sufficient for metric-driven remeshing.
 */
template <typename T>
__device__ __forceinline__ void
interpolate_metric(const rxmesh::VertexAttribute<T>& metric,
                   rxmesh::VertexAttribute<T>&       metric_out,
                   const rxmesh::VertexHandle         va,
                   const rxmesh::VertexHandle         vb,
                   const rxmesh::VertexHandle         v_new,
                   int                                met_size)
{
    for (int i = 0; i < met_size; ++i) {
        metric_out(v_new, i) = T(0.5) * (metric(va, i) + metric(vb, i));
    }
}

/**
 * Anisotropic triangle quality: area_M / (l0^2 + l1^2 + l2^2)
 * where area_M = sqrt(det(J^T * M * J)) and li are metric edge lengths.
 */
template <typename T>
__device__ __forceinline__ T
aniso_tri_quality(const rxmesh::VertexAttribute<T>& coords,
                  const rxmesh::VertexAttribute<T>& metric,
                  const rxmesh::VertexHandle         va,
                  const rxmesh::VertexHandle         vb,
                  const rxmesh::VertexHandle         vc)
{
    // Average metric
    T m[6];
    for (int i = 0; i < 6; i++)
        m[i] = (metric(va, i) + metric(vb, i) + metric(vc, i)) / T(3);

    T abx = coords(vb, 0) - coords(va, 0);
    T aby = coords(vb, 1) - coords(va, 1);
    T abz = coords(vb, 2) - coords(va, 2);
    T acx = coords(vc, 0) - coords(va, 0);
    T acy = coords(vc, 1) - coords(va, 1);
    T acz = coords(vc, 2) - coords(va, 2);
    T bcx = coords(vc, 0) - coords(vb, 0);
    T bcy = coords(vc, 1) - coords(vb, 1);
    T bcz = coords(vc, 2) - coords(vb, 2);

    // Anisotropic area: sqrt(det(J^T * M * J))
    T d00 = m[0]*abx*abx + m[3]*aby*aby + m[5]*abz*abz
          + T(2)*(m[1]*abx*aby + m[2]*abx*abz + m[4]*aby*abz);
    T d01 = m[0]*abx*acx + m[3]*aby*acy + m[5]*abz*acz
          + m[1]*(abx*acy+aby*acx) + m[2]*(abx*acz+abz*acx) + m[4]*(aby*acz+abz*acy);
    T d11 = m[0]*acx*acx + m[3]*acy*acy + m[5]*acz*acz
          + T(2)*(m[1]*acx*acy + m[2]*acx*acz + m[4]*acy*acz);

    T area_sq = d00*d11 - d01*d01;
    if (area_sq <= T(0)) return T(0);
    T area = sqrt(area_sq);

    // Edge lengths squared in metric
    T l0 = m[0]*abx*abx + m[3]*aby*aby + m[5]*abz*abz
         + T(2)*(m[1]*abx*aby + m[2]*abx*abz + m[4]*aby*abz);
    T l1 = m[0]*acx*acx + m[3]*acy*acy + m[5]*acz*acz
         + T(2)*(m[1]*acx*acy + m[2]*acx*acz + m[4]*acy*acz);
    T l2 = m[0]*bcx*bcx + m[3]*bcy*bcy + m[5]*bcz*bcz
         + T(2)*(m[1]*bcx*bcy + m[2]*bcx*bcz + m[4]*bcy*bcz);

    T rap = l0 + l1 + l2;
    if (rap <= T(0)) return T(0);

    return area / rap;
}
