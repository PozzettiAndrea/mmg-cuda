# mmg-cuda — GPU-Accelerated Surface Remeshing

## What This Is

CUDA adaptation of [mmg](https://www.mmgtools.org/) surface/volume remeshing, plus an RXMesh-based GPU remeshing backend. Part of the `cudageom` project adapting remeshing libraries to CUDA.

## Architecture

Two approaches co-exist:

### 1. mmg + CUDA kernels (Sprint 1-5)
- Strategy flags: `-quality cuda`, `-metvol cuda`, `-gradation-strategy cuda`
- Checkpoint system: `-save-dir`, `-save-all`, `-run-from`, `-run-to`, `-list-stages`
- GPU quality kernel (verified, matches CPU on dragon/gargoyle)
- GPU edge length batch kernel (verified)
- GPU persistent context (upload once, compute many)
- Timing instrumentation on all pipeline phases

### 2. RXMesh GPU remeshing backend (Sprint 6+)
- Vendored RXMesh + all deps in `RXMesh/vendor/`
- Builds on CUDA 13.0 with `--extended-lambda --expt-relaxed-constexpr`
- `rx_init(0)` MUST be called before any RXMesh operations
- Isotropic remesh works via `Remesh/remesh_rxmesh.cuh`
- Aniso metric device functions in `src/mmgs/cuda/rxmesh_remesh/aniso_metric.cuh`
- **Status**: RXMeshDynamic loads dragon (871K faces), but `cudaFuncSetAttribute` fails on shared memory size for boundary detection kernel. Needs SM 86 shared memory limit check.

## Build

```bash
# Without RXMesh (mmg CUDA kernels only):
mkdir build && cd build
cmake .. -DBUILD_CUDA=ON -DBUILD_MMGS=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11
make mmgs -j$(nproc)

# With RXMesh backend:
cmake .. -DBUILD_CUDA=ON -DBUILD_RXMESH=ON -DBUILD_MMGS=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make rxmesh_aniso_remesh -j$(nproc)
```

## Testing

```bash
# mmg CPU baseline (dragon 871K → ~95K):
./bin/mmgs_O3 /tmp/dragon_clean.mesh -hsiz 0.0015 -hausd 0.001 -nr -out /dev/null -v 1

# mmg with CUDA quality + edge marking:
./bin/mmgs_O3 /tmp/dragon_clean.mesh -hsiz 0.0015 -hausd 0.001 -nr -quality cuda -out /dev/null -v 1

# RXMesh standalone test:
./bin/rxmesh_aniso_remesh gargoyle.obj 0.5 3

# Always test on real meshes (dragon, gargoyle), not toy cubes.
```

## Key Files

| File | What |
|------|------|
| `src/mmgs/cuda/mmgs_cuda.h` | Stage enum, checkpoint API, GPU dispatch declarations |
| `src/mmgs/cuda/mmgs_checkpoint.c` | Binary .msc checkpoint save/load |
| `src/mmgs/cuda/quality_s.cu` | GPU triangle quality kernel |
| `src/mmgs/cuda/anisosiz_s.cu` | GPU edge length + metric init kernels |
| `src/mmgs/cuda/mmgs_gpu_context.cu` | Persistent GPU context (upload once) |
| `src/mmgs/cuda/split_s.cu` | GPU parallel split pass (QuadriFlow pattern) |
| `src/mmgs/cuda/rxmesh_remesh/` | RXMesh-based GPU remeshing |
| `src/mmgs/cuda/rxmesh_remesh/aniso_metric.cuh` | Device functions for metric-space operations |
| `RXMesh/` | Vendored RXMesh with CUDA 13 patches |

## Known Issues

1. **GPU split writeback bug**: GPU-created midpoint vertices cause `adptri` explosion on dragon (works on gargoyle). Root cause: spdlog null logger crash was the RXMesh blocker, now fixed.
2. **RXMesh shared memory**: `cudaFuncSetAttribute` fails with 89KB on SM 86 (limit is 100KB — should work, investigating).
3. **Anisotropic remeshing via RXMesh**: Kernels written but not yet wired — need to replace `glm::distance2` with metric-space edge length in split/collapse/flip.

## Performance (Dragon 871K → 95K tris)

| Operation | CPU | GPU | Notes |
|-----------|-----|-----|-------|
| Quality computation | ~700ms | 9ms | 77x faster (persistent ctx) |
| Edge length batch | ~400ms/iter | 2-5ms | 80-200x faster (no upload) |
| anatri(geom) | 7.0s | 7.0s | Not GPU-accelerated (needs Bezier) |
| anatri(comp) | 3.3s | 3.1s | GPU marking saves ~200ms |
| adptri | 2.7s | 2.7s | Sequential topology changes |
| **Total** | **16.9s** | **17.8s** | GPU overhead > savings |

Real speedup requires RXMesh backend (full GPU-resident split/collapse/flip/smooth).
