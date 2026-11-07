# CUDA-Accelerated Multi-Exposure Image Fusion

**Team**

Estella Xie (ziyoux), Junhao Li (junhaoli)

**Project URL**

[Project webpage](https://Lijunhaohz.github.io/15618-project/)

## Summary

We propose to implement and optimize a CUDA-based multi-exposure image fusion pipeline that combines photographs of the same scene taken at different exposure levels into a single image with well-preserved details in both bright and dark regions. Using NVIDIA GPUs, we will parallelize multi-scale image processing operations and investigate optimizations including kernel fusion, shared-memory tiling, and intermediate-buffer reuse to improve execution time and memory efficiency. We will evaluate the performance and scalability of our optimized implementation against a CPU reference and an unoptimized CUDA baseline.

## Background

Photographs captured at a single exposure often lose details in either bright or dark regions. **Exposure fusion** addresses this problem by combining multiple images of the same scene at different exposures into a single image.

We will implement the [Exposure Fusion algorithm](https://www.tommertens.com/old-academic/papers/exposure_fusion_reduced.pdf) by Mertens et al., which consists of four main stages:

1. **Weight computation:** Compute per-pixel weights based on contrast, saturation, and well-exposedness, then normalize weights across input images.
2. **Pyramid construction:** Build Gaussian pyramids of the weight maps and Laplacian pyramids of the input images using filtering and downsampling.
3. **Multi-scale blending:** Compute weighted combinations of the Laplacian pyramids at each level.
4. **Reconstruction:** Progressively upsample and combine pyramid levels to produce the final image.

These operations offer substantial GPU parallelism because most per-pixel computations and neighborhood filters can execute independently. But the pipeline also introduces dependencies between pyramid levels and repeated accesses to large intermediate arrays.

We will first implement a CPU reference and a straightforward CUDA baseline with separate kernels for each stage, then optimize the GPU pipeline while preserving numerical correctness.

## The Challenge

The main challenge is efficiently mapping a multi-scale exposure fusion pipeline onto the GPU. Although individual pixel operations are highly parallel, dependencies across images, neighboring pixels, and pyramid levels introduce synchronization and memory overhead.

**1. Algorithmic Dependencies and Parallelism.** Weight normalization requires information from all input exposures, while Gaussian/Laplacian pyramid construction and reconstruction introduce sequential dependencies between levels. Although pixels within each level can be processed in parallel, these dependencies limit parallelism across stages and require synchronization. We will investigate how to minimize this overhead while preserving correctness.

**2. Memory Access Patterns and Data Reuse.** The pipeline repeatedly accesses large images and intermediate pyramids, with relatively little computation per memory access, making it potentially memory-bandwidth-bound. Neighborhood filtering exhibits spatial locality, but naive implementations repeatedly load overlapping pixels from global memory. We will explore kernel fusion and shared-memory tiling to reduce memory traffic, while balancing register pressure, shared-memory usage, and GPU occupancy.

**3. Varying Parallelism Across Pyramid Levels.** Image dimensions shrink at each pyramid level, creating different GPU utilization challenges. Fine levels offer abundant parallelism but high memory traffic, while coarse levels may underutilize the GPU. Boundary handling can also introduce branch divergence. We will investigate level-specific kernel configurations to improve performance across different workload sizes.

**4. Memory Capacity and Buffer Management.** Storing Gaussian and Laplacian pyramids for multiple exposures can consume substantial GPU memory. Reusing buffers and accumulating contributions requires careful handling of intermediate dependencies. For our streaming extension, cross-image weight normalization further complicates incremental processing and may require multiple passes.

Through this project, we aim to understand how **algorithmic dependencies, memory bandwidth, and GPU resource constraints** affect performance, and how to balance execution speed, parallelism, and memory efficiency through CUDA optimizations.

## Resources
We will develop the core CPU reference and CUDA implementations from scratch. Existing libraries will be used for image loading and saving, rather than for the core GPU fusion pipeline.

Planned resources include:
- An available course NVIDIA GPU; the exact model and memory capacity will be confirmed.
- CUDA development tools, CUDA-event timing, and available NVIDIA profiling tools.
- Pre-aligned exposure brackets from public datasets and/or controlled tripod captures.
- Synthetic constant images, impulses, and odd-sized images for boundary and reconstruction checks.
No special hardware beyond a CUDA-capable GPU is initially required.

## Goals and Deliverables
50% Target — Functional Baseline / Reduced-Scope Outcome
- Complete a CPU reference and a separate-kernel CUDA implementation.
- Support weight computation, normalization, pyramid construction, blending, and reconstruction.
- Validate intermediate stages and final outputs against the reference.
- Produce stage-level timings and identify the dominant bottlenecks.
100% Target — Optimized Pipeline
- Implement shared-memory tiling for selected neighborhood operations.
- Implement at least one substantive fusion strategy, such as filtering with downsampling or compatible weight computations.
- Reuse intermediate storage to reduce peak memory consumption.
- Compare alternative execution configurations across resolutions, exposure counts, and pyramid depths.
- Deliver a command-line application, correctness checks, visual examples, and a measurement-supported performance evaluation.
We will measure improvements rather than promise a fixed speedup before profiling.

150% Target — Incremental and Overlapped Execution
- Process exposures incrementally without retaining all input pyramids simultaneously.
- Handle cross-image normalization through an explicitly designed multi-pass strategy.
- Overlap transfers with computation using CUDA streams and double buffering.
- Evaluate the trade-off among memory savings, recomputation, additional transfers, and runtime.
A poster demonstration will show individual exposures and the fused output, accompanied by runtime and memory comparisons. A live demo is optional.

## Evaluation
We will compare the CPU reference, baseline CUDA pipeline, and optimized variants using identical algorithm parameters and boundary rules.

Measurements will include:
- GPU pipeline latency and processing time including host–device transfers.
- Stage-level runtime and peak device-memory allocation.
- Numerical error relative to the reference.
- Ablations isolating tiling, fusion, and storage reuse.
Experiments will vary image resolution, exposure count, and pyramid depth. Profiling evidence will be used to explain whether performance is limited by memory traffic, launch overhead, synchronization, or available parallelism. Visual comparisons will illustrate application behavior separately from implementation correctness.

## Platform Choice
CUDA provides explicit control over thread-block organization, shared memory, synchronization, and device buffers. These controls are well matched to the project’s central trade-offs: neighborhood reuse, multi-stage fusion, and memory scheduling. The CPU implementation will provide a correctness reference and a clearly defined performance baseline.

## Schedule
| Week | Planned work |
| --- | --- |
| Nov 7–13 | Finalize algorithm definitions, obtain data, confirm GPU access, and build the CPU reference. |
| Nov 14–20 | Implement independent CUDA stages and validate intermediate results. |
| Nov 21–27 | Complete the baseline, profile bottlenecks, and submit the Nov 23 milestone with preliminary results and an updated plan. Begin tiling/fusion experiments. |
| Nov 28–Dec 4 | Complete core optimizations and buffer reuse; run controlled comparisons. Attempt incremental execution if ahead of schedule. |
| Dec 5–10 | Finish experiments, document limitations, and complete the Dec 10 final submission. |
| Dec 11 | Present visual results and performance findings at the poster session. |


## Proposed division: 
- Estella: initially lead quality weights and cross-image accumulation; 
- Junhao: lead pyramid construction and reconstruction. 

Both will participate in reference validation, optimization, and experiments.

## Reference
Tom Mertens, Jan Kautz, and Frank Van Reeth. Exposure Fusion. Pacific Graphics, 2007.