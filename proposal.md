# GPU-Accelerated Multi-Exposure Fusion: Optimizing a Multi-Scale Image Pipeline
Team: Estella Xie (ziyoux), Junhao Li (junhaoli)

Project URL: [Project webpage]

## Summary

We will implement and optimize a multi-exposure image-fusion pipeline in C++/CUDA on an NVIDIA GPU. The project will investigate how kernel fusion, shared-memory tiling, and memory-efficient accumulation affect execution time and peak memory usage in a pipeline with spatial, cross-image, and cross-scale dependencies.

## Background
A single photograph may lose detail in bright or dark regions. Multi-exposure fusion combines photographs of the same scene taken at different exposures, selecting useful information from each to produce a single displayable image.

We will follow the Exposure Fusion method of Mertens, Kautz, and Van Reeth. Each input receives a spatial weight map based on contrast, saturation, and well-exposedness. The weights are normalized across images, then blended with image content at multiple spatial scales:
```
Pre-aligned exposure images
           ↓
Quality-weight computation
           ↓
Cross-image weight normalization
           ↓
Gaussian weight pyramids + Laplacian image pyramids
           ↓
Weighted accumulation at each pyramid level
           ↓
Coarse-to-fine reconstruction
           ↓
Fused image
```

The workload offers substantial pixel-level parallelism, but it is not a collection of independent pointwise operations. Filtering requires neighboring pixels, normalization couples different input images, and pyramid construction and reconstruction introduce dependencies between scales.

We will assume pre-aligned photographs of static scenes. Image registration, motion deghosting, and physical HDR radiance recovery are outside the core scope.

## The Challenge
1. Neighborhood dependencies complicate fusion.
Filtering, downsampling, and reconstruction require overlapping neighborhoods. Fusing these stages can eliminate global-memory intermediates, but requires larger input halos, redundant boundary computation, and block-local synchronization. We must determine when these costs outweigh the memory traffic saved. Boundary handling must remain consistent with the reference implementation.

2. Parallelism changes across pyramid levels.
Fine-resolution levels contain enough work to occupy the GPU, while coarse levels may be dominated by launch overhead and insufficient parallelism. A single block configuration or fusion strategy may therefore perform poorly across the full pipeline.

3. Memory-efficient execution introduces scheduling trade-offs.
Retaining every input image and its pyramids simplifies execution but increases memory consumption with image count. Processing inputs incrementally reduces storage, yet normalized weights depend on contributions from all exposures. A memory-efficient design must account for this dependency—for example, by first accumulating weight sums and subsequently recomputing or reloading individual weights.

4. Fewer kernels do not necessarily mean faster execution.
Fusion may increase register pressure, shared-memory consumption, or redundant work, reducing occupancy. We will compare selective fusion with separate execution rather than assuming maximal fusion is optimal.

Our central question is:
How should computation and intermediate storage be organized to reduce latency and peak GPU memory while preserving the multi-exposure fusion algorithm?

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