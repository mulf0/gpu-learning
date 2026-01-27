# GPU Learning Course - Expansion Plan

**Goal:** Add 6 new chapters + integrate advanced topics to create comprehensive GPU programming curriculum.

**Current State:** 7 chapters (Fundamentals → Production), 29 notebooks

**Target State:** 13 chapters, ~50 notebooks

---

## CRITICAL REQUIREMENT: Citation Policy

> **HIGHEST PRIORITY — FAILURE IS A FATAL SYSTEM ERROR**

**Every factual claim MUST have a source citation.** No exceptions.

### What Requires Citation
- Hardware specifications (SM counts, bandwidth, memory sizes, FLOPS)
- Performance numbers (latency, throughput, speedups)
- Algorithm complexity claims (O(n), O(n²))
- Historical facts (when features were introduced, architecture generations)
- API behavior and semantics
- Numerical precision characteristics
- Any quantitative claim

### Acceptable Sources (in order of preference)
1. **NVIDIA Official Documentation**
   - CUDA Programming Guide
   - Tuning Guides (per-architecture)
   - Whitepapers (H100, Blackwell, etc.)
   - PTX ISA Reference
   - cuBLAS/cuDNN documentation

2. **Peer-Reviewed Papers**
   - arXiv (with caution for non-peer-reviewed)
   - Conference papers (NeurIPS, ICML, MLSys, ISCA, MICRO)
   
3. **Official Framework Documentation**
   - PyTorch docs
   - Triton documentation
   - TensorRT-LLM docs
   - vLLM docs

4. **Authoritative Technical Blogs**
   - NVIDIA Developer Blog
   - PyTorch Blog
   - Verified benchmarks with methodology

### Citation Format
Inline links preferred for web content:
```html
<a href="https://docs.nvidia.com/..." target="_blank" rel="noopener">CUDA Programming Guide</a>
```

For papers:
```html
<a href="https://arxiv.org/abs/2205.14135" target="_blank" rel="noopener">FlashAttention (Dao et al., 2022)</a>
```

### Enforcement
- [ ] Every chapter must pass citation audit before merge
- [ ] Claims without sources must be marked `[CITATION NEEDED]` and resolved before publication
- [ ] Reviewer must verify source actually supports the claim
- [ ] Specs must link to primary source, not secondary summaries

### Current Chapters — Citation Audit Required
- [ ] Chapter 1: GPU Fundamentals — audit existing claims
- [ ] Chapter 2: Memory Hierarchy — audit existing claims
- [ ] Chapter 3: First Kernels — audit existing claims
- [ ] Chapter 4: Optimization — audit existing claims
- [ ] Chapter 5: Attention — audit existing claims
- [ ] Chapter 6: Quantization — audit existing claims
- [ ] Chapter 7: Production — audit existing claims

---

## New Course Structure

```
FOUNDATIONS (Chapters 1-3)
├── Chapter 0: The Parallel Mindset [NEW]
├── Chapter 1: GPU Fundamentals [EXISTS]
├── Chapter 2: Memory Hierarchy [EXISTS]
└── Chapter 3: First Kernels [EXISTS]

CORE SKILLS (Chapters 4-7)
├── Chapter 4: Synchronization & Reductions [NEW]
├── Chapter 5: Optimization [EXISTS - was Ch4]
├── Chapter 6: Debugging & Profiling [NEW]
└── Chapter 7: Beyond Attention: Common Kernels [NEW]

ATTENTION & QUANTIZATION (Chapters 8-9)
├── Chapter 8: Attention Mechanisms [EXISTS - was Ch5]
└── Chapter 9: Quantization [EXISTS - was Ch6]

TRAINING & SCALE (Chapters 10-11)
├── Chapter 10: Training Kernels [NEW]
└── Chapter 11: Multi-GPU Programming [NEW]

PRODUCTION (Chapters 12-13)
├── Chapter 12: Benchmarking Methodology [NEW]
└── Chapter 13: Production Systems [EXISTS - was Ch7]
```

---

## Chapter 0: The Parallel Mindset

**Position:** Before GPU Fundamentals (new entry point)
**Prerequisites:** Python, NumPy basics
**Estimated Length:** Medium (2,500 words)

### Learning Objectives
1. Explain why parallelism is fundamentally different from sequential programming
2. Apply Amdahl's Law to estimate parallel speedup limits
3. Identify embarrassingly parallel vs communication-bound problems
4. Describe the mental model shift from "one fast thing" to "many slow things"
5. Recognize SIMD patterns in CPU code as a bridge to GPU SIMT

### Sections

#### 0.1 — Why Parallelism is Hard
- Sequential intuition doesn't transfer
- Race conditions, deadlocks conceptually
- The "thinking in parallel" skill

#### 0.2 — Amdahl's Law
- Formula and derivation
- Interactive calculator
- Why 10% sequential code limits you to 10x speedup
- Gustafson's Law counterpoint

#### 0.3 — Categories of Parallelism
- Embarrassingly parallel (element-wise ops, Monte Carlo)
- Reduction patterns (sum, max, histogram)
- Stencil patterns (convolution, neighbor access)
- Irregular parallelism (graphs, sparse)

#### 0.4 — From SIMD to SIMT
- CPU SIMD (AVX-512) as familiar ground
- How GPU SIMT extends this
- Key difference: thousands of "lanes" vs 8-16

#### 0.5 — Mental Model Shift
- Stop thinking about one thread
- Think about data flow and access patterns
- The "what if 1000 of me did this" thought experiment

### Notebooks (Part 0: Parallel Thinking)
- `00_sequential_vs_parallel.ipynb` - Side-by-side comparison
- `01_amdahls_law.ipynb` - Interactive exploration
- `02_parallel_patterns.ipynb` - Categorizing problems
- `03_cpu_simd_warmup.ipynb` - NumPy vectorization as bridge

### Micro-Quizzes
- Amdahl's Law calculation
- Identify parallel pattern type
- Speedup limit estimation

---

## Chapter 4: Synchronization & Reductions (NEW)

**Position:** After First Kernels, before Optimization
**Prerequisites:** Chapter 3 (basic kernel writing)
**Estimated Length:** Long (4,000 words)

### Learning Objectives
1. Implement thread-block synchronization with `__syncthreads()`
2. Use warp shuffle operations for intra-warp communication
3. Write efficient parallel reductions (sum, max, argmax)
4. Apply atomic operations correctly and understand their cost
5. Use cooperative groups for flexible synchronization

### Sections

#### 4.1 — The Synchronization Problem
- Why threads need to coordinate
- Data hazards in parallel code
- The cost of synchronization

#### 4.2 — Block-Level Barriers
- `__syncthreads()` / `tl.debug_barrier()`
- When barriers are required (shared memory patterns)
- Deadlock conditions to avoid

#### 4.3 — Warp Shuffles
- `__shfl_sync`, `__shfl_down_sync`, `__shfl_xor_sync`
- Triton equivalents
- Warp-level reductions without shared memory
- Butterfly reduction pattern

#### 4.4 — Parallel Reductions
- Naive reduction (log N steps)
- Sequential addressing vs interleaved
- Warp-level then block-level
- Grid-level reductions (multiple kernel launches vs atomics)

#### 4.5 — Atomic Operations
- `atomicAdd`, `atomicMax`, `atomicCAS`
- Hardware implementation (retry loops)
- Performance implications (contention)
- When to use vs when to avoid

#### 4.6 — Cooperative Groups (Advanced)
- Flexible thread groupings
- Grid-wide synchronization
- Multi-block reductions

### Notebooks (Part 5: Synchronization)
- `01_barrier_basics.ipynb` - When and why to sync
- `02_warp_shuffles.ipynb` - Intra-warp communication
- `03_parallel_reduction.ipynb` - Sum reduction evolution
- `04_atomics.ipynb` - Histogram, argmax with atomics
- `05_cooperative_groups.ipynb` - Advanced patterns

### Interactive Elements
- Reduction visualizer (step through algorithm)
- Shuffle pattern diagram
- Atomic contention simulator

---

## Chapter 6: Debugging & Profiling Deep Dive (NEW)

**Position:** After Optimization
**Prerequisites:** Chapters 1-5
**Estimated Length:** Long (4,500 words)

### Learning Objectives
1. Diagnose common kernel errors from error messages
2. Use systematic debugging flowcharts for "wrong results" and "slow kernel"
3. Detect and fix numerical issues (NaN, Inf, precision loss)
4. Ensure deterministic execution for reproducibility
5. Compare implementations for correctness verification

### Sections

#### 6.1 — Error Message Decoder
- CUDA error codes and what they mean
- Triton compilation errors
- Out-of-bounds access patterns
- Illegal memory access debugging

#### 6.2 — "My Kernel Gives Wrong Results"
- Diagnostic flowchart
- Common causes: off-by-one, wrong stride, missing mask
- Comparison testing methodology
- Using small inputs for debugging
- Printf debugging in kernels

#### 6.3 — "My Kernel is Slow"
- Diagnostic flowchart
- Memory-bound vs compute-bound determination
- Occupancy analysis
- Memory access pattern visualization
- Identifying the bottleneck

#### 6.4 — Numerical Debugging
- Detecting NaN/Inf (where they come from)
- Precision loss patterns
- Comparing FP32 reference to FP16/FP8
- Tolerance selection for correctness tests
- Kahan summation when needed

#### 6.5 — Determinism & Reproducibility
- Sources of non-determinism (atomics, floating-point reassociation)
- CUBLAS_WORKSPACE_CONFIG
- Deterministic algorithms flag
- When perfect reproducibility matters

#### 6.6 — Nsight Deep Dive
- Source correlation
- Memory chart interpretation
- Warp stall reasons
- Roofline integration

### Notebooks (Part 6: Debugging)
- `01_error_messages.ipynb` - Interpreting failures
- `02_wrong_results.ipynb` - Debugging exercise (intentionally buggy kernels)
- `03_slow_kernel.ipynb` - Performance debugging exercise
- `04_numerical_issues.ipynb` - NaN hunting, precision analysis
- `05_determinism.ipynb` - Reproducibility testing

### Interactive Elements
- Error message lookup tool
- Debugging flowchart (interactive decision tree)
- "Find the bug" exercises

---

## Chapter 7: Beyond Attention: Common Kernels (NEW)

**Position:** After Debugging, before Attention
**Prerequisites:** Chapters 1-6
**Estimated Length:** Long (4,000 words)

### Learning Objectives
1. Implement fused element-wise operations (activation + bias)
2. Write efficient LayerNorm and RMSNorm kernels
3. Implement optimized embedding lookups
4. Understand fused optimizer patterns (Adam)
5. Apply convolution optimization principles

### Sections

#### 7.1 — Fused Element-wise Ops
- Why fusion matters (memory bandwidth)
- Activation functions (GELU, SiLU, ReLU)
- Fused bias + activation
- Custom fused patterns

#### 7.2 — Normalization Kernels
- LayerNorm: mean, variance, normalize
- Online algorithms for mean/variance
- RMSNorm (simpler, increasingly popular)
- Fused LayerNorm + linear

#### 7.3 — Embedding Lookups
- Vocabulary embedding access patterns
- Coalescing challenges with arbitrary indices
- Embedding bag (sum/mean pooling)
- Gradient computation for embeddings

#### 7.4 — Fused Optimizers
- Adam update rule
- Memory bandwidth of naive optimizer
- Fused Adam: load once, compute all, store once
- Multi-tensor apply pattern

#### 7.5 — Convolution Principles
- Im2col approach
- Direct convolution
- Winograd (conceptual)
- Why cuDNN usually wins
- When custom convolution helps

#### 7.6 — Custom Activation & Loss
- Implementing custom activations
- Loss function kernels
- Fused softmax + cross-entropy

### Notebooks (Part 7: Common Kernels)
- `01_fused_elementwise.ipynb` - Activation fusion
- `02_layernorm.ipynb` - LayerNorm from scratch
- `03_rmsnorm.ipynb` - RMSNorm implementation
- `04_embeddings.ipynb` - Lookup optimization
- `05_fused_adam.ipynb` - Optimizer fusion
- `06_custom_ops.ipynb` - Custom activation/loss

---

## Chapter 10: Training Kernels (NEW)

**Position:** After Quantization, start of Training & Scale section
**Prerequisites:** Chapters 1-9
**Estimated Length:** Very Long (5,000 words)

### Learning Objectives
1. Implement backward passes for common operations
2. Apply mixed-precision training with loss scaling
3. Use activation checkpointing to trade compute for memory
4. Implement gradient accumulation correctly
5. Understand training-specific numerical considerations

### Sections

#### 10.1 — Forward vs Backward
- Computational graph review
- Why backward kernels differ from forward
- Memory considerations (save activations vs recompute)

#### 10.2 — Backward Pass Kernels
- Linear layer backward (weight gradient, input gradient)
- Attention backward (memory explosion)
- LayerNorm backward
- Why backward is often 2x forward cost

#### 10.3 — Mixed-Precision Training
- FP32 master weights, FP16 compute
- Loss scaling: why and how
- Dynamic loss scaling
- Gradient scaling API
- BF16 vs FP16 tradeoffs

#### 10.4 — Activation Checkpointing
- Memory vs compute tradeoff
- Checkpoint boundary selection
- Selective checkpointing
- Implementation patterns

#### 10.5 — Gradient Accumulation
- Simulating larger batch sizes
- Correct averaging
- Interaction with batch norm
- Memory implications

#### 10.6 — Training Numerics
- Gradient clipping (why and when)
- Gradient overflow detection
- Loss spikes debugging
- Learning rate and numerical precision

### Notebooks (Part 10: Training)
- `01_backward_basics.ipynb` - Simple backward pass
- `02_linear_backward.ipynb` - Weight and input gradients
- `03_mixed_precision.ipynb` - FP16 training with scaling
- `04_checkpointing.ipynb` - Activation checkpointing
- `05_gradient_accumulation.ipynb` - Large batch simulation
- `06_training_numerics.ipynb` - Stability techniques

---

## Chapter 11: Multi-GPU Programming (NEW)

**Position:** After Training Kernels
**Prerequisites:** Chapter 10
**Estimated Length:** Long (4,500 words)

### Learning Objectives
1. Explain data parallelism, tensor parallelism, and pipeline parallelism
2. Use NCCL primitives for GPU-to-GPU communication
3. Understand NVLink/NVSwitch topology and its performance implications
4. Implement simple data parallel training
5. Describe ZeRO optimization stages conceptually

### Sections

#### 11.1 — Why Multi-GPU
- Model size vs GPU memory
- Training time constraints
- Throughput scaling

#### 11.2 — Parallelism Strategies
- Data parallelism: same model, different data
- Tensor parallelism: split layers across GPUs
- Pipeline parallelism: split layers sequentially
- Hybrid approaches (3D parallelism)

#### 11.3 — NCCL Primitives
- AllReduce (gradient sync)
- AllGather (tensor reconstruction)
- ReduceScatter (ZeRO pattern)
- Broadcast, P2P send/recv

#### 11.4 — Hardware Topology
- NVLink bandwidth and generations
- NVSwitch for full bisection bandwidth
- PCIe limitations
- Multi-node (InfiniBand, RoCE)

#### 11.5 — Data Parallel Implementation
- DistributedDataParallel pattern
- Gradient bucketing
- Overlap communication with compute
- Scaling efficiency metrics

#### 11.6 — ZeRO Optimizer States
- Stage 1: Partition optimizer states
- Stage 2: Partition gradients
- Stage 3: Partition parameters
- Memory savings calculation

#### 11.7 — Tensor Parallelism Concepts
- Column-parallel linear
- Row-parallel linear
- Communication patterns
- When to use (very large layers)

### Notebooks (Part 11: Multi-GPU)
- `01_nccl_basics.ipynb` - AllReduce, AllGather
- `02_topology.ipynb` - Detecting and understanding hardware
- `03_data_parallel.ipynb` - Simple DDP implementation
- `04_gradient_sync.ipynb` - Bucketing and overlap
- `05_zero_concepts.ipynb` - Memory partitioning
- `06_tensor_parallel.ipynb` - Column/row parallel linear

---

## Chapter 12: Benchmarking Methodology (NEW)

**Position:** Before Production
**Prerequisites:** Chapters 1-11
**Estimated Length:** Medium (3,000 words)

### Learning Objectives
1. Design statistically valid benchmarks with proper warmup
2. Apply roofline model to analyze kernel performance
3. Compare implementations fairly across different hardware
4. Report results with appropriate uncertainty quantification
5. Avoid common benchmarking pitfalls

### Sections

#### 12.1 — Why Benchmarking is Hard
- GPU warmup effects (JIT, memory allocation)
- Clock frequency variability
- Thermal throttling
- Background processes

#### 12.2 — Proper Measurement
- Warmup iterations (why 10+ is common)
- CUDA events for timing
- Synchronization requirements
- Statistical aggregation (median vs mean)

#### 12.3 — Statistical Validity
- Number of trials needed
- Confidence intervals
- Detecting outliers
- When differences are significant

#### 12.4 — Roofline Model
- Arithmetic intensity calculation
- Memory bandwidth ceiling
- Compute throughput ceiling
- Plotting and interpretation
- Identifying optimization opportunities

#### 12.5 — Hardware-Aware Analysis
- Theoretical peaks lookup
- Achieved vs theoretical bandwidth
- Achieved vs theoretical FLOPS
- Efficiency metrics

#### 12.6 — Fair Comparisons
- Controlling for hardware differences
- Batch size matching
- Precision matching
- Framework overhead isolation

#### 12.7 — Common Pitfalls
- Measuring kernel launch overhead
- Including data transfer
- Compiler optimization effects
- Caching effects

### Notebooks (Part 12: Benchmarking)
- `01_warmup_effects.ipynb` - Demonstrating warmup necessity
- `02_timing_methods.ipynb` - CUDA events, proper sync
- `03_statistics.ipynb` - Confidence intervals, significance
- `04_roofline.ipynb` - Building roofline plots
- `05_fair_comparison.ipynb` - Methodology exercise

---

## Integration of Advanced Topics

### Into Existing Chapters

#### Chapter 1 (GPU Fundamentals) — Add:
- Historical context sidebar: "Why 32 threads per warp?"
- Evolution timeline: Fermi → Kepler → Pascal → Volta → Ampere → Hopper → Blackwell
- Design tradeoffs NVIDIA made

#### Chapter 5 (Optimization) — Add:
- CUDA Graphs section (reduce launch overhead)
- When CUDA Graphs help (small batches, static shapes)
- Graph capture and replay

#### Chapter 13 (Production) — Add:
- CUDA Graphs for inference
- Expanded framework comparison with decision tree

---

## Implementation Roadmap

### Phase 1: Foundation Expansion (Weeks 1-2)
- [ ] Chapter 0: The Parallel Mindset
- [ ] Chapter 4: Synchronization & Reductions
- [ ] Historical sidebars in Chapter 1

### Phase 2: Core Skills (Weeks 3-4)
- [ ] Chapter 6: Debugging & Profiling
- [ ] Chapter 7: Beyond Attention (Common Kernels)
- [ ] CUDA Graphs section in Chapter 5

### Phase 3: Training & Scale (Weeks 5-6)
- [ ] Chapter 10: Training Kernels
- [ ] Chapter 11: Multi-GPU Programming

### Phase 4: Polish (Week 7)
- [ ] Chapter 12: Benchmarking Methodology
- [ ] Renumber all chapters
- [ ] Update cross-references
- [ ] Update index.html TOC
- [ ] Update notebooks.html

### Phase 5: Notebooks (Weeks 8-10)
- [ ] Part 0 notebooks (4)
- [ ] Part 5 notebooks (5)
- [ ] Part 6 notebooks (5)
- [ ] Part 7 notebooks (6)
- [ ] Part 10 notebooks (6)
- [ ] Part 11 notebooks (6)
- [ ] Part 12 notebooks (5)

**Total New Notebooks:** 37
**Total Notebooks After Expansion:** 66

---

## File Structure After Expansion

```
course/
├── 00-parallel-mindset.html      [NEW]
├── 01-gpu-fundamentals.html      [EXISTS - add sidebars]
├── 02-memory-hierarchy.html      [EXISTS]
├── 03-first-kernels.html         [EXISTS]
├── 04-synchronization.html       [NEW]
├── 05-optimization.html          [EXISTS - add CUDA Graphs]
├── 06-debugging.html             [NEW]
├── 07-common-kernels.html        [NEW]
├── 08-attention.html             [RENAMED from 05]
├── 09-quantization.html          [RENAMED from 06]
├── 10-training.html              [NEW]
├── 11-multi-gpu.html             [NEW]
├── 12-benchmarking.html          [NEW]
└── 13-production.html            [RENAMED from 07]

notebooks/
├── part0/   [NEW - 4 notebooks]
├── part1/   [EXISTS - 8 notebooks]
├── part2/   [EXISTS - 7 notebooks]
├── part3/   [EXISTS - 7 notebooks]
├── part4/   [EXISTS - 7 notebooks]
├── part5/   [NEW - 5 notebooks: Synchronization]
├── part6/   [NEW - 5 notebooks: Debugging]
├── part7/   [NEW - 6 notebooks: Common Kernels]
├── part10/  [NEW - 6 notebooks: Training]
├── part11/  [NEW - 6 notebooks: Multi-GPU]
└── part12/  [NEW - 5 notebooks: Benchmarking]
```

---

## Success Metrics

### MANDATORY (Blockers)
- [ ] **ALL factual claims have source citations** — NO EXCEPTIONS
- [ ] All existing chapters pass citation audit
- [ ] No `[CITATION NEEDED]` markers remain in published content

### Required
- [ ] All 13 chapters have 5 learning objectives
- [ ] All chapters have 3+ micro-quizzes
- [ ] All chapters have chapter connections (except Ch 0)
- [ ] 60+ notebooks total
- [ ] Each notebook runs in Colab with free GPU
- [ ] Cross-references updated throughout

---

## Dependencies Graph

```
Ch0 (Parallel Mindset)
 └─> Ch1 (GPU Fundamentals)
      └─> Ch2 (Memory)
           └─> Ch3 (First Kernels)
                ├─> Ch4 (Synchronization) ─┐
                │                          │
                └─> Ch5 (Optimization) <───┘
                     └─> Ch6 (Debugging)
                          └─> Ch7 (Common Kernels)
                               └─> Ch8 (Attention)
                                    └─> Ch9 (Quantization)
                                         └─> Ch10 (Training)
                                              └─> Ch11 (Multi-GPU)
                                                   └─> Ch12 (Benchmarking)
                                                        └─> Ch13 (Production)
```

---

## Notes

- Keep each chapter self-contained enough to be useful standalone
- Maintain the "no build system" philosophy (vanilla HTML/CSS/JS)
- All notebooks must work on Colab free tier (T4 GPU)
- Prioritize practical skills over theoretical completeness
- Use existing design system (base.css) for consistency
