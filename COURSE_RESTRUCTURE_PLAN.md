# GPU Learning Course: Interleaved Coding-First Restructure Plan

> **Document Purpose:** Master plan for restructuring the GPU/CUDA learning course from a traditional "prerequisites first" approach to an interleaved, programmer-first curriculum with Python.

## Executive Summary

Transform the existing course from separate math prerequisites + theory sections into a **coding-first, just-in-time learning** experience where concepts are introduced precisely when learners need them to solve real problems.

---

## Current State Analysis

### Existing Content Inventory

| File | Content | Lines |
|------|---------|-------|
| `math-prerequisites.html` | Diagnostic quiz, resource links, learning paths | 1351 |
| `attention-math.html` | Dot product, softmax, online softmax, FP formats | 1004 |
| `lessons/gpu-architecture.html` | SMs, warps, warpgroups, occupancy calculator | 867 |
| `lessons/memory-hierarchy.html` | Coalescing, bank conflicts, TMA, bandwidth | 1166 |
| `lessons/math-foundations.html` | Matrices, index arithmetic, FP, quantization | 1500+ |
| `kernel-course.html` | 4-week curriculum, foundations, resources | 1700+ |

### Problems with Current Structure

1. **Front-loaded prerequisites** — Learners encounter abstract concepts without context
2. **Motivation gap** — "Why am I learning row-major order?" before seeing it matter
3. **Poor retention** — Math learned in isolation doesn't transfer to coding
4. **Separate interactive content** — Visualizations are in lesson pages, not integrated
5. **No Python entry point** — Jumps straight to CuTe/CUDA concepts

---

## Design Principles

### 1. No Separate Prerequisites
Every concept introduced at the moment it's needed to solve a coding problem.

**Old:** "Learn matrix indexing" → (weeks later) → "Now use it in your kernel"  
**New:** "Your kernel outputs garbage. Why? Let's understand row-major indexing."

### 2. Python First
Start with familiar tools, progressively add GPU-specific knowledge.

```
NumPy → CuPy → Triton → (optional) CUDA C++
```

### 3. Code Before Theory
See the problem in running code, then understand why it happens.

**Old:** "Coalescing means adjacent threads access adjacent memory addresses"  
**New:** "Run this kernel. See how it's 10x slower? Here's why..."

### 4. Profile-Driven Learning
Every optimization claim backed by actual measurement.

```python
# Before optimization
%timeit kernel_v1[grid](x, y)  # 2.3 ms

# After understanding coalescing
%timeit kernel_v2[grid](x, y)  # 0.23 ms ← 10x faster!
```

### 5. Interleaved Interactives
Existing visualizations embedded in the learning flow, not separate pages.

---

## Course Structure

### Overview

```
Week 1: "Why GPUs?" — From Slow to Fast
  └── Project: Matrix Multiplication Journey

Week 2: "The Memory Game" — Optimization Deep Dive  
  └── Project: Achieving 80% Peak Bandwidth

Week 3: "Numbers & Attention" — Precision Meets Performance
  └── Project: Build FlashAttention from Scratch

Week 4: "Compression & Production" — Real-World Deployment
  └── Project: Mixed-Precision KV Cache Kernel
```

---

### Week 1: "Why GPUs?" — From Slow to Fast

**Driving Project:** Take a naive matmul from 1 GFLOPS to 1000+ GFLOPS

#### Day 1: Python Baseline
- **Challenge:** Multiply two 1024×1024 matrices in NumPy
- **Measure:** Time it, calculate GFLOPS
- **Concepts pulled in:** None (baseline)
- **Code artifact:** `01_numpy_baseline.py`

#### Day 2: CuPy — Instant Speedup
- **Challenge:** Same matmul with CuPy
- **Observe:** 50-100x speedup with zero code changes
- **Concepts pulled in:** GPU vs CPU, memory transfers, device arrays
- **Explore:** What happens if you transfer data every iteration?

#### Day 3: How GPUs Actually Work
- **Challenge:** Why is CuPy fast? Let's peek inside.
- **Concepts pulled in:**
  - SMs (streaming multiprocessors) — like CPU cores, but thousands
  - Warps — 32 threads that move in lockstep
  - SIMT execution model
- **Interactive:** Warp execution visualization (from `gpu-architecture.html`)
- **Code artifact:** Simple profiling to see SM utilization

#### Day 4: Your First Triton Kernel
- **Challenge:** Write vector addition in Triton
- **Concepts pulled in:**
  - Index arithmetic: `linear = row * cols + col`
  - Program IDs, block sizes
  - Row-major vs column-major (why it matters for access)
- **Code artifact:** `04_triton_vector_add.py`
- **Quiz:** Convert 2D coordinate to linear index

#### Day 5: Naive Matmul — Why It's Slow
- **Challenge:** Write matmul in Triton (naive triple loop)
- **Observe:** 10-50x slower than CuPy!
- **Profile:** Show memory bottleneck in Nsight Compute
- **Concepts pulled in:**
  - Memory hierarchy (registers → shared → L2 → HBM)
  - Latency numbers (1 cycle vs 400 cycles)
- **Interactive:** Memory hierarchy diagram (from `memory-hierarchy.html`)

#### Day 6: Tiling — The Key Insight
- **Challenge:** Why do we need tiles?
- **Concepts pulled in:**
  - Data reuse — load once, use many times
  - Shared memory as programmer-managed cache
  - Tile coordinates: `(blockIdx.x * TILE_SIZE + threadIdx.x)`
- **Visual:** Tiled matmul animation
- **Code artifact:** `06_tiled_matmul_v1.py`

#### Day 7: Your First Fast Matmul
- **Challenge:** Implement tiled matmul with shared memory
- **Concepts pulled in:**
  - Coalescing — why adjacent threads must access adjacent memory
  - Bank conflicts — 32 banks, stride patterns
- **Interactive:** Coalescing and bank conflict visualizations (from `memory-hierarchy.html`)
- **Milestone:** Achieve 500+ GFLOPS

**Week 1 Math Summary:**
- Linear indexing: `idx = row * cols + col`
- Row-major vs column-major storage
- Tile coordinate transformation
- Basic profiling metrics (GFLOPS, bandwidth)

---

### Week 2: "The Memory Game" — Optimization Deep Dive

**Driving Project:** Push matmul to 80%+ of theoretical peak

#### Day 1: Profile Like a Pro
- **Tools:** `nsys profile`, `ncu` analysis
- **Metrics that matter:**
  - Memory throughput (GB/s achieved vs peak)
  - Compute throughput (TFLOPS achieved vs peak)
  - Occupancy (active warps / max warps)
- **Code artifact:** Profiling scripts, metric interpretation guide

#### Day 2: Coalescing Experiments
- **Challenge:** Create kernels with good/bad access patterns
- **Measure:** Actual bandwidth difference
- **Concepts pulled in:**
  - 128-byte cache lines
  - Memory transactions
  - Alignment requirements
- **Interactive:** Coalescing visualization with thread/memory mapping

#### Day 3: Bank Conflict Laboratory
- **Challenge:** Deliberately create bank conflicts, measure penalty
- **Concepts pulled in:**
  - 32 shared memory banks
  - Stride analysis: `smem[tid * 32]` = worst case
  - Broadcast exception (same address = no conflict)
- **Interactive:** Bank conflict visualization (from `memory-hierarchy.html`)
- **Fix:** Padding technique

#### Day 4: Hiding Latency — Pipelining
- **Challenge:** Overlap memory loads with compute
- **Concepts pulled in:**
  - Async copy operations
  - Multi-buffering (double/triple buffering)
  - Producer-consumer pattern
- **Code artifact:** Pipelined matmul kernel

#### Day 5: TMA — Tensor Memory Accelerator (Hopper+)
- **Challenge:** Use TMA for bulk data movement
- **Concepts pulled in:**
  - TMA descriptors
  - Multicast to multiple SMs
  - Address generation offloaded from SM
- **Code artifact:** TMA-based copy kernel

#### Day 6: Tensor Cores — Matrix Math Units
- **Challenge:** Use Tensor Cores for matmul
- **Concepts pulled in:**
  - MMA (matrix-multiply-accumulate) operations
  - Supported shapes (16×8×16, warpgroup 64×64)
  - Type constraints (FP16, BF16, FP8)
- **Code artifact:** Tensor Core matmul

#### Day 7: Optimized GEMM — Putting It Together
- **Challenge:** Combine all optimizations
- **Milestone:** Achieve 80% of theoretical peak
- **Compare:** Profile vs cuBLAS, understand remaining gap
- **Code artifact:** Production-quality GEMM kernel

**Week 2 Math Summary:**
- Arithmetic intensity: `FLOPS / bytes`
- Roofline model: memory-bound vs compute-bound
- Bandwidth calculations: `data_transferred / time`
- Occupancy: `active_warps / max_warps`

---

### Week 3: "Numbers & Attention" — Precision Meets Performance

**Driving Project:** Build FlashAttention from scratch

#### Day 1: The Dot Product — Similarity in Code
- **Challenge:** Compute Q·K for a single query against multiple keys
- **Concepts pulled in:**
  - Dot product as similarity measure
  - Geometric interpretation: `|Q||K|cos(θ)`
- **Interactive:** Dot product visualization (from `attention-math.html`)
- **Code artifact:** Dot product kernel

#### Day 2: The Softmax Problem
- **Challenge:** Run softmax on large values
- **Observe:** NaN/Inf explosion
- **Concepts pulled in:**
  - Floating point overflow: `exp(100)` overflows FP16
  - Range vs precision tradeoff
- **Interactive:** FP format comparison (from `attention-math.html`)
- **Code artifact:** Naive softmax that breaks

#### Day 3: Stable Softmax
- **Challenge:** Fix the overflow
- **Concepts pulled in:**
  - Max-subtraction trick: `exp(x - max)` stays bounded
  - Mathematical proof: why this doesn't change the result
  - Log-sum-exp for numerical stability
- **Code artifact:** Stable softmax kernel

#### Day 4: Attention — The Complete Formula
- **Challenge:** Implement full attention: `softmax(QK^T / √d) @ V`
- **Concepts pulled in:**
  - Why √d scaling? Variance normalization
  - Quadratic memory problem: [seq × seq] is huge
- **Interactive:** Attention step visualization (from `attention-math.html`)
- **Code artifact:** Naive attention (materializes full matrix)

#### Day 5: Online Softmax — The Streaming Algorithm
- **Challenge:** Compute softmax without seeing all values
- **Concepts pulled in:**
  - Running max and sum tracking
  - Rescaling when max changes
  - Single-pass algorithm
- **Interactive:** Online softmax simulation (from `attention-math.html`)
- **Code artifact:** Online softmax kernel

#### Day 6: Tiled Attention — Block by Block
- **Challenge:** Apply online softmax to tiled attention
- **Concepts pulled in:**
  - Block-wise computation in SRAM
  - Output accumulation with rescaling
  - FlashAttention algorithm overview
- **Code artifact:** Tiled attention kernel (single pass)

#### Day 7: FlashAttention Kernel
- **Challenge:** Full FlashAttention implementation
- **Milestone:** O(N) memory, competitive speed
- **Profile:** Compare vs naive attention
- **Code artifact:** Complete FlashAttention

**Week 3 Math Summary:**
- Dot product geometry: `a·b = |a||b|cos(θ)`
- Floating point: sign, exponent, mantissa
- Machine epsilon: smallest distinguishable difference
- Softmax stability: `softmax(x) = softmax(x - c)` for any constant c
- Variance normalization: why divide by √d

---

### Week 4: "Compression & Production" — Real-World Deployment

**Driving Project:** Mixed-precision KV cache kernel (K=FP8, V=NVFP4)

#### Day 1: FP16 → FP8 — Halving Precision
- **Challenge:** Convert FP16 tensor to FP8, measure error
- **Concepts pulled in:**
  - E4M3 format: 4 exponent bits, 3 mantissa bits, max=448
  - E5M2 format: 5 exponent bits, 2 mantissa bits, max=57344
  - When to use which (activations vs gradients)
- **Interactive:** FP8 bit toggle (from `math-foundations.html`)
- **Code artifact:** FP8 quantization kernel

#### Day 2: Quantization Fundamentals
- **Challenge:** Implement quantize/dequantize round-trip
- **Concepts pulled in:**
  - Scale factor computation: `scale = max(|x|) / max_representable`
  - Rounding modes (nearest, stochastic)
  - Quantization error analysis
- **Interactive:** Quantization visualization (from `math-foundations.html`)
- **Code artifact:** General quantization utilities

#### Day 3: INT8 and INT4
- **Challenge:** Implement block-wise INT4 quantization
- **Concepts pulled in:**
  - Per-channel vs per-tensor scaling
  - Block scaling (groups of 32 or 128)
  - Activation-aware quantization (AWQ concepts)
- **Code artifact:** INT4 quantization with block scales

#### Day 4: NVFP4 — Blackwell's Native Format
- **Challenge:** Implement NVFP4 two-level scaling
- **Concepts pulled in:**
  - E2M1 format: only 6 distinct positive values
  - Per-block FP8 scale factors (16-element blocks)
  - Per-tensor FP32 scale
- **Code artifact:** NVFP4 quantization kernel

#### Day 5: KV Cache Quantization Strategy
- **Challenge:** Design mixed-precision KV cache
- **Concepts pulled in:**
  - Why K is more sensitive than V (softmax amplification)
  - K at FP8, V at NVFP4: ~2.6× memory reduction
  - Quality vs compression tradeoff
- **Code artifact:** Mixed-precision KV cache storage

#### Day 6: Full Attention + Quantization
- **Challenge:** Fuse attention with quantized KV cache
- **Concepts pulled in:**
  - Dequantize-on-load vs precompute
  - Fused kernels for bandwidth efficiency
  - Numerical accuracy testing
- **Code artifact:** Quantized attention kernel

#### Day 7: Production Integration
- **Challenge:** Integrate with real inference framework
- **Topics:**
  - TensorRT-LLM plugin architecture
  - vLLM quantization hooks
  - Benchmarking methodology
- **Final milestone:** End-to-end quantized inference

**Week 4 Math Summary:**
- FP format calculations: value = (-1)^sign × 2^(exp-bias) × (1 + mantissa)
- Quantization error: `|x - Q(x)| ≤ scale / 2`
- Block scaling: separate scale per N elements
- Precision/range tradeoff: more exponent bits = more range, less precision

---

## Module Micro-Structure

Each day follows this 5-step pattern (~1 hour total):

### Step 1: The Challenge (5 minutes)
Present a coding problem that reveals the concept need.

```python
# Today's challenge: This kernel is 10x slower than expected. Why?
@triton.jit
def slow_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * 32 * BLOCK + tl.arange(0, BLOCK) * 32  # Bad access!
    ...
```

### Step 2: Explore (15 minutes)
Interactive visualization or hands-on experiment.

- Embed existing interactive components
- Or provide code cells to experiment with variations
- Encourage "what if?" exploration

### Step 3: The Concept (10 minutes)
Just-in-time explanation of the underlying principle.

- Keep it brief — learner now has context
- Link to deeper resources for those who want more
- Connect to GPU hardware/architecture

### Step 4: Code It (30+ minutes)
Implement the solution.

- Scaffolded code with TODO sections
- Clear success criteria
- Test cases provided

### Step 5: Verify (10 minutes)
Confirm understanding.

- Quiz (can reuse existing quiz components)
- Profile to verify improvement
- Reflection: "What would happen if...?"

---

## Content Migration Plan

### Files to Dissolve (Content Distributed Across Weeks)

| Source File | Destination |
|-------------|-------------|
| `math-prerequisites.html` | Diagnostic quiz → Week 1 Day 1 pre-assessment |
| | Index arithmetic → Week 1 Days 4-6 |
| | FP basics → Week 3 Days 1-3 |
| | Learning paths → Remove (integrated now) |

### Files to Restructure

| Source File | Destination |
|-------------|-------------|
| `attention-math.html` | Week 3, Days 1-7 (primary source) |
| `lessons/gpu-architecture.html` | Week 1, Days 3-6 |
| `lessons/memory-hierarchy.html` | Week 1 Day 5, Week 2 Days 1-5 |
| `lessons/math-foundations.html` | Weeks 3-4 (numerics content) |

### Interactive Components to Preserve

| Component | Current Location | New Location |
|-----------|-----------------|--------------|
| Warp execution viz | `gpu-architecture.html` | Week 1, Day 3 |
| Occupancy calculator | `gpu-architecture.html` | Week 2, Day 1 |
| Coalescing visualization | `memory-hierarchy.html` | Week 1, Day 7 |
| Bank conflict viz | `memory-hierarchy.html` | Week 2, Day 3 |
| Memory hierarchy diagram | `memory-hierarchy.html` | Week 1, Day 5 |
| Dot product calculator | `attention-math.html` | Week 3, Day 1 |
| Softmax visualization | `attention-math.html` | Week 3, Day 2 |
| Online softmax simulation | `attention-math.html` | Week 3, Day 5 |
| FP bit toggle | `math-foundations.html` | Week 3, Day 2 |
| Quantization viz | `math-foundations.html` | Week 4, Day 2 |
| Tile coordinate viz | `math-foundations.html` | Week 1, Day 6 |

---

## Delivery Format

### Primary: Jupyter Notebooks

Each week consists of 7 notebooks:
```
week1/
  01_numpy_baseline.ipynb
  02_cupy_intro.ipynb
  03_gpu_architecture.ipynb
  04_first_triton_kernel.ipynb
  05_memory_hierarchy.ipynb
  06_tiling_basics.ipynb
  07_fast_matmul.ipynb
```

Notebooks include:
- Markdown cells for concepts
- Code cells for implementation
- Embedded visualizations (ipywidgets or iframe to HTML)
- Quiz cells (ipywidgets checkboxes/buttons)

### Supporting: HTML Interactive Pages

Keep existing HTML pages as standalone references:
- Link from notebooks for deep dives
- Embed via IFrame where appropriate
- Update navigation to point to new course structure

### Prerequisites Check

Single notebook at start: `00_ready_check.ipynb`
- Test Python/NumPy proficiency
- Test GPU availability (CuPy/Triton install)
- Brief diagnostic quiz (adapted from `math-prerequisites.html`)
- Recommended path based on results

---

## Implementation Phases

### Phase 1: Foundation (2 weeks)
- [ ] Create Week 1 notebooks (Days 1-7)
- [ ] Migrate GPU architecture content
- [ ] Migrate memory hierarchy content
- [ ] Test on fresh environment

### Phase 2: Core Content (2 weeks)
- [ ] Create Week 2 notebooks
- [ ] Create Week 3 notebooks
- [ ] Migrate attention math content
- [ ] Migrate math foundations content

### Phase 3: Advanced + Polish (2 weeks)
- [ ] Create Week 4 notebooks
- [ ] Embed interactive visualizations
- [ ] Create quizzes/assessments
- [ ] Write solutions notebooks

### Phase 4: Integration (1 week)
- [ ] Update main course navigation
- [ ] Cross-link between notebooks and HTML pages
- [ ] Final testing and polish
- [ ] Write instructor notes

---

## Success Metrics

1. **Engagement:** Learners complete Week 1 at higher rate than current prereqs
2. **Retention:** Week 4 quiz scores on earlier concepts stay high
3. **Time-to-working-kernel:** Learners write first GPU kernel by Day 4 (not Day 14+)
4. **Practical outcomes:** Learners can profile and optimize a real kernel by Week 2 end

---

## Appendix: Python Environment Setup

### Required Packages
```bash
pip install numpy cupy-cuda12x triton torch
pip install jupyterlab ipywidgets matplotlib
```

### GPU Requirements
- NVIDIA GPU with CUDA 12+ support
- For Tensor Cores: Volta or newer (V100, A100, H100)
- For TMA: Hopper or newer (H100, H200)
- For NVFP4: Blackwell (B100, B200)

### Fallback Path
For learners without GPU access:
- Google Colab with T4/A100 runtime
- Simulation mode for some visualizations
- Focus on concepts, run expensive experiments in cloud

---

## Document History

- **Created:** 2025-01-26
- **Authors:** Expert Council (Curriculum Design, GPU Engineering, Educational Technology)
- **Status:** Planning complete, ready for implementation
