# Unified Course Architecture Plan

## Vision

Transform the GPU learning course from disconnected islands (syllabus + interactives + notebooks) into **ONE expertly woven learning experience** where every element builds on the previous and enhances the learner's understanding.

---

## Architecture Decision

**Primary vehicle:** Rich HTML course page (`kernel-course.html`)
**Interactives:** Embedded directly via JS components (not iframes)
**Notebooks:** Positioned as hands-on labs after concepts are taught
**Existing pages:** Become reference materials for deep dives

---

## Implementation Approach

### Phase 1: Component Extraction

Extract reusable interactive components from lesson pages into `scripts/components.js`:

| Component | Source | Target Week |
|-----------|--------|-------------|
| `HierarchyExplorer` | gpu-architecture.html | Week 1 |
| `WarpVisualizer` | gpu-architecture.html | Week 1 |
| `OccupancyCalculator` | gpu-architecture.html | Week 2 |
| `MemoryTierExplorer` | memory-hierarchy.html | Week 1 |
| `CoalescingVisualizer` | memory-hierarchy.html | Week 1, 2 |
| `BankConflictSimulator` | memory-hierarchy.html | Week 2 |
| `BandwidthCalculator` | memory-hierarchy.html | Week 2 |
| `LiveComputation` | attention-math.html | Week 3 (exists) |
| `SoftmaxViz` | attention-math.html | Week 3 (exists) |
| `OnlineSoftmaxSim` | attention-math.html | Week 3 (exists) |
| `FPBitToggle` | math-foundations.html | Week 3, 4 (exists) |
| `MatrixMultViz` | math-foundations.html | Week 1 |
| `IndexCalculator` | math-foundations.html | Week 1 |
| `TileCoordViz` | math-foundations.html | Week 1 |
| `QuantizationCalc` | math-foundations.html | Week 4 |

### Phase 2: Course Page Transformation

Transform `kernel-course.html` from syllabus to learning experience.

#### Part 1: "From Python to GPU Mastery"

**Narrative Arc:** "You start with slow Python. By the end, you've achieved 500,000x speedup and understand exactly why."

```
1. THE CHALLENGE
   - "Your code is slow. How slow?"
   - [Code block] NumPy matmul timing
   - Reveal: 0.001 GFLOPS on a $2000 machine

2. THE INSTANT FIX
   - "One import changes everything"
   - [Code block] CuPy drop-in replacement
   - Profile: 50-100x speedup, zero code changes
   - "But why? Let's look inside."

3. GPU ARCHITECTURE (concepts woven in)
   - [INTERACTIVE] GPU Hierarchy Explorer
     - Click through: GPU → SM → Warp → Thread
     - Key insight: Thousands of simple cores vs few complex ones
   - [INTERACTIVE] Warp Execution Visualizer
     - See 32 threads moving in lockstep
     - Demonstrate divergence cost
   - Quiz: "How many threads in a warp?"

4. YOUR FIRST KERNEL
   - "Now write your own GPU code"
   - [Code block] Triton vector addition, annotated
   - Key concepts inline: program_id, blocks, offsets
   - [INTERACTIVE] Index Calculator
     - Practice: 2D → 1D index conversion

5. WHY NAIVE KERNELS ARE SLOW
   - "Your kernel works. It's also 50x slower than CuPy. Why?"
   - [INTERACTIVE] Memory Hierarchy Explorer
     - Click through: Registers → SMEM → L2 → HBM
     - See latency numbers: 1 vs 20 vs 400 cycles
   - [INTERACTIVE] Coalescing Visualizer
     - Good pattern: adjacent threads → adjacent memory
     - Bad pattern: strided access → wasted bandwidth
   - The insight: "Memory is the bottleneck, not compute"

6. THE SOLUTION: TILING
   - "Load once, use many times"
   - [INTERACTIVE] Tile Coordinate Visualizer
     - See how tiles map to blocks
   - [Code block] Tiled matmul structure
   - [INTERACTIVE] Data Reuse Animation
     - Watch tiles being loaded and reused

7. YOUR FAST KERNEL
   - [Code block] Complete tiled matmul
   - Profile: 500+ GFLOPS achieved
   - "You just went from 0.001 to 500+ GFLOPS"

→ [LAB SECTION] "Now build it yourself"
   Links to notebooks/part1/ with clear objectives:
   - Environment check
   - Establish baselines (NumPy, CuPy)
   - First kernels (vector add, naive matmul)
   - Tiled matmul mastery
```

#### Part 2: "Optimization Mastery"

**Narrative Arc:** "500 GFLOPS is good. 80% of peak is mastery. Here's how."

```
1. PROFILE LIKE A PRO
   - "You can't optimize what you can't measure"
   - Key metrics: throughput, occupancy, stalls
   - [Code block] nsys/ncu commands
   - [INTERACTIVE] Bandwidth Calculator
     - Input: data size, kernel time
     - Output: achieved bandwidth, % of peak

2. COALESCING EXPERIMENTS
   - "Same algorithm, 10x different performance"
   - [INTERACTIVE] Coalescing Visualizer (expanded)
     - Pattern 1: Coalesced (1 transaction)
     - Pattern 2: Strided (32 transactions)
     - Pattern 3: Random (worst case)
   - [Code block] Measure the difference

3. BANK CONFLICTS
   - "32 banks, 32 ways to slow down"
   - [INTERACTIVE] Bank Conflict Simulator
     - See conflicts happen in real-time
     - Try different stride patterns
   - The fix: padding

4. HIDING LATENCY
   - "Don't wait for memory—overlap!"
   - Software pipelining concept
   - Double buffering visualization

5. TENSOR CORES
   - "The secret weapon: matrix math units"
   - MMA operations explained
   - Type constraints (FP16, BF16, FP8)

6. PRODUCTION GEMM
   - Putting it all together
   - [Code block] Optimized kernel structure
   - "You've reached 80% of theoretical peak"

→ [LAB SECTION] notebooks/part2/
```

#### Part 3: "Building FlashAttention"

**Narrative Arc:** "Attention is THE bottleneck. You'll build the algorithm that changed LLM inference."

```
1. THE ATTENTION BOTTLENECK
   - "Every token attends to every other token: O(N²)"
   - Memory explodes with sequence length
   - "We need a better algorithm"

2. ATTENTION = SIMILARITY
   - "Which tokens matter? The ones that 'agree' with your query."
   - [INTERACTIVE] Q·K Calculator
     - Input query vector
     - See dot products with keys
     - Highest scores = most relevant
   - The formula: QK^T

3. SOFTMAX: PROBABILITIES FROM SCORES
   - "Turn scores into attention weights"
   - [INTERACTIVE] Softmax Visualizer
     - Adjust scores, watch probabilities
     - See how one high score dominates

4. THE OVERFLOW PROBLEM
   - "Run this code. Watch it explode."
   - [Code block] Naive softmax, NaN result
   - [INTERACTIVE] FP Bit Toggle
     - See FP16 range limits
     - exp(100) overflows
   - "We need numerical stability"

5. STABLE SOFTMAX
   - The max-subtraction trick
   - [Code block] Stable implementation
   - Mathematical proof: same result, bounded intermediate values

6. ONLINE SOFTMAX: THE KEY INSIGHT
   - "What if you can't see all values at once?"
   - [INTERACTIVE] Online Softmax Simulation
     - Watch streaming algorithm in action
     - See rescaling when max changes
   - This enables: tiled, memory-efficient attention

7. FLASHATTENTION
   - "O(N) memory, competitive speed"
   - Block-by-block algorithm walkthrough
   - [Code block] Core FlashAttention loop
   - Performance comparison vs naive

→ [LAB SECTION] notebooks/part3/
```

#### Part 4: "Production Quantization"

**Narrative Arc:** "16 bits per weight × 70B weights = 140GB. Your GPU has 80GB. Let's fix that."

```
1. THE MEMORY WALL
   - LLM parameter counts vs GPU memory
   - "We need compression that preserves quality"

2. FLOATING POINT REFRESHER
   - [INTERACTIVE] FP Bit Toggle
     - FP32 → FP16 → FP8 comparison
     - See precision loss directly
   - E4M3 vs E5M2: range vs precision tradeoff

3. QUANTIZATION FUNDAMENTALS
   - Scale factor computation
   - [INTERACTIVE] Quantization Calculator
     - Input: tensor, target format
     - Output: quantized values, error
   - Rounding modes: nearest, stochastic

4. INT8 AND INT4
   - Block-wise quantization
   - Per-channel vs per-tensor scaling
   - When INT4 works, when it doesn't

5. NVFP4: BLACKWELL'S SECRET
   - E2M1: only 6 distinct positive values
   - Two-level scaling: per-tensor FP32 + per-block FP8
   - Why this works for values (V), not keys (K)

6. MIXED-PRECISION KV CACHE
   - K at FP8 (softmax amplifies errors)
   - V at NVFP4 (averaging smooths errors)
   - ~2.6× memory reduction, quality preserved

7. FUSED QUANTIZED ATTENTION
   - Dequantize-on-load
   - Fused kernel structure
   - Benchmarking methodology

→ [LAB SECTION] notebooks/part4/
```

### Phase 3: Create Part 4 Notebooks

After course page is transformed, create notebooks in `notebooks/part4/`:
1. `01_fp8_conversion.ipynb`
2. `02_quantization_fundamentals.ipynb`
3. `03_int8_int4.ipynb`
4. `04_nvfp4.ipynb`
5. `05_kv_cache_strategy.ipynb`
6. `06_fused_quantized_attention.ipynb`
7. `07_production_integration.ipynb`

### Phase 4: Update Navigation & Cross-Links

1. Sidebar: Direct links to week anchors + lab sections
2. Reference section: Links to standalone pages for deep dives
3. Notebooks: Clear "Back to course" links

---

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `kernel-course.html` | **Major rewrite** | Transform from syllabus to learning experience |
| `scripts/components.js` | **Expand** | Add extracted interactive components |
| `base.css` | **Minor additions** | Styles for new embedded components |
| `notebooks/week4/*` | **Create** | 7 new notebooks |
| `lessons/*.html` | **Keep** | Become reference materials |
| `COURSE_RESTRUCTURE_PLAN.md` | **Update** | Mark as superseded by this plan |

---

## Success Criteria

1. **Single entry point:** Learner starts at kernel-course.html, never feels lost
2. **Narrative flow:** Each concept leads naturally to the next
3. **Interactives at the right moment:** Appear when they reinforce the current concept
4. **Clear lab transitions:** "Now build it yourself" is explicit and positioned after concepts
5. **No dead ends:** Every section links forward
6. **Progressive complexity:** Week 1 is gentler, Week 4 is advanced

---

## Implementation Order

1. [ ] Extract remaining components to components.js
2. [ ] Rewrite Week 1 section with embedded interactives
3. [ ] Rewrite Week 2 section with embedded interactives
4. [ ] Rewrite Week 3 section with embedded interactives
5. [ ] Create Week 4 notebooks
6. [ ] Rewrite Week 4 section with embedded interactives
7. [ ] Update navigation and cross-links
8. [ ] Test complete flow
9. [ ] Commit

---

## Document History

- **Created:** 2025-01-26
- **Status:** Planning complete, ready for implementation
