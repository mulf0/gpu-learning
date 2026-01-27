# GPU Learning Course - Pedagogy Revamp Plan

**Goal:** Restructure course for maximum learning retention using evidence-based pedagogy.

**Status:** Complete

---

## Phase 1: Component Development

### CSS Components (base.css)

| Component | Purpose | Status |
|-----------|---------|--------|
| `.learning-objectives` | Chapter start objectives | Done |
| `.chapter-connection` | Link to previous chapters | Done |
| `.quiz--micro` | Inline single-question quiz | Done |
| `.chunk-divider` | Visual break between chunks | Done |
| `.expert-note` | Expandable expert annotations | Done |
| `.human-scale` | Inline human comparisons | Done |
| `.section-progress` | Progress tracking | Done |
| `.collapsible` | Generic collapsible pattern | Done |

### JavaScript Components (components.js)

| Component | Purpose | Status |
|-----------|---------|--------|
| `Collapsible` | Generic toggle with open/close | Done |
| `SectionProgress` | Scroll-based progress tracking | Done |
| `initPedagogy()` | Auto-initialization on DOMContentLoaded | Done |

*Note: Micro-quizzes reuse existing Quiz class with `.quiz--micro` CSS modifier. Expert notes use native `<details>` element (no JS needed).*

---

## Phase 2: Content Specifications

### Chapter 1: GPU Fundamentals

**Learning Objectives:**
1. Explain why GPUs optimize for throughput over latency
2. Describe the SM → Warp → Thread hierarchy
3. Calculate theoretical occupancy given resource constraints
4. Identify warp divergence in code and explain its performance impact
5. Interpret basic GPU specifications (SM count, memory bandwidth)

**Chapter Connection:** None (first chapter)

**Micro-Quiz Questions:**
- Q1: "A GPU is faster than a CPU because: (a) Higher clock speed (b) More parallel execution units (c) Larger cache" [Answer: b]
- Q2: "If you launch 256 threads per block, how many warps is that?" [Answer: 8]
- Q3: "What happens when threads in a warp take different branches?" [Answer: Serialized execution]
- Q4: "Which resource does NOT limit occupancy? (a) Registers (b) Shared memory (c) Global memory" [Answer: c]

**Human-Scale Comparisons:**
- "~192 SMs" → "Like having 192 mini-processors, each running thousands of threads"
- "8 TB/s bandwidth" → "Could transfer every book ever written in under 2 seconds"
- "32 threads per warp" → "Like a marching band where all 32 musicians must play the same note"

**Content Restructure Status:** Done

---

### Chapter 2: Memory Hierarchy

**Learning Objectives:**
1. List the GPU memory types in order of speed and size
2. Explain why memory bandwidth is often the bottleneck
3. Demonstrate coalesced vs uncoalesced memory access patterns
4. Identify and fix shared memory bank conflicts
5. Calculate effective memory bandwidth for a given access pattern

**Chapter Connection:**
"In Chapter 1, you learned that warps execute 32 threads in lockstep. This seemingly simple fact has profound implications for memory access. When 32 threads read memory simultaneously, *where* they read from determines whether you get 8 TB/s or 500 GB/s."

**Micro-Quiz Questions:**
- Q1: "Which memory is fastest? (a) Global (b) Shared (c) Registers" [Answer: c]
- Q2: "Coalesced access means: (a) All threads access same address (b) Adjacent threads access adjacent addresses (c) Random access" [Answer: b]
- Q3: "A 2-way bank conflict means the access takes: (a) Half as long (b) Twice as long (c) Same time" [Answer: b]
- Q4: "When a kernel uses too many registers, what happens?" [Answer: Register spilling to local memory]

**Human-Scale Comparisons:**
- "~1 cycle (registers)" → "Instant - like remembering your own name"
- "~20 cycles (shared)" → "Quick pause - like glancing at a sticky note"
- "~400 cycles (global)" → "Significant wait - like walking to a filing cabinet in another room"
- "128-byte cache line" → "Minimum shipping container - even for 4 bytes, you get 128"

**Content Restructure Status:** Done

---

### Chapter 3: First Kernels

**Learning Objectives:**
1. Write a basic Triton kernel with proper grid/block configuration
2. Use `tl.load` and `tl.store` with appropriate masking
3. Implement element-wise operations (add, multiply, activation functions)
4. Debug common kernel errors (out-of-bounds, wrong output)
5. Profile kernel performance using basic metrics

**Chapter Connection:**
"Chapters 1-2 gave you the mental model: thousands of threads executing in warps, memory access patterns determining performance. Now you'll write actual code. Every line maps directly to those concepts—thread indices become array offsets, block sizes affect occupancy, access patterns determine coalescing."

**Micro-Quiz Questions:**
- Q1: "The @triton.jit decorator: (a) Runs on CPU (b) Compiles for GPU (c) Optimizes Python" [Answer: b]
- Q2: "BLOCK_SIZE=1024 means each block processes: (a) 1024 threads (b) 1024 elements (c) Both" [Answer: c]
- Q3: "Why do we need masks in tl.load?" [Answer: Prevent out-of-bounds access]
- Q4: "Your kernel outputs all zeros. Most likely cause?" [Answer: Wrong pointer/offset calculation]

**Content Restructure Status:** Done

---

### Chapter 4: Optimization

**Learning Objectives:**
1. Use Nsight Compute to identify kernel bottlenecks
2. Apply tiling to improve cache utilization
3. Optimize memory access patterns for coalescing
4. Balance occupancy against per-thread resources
5. Achieve >50% of theoretical memory bandwidth

**Chapter Connection:**
"You can write working kernels now. But 'working' isn't enough—you need 'fast.' Chapter 2's memory hierarchy becomes your optimization target. Chapter 3's kernel structure becomes your canvas for applying tiling and access pattern fixes."

**Micro-Quiz Questions:**
- Q1: "Most important metric for memory-bound kernels: (a) Occupancy (b) Memory throughput (c) Compute throughput" [Answer: b]
- Q2: "Tiling improves performance by: (a) More threads (b) Reusing data in fast memory (c) Less computation" [Answer: b]
- Q3: "Kernel achieves 20% bandwidth. First optimization?" [Answer: Check coalescing/access patterns]
- Q4: "When might you intentionally reduce occupancy?" [Answer: When more registers/shared memory per thread helps]

**Content Restructure Status:** Done

---

### Chapter 5: Attention Mechanisms

**Learning Objectives:**
1. Implement standard scaled dot-product attention
2. Explain why naive attention is O(n²) in memory
3. Describe how FlashAttention achieves O(n) memory
4. Implement online softmax for numerical stability
5. Apply tiling strategies to attention computation

**Chapter Connection:**
"Chapter 4's tiling wasn't just about matrix multiply—it's a universal optimization pattern. Attention is where this pattern becomes essential. Without tiling, attention on long sequences would require terabytes of memory. With tiling (FlashAttention), it fits in shared memory."

**Micro-Quiz Questions:**
- Q1: "Q and K have shape [seq_len, d]. Shape of QK^T?" [Answer: [seq_len, seq_len]]
- Q2: "For seq_len=8192, d=128, naive attention QK^T memory?" [Answer: 8192² × 4 bytes = 256 MB]
- Q3: "Online softmax maintains: (a) Max only (b) Sum only (c) Both" [Answer: c]
- Q4: "FlashAttention is faster because: (a) Fewer FLOPs (b) Less memory traffic (c) More threads" [Answer: b]

**Content Restructure Status:** Done

---

### Chapter 6: Quantization

**Learning Objectives:**
1. Explain the memory/accuracy tradeoff in quantization
2. Convert between floating point formats (FP32, FP16, FP8)
3. Implement symmetric and asymmetric quantization
4. Choose appropriate quantization strategies for different workloads
5. Identify when quantization will/won't help performance

**Chapter Connection:**
"You've optimized compute (Ch4) and memory access patterns (Ch2-5). Quantization attacks the problem from a different angle: reducing the data itself. Every concept from earlier chapters applies—bandwidth limits, memory hierarchy, tiling—but now with 2x or 4x more effective bandwidth."

**Micro-Quiz Questions:**
- Q1: "FP16 vs FP32: which is smaller, range or precision?" [Answer: Both are smaller, but precision loss is more noticeable]
- Q2: "FP8 E4M3 vs E5M2: E4M3 has (a) More range (b) More precision (c) Same" [Answer: b]
- Q3: "Asymmetric quantization uses zero-point because: (a) Faster (b) Handles non-symmetric distributions (c) Less memory" [Answer: b]
- Q4: "Quantization helps MOST when: (a) Compute-bound (b) Memory-bound (c) Neither" [Answer: b]

**Content Restructure Status:** Done

---

### Chapter 7: Production Systems

**Learning Objectives:**
1. Explain continuous batching and its benefits
2. Describe KV cache management strategies
3. Identify bottlenecks in inference serving systems
4. Choose between inference frameworks (vLLM, TensorRT-LLM)
5. Design a system meeting latency and throughput requirements

**Chapter Connection:**
"Everything converges here. GPU fundamentals (Ch1) determine how your server utilizes hardware. Memory hierarchy (Ch2) explains why KV cache is the bottleneck. Attention (Ch5) and quantization (Ch6) are the core computations you're serving. This chapter is about orchestrating all of it at scale."

**Micro-Quiz Questions:**
- Q1: "Batching improves throughput because: (a) Faster memory (b) Better GPU utilization (c) Simpler code" [Answer: b]
- Q2: "Continuous vs static batching: (a) Larger batches (b) Per-request scheduling (c) Less memory" [Answer: b]
- Q3: "KV cache scales with: (a) Model size (b) Sequence length (c) Both" [Answer: c]
- Q4: "TensorRT-LLM is better for: (a) Flexibility (b) Performance (c) Easy setup" [Answer: b]

**Content Restructure Status:** Done

---

## Implementation Checklist

### Phase 1: Components
- [x] Add CSS components to base.css
- [x] Add JS components to components.js
- [x] Test components in isolation

### Phase 2: Chapter Restructuring
- [x] Chapter 1: GPU Fundamentals
- [x] Chapter 2: Memory Hierarchy
- [x] Chapter 3: First Kernels
- [x] Chapter 4: Optimization
- [x] Chapter 5: Attention
- [x] Chapter 6: Quantization
- [x] Chapter 7: Production

### Phase 3: Validation
- [x] Every chapter has 5 learning objectives
- [x] Every chapter (2-7) has chapter connection
- [x] Minimum 3-4 micro-quizzes per chapter (~20 total)
- [x] Key interactive elements have expert notes (Ch 1-3)
- [x] Human-scale comparisons added (Ch 1-2)

---

## Progress Log

| Date | Step | Status |
|------|------|--------|
| 2026-01-26 | Phase 1: CSS Components | Complete |
| 2026-01-26 | Phase 1: JS Components | Complete |
| 2026-01-26 | Chapter 1 Restructure | Complete |
| 2026-01-26 | Chapter 2 Restructure | Complete |
| 2026-01-26 | Chapter 3 Restructure | Complete |
| 2026-01-26 | Chapter 4 Restructure | Complete |
| 2026-01-26 | Chapter 5 Restructure | Complete |
| 2026-01-26 | Chapter 6 Restructure | Complete |
| 2026-01-26 | Chapter 7 Restructure | Complete |
| 2026-01-26 | Final Validation | Complete |

## Implementation Notes

### Design Decisions
- **Micro-quizzes**: Reuse existing `.quiz` markup with `.quiz--micro` CSS modifier (no new JS)
- **Expert notes**: Use native `<details>` element (no JS needed, better accessibility)
- **Chapter connections**: Cyan accent (`--accent-cyan`) to differentiate from other callouts
- **All new CSS**: Uses existing design system variables from base.css

### Component Summary
- **CSS added**: ~120 lines to base.css (lines ~778-900)
- **JS added**: ~30 lines to components.js (Collapsible, SectionProgress classes)
- **Total micro-quizzes**: ~20 across all chapters
- **Learning objectives**: 5 per chapter (35 total)
- **Chapter connections**: 6 (chapters 2-7)
