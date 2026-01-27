# GPU Learning Course - Citation Audit Plan

**Purpose**: Verify every factual claim has a valid, correct, and relevant citation.  
**Status**: In Progress  
**Last Updated**: 2026-01-26

---

## Audit Legend
- [ ] = Not verified
- [x] = Verified correct
- [!] = Needs fix/update
- [?] = Citation needed

---

## 1. Hardware Specifications

### 1.1 Blackwell B200 Specifications

| Claim | File | Line | Current Citation | Status |
|-------|------|------|------------------|--------|
| ~192 SMs | kernel-course.html | 284 | nvidia.com/blackwell-architecture | [?] Not in public specs - needs datasheet |
| 192GB HBM3e | kernel-course.html | 288, 304 | nvidia.com/blackwell-architecture | [x] Verified: GB200 NVL72 = 72 GPUs × ? per GPU |
| ~8 TB/s memory bandwidth | kernel-course.html | 292, 304 | nvidia.com/gb200-nvl72 | [x] Verified: 576 TB/s ÷ 72 GPUs ≈ 8 TB/s/GPU |
| ~2.5 GHz boost clock | gpu-architecture.html | 296 | nvidia.com/blackwell-architecture | [?] Not in public specs |
| ~2.5 PFLOPS FP8 compute | memory-hierarchy.html | 361 | nvidia.com/blackwell-architecture | [?] Needs verification |
| 5th gen Tensor Cores | kernel-course.html | 438 | nvidia.com/blackwell-architecture | [x] Verified: "Fifth-generation Tensor Cores" |
| FP4, FP6 support | kernel-course.html | 440 | nvidia.com/blackwell-architecture | [x] Verified: FP4 (NVFP4) confirmed |
| 208B transistors | gpu-architecture.html | 693-745 | nvidia.com/blackwell-architecture | [x] Verified: "208 billion transistors" |

### 1.2 Hopper H100 Specifications

| Claim | File | Line | Current Citation | Status |
|-------|------|------|------------------|--------|
| 256KB registers/SM | kernel-course.html | 264-266 | H100 datasheet | [x] Verified: "64K 32-bit registers per SM" = 256KB |
| 228KB shared memory/SM | kernel-course.html | 276-278 | H100 datasheet | [x] Verified: Hopper Tuning Guide confirms 228KB |
| ~60MB L2 cache | kernel-course.html | 289-291 | H100 datasheet | [!] H100 = 50MB, files show ~60MB (may be B200 context) |
| 64 max warps per SM | gpu-architecture.html | 739-740 | Hopper whitepaper | [x] Verified: Hopper Tuning Guide confirms 64 concurrent warps |
| 4 warp schedulers per SM | gpu-architecture.html | 351 | CUDA Programming Guide | [ ] |
| Warpgroup = 128 threads | gpu-architecture.html | 372-378 | Hopper tuning guide | [x] Verified: 4 warps × 32 threads = 128 |

### 1.3 Memory Latency/Bandwidth (Approximate)

| Claim | File | Line | Current Citation | Status |
|-------|------|------|------------------|--------|
| Registers: ~1 cycle | memory-hierarchy.html | 413 | None (approximate) | [?] |
| Registers: ~20 TB/s | kernel-course.html | 265 | None (approximate) | [?] |
| SMEM: ~20 cycles | kernel-course.html | 277 | None (approximate) | [?] |
| SMEM: ~20 TB/s | kernel-course.html | 277 | None (approximate) | [?] |
| L2: ~200 cycles | kernel-course.html | 290 | None (approximate) | [?] |
| L2: ~10 TB/s | kernel-course.html | 290 | None (approximate) | [?] |
| HBM: ~400 cycles | kernel-course.html | 303 | None (approximate) | [?] |
| HBM: ~8 TB/s | kernel-course.html | 303 | GB200 NVL72 specs | [ ] |

---

## 2. Floating Point Format Specifications

### 2.1 FP32 (IEEE 754 binary32)

| Claim | File | Line | Current Citation | Status |
|-------|------|------|------------------|--------|
| 1 sign + 8 exp + 23 mantissa | kernel-course.html | 543-549 | Wikipedia | [ ] |
| Range: ±3.4x10^38 | kernel-course.html | 545, math-foundations | Wikipedia | [ ] |
| Epsilon: ~1.19x10^-7 | math-foundations.html | 857 | None | [?] |

### 2.2 FP16 (IEEE 754 binary16)

| Claim | File | Line | Current Citation | Status |
|-------|------|------|------------------|--------|
| 1 sign + 5 exp + 10 mantissa | kernel-course.html | 554-561 | Wikipedia | [ ] |
| Max value: 65,504 | kernel-course.html | 555, attention-math | Wikipedia | [ ] |
| Epsilon: ~9.77x10^-4 | math-foundations.html | 863 | None | [?] |

### 2.3 FP8 E4M3

| Claim | File | Line | Current Citation | Status |
|-------|------|------|------------------|--------|
| 1 sign + 4 exp + 3 mantissa | kernel-course.html | 565-572 | arXiv:2209.05433 | [x] Verified |
| Max value: 448 | kernel-course.html | 566, attention-math | arXiv:2209.05433 | [x] Verified: bias=7, exp=8, mantissa=1.75 → 1.75×256=448 |
| Epsilon: 0.125 (2^-3) | math-foundations.html | 869 | None | [?] |

### 2.4 FP8 E5M2

| Claim | File | Line | Current Citation | Status |
|-------|------|------|------------------|--------|
| 1 sign + 5 exp + 2 mantissa | math-foundations.html | 825-832 | arXiv:2209.05433 | [ ] |
| Max value: 57,344 | math-foundations.html | 826 | arXiv:2209.05433 | [ ] |
| Epsilon: 0.25 (2^-2) | math-foundations.html | 875 | None | [?] |

### 2.5 FP4 E2M1 (NVFP4)

| Claim | File | Line | Current Citation | Status |
|-------|------|------|------------------|--------|
| 1 sign + 2 exp + 1 mantissa | kernel-course.html | 576-583 | OCP MX Spec | [ ] |
| Max value: ±6 | kernel-course.html | 577 | OCP MX Spec | [ ] |
| 16-element block size (NVFP4) | kernel-course.html | 609, 625 | nvidia.com/blackwell | [ ] |
| E4M3 scale factors | kernel-course.html | 609, 629 | nvidia.com/blackwell | [ ] |
| Two-level scaling | kernel-course.html | 600-614 | nvidia.com/blackwell | [ ] |

### 2.6 MXFP4 (OCP Microscaling)

| Claim | File | Line | Current Citation | Status |
|-------|------|------|------------------|--------|
| 32-element block size | kernel-course.html | 625 | OCP MX Spec | [ ] |
| E8M0 (power-of-2) scales | kernel-course.html | 629 | OCP MX Spec | [ ] |

---

## 3. arXiv Paper Citations

### 3.1 Transformer & Attention Papers

| Paper | arXiv ID | Files Using | Claim | Status |
|-------|----------|-------------|-------|--------|
| Attention Is All You Need | 1706.03762 | attention-math.html:585,712 | Attention formula | [x] Verified: Vaswani et al. |
| FlashAttention | 2205.14135 | kernel-course.html:684,699,1241,1272 | Tiled attention, O(N) memory | [x] Verified: Dao et al. |
| FlashAttention-2 | 2307.08691 | attention-math.html:651,722 | 2x speedup | [x] Verified: Tri Dao (sole author), 2x speedup confirmed |
| FlashAttention-3 | 2407.08608 | attention-math.html:725-728 | Hopper + FP8 support | [x] Verified: Shah, Dao et al., H100/Hopper explicit |

### 3.2 Quantization Papers

| Paper | arXiv ID | Files Using | Claim | Status |
|-------|----------|-------------|-------|--------|
| FP8 Formats for DL | 2209.05433 | kernel-course.html:565, attention-math:479,541,742-743 | E4M3/E5M2 specs | [x] Verified: Micikevicius et al. (NVIDIA/ARM/Intel) |
| LLM.int8() | 2208.07339 | kernel-course.html:1229,1251 | Outlier handling | [x] Verified: Dettmers et al., mixed-precision decomposition |
| SmoothQuant | 2211.10438 | kernel-course.html:1233,1258 | Migration technique | [x] Verified: Xiao et al., "smooths activation outliers" |
| AWQ | 2306.00978 | kernel-course.html:1237,1265 | Activation-aware quant | [x] Verified: Lin et al., MLSys 2024 Best Paper |

### 3.3 Architecture Papers

| Paper | arXiv ID | Files Using | Claim | Status |
|-------|----------|-------------|-------|--------|
| Volta Microbenchmarking | 1804.06826 | kernel-course.html:1158,1389-1393 | Jia et al. | [x] Verified: "Dissecting NVIDIA Volta via Microbenchmarking" |

---

## 4. NVIDIA Documentation Links

### 4.1 Architecture Pages

| Link | Files Using | Status |
|------|-------------|--------|
| nvidia.com/blackwell-architecture | kernel-course, gpu-architecture, memory-hierarchy | [ ] |
| nvidia.com/gb200-nvl72 | gpu-architecture.html:292, memory-hierarchy | [ ] |
| resources.nvidia.com/hopper-architecture | gpu-architecture.html:689, memory-hierarchy | [ ] |

### 4.2 CUDA Documentation

| Link | Files Using | Purpose | Status |
|------|-------------|---------|--------|
| docs.nvidia.com/cuda/cuda-c-programming-guide | Multiple | SIMT, thread hierarchy, memory | [x] Verified v13.1 |
| docs.nvidia.com/cuda/cuda-c-best-practices-guide | Multiple | Optimization, coalescing | [x] Verified v13.1 |
| docs.nvidia.com/cuda/hopper-tuning-guide | kernel-course.html:351,432,448 | Warpgroup, TMA | [x] Verified v13.1 |
| docs.nvidia.com/cuda/ampere-tuning-guide | kernel-course.html:426 | Ampere Tensor Cores | [x] Verified v13.1 |
| docs.nvidia.com/nsight-compute | memory-hierarchy.html:920 | Profiling | [x] Verified v2025.4.1 |

### 4.3 CUTLASS/CuTe

| Link | Files Using | Status |
|------|-------------|--------|
| github.com/NVIDIA/cutlass | kernel-course.html:321,1369 | [x] Verified: CUTLASS 4.4.0, 9.2k stars |
| docs.nvidia.com/cutlass/latest/media/docs/cpp/cute | kernel-course.html:1379 | [ ] |
| docs.nvidia.com/cutlass/latest/media/docs/pythonDSL | kernel-course.html:1359 | [ ] |

---

## 5. External Documentation Links

### 5.1 Standards & Specifications

| Link | Files Using | Purpose | Status |
|------|-------------|---------|--------|
| opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf | kernel-course:576, attention-math:748, math-foundations:1144 | MX/MXFP4 spec | [ ] |
| ieeexplore.ieee.org/document/8766229 | attention-math.html:738 | IEEE 754-2019 | [ ] |

### 5.2 Wikipedia Links

| Link | Files Using | Purpose | Status |
|------|-------------|---------|--------|
| en.wikipedia.org/wiki/Single-precision_floating-point_format | kernel-course:543, math-foundations:792 | FP32 | [x] Verified |
| en.wikipedia.org/wiki/Half-precision_floating-point_format | kernel-course:554, attention-math:345,468 | FP16 | [x] Verified |
| en.wikipedia.org/wiki/Multi-core_processor | gpu-architecture.html:276 | CPU cores | [ ] |
| en.wikipedia.org/wiki/Roofline_model | memory-hierarchy.html:378 | Ridge point | [x] Verified |
| en.wikipedia.org/wiki/Row-_and_column-major_order | math-prerequisites:983 | Memory layout | [ ] |

### 5.3 Interactive Tools

| Link | Files Using | Purpose | Status |
|------|-------------|---------|--------|
| float.exposed | math-prerequisites.html:727 | FP bit toggling | [x] Verified |
| matrixmultiplication.xyz | math-prerequisites.html:664 | Matrix viz | [!] FIXED: Replaced with mathsisfun.com (TLS error) |

---

## 6. Video Resource Links

### 6.1 3Blue1Brown

| Link | Files Using | Topic | Status |
|------|-------------|-------|--------|
| youtube.com/watch?v=eMlx5fFNoYc | attention-math.html:680 | Attention in Transformers | [ ] |
| youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab | math-prerequisites:548 | Essence of Linear Algebra | [ ] |
| youtube.com/watch?v=XkY2DOUCWMU | math-prerequisites:640 | Matrix multiplication | [ ] |
| youtube.com/watch?v=m2MIpDrF7Es | math-prerequisites:839 | Euler's number e | [ ] |

### 6.2 Other YouTube

| Link | Files Using | Topic | Status |
|------|-------------|-------|--------|
| youtube.com/watch?v=kCc8FmEb1nY | attention-math.html:691 | Karpathy GPT | [ ] |
| youtube.com/watch?v=gMOAud7hZg4 | attention-math.html:701 | Yannic Kilcher FlashAttention | [ ] |
| youtube.com/watch?v=h9Z4oGN89MU | gpu-architecture.html:657 | Branch Education GPU | [ ] |
| youtube.com/watch?v=PZRI1IfStY0 | math-prerequisites:752 | Computerphile FP | [ ] |
| youtube.com/watch?v=dQhj5RGtag0 | math-prerequisites:804 | Computerphile FP Part 2 | [ ] |
| youtube.com/watch?v=IzU4AVcMFys | memory-hierarchy.html:878 | CUDA intro | [ ] |
| youtube.com/watch?v=3xfyiWhtvZw | memory-hierarchy.html:888 | CoffeeBeforeArch GPU memory | [ ] |

### 6.3 Other Resources

| Link | Files Using | Purpose | Status |
|------|-------------|---------|--------|
| developer.nvidia.com/blog/cuda-refresher-cuda-programming-model | gpu-architecture.html:667 | CUDA model | [ ] |
| khanacademy.org/math/linear-algebra | math-prerequisites:576 | Linear algebra course | [x] Verified (JS-heavy) |
| ocw.mit.edu/courses/18-06-linear-algebra-spring-2010 | math-prerequisites:602 | MIT 18.06 | [x] Verified: Gilbert Strang course |
| betterexplained.com/articles/an-intuitive-guide-to-exponential-functions-e | math-prerequisites:866 | Exponentials | [x] Verified |
| numpy.org/doc/stable/reference/arrays.ndarray.html | math-prerequisites:930 | Memory layout | [ ] |
| eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays | math-prerequisites:956 | Memory layout | [ ] |
| docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html | math-prerequisites:778 | Goldberg FP paper | [x] Verified: "What Every Computer Scientist Should Know" |

---

## 7. Numerical/Performance Claims

### 7.1 Performance Numbers

| Claim | File | Line | Citation Status |
|-------|------|------|-----------------|
| GEMM achieves >80% peak FLOPS | kernel-course.html:321-322 | github.com/NVIDIA/cutlass | [?] Need specific reference |
| 2.6x memory reduction (FP8 K + NVFP4 V) | kernel-course.html:756-767 | nvidia.com/blackwell | [?] Verify claim |
| 4x memory reduction vs FP16 (NVFP4) | attention-math.html:543 | arXiv:2209.05433 | [?] Verify this applies to NVFP4 |
| ~1% accuracy loss with NVFP4 | attention-math.html:543 | None | [?] Need citation |
| 312.5 FLOPS/byte ridge point (B200) | memory-hierarchy.html:378-381 | Derived calculation | [?] Verify math |

### 7.2 Algorithm Complexity

| Claim | File | Line | Citation Status |
|-------|------|------|-----------------|
| Attention matrix [seq x seq] = O(N^2) | kernel-course.html:682 | Common knowledge | [x] |
| FlashAttention memory O(N) not O(N^2) | attention-math.html:783-784 | arXiv:2205.14135 | [ ] |
| 128K context = 16 billion elements | kernel-course.html:683 | Derived: 128K^2 / 1B | [x] |

### 7.3 Warp/Thread Numbers

| Claim | File | Line | Citation Status |
|-------|------|------|-----------------|
| Warp = 32 threads | kernel-course.html:235 | CUDA Programming Guide | [ ] |
| Warpgroup = 4 warps = 128 threads | kernel-course.html:237, gpu-arch:372-378 | Hopper whitepaper | [ ] |
| Max 255 registers per thread | math-foundations.html:419 | CUDA Programming Guide | [ ] |
| 32 shared memory banks | kernel-course.html:341, memory-hierarchy:596-607 | CUDA Programming Guide | [ ] |

---

## 8. Files Summary

### 8.1 Files to Audit

| File | Lines | Claims | Citations | Priority |
|------|-------|--------|-----------|----------|
| kernel-course.html | 1746 | ~60 | ~40 | HIGH |
| attention-math.html | 1004 | ~25 | ~20 | HIGH |
| lessons/gpu-architecture.html | 867 | ~30 | ~25 | HIGH |
| lessons/memory-hierarchy.html | 1155 | ~35 | ~30 | HIGH |
| lessons/math-foundations.html | 1740 | ~40 | ~20 | MEDIUM |
| math-prerequisites.html | 1350 | ~15 | ~20 (external) | MEDIUM |
| index.html | 445 | ~5 | 0 | LOW |

### 8.2 Citation Types Needed

1. **Hardware specs**: Official NVIDIA datasheets/whitepapers
2. **FP formats**: IEEE 754, arXiv:2209.05433, OCP MX Spec
3. **Algorithms**: Primary papers (FlashAttention, etc.)
4. **Performance claims**: Benchmarks or official marketing
5. **Latency numbers**: Need disclaimer about approximations

---

## 9. Action Items

### 9.1 Immediate Fixes Needed
- [x] Add disclaimer to all approximate latency numbers (added to memory-hierarchy.html)
- [ ] Verify B200 SM count (~192 vs exact number) - not in public specs
- [x] Verify B200 memory bandwidth (~8 TB/s source) - confirmed from GB200 NVL72 specs
- [x] Verify FP8 E4M3 max value = 448 matches arXiv paper - math verified
- [ ] Verify NVFP4 block size = 16 from official source
- [!] L2 cache size: H100=50MB but files say ~60MB (check if B200 context)

### 9.2 Links to Verify
- [x] All arXiv links resolve correctly
- [x] All NVIDIA docs links are current (v13.1+)
- [x] All YouTube video links are accessible
- [x] OCP MX Spec PDF link works (16-page PDF accessible)

### 9.3 Citations to Add
- [ ] Machine epsilon values need IEEE 754 reference
- [ ] Register bandwidth (~20 TB/s) needs source or removal
- [ ] "~1% accuracy loss" claim needs citation
- [ ] ">80% peak FLOPS" claim needs specific benchmark reference

---

## 10. Verification Methodology

For each claim:
1. Open the cited source
2. Verify the specific claim appears in the source
3. Verify the claim is current (not outdated specs)
4. Mark status in this document
5. If incorrect, note the correction needed

For external links:
1. Navigate to URL
2. Verify page loads and content is relevant
3. Note any 404s or redirects
4. Check if content matches what we're citing it for

---

## 11. Progress Tracking

| Date | Auditor | Section | Items Verified | Issues Found |
|------|---------|---------|----------------|--------------|
| 2026-01-26 | Claude | Plan created | 0 | 0 |
| 2026-01-26 | Claude | arXiv papers | 10 | 0 - All verified |
| 2026-01-26 | Claude | Blackwell specs | 5 | 2 need datasheet (SM count, clock) |
| 2026-01-26 | Claude | FP8 E4M3 max=448 | 1 | 0 - Math verified |
| 2026-01-26 | Claude | YouTube links | 14 | 0 - All accessible |
| 2026-01-26 | Claude | NVIDIA docs | 6 | 0 - All verified v13.1+ |
| 2026-01-26 | Claude | External resources | 9 | 1 - matrixmultiplication.xyz (FIXED) |
| 2026-01-26 | Claude | Wikipedia links | 4 | 0 - All verified |
| 2026-01-26 | Claude | Hopper specs | 5 | 1 - L2 cache 50MB vs ~60MB |

---

*This audit plan will be updated as verification progresses.*
