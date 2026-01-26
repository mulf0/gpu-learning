# GPU Learning Course - Progress & TODO

## Vision

One unified course structured as **chapters of a book**, organized from least to most complex. Each chapter flows into the next. Prerequisites exist as a separate reference, linked contextually where relevant.

---

## Current State

### Chapter Files (7 of 7 complete)

| File | Content | Status |
|------|---------|--------|
| `course/01-gpu-fundamentals.html` | GPU hierarchy, warps, blocks, occupancy | Complete |
| `course/02-memory.html` | Memory tiers, coalescing, bank conflicts, TMA | Complete |
| `course/03-kernels.html` | First Triton kernels, index arithmetic, tiling | Complete |
| `course/04-optimization.html` | Profiling, bank conflicts, Tensor Cores, TMA | Complete |
| `course/05-attention.html` | Dot products, softmax, FlashAttention | Complete |
| `course/06-quantization.html` | FP formats, INT8/INT4, FP8, numerical stability | Complete |
| `course/07-production.html` | KV cache, PagedAttention, TensorRT-LLM, vLLM | Complete |

### Entry Points

| File | Purpose | Status |
|------|---------|--------|
| `index.html` | Book intro with TOC and "Start Chapter 1" CTA | Complete |
| `math-prerequisites.html` | Reference appendix (linked contextually) | Existing |

### Legacy Files (to be cleaned up)

| File | Notes |
|------|-------|
| `lessons/gpu-architecture.html` | Content moved to Chapter 1 |
| `lessons/memory-hierarchy.html` | Content moved to Chapter 2 |
| `lessons/math-foundations.html` | Content moved to Chapter 6 |
| `attention-math.html` | Content moved to Chapter 5 |
| `kernel-course.html` | Replaced by chapter structure |

### Notebooks

| Directory | Files | Content | Status |
|-----------|-------|---------|--------|
| `notebooks/part1/` | 8 | NumPy → CuPy → Triton matmul | Complete |
| `notebooks/part2/` | 7 | Profiling, coalescing, bank conflicts | Complete |
| `notebooks/part3/` | 7 | Dot products, softmax, FlashAttention | Complete |
| `notebooks/part4/` | 7 | Quantization, KV cache, production | Complete |

### Branch

`course-restructure`

---

## Target Structure

```
index.html                      ← Book intro / Start Here (DONE)
course/
├── 01-gpu-fundamentals.html    ← DONE
├── 02-memory.html              ← DONE
├── 03-kernels.html             ← DONE
├── 04-optimization.html        ← DONE
├── 05-attention.html           ← DONE
├── 06-quantization.html        ← DONE
└── 07-production.html          ← DONE

math-prerequisites.html         ← Reference appendix (existing)

notebooks/
├── part1/                      ← DONE
├── part2/                      ← DONE
├── part3/                      ← DONE
└── part4/                      ← TODO
```

---

## TODO

### Phase 1: Chapter Structure
- [x] Create `course/` directory
- [x] Create shared chapter navigation styles (in base.css)
- [x] Create `course/01-gpu-fundamentals.html`
- [x] Create `course/02-memory.html`
- [x] Create `course/03-kernels.html`
- [x] Create `course/04-optimization.html`
- [x] Create `course/05-attention.html`
- [x] Create `course/06-quantization.html`
- [x] Create `course/07-production.html`

### Phase 2: Navigation & Flow
- [x] Add chapter-to-chapter navigation to all chapters
- [x] Add prereq callouts with links to `math-prerequisites.html#section`
- [x] Add notebook links to chapters
- [x] Update `index.html` to be book intro/start page

### Phase 3: Cleanup
- [x] Remove or redirect old `kernel-course.html`
- [x] Clean up `lessons/` folder (move or delete)
- [x] Remove old `attention-math.html`

### Phase 4: Part 4 Notebooks
- [x] Create `notebooks/part4/01_fp8_conversion.ipynb`
- [x] Create `notebooks/part4/02_quantization_fundamentals.ipynb`
- [x] Create `notebooks/part4/03_int8_int4.ipynb`
- [x] Create `notebooks/part4/04_nvfp4.ipynb`
- [x] Create `notebooks/part4/05_kv_cache_strategy.ipynb`
- [x] Create `notebooks/part4/06_fused_quantized_attention.ipynb`
- [x] Create `notebooks/part4/07_production_integration.ipynb`

### Phase 5: Polish
- [ ] Test full learning flow
- [ ] Final commit and merge to master

---

## Prereq Link Targets

For contextual callouts, link to these sections in `math-prerequisites.html`:

| Topic | Link |
|-------|------|
| Linear algebra basics | `math-prerequisites.html#linear-algebra` |
| Matrix operations | `math-prerequisites.html#matrix-ops` |
| Floating point | `math-prerequisites.html#floating-point` |
| Exponentials/softmax | `math-prerequisites.html#exponentials` |
| Index arithmetic | `math-prerequisites.html#index-arithmetic` |

---

## Session Log

### 2025-01-26
- Created 22 notebooks (Parts 1-3)
- Renamed `notebooks/week*` → `notebooks/part*`
- Partial restructure of `kernel-course.html` with Part 1 content
- Established chapter-based book structure plan
- Decided: transform existing pages into chapters (not rewrite from scratch)

### 2025-01-26 (continued)
- Added chapter navigation styles to `base.css`
- Created `course/01-gpu-fundamentals.html` from `lessons/gpu-architecture.html`
- Created `course/02-memory.html` from `lessons/memory-hierarchy.html`
- Created `course/05-attention.html` from `attention-math.html`

### 2025-01-26 (session 3)
- Created `course/06-quantization.html` from `lessons/math-foundations.html`
- Created `course/03-kernels.html` (new content, Triton basics from kernel-course.html)
- Created `course/04-optimization.html` (new content, profiling/bank conflicts/TMA)
- Refactored `index.html` as book intro with chapter TOC
- All chapters have: navigation, prereq callouts, notebook links, prev/next footer

### 2025-01-26 (session 4)
- Added citations to Chapter 3 and Chapter 4 (Triton, CuPy, CUDA docs, H100 datasheet, PTX ISA)
- Created `course/07-production.html` (KV cache, PagedAttention, continuous batching, TensorRT-LLM, vLLM)
- All claims in new chapters properly cited with arXiv papers, official docs, and datasheets
- Updated index.html to enable Chapter 7
- **All 7 chapters complete!**
- Increased base font size to 20px in `base.css` for better readability
- Remaining: Part 4 notebooks, cleanup legacy files
