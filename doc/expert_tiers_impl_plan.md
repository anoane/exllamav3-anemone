# Expert tiers: implementation plan

**Status.** This is the plan the expert tier was built from, kept for its design reasoning, its
exact policy semantics (section 3) and the micro-scenarios the tests cite (section 3.7). It is not
the reference: [expert_tiers.md](expert_tiers.md) describes what the code does, and where the two
differ, that document is right. The implementation departs from the plan where measurement or
review said so: the layer-mode staging is planned by the caller's thread one layer ahead (the tier
thread never calls CUDA), the fetch kernel copies decode and routed misses with the SMs rather than
the copy engines, `prefetch=router` is implemented (section 0 lists it as deferred), hot pins are
chosen at load only (T5.1's re-choice between generations is not implemented; the VRAM cache
adapts instead), and the commit series of section 11 was regrouped. File and line references are
to the tree the plan was written against.

Date: 2026-09-25. Worktree `dsv41-release/engine-tier`, branch `expert-tier`, based on `disk-pr3`
at `bfba0e0` (upstream `dev` + PR 1 + PR 2 + the disk engine). Every `file:line` below is at
`bfba0e0` in this worktree. The branch is rebased onto the final stack later, so the tier lives
in new files and touches shared files only through small, separate hooks (Appendix A lists every
one).

Design this plan implements: `disk-tier/DESIGN.md` (tier model 2, grammar 3, I/O 4, decode and
prefill 5, bidirectional swap 6, plan 8), with `grammar_extend-rules.md`, `grammar_tiers-first.md`,
`analysis_bidirectional.md`, `sim_demotion.py` / `sim_presets.py` (`sim_results/`), `DISK_ENGINE.md`,
`ENGRAM_STREAMING.md`, `unified_engine.md`, `HOST_CONFIG.md`. Where this plan departs from
`DESIGN.md` it says so and why (section 1.3).

Labels: **[M]** measured on the AI VM (source named); **[C]** read from code (`file:line`);
**[S]** simulated (`sim_presets.py`); **[R]** computed for this plan by the scratch reference
`dsv41-release/tmp/tier_plan/micro_scenarios.py` on the AI VM (CPU only,
`/root/rnd/tier/plan/`); **[E]** estimate, inputs shown.

---

## 0. Summary

- **What gets built.** A three-level home for the routed experts of `experts=cache` layers: a
  per-GPU VRAM pool (plus optional per-layer `hot=` pins), a parent-owned pinned RAM tier, and the
  checkpoint shards on the SSD, read in place through the disk engine. Every policy is
  selectable: `policy=lazy-exclusive|exclusive|inclusive`, `demote=swap|heat|all|off` plus an
  explicit master switch `EXL3_MOE_TIER_DEMOTE=0|1`, `admit=adaptive|heat|always`,
  `evict=lru|lfu` per level. RAM budgets are first-class: `-er/--expert_ram`,
  `-der/--draft_expert_ram`, `-ngr/--ngram_ram [SIZE]`, validated against `MemAvailable` (and the
  memory cgroup) before any pinned allocation.
- **The central choice.** A tiered layer runs the *resident* compute path unchanged
  (`BlockSparseMLP.forward`, `block_sparse_mlp.py:994-1416`). The tier only rewrites the layer's
  per-expert trellis pointer tables (`MultiLinear.ptrs_trellis`, `multilinear.py:32`), which every
  quantized fast path reads on the device (`run_bszN` through `coop_p`,
  `exl3_moe_coop.cu:235-243`; `exl3_moe`; the batched reconstruct tier through `index_select`,
  `moe_batch_recon.py:250-252`). Hits, admitted misses and transients therefore compute exactly
  what a resident layer computes, bit for bit, in default and in stable arithmetic. The one path
  that reads host-held per-expert tensors (`run_single_expert_dq`,
  `libtorch/blocksparse_mlp.cpp:481-536`) gets a sibling that takes the trellis tensors
  explicitly (commit T0.3).
- **Decode.** A single-CTA lookup kernel inside the MoE dispatch (after routing,
  `block_sparse_mlp.py:1067`) resolves hits, admits misses into spare slots, retires LRU victims,
  rewrites pointers and publishes a call record to mapped pinned memory. When nothing is missing it
  releases the layer itself; otherwise a native tier thread copies the missing experts with the
  copy engines (RAM) or through the disk engine (SSD) and releases the layer with a stream memop
  flag. Demotions are D2H copies on a low-priority stream, off the critical path.
- **Prefill.** Chunks of at least `EXL3_MOE_TIER_PREFILL_ROWS` rows use FreeToken's whole-layer
  double buffering: while layer L computes, every expert of layer L+1 that VRAM does not hold is
  copied into the other half of a dedicated staging area; no admission, eviction or demotion.
  SSD-only experts are read `prefetch=layer:D` layers ahead (class 2, promoted to 1).
- **Series.** 19 commits in 7 groups (section 11): T0 prerequisites (3), T1 budgets and grammar
  (4), T2 CPU core (4), T3 the GPU tier over a RAM tier that holds every cached expert (4),
  T4 the SSD tier (2), T5 quiescent points and the heat file (1), T6 defaults from the AI VM
  measurements (1, last). Groups stop cleanly; no commit accepts a grammar word it cannot run.
- **The three test configurations** (section 13): (a) PRO 6000 + RAM holding every non-VRAM
  expert (160 GB VM), no CMP; (b) PRO 6000 + RAM <= 128 GiB + SSD expert tier, no CMP; (c) PRO 6000
  + CMP 170HX with resident experts + RAM <= 128 GiB + `disk experts=off` (zero expert SSD reads,
  the daily recipe family). Engram tables stay on the SSD in all three (`ram ngram=0`).
- **Deferred** (not in this series): `profile=` seeding, `prefetch=router`, `-mcm cache`, draft
  (MTP head) tiers, staging smaller than one layer, staging borrowed from the pool, the n-gram row
  cache, Windows.

---

## 1. Inputs, facts, and departures from DESIGN.md

### 1.1 Facts the plan is built on

| Fact | Value | Source |
|---|---|---|
| V4.1 EXL3 3.0 bpw | 40 layers x 384 routed experts, top-6; hidden 5120, intermediate 2304 | [M] `research_freetoken.md` 9.1 |
| One expert on disk | 13,315,596 B, one contiguous span per expert; trellis tensors at odd offsets | [M] shard headers |
| VRAM slot (3 trellis, compact) | 3 x 4,423,680 = 13,271,040 B | [C] trellis `[320, 144, 48]` int16 |
| RAM slot (extent rounded to 4 KiB) | 13,320,192 B, 80 per 1 GiB chunk | `unified_engine.md` 6.2 |
| Resident aux per expert (suh/svh as fp16) | ~44.5 KB; 479 MB for 28 layers, 684 MB for 40 | [E] |
| PRO 6000 copy engines | `asyncEngineCount` 2 | [M] `/root/rnd/bench-window/duplex_register.json` |
| PRO H2D retained under D2H | 1:1 0.49-0.52; 1:4 0.81; 1:10 0.92-0.94; **1:20 0.956-0.959**; duplex aggregate 52.8-54.3 GB/s | [M] same (alloc and register runs) |
| CMP 170HX link | ~0.7 GB/s duplex aggregate, x1; H2D retained ~0.91 even at 1:1 (link-bound) | [M] same |
| Misaligned-source H2D (`--host-misalign 11`) | **not measured** | gate in T3.2's window |
| Disk engine | `auto` = io_uring, engine route, 5 ms keep-alive; decode 48-row p50 267 us at 75 % cached; extents 10.9 GB/s at 4 in flight; next to a class-1 stream with an 8M window: rows 750 us, bulk 8.9 GB/s | [M] `/root/rnd/bench-window/disk/decision.md` |
| Registered io_uring buffers | at most 63 user buffers of <= 1 GiB (`kMaxUserBuffers = 64`) | [C] `exllamav3_ext/disk/disk_internal.h:52` |
| VM memory | 160 GiB total, swap 0, pinned memory unreclaimable; daily budget <= 128-130 GiB | [M] `free -g` 2026-09-25 |
| Serving device order | `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,0`: cuda:0 = CMP, cuda:1 = PRO | [C] `ds41-exl3.service` |

### 1.2 What `DESIGN.md` settles and this plan keeps

One placement string (3.1); `experts=cache`, `hot=`, `cuda:<n> cache= spare= evict= admit=`,
`ram experts= ngram= pagecache= policy= demote= evict=`, `disk experts= ngram= io=` (3.2-3.6);
sizes with strict units (3.6); canonical form (3.7); the flag table (3.9); the refusal texts
(3.10); exclusive-by-default tiers with lazy duplicates (2.2); adaptive admission (2.3); global
pools per GPU, floors by `hot=` (2.4); heat as a decayed count (2.5); cold start from the SSD (2.6);
O_DIRECT extents into registered pinned slots through one disk engine (4.2); copy-engine promotion
(4.3); no admission or demotion at prefill (5.3); `demote=swap` (6); CMP never in the dynamic tier
(2.4, 6.1).

The duplex measurement now decides DESIGN 6.4's rule: at 1:20 the PRO keeps 0.956-0.959 of its
H2D rate, above 0.95, so **`demote=swap` stays the default and runs inline** on the low-priority
stream (1.1). (Superseded by the measurement on the test host: the defaults are now `policy=exclusive`
and `demote=heat`, see doc/expert_tiers.md, "Tuning".)

### 1.3 Departures and refinements (each with its reason)

| # | DESIGN.md | This plan | Why |
|---|---|---|---|
| R1 | B3: `experts=cache` maps onto the streamed path of `MoeCpuHost` (`("tiered", "stream", k)`), pointer redirect in its `tbl` (`moe_cpu_host.py:1681-1701`) | A tiered layer is a GPU-resident `BlockSparseMLP` whose expert trellis pointers are rewritten per call; `MoeCpuHost` is not involved | The streamed path costs ~1.0 ms per expert at decode (DESIGN risk 2), needs the spawned CPU worker and its handoff segment, plans per call on the host (`counts1.tolist()`, `moe_cpu_host.py:1388`), and equals the resident arithmetic only under the stable profile. The resident path reads per-expert pointers on the device in every quantized tier, so redirecting them gives bitwise identity with `experts=vram` by construction |
| R2 | RAM tier = memfd arena chunks shared with the MoE child (`_HugeArena`) | Parent-owned anonymous chunks (1 GiB, 80 slots), registered with CUDA after they are filled and with the disk engine before | No child process exists for `cache` layers; the engine lives in the parent (`unified_engine.md` 9.4) |
| R3 | (not specified) | Expert `Linear`s of a tiered layer get an `LinearEXL3` inner whose trellis is a view into one per-GPU **sentinel slot**; suh/svh are loaded resident | `MultiLinear`, `BC_BlockSparseMLP` and the fused buffers are then built by the unchanged `load_local` (`block_sparse_mlp.py:497-760`); `pin_linears` (`linear.py:448-471`) is the precedent for swapping a trellis |
| R4 | VRAM victims chosen at admission | Spare-slot protocol: an admission takes a FREE slot; the kernel retires one LRU victim per admission once the pool is at `S - spare`; the host decides drop or D2H and returns the slot through a return ring | Demotion needs the victim's bytes intact until its D2H lands; the kernel cannot know whether the host will demote (RAM state is host-side) |
| R5 | 5.3: prefill streams RAM-owned experts through a bounded ring of 64 slots | A dedicated staging area of `2 x (E - hot)` slots: FreeToken's whole-layer double buffer | A layer's MoE must then run as one resident call (bitwise identity); a smaller ring needs per-batch launches whose arithmetic is equal only under the stable profile. Costs 9.49 GiB on the PRO for V4.1 (`EXL3_MOE_TIER_STAGING=single` halves it, without copy/compute overlap). Smaller staging is deferred behind a bitwise gate |
| R6 | RAM admission of an SSD fill: "over RAM's coldest sampled entry **when warmer**" | Default = the simulated rule (a fill always takes a free slot, else a reclaimable duplicate, else the coldest sampled unique entry); the heat-gated rule is `EXL3_MOE_TIER_RAM_ADMIT=heat` | The defaults were chosen on `sim_presets.py`, whose `room_for_unique` is ungated; keep what was measured, offer the text's rule as a knob |
| R7 | `policy=inclusive` back-invalidates a duplicate and its VRAM copy when RAM is full (sim `room_incl`) | Never back-invalidates; the load requires `ram experts` > the VRAM pool for `inclusive` | The host cannot evict a VRAM slot without racing the lookup kernel. With R > S + spare a unique or free RAM slot always exists (4.4) |
| R8 | Heat kept by a host thread from a per-token histogram D2H | Heat kept twice with one integer arithmetic: on the device (for `admit=heat`, `evict=lfu`) and on the host (from the call records, for RAM decisions); a test asserts they are equal | The call records already carry every call's unique experts and counts; no extra D2H |
| R9 | `EXL3_MOE_PREFILL_SLOTS`, `EXL3_MOE_PREFILL_ADMIT` | `EXL3_MOE_TIER_STAGING=double|single`, `EXL3_MOE_TIER_PREFILL_ROWS`; prefill admission stays off (no switch) | Staging is whole layers now (R5); an admission switch at prefill had no evidence behind it |
| R10 | (not specified) | `kMaxUserBuffers` 64 -> 1024 with a sorted lookup (T0.1) | Configuration (a) needs ~126 registered 1 GiB chunks |
| R11 | Loader warns when a `cache` layer sits behind a slow link | Refuses below `EXL3_MOE_TIER_MIN_LINK_GBS` (default 2.0; 0 allows) | "CMP: static placement only, never a cache/stream target"; `stream` keeps its warning only (legacy) |
| R12 | Stamps (LRU) updated by every use | Only decode calls update LRU stamps; prefill (layer and routed modes) only adds heat | A scan must not refresh recency of every expert it touches (scan resistance) |
| R13 | A1 routing trace first | Kept as T2.1, and the tier's own call records double as a trace (`EXL3_MOE_TIER_TRACE`) | Risk 1 (V4.1 locality unmeasured) |
| R14 | `admit=adaptive` adapts p from cache lives | Same rule; tests pin p with `EXL3_MOE_TIER_ADMIT_P` because a live p depends on host timing | Tier decisions change timing only, never logits; replay tests need determinism (3.6) |

---

## 2. Architecture

### 2.1 Components

```
 Python (main / pipeline thread)                      native tier runtime (parent process)
 ------------------------------------------------     ------------------------------------------------
 BlockSparseMLP.forward                               TierRuntime: one per model component
   routing -> selected_experts                          tier thread (spins while a pass is active)
   tier_resolve()  --- decode / routed ---> lookup kernel (compute stream, 1 CTA)
                                              |  hits: stamp, ptr <- slot           call records
                                              |  misses: admit into FREE / transient  ------------>  mapped pinned
                                              |  retire LRU victims                   ctl page       ring (host reads)
                                              |  no host work -> ready := seq  <--------------------  |
                   flag_wait(ready >= seq) <--+                                                        v
   resident compute (run_bszN / exl3_moe /      host mirror of the VRAM directory, RamTier, heat
     batched recon / DQ views) reads ptr tables  per miss: RAM slot --H2D (copy stream A)--> VRAM slot
   tier_after_compute(): consumed := seq                 SSD -> disk engine (class 1) -> RAM or slab
                                                          --flag--> copy stream B -> H2D
   --- layer mode (prefill) ---> host plan             ready := seq (memop on stream B, after A)
        (after directory sync): ptr upload,           victims: drop, or D2H (low-priority stream)
        copies for L+1 into staging half,               into the RAM slot the promotion left;
        class-2 SSD reads for L+1+D                     slot returned through the return ring
```

### 2.2 Where one routed expert of a `cache` layer can be

| Place | State | Owner | Notes |
|---|---|---|---|
| VRAM hot pin | `PINNED` slot | the layer, fixed between generations | `hot=<k>`; never a victim; re-chosen at quiescent points (T5.1) |
| VRAM pool slot | `VALID` | VRAM | stamp and heat decide eviction |
| VRAM pool slot being retired | `RETIRING` | nobody (bytes still valid) | `slot_of` already -1; a miss on it is served D2D from this slot ("rescue") until the host returns it |
| VRAM staging slot | transient | the call that uses it | decode transients use staging half 0; prefill uses both halves |
| RAM slot, unique | `OWN` | RAM | extent layout (filled from the SSD) or compact layout (filled by a D2H demotion, or by cold-fill compaction if risk 4 turns it on) |
| RAM slot, duplicate | `DUP` | VRAM (RAM copy reclaimable) | `lazy-exclusive` promotions, `inclusive` always |
| Disk slab slot | transient | the read that fills it | SSD reads that bypass RAM (prefill, refused RAM admission) |
| SSD | always (unless `disk experts=off`) | the checkpoint | read in place, never written |

### 2.3 Streams, flags and events (per GPU)

| Name | Kind | Written by | Waited by | Purpose |
|---|---|---|---|---|
| compute stream | the caller's current stream | kernels | - | lookup kernel, MoE compute |
| copy stream A | high priority | tier thread, prefill planner | - | RAM -> VRAM (pool, staging), D2D rescues |
| copy stream B | high priority | tier thread, prefill planner | - | waits on disk flags, then slab/RAM -> VRAM; writes `ready` after waiting on A |
| D2H stream | lowest priority | tier thread | - | demotions VRAM -> RAM |
| `ready` (u32, mapped) | memop GEQ | lookup kernel (no host work) or stream B | compute stream before the MoE compute | call `seq` may compute |
| `consumed[2]` (u32, mapped) | memop GEQ | compute stream after the MoE compute | copy streams before overwriting a staging half | staging reuse |
| `prefill_ready[2]` (u32, mapped) | memop GEQ | stream B after a layer's staging copies | compute stream | layer-mode prefill |
| `abort` (u32, mapped) | plain | tier thread on failure | memop fallback kernel (`moe_handoff.cu:37`) | unblocks waits after an unrecoverable error |
| disk flags (u32, mapped, one per ticket slot) | release store | disk engine (`Options.flag`, `disk_engine.h:145`) | copy stream B (memop GEQ) | SSD read done |
| events | CUDA events | tier thread | tier thread (`cudaEventQuery`) | "H2D from RAM slot r done" before a D2H overwrites r; "D2H done" before a slot returns |

Flags are 64 B apart in one mapped pinned control page per GPU, as the handoff segment does
(`moe_cpu_host.py:890-892`). Waits and writes go through the existing memop helpers
(`cpu/moe_handoff.cu:108-142`), generalized to an explicit stream by T0.2.

### 2.4 Threads

- **The caller's thread** (Python main thread, or the V4.1 pipeline's stage-2 worker,
  `architecture/dsv41/pipeline.py:343-352`) launches lookups, uploads prefill pointer tables and
  enqueues layer-mode copies (through a native call, GIL released). One caller thread per GPU at
  a time; the tier refuses to be driven from two threads concurrently (checked by thread id,
  debug builds and `EXL3_MOE_TIER_VERIFY=1`).
- **The tier thread** (C++ `std::thread`, one per component): consumes call records, issues
  copies and disk reads, decides demotions, returns slots, keeps host heat and the RAM tier,
  schedules refills. It spins while a pass is active (`begin_pass` .. `end_pass`), then sleeps on
  a condition variable; `EXL3_MOE_TIER_SPIN_US` bounds the spin, `EXL3_MOE_TIER_AFFINITY` pins it.
  Every CUDA call it makes is preceded by `cudaSetDevice`.
- **Disk engine threads** (existing): the io_uring reaper sets ticket flags.
- **Registration thread** (load only): `cudaHostRegister` of RAM chunks as their cold fill
  completes (~0.2 s per GiB, `moe_cpu_host.py:113-119`), overlapping the reads.

---

## 3. Normative policy semantics

Every implementation (Python core, C++ core, GPU kernel + tier thread) must produce exactly
these decisions. Keys are `lc * E + e` (lc = index of the cache layer on its GPU, in forward
order). `S` = dynamic pool slots, `spare` from `cuda:<n> spare=`, capacity `C = S - spare`.

### 3.1 One lookup call (layer lc, ids = the call's `selected_experts`, row-major)

1. `now = seq`: the caller numbers every tier call on a GPU, one increment per call (decode, routed and
   layer mode alike). `U` = unique ids in order of first occurrence; `cnt[e]` = occurrences.
2. Heat: for e in U, `heat[e] += cnt[e] * inc` (decode inc = 1.0, prefill inc = w; 3.4).
3. **Phase 1 (protection).** In decode mode, every e in U with `slot_of[e] >= 0` gets
   `stamp[slot] = now`. A barrier follows (FreeToken found that without it a hit of the same
   call is evicted; the scratch reference reproduced exactly that bug before this rule, [R]).
4. **Phase 2**, e in U order, `access += 1` for each:
   - `slot_of[e] >= 0` -> **hit**.
   - routed mode (prefill below `EXL3_MOE_TIER_PREFILL_ROWS`) -> **transient**.
   - decode mode: admit if `occupied < C`; else by `admit=`: `always` admits; `heat` admits iff
     `heat[e] >= heat[v]`, v = the slot the eviction rule would pick now; `adaptive` admits iff
     `u(access) < p`, `u = splitmix64(seed ^ (access * 0x9E3779B97F4A7C15)) / 2^64`.
   - An admitted miss takes the **lowest-index FREE slot**; none free -> **transient**
     (counted `starved`). The slot becomes `VALID`, key e, stamp `now`.
   - If now `occupied > C`, **retire one victim**: `evict=lru` = min `(stamp, slot index)` over
     `VALID` slots with `stamp < now`; `evict=lfu` = min `(decayed heat, stamp, slot index)` over
     the same set. The victim becomes `RETIRING`, `slot_of[victim] = -1`, and is **paired** with e
     in the record.
   - A transient takes the next staging slot of half 0 (decode and routed mode alike).
5. Pointer rewrite: for every e in U, gate/up/down entries of the layer's three tables =
   slot address + `proj_off[0..2]`.
6. Record: `{seq, lc, mode, n, entries[e, kind, dst, cnt, heat_e, paired victim key/slot/stamp/heat]}`.
   If no entry needs the host (all hits), the kernel writes `ready = seq` itself.

Occupancy counts `VALID` only. `RETIRING` slots return to `FREE` when the host returns them.

### 3.2 Host processing of a record (tier thread, in record order)

For each non-hit entry e (source first, then its paired victim v):

**Source of e.** A `RETIRING` VRAM slot holding e -> D2D on stream A (`rescued`). Else RAM
(`ram_of[e] >= 0`) -> H2D on stream A (3 sub-range copies from an extent-layout slot, 1 from a
compact one). Else SSD -> class-1 read (disk flag) into a RAM slot when RAM admission says so,
else into a slab slot, then stream B waits on the flag and copies. With `disk experts=off` an SSD
source is an internal error (the load guarantees it cannot happen).

**RAM side of e.**

| e was | kind | exclusive | lazy-exclusive | inclusive |
|---|---|---|---|---|
| in RAM | admitted | RAM slot r freed after the H2D event | r becomes `DUP` | r becomes `DUP` |
| in RAM | transient | unchanged | unchanged | unchanged |
| SSD | admitted | bypass (slab) | into a free slot, else over a reclaimed `DUP`, as `DUP`; else bypass | into a free slot, else over the coldest unique, as `DUP` |
| SSD | transient | RAM admission (below) as `OWN`, else bypass | same | same |

**RAM admission** (`EXL3_MOE_TIER_RAM_ADMIT=always`, default): a free slot, else reclaim a `DUP`,
else evict the coldest sampled `OWN`. With `heat`: the last step only when e is warmer.

**Paired victim v** (skipped entirely when v has a `DUP` in RAM: the duplicate becomes `OWN`):

| `demote=` | Rule |
|---|---|
| `off` | drop |
| `swap` (default) | only if e came from RAM slot r (freed or made `DUP`) and `heat[v] > heat[coldest sampled OWN]`: D2H v into r (overwriting e's `DUP`) after e's H2D event; else drop |
| `heat` | D2H into a free slot, else over a reclaimed `DUP`, else over the coldest sampled `OWN` when v is warmer; else drop |
| `all` | D2H: free slot, else reclaimed `DUP`, else over the coldest sampled `OWN` |
| any, with `disk experts=off` | unconditional: into e's `DUP`/freed slot, else a free slot, else a reclaimed `DUP` (sizing guarantees one) |
| `EXL3_MOE_TIER_DEMOTE=0` | as `off` (refused with `disk experts=off`) |

`inclusive` never demotes (v always has a `DUP`). A demoted slot holds v in compact layout. v's
VRAM slot returns to `FREE` after the D2H event (or at once on a drop). After the call's copies:
stream B waits on stream A's event, then writes `ready = seq`.

**Sampling.** "Coldest sampled" = min `(heat, key)` over `min(EXL3_MOE_TIER_SAMPLE, n)` draws with
replacement from the `OWN` list (all of them, sorted by key, when n <= the sample size); the
draw index is `splitmix64(seed ^ (0xD1B54A32D192ED03 * draw_counter)) % n`. "Reclaim a `DUP`" = the
duplicate whose VRAM copy has the most recent stamp (duplicates no longer in VRAM count as -1),
ties to the lowest key. RAM slots are allocated lowest index first. `ram evict=lru` replaces heat
by the RAM recency stamp in "coldest".

### 3.3 Refill (class 3)

When a record shows an SSD-sourced expert that went to the slab (bypass) and whose host heat
exceeds the coldest sampled `OWN`, the thread schedules one class-3 read into RAM (as `OWN`), at
most `EXL3_MOE_TIER_REFILL_INFLIGHT` (default 2) at a time; beyond that the candidate is dropped
and counted, never queued (DESIGN 5.2). `EXL3_MOE_TIER_REFILL=0` turns it off.

### 3.4 Heat arithmetic (both sides)

`u32` 16.16 fixed point plus a `u32` epoch per key; global epoch `T = decode_tokens / H`
(`EXL3_MOE_HEAT_HALFLIFE`, default 256). Read = `v >> min(31, T - epoch)`; add = `read + inc`,
epoch = T. Decode inc = `cnt << 16`; prefill inc = `cnt * w * 65536` with `w` =
`EXL3_MOE_HEAT_PREFILL` (default 1/16 = 4096). `decode_tokens` advances once per decode pass
(`begin_pass` with rows <= `EXL3_MOE_TIER_DECODE_ROWS`) and is passed to the kernel in the control
page.

### 3.5 Adaptive admission probability

Every `EXL3_MOE_TIER_ADAPT_EVERY` accesses (default 2048) the host computes, in call units,
`lv = call - stamp(VRAM LRU slot)` (host mirror) and `lr = call - recency(RAM LRU entry)`, then
`p = min(pmax, max(pmin, p * (1 + (lv / max(1, lv + lr) - 0.5))))`, `pmin` =
`EXL3_MOE_TIER_ADMIT_PMIN` (0.05), `pmax = C / (C + R)`, initial p = pmax (sim_presets' rule,
`disk-tier/sim_presets.py:91, 151-155`, with calls instead of accesses as the clock). It
publishes `p` as u32 Q0.32 in the control page; the kernel reads it at launch.
`EXL3_MOE_TIER_ADMIT_P=<float>` pins p (tests, A/B).

### 3.6 Deterministic mode

`EXL3_MOE_TIER_DETERMINISTIC=1`: the kernel never writes `ready` itself; the tier thread
processes every record, waits for its own D2H events, returns the slots, then writes `ready`. The
next lookup therefore sees the same state as the reference cores. Used by every replay test; also
a debugging aid. Production mode differs only in timing (returned slots may arrive a call later,
so an admission may become `starved`) and never in logits.

### 3.7 Micro-scenarios with exact expectations [R]

All in deterministic mode, `E = 8`, one cache layer, no decay unless stated, experts not in VRAM
or RAM come from the SSD. Records list `(kind, expert[, paired victim])`. These tables are the
expected values of `tests/test_expert_tier_policy_.py` (Python core), `tests/test_expert_tier_native_.py`
(C++ core) and `tests/expert_tier/tier_kernel_gpu_.py` (kernel + thread, VRAM side). The scratch
reference that produced them is `dsv41-release/tmp/tier_plan/micro_scenarios.py` (also at
`/root/rnd/tier/plan/`; not engine code).

**S1: LRU, protection, spare, ties** (`S = 5, spare = 2, admit=always, evict=lru, R = 0`)

| Call | Record | After the call |
|---|---|---|
| c1 [0, 1] | admit 0, admit 1 | slots 0:0, 1:1 |
| c2 [2, 3] | admit 2, admit 3 (victim 0) | 0 retired (stamp tie with 1, lower slot) |
| c3 [0, 2] | admit 0 (victim 1), hit 2 | |
| c4 [1, 3] | admit 1 (victim 0), hit 3 | 3 protected by phase 1 |
| c5 [4, 5] | admit 4 (victim 2), admit 5 (victim 1) | slots 0:4 (5), 3:3 (4), 4:5 (5); 1, 2 FREE |

Counters: hits 2, admits 8, retires 5, drops 5, SSD reads 8, D2H 0.

**S2: `admit=heat`** (`S = 4, spare = 2`, others as S1)

| Call | Record |
|---|---|
| c1 [0, 1] | admit 0, admit 1 |
| c2 [0, 2] | hit 0, admit 2 (victim 1) — heat 1 >= 1 |
| c3 [3, 0] | admit 3 (victim 2), hit 0 |
| c4 [2, 4] | transient 2, transient 4 — candidate is expert 0 (heat 3) |
| c5 [2, 5] | admit 2 (victim 0) — heat 3 >= 3; admit 5 (victim 3) |
| c6 [2, 4] | hit 2, admit 4 (victim 5) |

Counters: hits 3, admits 7, transients 2, retires 5, SSD reads 9. Final: slot 0:4, slot 2:2.

**S3: one promotion from RAM, every policy x demote** (`S = 3, spare = 1, R = 2`; VRAM holds 0
(stamp 1) and 1 (stamp 2); RAM holds 2 (slot 0) and 3 (slot 1) as `OWN`; heat 0 = h, 1 = 1,
2 = 4, 3 = 2; call [2]: admit 2 from RAM slot 0, victim 0)

| policy | demote | h = 5 (warmer than RAM's coldest, 3 at 2) | h = 1 (colder) |
|---|---|---|---|
| exclusive | off | RAM {3}; drop | same |
| exclusive | swap | RAM {0: slot 0, 3}; D2H 1 | RAM {3}; drop |
| exclusive | heat | RAM {0: slot 0, 3}; D2H 1 (free slot) | same |
| exclusive | all | RAM {0: slot 0, 3}; D2H 1 | same |
| lazy-exclusive | off | RAM {2: DUP slot 0, 3}; drop | same |
| lazy-exclusive | swap | RAM {0: slot 0, 3}; D2H 1 (over 2's DUP) | RAM {2: DUP, 3}; drop |
| lazy-exclusive | heat | RAM {0: slot 0, 3}; D2H 1; 1 DUP reclaimed | same |
| lazy-exclusive | all | RAM {0: slot 0, 3}; D2H 1; 1 DUP reclaimed | same |

Every row: admits 1, retires 1, H2D from RAM 1, SSD 0.

**S4: `disk experts=off`** (`E = 6, S = 4, spare = 2, R = 4`, lazy-exclusive, swap; VRAM 0, 1;
RAM 2, 3, 4, 5): calls [2,3] [4,0] [5,1] [2,4] [0,1] [3,5] each admit two and retire two (victims
0,1 / 2,3 / 4,0 / 5,1 / 2,4 / 0,1). Counters: admits 12, retires 12, **D2H 12, H2D from RAM 12,
SSD 0**; final VRAM {3, 5}, RAM {0: slot 0, 1: slot 2, 2: slot 1, 4: slot 3}. Invariant after
every call: each expert is `VALID` in VRAM or `OWN` in RAM.

**S5: `inclusive`** (`S = 4, spare = 2, R = 5`, demote off): [0,1] admit, admit; [2,0] admit 2
(victim 1), hit 0; [3,4] admit 3 (victim 0), admit 4 (victim 2); [1,2] admit 1 (victim 3), admit 2
(victim 4). Counters: hits 1, admits 7, retires 5, drops 5, H2D from RAM 2, SSD 5. Final RAM
{0 OWN, 1 DUP, 2 DUP, 3 OWN, 4 OWN}; every VRAM expert has a RAM copy after every call.

**S6: `admit=adaptive` with a pinned p** (`S = 4, spare = 2`): p = 0: [0,1] admit, admit;
[2,3] transient, transient; [4,0] transient 4, hit 0 (admits 2, transients 3, SSD 5). p = 1:
[2,3] admit 2 (victim 0), admit 3 (victim 1); [4,0] admit 4 (victim 2), admit 0 (victim 3)
(admits 6, retires 4, SSD 6).

**S7: heat** (H = 2 tokens; values 16.16): tok 0 [0,1] -> 0: 65536, 1: 65536; tok 1 [0] -> 0:
131072; tok 2 (read) -> 0: 65536, 1: 32768; tok 5 prefill [1,1,2] with w = 1/16 -> 0: 32768,
1: 24576, 2: 4096; tok 40 -> all 0.

**S8: `evict=lfu`, `demote=heat`** (`S = 4, spare = 2, R = 1`, lazy-exclusive): [0,1] admit,
admit; [0,2] hit 0, admit 2 (victim 1); [0,3] hit 0, admit 3 (victim 2); [3,4] hit 3, admit 4
(victim 0). Counters: hits 3, admits 5, retires 3, drops 1, D2H 2, SSD 5, RAM evictions 1,
duplicates reclaimed 3; final RAM {0 OWN}.

**S9: starvation** (`S = 3, spare = 1`): [0,1] admit, admit; [2,3] admit 2 (victim 0), transient 3
(starved: the only FREE slot is taken, the victim is still RETIRING). Counters: admits 3,
transients 1, starved 1.

---

## 4. Data structures

### 4.1 Device, per GPU (`TierDeviceState`, allocated by torch so `-gs` and the VRAM report see it)

| Array | Type, size | V4.1 T5 (Lc = 28, S ~ 5,780) |
|---|---|---|
| `slot_state` | u8 [S_total] (FREE 0, VALID 1, RETIRING 2, PINNED 3) | 6 KB |
| `slot_key` | i32 [S_total] | 23 KB |
| `slot_stamp` | u32 [S_total] | 23 KB |
| `slot_addr` | i64 [S_total] (hot pins after the pool) | 46 KB |
| `slot_of` | i32 [Lc x E] | 43 KB |
| `heat_v`, `heat_ep` | u32 [Lc x E] each | 86 KB |
| `staging_addr` | i64 [2 x (E - hot)] | 6 KB |
| `layer_ptrs` | i64 [Lc][3] (device addresses of each layer's gate/up/down `ptrs_trellis`) | < 1 KB |
| `proj_off` | i32 [3] (0, gb, gb + ub; one expert geometry per GPU) | - |
| `ctr` | u32/u64 counters: call, access, occupied, free, ret_head, rec_head, overflow | - |
| pool | u8 chunks of up to 4 GiB, 323 slots each | ~71.5 GiB [E] |
| staging | 2 x (E - hot) slots | 9.49 GiB (double), 4.75 (single) |
| sentinel | one slot, zero-filled | 12.7 MiB |
| hot pins | `hot` slots per layer, allocated with the layer | 0 by default |

### 4.2 Mapped pinned control page, per GPU (`TierCtl`, `cudaHostAlloc` mapped + portable)

| Field | Layout |
|---|---|
| params | admit mode, `p` (Q0.32), spare, evict mode, mode flags (deterministic, verify), `decode_tokens`, halflife shift, seed |
| flags | `ready`, `consumed[2]`, `prefill_ready[2]`, `abort`, each on its own 64 B line |
| record ring | 1,024 records of a 32 B header `{seq u32, lc u16, mode u8, n u8, flags u32, ...}` and up to 64 entries of 32 B `{key u32, kind u8, cnt u16, dst u32, heat u32, victim key i32, victim slot u32, victim stamp u32, victim heat u32}` (2 MiB); `rec_head` written by the kernel after `__threadfence_system()` |
| return ring | 4,096 u32 slot ids + `ret_tail` (host, release store); the kernel harvests up to `ret_tail` at launch |
| disk flags | 512 u32 flags for tickets (ring) |

If the host falls behind by a whole ring, only all-hit records can be overwritten (a record with
host work blocks its own layer), so the loss is heat only; `overflow` counts it.

### 4.3 Host, native (`TierRuntime`)

- **Host mirror** of 4.1 (`VramDirectoryRef`, the same class as the C++ policy core): applied from
  records, never re-decided.
- **RamTier:** `ram_key[R]`, `ram_dup[R]`, `ram_layout[R]` (`EXTENT{payload_off, off_g, off_u,
  off_d}` or `COMPACT`), `ram_of[K]`, a min-heap of free indices, `OWN` and `DUP` lists with
  position maps (O(1) sampling), RAM recency stamps, host heat.
- **RAM arena:** 1 GiB anonymous chunks (`mmap`, `MADV_HUGEPAGE` when
  `EXL3_MOE_ARENA_HUGEPAGE=1`), each registered with the disk engine (`disk_register_buffer`,
  `disk_ext.cpp:577`) before its fill and with CUDA (`cudaHostRegister(PORTABLE)`) after it.
- **Extent table** `extents[K]`: file index, payload offset, length, the three trellis
  sub-offsets, trellis bytes (from T2.2); `files[]`: path and read-only fd.
- **Disk slab:** `EXL3_MOE_TIER_DISK_SLAB` slots (default `auto` = 8 decode + the prefill
  read-ahead need, see 9.2), states FREE / READING(ticket) / READY / COPYING(event).
- **In flight:** disk tickets, demotion events, pending returns, refill tickets.
- **Stats** (per GPU): hits, admits, transients, starved, retires, drops, D2H, H2D from RAM, SSD
  reads and bytes, rescues, prefill copies and SSD reads, refills (done, dropped), RAM evictions,
  duplicates reclaimed, record overflows, host lag (records), lookup-to-ready latency histogram.

### 4.4 Sizing rules

- `S_total = S + hot pins`; `C = S - spare`.
- `ram experts=all` with `disk experts=off`: `R = N_c - C` where `N_c` = cached experts (`sum (E - hot)`
  over `cache` layers). With `lazy-exclusive` promotions keep `DUP`s, so R must also hold the
  in-flight victims: the kernel retires at most one victim per admission and admissions draw on
  the spare, so `R = N_c - C` suffices (proof sketch in `doc/expert_tiers.md`; asserted by S4 and by
  the churn test).
- `policy=inclusive` needs `R > S + spare` (R7).
- RAM chunks hold 80 slots; the tier's pinned bytes are `ceil(R / 80)` GiB.

### 4.5 Python

| Module (new) | Contents |
|---|---|
| `exllamav3/util/host_budget.py` | `Size`, `parse_size`, `fmt_size`, `LedgerItem`, `HostLedger`, `memory_available()` (MemAvailable and the memory cgroup) |
| `exllamav3/model/placement_storage.py` | storage rules (`DeviceStore`, `RamStore`, `DiskStore`), attribute parsing, did-you-mean, checks, canonical printing; imported by `placement.py` |
| `exllamav3/model/expert_extents.py` | `ExpertExtent`, `SlotGeometry`, `build_extent_index` |
| `exllamav3/model/expert_tier_policy.py` | torch-free Python core: `SplitMix64`, `Heat`, `VramDirectory`, `RamTier`, `TierCore`, `replay` |
| `exllamav3/model/expert_tier.py` | `TierConfig`, `ExpertTierSet` (per component, `config.expert_tiers`), `TierLayer`, `TierDevice` |
| `exllamav3/modules/block_sparse_mlp_tier.py` | `BlockSparseMLP_Tier` mixin (load, resolve, after-compute, DQ views, unload) |
| `exllamav3/model/moe_trace.py` | `EXL3_MOE_TRACE` routing trace writer |

---

## 5. Native code

New directory `exllamav3/exllamav3_ext/tier/`; both `ext.py:133-142` and `setup.py:58-61` compile
every `.cpp` / `.cu` under the extension directory, so no build-list change is needed.

| File | Contents |
|---|---|
| `tier_policy.h/.cpp` | torch-free C++ core: `SplitMix64`, `Heat`, `VramDirectoryRef` (3.1 on the CPU; also the host mirror), `RamTier` (3.2-3.3), `AdaptiveP` (3.5), `replay()` for tests |
| `tier_kernels.cuh/.cu` | `tier_lookup_kernel`, `tier_heat_add_kernel` (layer-mode prefill heat), `tier_verify_kernel` (invariants, `EXL3_MOE_TIER_VERIFY=1`), `tier_fill_kernel` (sentinel zero) |
| `tier_device.h/.cpp` | `TierDevice`: device state, control page, streams, event pool, `lookup(...)` launcher |
| `tier_runtime.h/.cpp` | `TierRuntime`: the tier thread, record processing, RAM arena and slab, disk submissions, promotion/demotion copies, returns, refill, prefill planner, cold fill, stats, error state |
| `tier_ext.h/.cpp` | Python-facing functions and the `TierRuntime` / `TierDevice` handles |
| `tier_bc.h` | pybind definitions, included inside `PYBIND11_MODULE` like `disk/disk_bc.h` (`bindings.cpp:121`) |

**Lookup kernel.** One CTA of 1,024 threads per call:

```cpp
void tier_lookup(
    TierDevice& dev,
    int lc,                          // cache-layer index on this GPU
    const at::Tensor& selected,      // [rows, topk] int64, device (routing output, never modified)
    int mode,                        // 0 decode (stamps, admission), 1 routed (no stamps, transients)
    uint32_t seq);                   // call sequence; the compute stream waits ready >= seq
```

Shared memory holds U (<= 64 ids: `MAX_BSZN` 8 x top-k <= 8) with counts; dedupe is an O(n^2)
compare in shared memory. The FREE-slot and victim choices are block reductions over packed
64-bit keys (`stamp << 32 | slot`), one pair per admission. Budget: <= 10 us per call at
S = 6,000 with <= 6 misses; if bsz-8 calls with many misses exceed it, the one-pass top-m
selection (FreeToken's `_insert` strategy) replaces the successive reductions. The kernel writes
pointer entries with plain stores; the MoE kernels that read them run later on the same stream.
It harvests the return ring (`ret_head .. ret_tail`) at launch and publishes the record with
`__threadfence_system()` before bumping `rec_head`.

**Promotion.** `cudaMemcpyAsync` H2D on stream A: three sub-range copies (payload + `off_g/u/d`,
4,423,680 B each for V4.1) from an extent-layout RAM slot, one from a compact slot. Sources are
unaligned (odd trellis offsets), destinations 256 B aligned. `cudaMemcpyBatchAsync` is used when
available (CUDA >= 12.8) for a call's copies; FreeToken's trap (a batch silently turns synchronous
when it mixes entries below ~256 KB with large ones on registered memory) does not apply (all
entries are 4.4 MB) and T3.2's window tests it.

**Demotion.** One D2H of the compact VRAM slot into the chosen RAM slot on the D2H stream, after
`cudaStreamWaitEvent` on the H2D event that last read that RAM slot. The slot is returned after
the D2H event completes (polled with `cudaEventQuery` by the tier thread).

**Hooks into existing native code:**

- T0.1 `disk/disk_internal.h:52` `kMaxUserBuffers` 64 -> 1024; `disk/disk_uring.cpp:477-560`:
  a sorted interval table with binary search in `find_buffer` (today a linear scan,
  `disk_uring.cpp:551-560`).
- T0.2 `cpu/moe_handoff.cu:108-142`, `cpu/moe_handoff.h:94-96`: `exl3_flag_write_on(cudaStream_t,
  uint32_t* flag, uint32_t value)` and `exl3_flag_wait_on(cudaStream_t, uint32_t* flag, uint32_t
  value, uint32_t* abort)`; the two existing entry points call them with the current stream
  (behaviour unchanged).
- T0.3 `libtorch/blocksparse_mlp.cpp:481-536`, `blocksparse_mlp.h:187-195`,
  `blocksparse_mlp_bc.h:116`: the body of `run_single_expert_dq` moves into a private
  `dq_impl(y, g_trellis, u_trellis, d_trellis, expert_idx, ...)`; `run_single_expert_dq` passes
  `gates[e]->trellis` etc. (same kernels, same arguments), and the new
  `run_single_expert_dq_views(y, expert_idx, g, u, d, yh, interm, interm_a, out)` passes the
  given views.
- `bindings.cpp:73` (`#include "tier/tier_ext.h"`) and after `bindings.cpp:121`
  (`#include "tier/tier_bc.h"`).

---

## 6. Decode path

Rows <= `EXL3_MOE_TIER_DECODE_ROWS` (default `MAX_BSZN` = 8, `block_sparse_mlp.py:45`).

1. Routing produces `selected_experts` (`block_sparse_mlp.py:1022-1065`).
2. Hook at `block_sparse_mlp.py:1067` (before the CPU split and the compute branches):
   `self.tier_resolve(y, selected_experts, bsz, params)` -> `tier_lookup(mode 0, seq)` on the
   compute stream, then `exl3_flag_wait_on(compute, ready, seq, abort)`.
3. Case all hits: the kernel wrote `ready = seq`; the wait passes as soon as the stream reaches
   it. Cost: the lookup (<= 10 us) plus one memop.
4. Case misses: the tier thread (spinning) sees the record (~1-2 us after the kernel's fence),
   issues the copies per 3.2 (RAM: stream A; SSD: class-1 ticket with a mapped flag, stream B waits
   on it), then `ready = seq` on stream B after A's event. Critical path per missed layer:
   ~10-30 us of host work + 0.24-0.25 ms per RAM-sourced expert at ~53 GB/s + ~1.4 ms per SSD read.
   Gate (DESIGN B3): <= 0.35 ms per RAM-sourced promotion measured end to end.
5. The unchanged compute runs: `run_bszN` (`block_sparse_mlp.py:1355-1359`, coop kernels read the
   pointer tables through `coop_p`) or, under `EXL3_STABLE_ARITHMETIC=1`, the fused path
   (`block_sparse_mlp.py:1204-1262`).
6. Hook at `block_sparse_mlp.py:1361` (before `cpu_split_combine`): `self.tier_after_compute()`
   writes `consumed[0] = seq` (the staging half transients use).
7. Victims are demoted or dropped in the background; returned slots reach the next lookups.

Transients (non-admitted misses) use staging half 0: a copy for call N+1 is issued only after the
lookup of N+1 ran, which the compute stream orders after call N's compute, so no extra wait is
needed; `EXL3_MOE_TIER_VERIFY=1` asserts it.

MTP verify shapes (rows 2-8) take this path, with admission. Rows 9 .. min(prefill rows, DQ
threshold) take the routed mode (7.3).

---

## 7. Prefill path

### 7.1 Mode choice per call

| Rows | Mode | Admission | Stamps | Staging |
|---|---|---|---|---|
| <= `EXL3_MOE_TIER_DECODE_ROWS` (8) | decode | yes | yes | transients in half 0 |
| 9 .. min(`EXL3_MOE_TIER_PREFILL_ROWS` (256) - 1, layer's DQ threshold) | routed | no | no | the call's misses in half 0 |
| >= `EXL3_MOE_TIER_PREFILL_ROWS`, or above the DQ threshold | layer | no | no | every non-VRAM expert of the layer, one layer ahead |

The DQ threshold is the row count above which the resident path would use
`run_single_expert_dq` (`recon.max_rows`, `moe_batch_recon.py:129-135`; 455 for V4.1 with the
default `EXL3_MOE_RECON_TILES=64`), so routed mode never needs host-known addresses.

### 7.2 Layer mode (FreeToken's whole-layer double buffering, adapted)

Per prefill forward, at its first tier layer:

1. **Directory sync.** If the previous tier call was a decode lookup, wait until the tier thread
   has applied every issued record (the GPU reaches the last lookup). Only at decode-to-prefill
   transitions; FreeToken's hidden per-layer sync (issue #501) is exactly what this avoids.
2. For the first two tier layers of the pass, plan and enqueue their staging copies.

Per tier layer L (half `h = L_index % 2`), in `tier_resolve`:

3. Enqueue layer L+1's copies into half `1 - h` (if not yet issued): stream A (and B for SSD
   sources) waits `consumed[1 - h] >=` the seq of the last call that read half `1 - h` (layer
   L-1 in steady state; at the start of a pass, the previous pass's last decode or routed call,
   whose transients live in half 0; the host tracks the last reader of each half), then copies
   every expert of L+1 whose key is not `VALID`/`PINNED` in VRAM (host mirror) from RAM
   (3 sub-copies) or from its slab slot (after its disk flag); stream B writes
   `prefill_ready[1 - h] = seq(L+1)` after A's event.
4. Submit class-2 reads for layer `L + 1 + D` (`prefetch=layer:D`, default `D = 2` when SSD-only
   experts exist) into slab slots, deadline = the expected start of that layer (previous pass's
   per-layer timings), and `promote(1)` the still-queued tickets of layer L+1.
5. Upload layer L's pointer tables (9 KiB for E = 384) from a pinned host ring with one H2D on
   the compute stream: VRAM-held experts -> their slots, the others -> their half-h staging slots.
6. Heat: `tier_heat_add_kernel(selected_experts, lc, w)` on the compute stream (device heat); the
   host adds the same counts when the compute stream's per-layer count D2H lands (a 1.5 KiB copy
   on the D2H stream after an event).
7. Compute stream waits `prefill_ready[h] >= seq(L)`; the unchanged compute runs. Experts above
   the DQ threshold use `run_single_expert_dq_views` with views the planner built (T0.3,
   `block_sparse_mlp.py:1320-1322`).
8. `tier_after_compute` writes `consumed[h] = seq(L)`.

**Where it helps.** A chunk of ~2K rows or more touches every expert of a layer (DESIGN 5.3), so
routing-independent whole-layer copies lose nothing and remove the per-call host plan. Estimated
H2D per 4K chunk: (c) ~61 GiB (~1.2 s at 53 GB/s) against ~3.9 s of compute; (a) ~124 GiB
(~2.5 s); fully hidden with `double` staging, serialized with `single` [E]. Below
`EXL3_MOE_TIER_PREFILL_ROWS` the routed mode copies only the touched experts.

**Differences from FreeToken** (`research_freetoken.md` 3.4): the two buffers are dedicated
(not borrowed from the first 2E pool slots, so decode-hot contents survive any prompt); the
"begin_prefill fence" is the `consumed` flag; hit experts are used in place (no D2D into the
buffer); SSD-only experts arrive through the slab.

### 7.3 Routed mode

The lookup kernel runs in mode 1 (no stamps, no admission): hits keep their slots, misses become
transients in half 0 (up to E - hot of them, which a half holds), the tier thread copies them as at
decode. A forward pass has one row count, so all its tier calls use one mode. No DQ path can be
reached (7.1).

---

## 8. The disk engine as the tier uses it

| Class | Tier traffic | Window | Notes |
|---|---|---|---|
| 0 engram | unchanged (engram rows; decode tickets hold bulk) | `EXL3_DISK_ENGRAM_QD` | the tier never submits class 0 |
| 1 expert demand | decode SSD misses; prefill reads promoted at their layer | `EXL3_DISK_WINDOW_EXPERT` (8M default) | measured next to decode rows: rows p50 750 us, bulk 8.9 GB/s [M] |
| 2 prefetch | prefill read-ahead `prefetch=layer:D` (EDF by deadline); cold fill at load | `EXL3_DISK_WINDOW_PREFETCH` (8M/0; lo under the hold rule) | promoted to 1 when its layer's copies are enqueued |
| 3 refill | RAM promotions by heat (3.3); heat-file warm (T5.1); hot-pin re-choice reads | `EXL3_DISK_WINDOW_REFILL` (8M/0), ages to 2 after `EXL3_DISK_REFILL_AGE_MS` (250) | dropped, not queued, beyond `EXL3_MOE_TIER_REFILL_INFLIGHT` |

- **Entry points:** `Engine::submit_extents` (`disk_engine.h:223-224`) with `ExtentReq{fd, offset,
  length, slot, slot_bytes}` (`disk_engine.h:124-131`), `Options{cls, hold = false, deadline_ns,
  flag, flag_value}` (`disk_engine.h:139-147`), `promote`, `cancel`, `release`, `extent_geometry`
  (`disk_engine.h:226`) for the payload offset. The tier thread calls the C++ API on
  `default_engine()` (`disk_engine.h:277`); Python never sits on the miss path.
- **Registered buffers:** every RAM chunk and slab chunk is registered (READ_FIXED, lookup by
  address, `disk_engine.cpp:1495-1506`), which needs T0.1 beyond 63 GiB.
- **Page cache:** extents are O_DIRECT (`EXL3_DISK_DIRECT=auto`); the engram keeps the page cache
  as its warm tier.
- **Cold fill** (end of load): class 2, deadline 0, in the order hot pins, VRAM pool, RAM tier
  (DESIGN 2.6: heat file, else round robin across layers by expert index); VRAM destinations go
  SSD -> slab -> H2D; RAM destinations are read in place. ~8.9-10.9 GB/s [M]: ~13 s for (c)
  (~71 GiB pool + ~62 GiB RAM), ~20 s for (a) [E], plus overlapped registration.
- **Failures:** a failed read is retried once through a plain `pread` of the extent by the tier
  thread; if that fails too, the tier sets its error state, writes `ready` (so the stream cannot
  hang) and the next `begin_pass` raises `RuntimeError` naming the layer, expert, file and errno.
  At most the token in flight is affected; the churn test injects `EXL3_DISK_FAULT_EIO`.

---

## 9. Memory accounting and refusals

### 9.1 VRAM (per GPU with `cache` layers), at the end of the autosplit (`model_ls.py:329`)

```
room  = min(device_budget[i] - allocated(i) - max_transient[i] - autosplit_margin,
            physical free + reserved - allocated - max_transient[i] - autosplit_margin)
      - staging (2 or 1 x (E - hot) slots)
      - EXL3_MOE_TIER_HEADROOM_MB (default 1024: generator statics and later allocations)
      - device tables (4.1)
S     = floor(room / vram_slot_bytes)              for cache=auto
S     = floor(size / vram_slot_bytes)              for cache=<size> (refused when > room)
```

`device_budget`, `max_transient` and `autosplit_margin` are the loader's own
(`model_ls.py:151-157`, `:197-199`, `:242`); the pool is allocated before
`unset_memory_fraction` (`model_ls.py:330`), so the `-gs` fraction still caps it. Hot pins are
allocated with their layer and counted as its weights. `vram_accounting` (`util/memory.py:372-489`)
gets a new category `tier` (pool, staging, sentinel, tables) next to weights / cache / statics.

### 9.2 Host: the ledger (`util/host_budget.py`), two stages

| Item | Size | Pinned |
|---|---|---|
| static arenas of `stream` / `cpu` / `split` layers (and `-mcl`/`-mcs`) | expert bytes of those layers (headers) | yes (pinned arena) or anonymous |
| RAM tier | `ceil(R / 80)` GiB | yes |
| disk slab | `(8 + prefill read-ahead) x 13,320,192 B`; read-ahead = `(D + 1) x` the largest per-layer SSD-only count at load, capped at 256 slots | yes |
| n-gram / engram budget | `ngram=` | table copies (pinned or anonymous per module) |
| engram pin sets, MoE handoff segment, control pages | measured constants | yes |
| page-cache headroom | `pagecache=` (auto budgets only) | no |
| `EXL3_HOST_MEM_RESERVE_MB` | 2048 | no |

- **Available** = `min(MemAvailable, memory.max - memory.current)` of the process's memory cgroup
  when it has a limit (cgroup v2 `/sys/fs/cgroup/<path>/memory.max`). Configurations (b) and (c) are
  run under `systemd-run --scope -p MemoryMax=128G -p MemorySwapMax=0` so the 128 GiB daily
  budget is enforced by the kernel and seen by the ledger.
- **Stage 1** (before any module loads, in `Model.load_gen` after `_resolve_placement`,
  `model.py:583-585`): explicit sizes exactly; `ram experts=all` estimated from `cuda:<n> cache=`
  when explicit, else from the device budget minus the header sizes of the placed modules.
  Refuses early with the DESIGN 3.10 load-time texts.
- **Stage 2** (at pool finalization, before the RAM tier or slab is allocated): exact R, exact
  slab, fresh `available`. Refusal unloads cleanly; nothing of the tier was pinned yet.
- `check_host_memory` (`util/memory.py:599-615`) keeps its message and becomes a one-item ledger
  check, so the existing callers (`moe_cpu_host.py:234`, `ngram_embedding.py:251-253`) are
  unchanged.

### 9.3 Refusals (exact texts; DESIGN 3.10 texts are kept verbatim)

Parse time: the 40 DESIGN cases (`sim_results/design_grammar_dump.txt`), ASCII digits kept (PR 2's
`[0-9]`, `placement.py:51-53`, and `isascii() and isdigit()`, `placement.py:209`).

Load time, new or adapted:

| Situation | Message |
|---|---|
| link below the threshold | `placement: cuda:0 reaches host memory at 0.38 GB/s (measured at load), below the 2.0 GB/s an expert cache needs (EXL3_MOE_TIER_MIN_LINK_GBS); keep its layers experts=vram, or set EXL3_MOE_TIER_MIN_LINK_GBS=0 to allow it` |
| ineligible layer | `placement: {key} cannot use experts=cache: it needs the fused MoE kernel (EXL3 mul1 codebook shared by gate, up and down; silu, silu_ref or gelu gated, or relu2 gateless; no per-expert biases; unpadded dims; at most 512 experts); use experts=vram, stream or cpu for this layer` |
| two expert geometries on one GPU | `placement: the experts=cache layers on cuda:1 have different expert shapes ({a} and {b}); one expert slot size per GPU is supported` |
| staging does not fit | `placement: cuda:1 has {x} left for the expert cache after the placed modules, the margins and the prefill staging ({y}, EXL3_MOE_TIER_STAGING=double); an expert cache needs at least top-k + spare = {n} slots ({z}); set EXL3_MOE_TIER_STAGING=single, raise -gs on cuda:1, or move layers` |
| cache below minimum / above room | DESIGN 3.10 rows 3-4 |
| `inclusive` too small | `placement: policy=inclusive keeps a RAM copy of every expert the VRAM cache holds, so ram experts= must hold more than the cache's {S} + {spare} slots ({x}); it holds {R} ({y})` |
| `disk experts=off`, tiers too small | DESIGN 3.10 row 1 (with S the non-spare slots) |
| `spare=0` with demotion | `cuda:1 spare=0 leaves no free slot for a demoted victim; spare=0 needs ram demote=off or policy=inclusive` |
| `EXL3_MOE_TIER_DEMOTE=0` with `disk experts=off` | `EXL3_MOE_TIER_DEMOTE=0 would drop VRAM victims, but disk experts=off keeps no other copy of them` |
| budgets above available | DESIGN 3.10 row 2, plus `(the memory cgroup allows {x} more)` when the cgroup is the limit |
| request above a flag, static arenas above a budget, tier below one decode step | DESIGN 3.10 rows 5-8 |
| not Linux | `experts=cache needs the disk engine, which is Linux-only` |
| tensor parallel / no placement | the existing placement refusals (`model.py:434-438`) |

---

## 10. Grammar, CLI, environment, documentation

### 10.1 Words, and the commit that makes each one runnable

| Word | Commit |
|---|---|
| `ram experts=` / `ngram=` / `pagecache=` (legacy layers: caps and requests), sizes and units, dict / JSON / `@file` forms, did-you-mean | T1.3 |
| `hot=<k>` / `hot=<p>%` on `stream` / `cpu` (-> `split` with `cpu = E - k`), `experts=hybrid hot=<k>` (-> `cpu hot=<k>`) | T1.3 |
| `experts=cache`, `hot=` on `cache`, `cuda:<n> cache= spare= evict= admit=`, `ram policy= demote= evict=`, `disk experts=off` (required until T4.1) | T3.3 |
| `disk experts=model|<dir> ngram=model|<dir>|off io=auto|direct|buffered`; `ram experts=<size>|auto` for cache layers (RAM smaller than all) | T4.1 |
| `prefetch=off|auto|layer[:D]` | T4.2 |
| `profile=`, `prefetch=router` | deferred (refused as unknown until then) |

`placement.py` changes, pinned: `EXPERT_MODES`, `_HOST_KIND`, `_PLAN`, `_ATTRS`
(`placement.py:43-50`: add `cache` -> kind `stream` (GPU-computed), plan `("cache", "gpu")`;
`hybrid`); `Experts` gains `hot` (`:56-64`); `Placement` gains `devices`, `ram`, `disk` and prints
them in the canonical order (`:95-146`); `parse` routes a chunk whose first token is `ram`, `disk`
or `cuda:<n>` followed by whitespace to `placement_storage.parse_rule` before the `=` split
(`:181-183`), accepts `hot=` / `prefetch=` (`:193-213`) and dict / JSON / `@path` (`:170-173`); the
end-of-parse checks call `placement_storage.check` (`:232-242`); `expert_plan` returns
`("cache", "gpu", hot)` and maps `stream|cpu hot=k` onto `split` (`:247-262`). The two
`tests/test_placement_.py` refusals that become valid words change their expectations
(`tests/test_placement_.py:75-76`).

### 10.2 CLI and Python

| Knob | CLI (`model_init.py`) | Environment | `config.infer_params` | Default |
|---|---|---|---|---|
| main expert RAM cap | `-er` / `--expert_ram <size>` (after `-mcl`, `model_init.py:64`) | `EXL3_EXPERT_RAM` | `expert_ram` | no cap |
| draft expert RAM cap | `-der` / `--draft_expert_ram <size>` (after `-dmcl`, `model_init.py:129`) | none | `draft_expert_ram` | no cap |
| n-gram / engram RAM | `-ngr` / `--ngram_ram [<size>]` (`model_init.py:69`: `nargs="?"`, `const="all"`) | `EXL3_NGRAM_RAM`; `EXL3_NGRAM_STREAM=0` kept = `all` | `ngram_ram` (`Size`); `ngram_stream_from_disk` becomes a derived property (`config.py:70-73`) | `0` |

`init` mapping at `model_init.py:227-228` (ngram) and `:232-251` (draft config, which copies
`draft_expert_ram` into the draft's `expert_ram`). `-er` without a placement caps `-mcl`/`-mcs`
arenas (and `-der` those of `-dmcl`): refused at stage 1 from headers, guarded again in
`MoeCpuHost.register_layer` (`moe_cpu_host.py:724-800`, cumulative expert bytes per component).

### 10.3 Environment (tier tuning; `doc/env_vars.md`, new section "Expert tiers" before "Multi-GPU", `doc/env_vars.md:1555`)

| Variable | Default | Meaning |
|---|---|---|
| `EXL3_MOE_TIER_DEMOTE` | `1` | master switch: `0` forces `demote=off` (refused with `disk experts=off`) |
| `EXL3_MOE_TIER_STAGING` | `double` | prefill staging: `double` = 2 x (E - hot) slots (overlap), `single` = E - hot (no overlap) |
| `EXL3_MOE_TIER_PREFILL_ROWS` | `256` | rows at which a call switches to layer mode |
| `EXL3_MOE_TIER_DECODE_ROWS` | `8` | rows up to which calls admit (decode mode) |
| `EXL3_MOE_HEAT_HALFLIFE` | `256` | decode tokens per heat halving |
| `EXL3_MOE_HEAT_PREFILL` | `0.0625` | heat per prefill assignment |
| `EXL3_MOE_HEAT_FILE` | unset | save / load heat (T5.1) |
| `EXL3_MOE_TIER_ADMIT_P` | unset | pin the adaptive probability |
| `EXL3_MOE_TIER_ADAPT_EVERY`, `EXL3_MOE_TIER_ADMIT_PMIN` | `2048`, `0.05` | adaptive rule |
| `EXL3_MOE_TIER_SAMPLE` | `16` | RAM eviction sample |
| `EXL3_MOE_TIER_RAM_ADMIT` | `always` | RAM admission of SSD fills (R6) |
| `EXL3_MOE_TIER_REFILL`, `EXL3_MOE_TIER_REFILL_INFLIGHT` | `1`, `2` | class-3 refills |
| `EXL3_MOE_TIER_DISK_SLAB` | `auto` | slab slots |
| `EXL3_MOE_TIER_HEADROOM_MB` | `1024` | VRAM kept free after the pool |
| `EXL3_MOE_TIER_MIN_LINK_GBS` | `2.0` | link refusal threshold (`0` allows) |
| `EXL3_MOE_TIER_SPIN_US` | `-1` | tier-thread spin: `-1` while a pass is active, `0` never, N us |
| `EXL3_MOE_TIER_AFFINITY` | unset | CPU list for the tier thread |
| `EXL3_MOE_TIER_DETERMINISTIC` | `0` | 3.6 |
| `EXL3_MOE_TIER_VERIFY` | `0` | invariant checks after every call (syncs; tests and debugging; precedent `EXL3_MOE_CPU_SWAP_VERIFY`) |
| `EXL3_MOE_TIER_TRACE` | unset | write call records to a file |
| `EXL3_MOE_TRACE` | unset | routing trace of every MoE layer (T2.1) |

### 10.4 Documentation

- `doc/expert_tiers.md` (new, created in T1.2 with the budgets, grown by every later commit): the
  tier model, the three levels, ownership and every policy with its trade-offs and the simulation
  and measurement behind the defaults, the grammar and flags with worked examples (DESIGN T1-T18
  plus (a), (b), (c)), what a load prints and how to read it, the decode and prefill mechanics,
  determinism guarantees, every refusal and what to do about it, tuning order, measured results.
- `doc/placement.md`: Syntax (`doc/placement.md:23-72`) gains the storage rules and the new
  words; Examples (`:74`); How a load applies it (`:147`) gains the tier sizing; Refusals (`:178`);
  Memory, performance and determinism (`:234`).
- `doc/env_vars.md`: new section (10.3); updated `EXL3_HOST_MEM_RESERVE_MB` (`:863`),
  `EXL3_NGRAM_STREAM` (`:1236`), `EXL3_PLACEMENT` (`:1272`), `EXL3_DISK_*` references to the
  tier's classes (`:1419-1553`).
- `doc/disk_engine.md`: registered-buffer limit (T0.1).

---

## 11. The commit series

Author and committer `Fabiano Anemone <anos@hotmail.it>`, neutral messages, no trailers. Tests
and builds on the AI VM only (section 12). Each commit: its docs, its CPU tests passing on the VM,
and where it has GPU tests, a GPU window before the next group starts.

| # | Title | Native | Tests |
|---|---|---|---|
| T0.1 | Disk engine: up to 1024 registered buffers, found by binary search | yes | CPU |
| T0.2 | MoE: stream-targeted flag wait and write for native callers | yes | GPU (short) |
| T0.3 | MoE: per-expert dequant path from explicit trellis tensors | yes | GPU |
| T1.1 | Memory: sizes with units and a host-memory ledger that reads the memory cgroup | | CPU |
| T1.2 | Loading: RAM budgets for routed experts and n-gram tables (-er, -der, -ngr SIZE) | | CPU |
| T1.3 | Placement: ram rule, hot= on stream and cpu, and the dict, JSON and file forms | | CPU |
| T1.4 | DeepSeek-V4.1: engram scale tables resident under the n-gram budget (optional) | | CPU + GPU |
| T2.1 | MoE: optional routing trace (EXL3_MOE_TRACE) | | CPU + GPU |
| T2.2 | Loader: expert extent index and slot geometry | | CPU |
| T2.3 | MoE: expert tier policy core (Python reference) | | CPU |
| T2.4 | MoE: expert tier policy core (native), with its standalone test driver | yes | CPU |
| T3.1 | MoE: tier directory and lookup kernel | yes | GPU |
| T3.2 | MoE: tier runtime (RAM arena, VRAM pool, cold fill, promotion, demotion) | yes | CPU + GPU |
| T3.3 | MoE: experts=cache over a RAM tier that holds every cached expert | | CPU + GPU |
| T3.4 | MoE: whole-layer double-buffered prefill for cached experts | yes | GPU |
| T4.1 | MoE: the SSD tier (disk experts=model, RAM tiers smaller than all) | yes | CPU + GPU |
| T4.2 | MoE: prefill read-ahead of SSD-only experts (prefetch=layer) | yes | GPU |
| T5.1 | MoE: hot pins chosen between generations, and the heat file | | CPU + GPU |
| T6 | MoE: expert tier defaults from the AI VM measurements | | (evidence) |

### T0.1 Disk engine: up to 1024 registered buffers, found by binary search

- **Files:** `exllamav3_ext/disk/disk_internal.h:52` (`kMaxUserBuffers = 1024`);
  `disk_uring.cpp:477-560` (`register_buffer` keeps a sorted `(base, idx)` vector under `c_.mx`;
  `find_buffer` binary-searches it; `unregister_buffer` removes the entry);
  `doc/disk_engine.md` (`EXL3_DISK_REGISTER`, "Memory ownership"); `doc/env_vars.md:1508`.
- **Why here:** configuration (a) registers ~126 RAM chunks (4.3). Belongs to the disk-engine PR;
  first in this branch so the restack can fold it there.
- **Tests:** `tests/disk_engine/disk_engine_test.cpp`: register 300 x 1 MiB buffers, read one
  extent into each (byte-exact), `find_buffer` at base, base + n - 1, one past the end and between
  buffers; unregister every third, reads still exact; ASan/UBSan and TSan builds
  (`tests/disk_engine/build.sh`). `tests/test_disk_engine_.py`: 1,023 registrations of 1 MiB
  through `disk_register_buffer` succeed (indices 1-1023; index 0 is the bounce arena) on io_uring;
  the next returns -1 with the existing warning.

### T0.2 MoE: stream-targeted flag wait and write for native callers

- **Files:** `cpu/moe_handoff.cu:108-142` (bodies move into `exl3_flag_wait_on` /
  `exl3_flag_write_on(cudaStream_t, ...)`; the existing functions pass the current stream),
  `cpu/moe_handoff.h:94-96` (declarations).
- **Tests:** existing `tests/test_moe_cpu_pool_.py` / split tests unchanged (CPU); GPU window
  T3.1's script also runs a two-stream check: stream X waits on a mapped flag written by a host
  thread and by stream Y (memop path and the fallback kernel with `EXL3_MOE_MEMOPS=0`).

### T0.3 MoE: per-expert dequant path from explicit trellis tensors

- **Files:** `libtorch/blocksparse_mlp.cpp:481-536`, `blocksparse_mlp.h:187-195`,
  `blocksparse_mlp_bc.h:114-116` (section 5).
- **Tests (GPU, T3.1 window):** on a synthetic layer, `run_single_expert_dq_views` with views of
  the module's own trellis tensors equals `run_single_expert_dq` bitwise for rows {33, 257, 1024}
  and gated / gateless experts; copying the trellis to another buffer and passing views of the
  copy gives the same bits.

### T1.1 Memory: sizes with units and a host-memory ledger that reads the memory cgroup

- **Files:** new `exllamav3/util/host_budget.py`; `util/memory.py:583-615` (`check_host_memory`
  delegates; message unchanged).
- **Functions:** `parse_size(text, what, allow=("auto", "all"))` (DESIGN 3.6: bare number = GiB;
  `KiB..TiB`, `Ki..Ti` binary, `KB..TB` decimal; bare `K M G T` refused; decimals; ASCII digits
  only), `fmt_size(n)` (largest exact unit, binary first), `memory_available()`,
  `HostLedger.add(name, nbytes, pinned)`, `HostLedger.check(what)`.
- **Tests (`tests/test_host_budget_.py`, torch-free):** `parse_size("48GiB") == 51539607552`;
  `"48GB" == 48000000000`; `"48" == 51539607552`; `"1.5"` prints `1536MiB`; `"512mi"` prints
  `512MiB`; `"48G"` -> `'48G' is ambiguous: write 48GiB (2^30 bytes) or 48GB (10^9 bytes)`;
  `"48GIGS"`, `"٣GiB"`, `"-1GiB"` refused with DESIGN's texts; ledger with MemAvailable 53 GiB and
  items 96 + 6 GiB -> DESIGN 3.10 load-time row 2 exactly; a fake cgroup directory
  (`memory.max` 137438953472, `memory.current` 21474836480) caps available at 108 GiB and the
  message says so; `memory.max` = `max` is ignored; reserve 0 disables the check.

### T1.2 Loading: RAM budgets for routed experts and n-gram tables (-er, -der, -ngr SIZE)

- **Files:** `model_init.py:64-69, 129, 227-251`; `model/config.py:41-73` (`expert_ram`,
  `draft_expert_ram`, `ngram_ram`, derived `ngram_stream_from_disk`); `model/model.py:583-585`
  (stage-1 ledger call for legacy arenas); `model/moe_cpu_host.py:724-800` (cumulative cap guard);
  `modules/ngram_embedding.py:213-253` (a table held in RAM when it fits the budget, streamed
  otherwise; `all` = today's `-ngr`); `doc/expert_tiers.md` (new: budgets), `doc/env_vars.md`,
  `doc/placement.md`.
- **Tests:** `tests/test_host_budget_.py` (flags): `-ngr` bare -> `all`; `-ngr 12GiB` -> Size;
  absent -> `EXL3_NGRAM_RAM`, then `0`; `EXL3_NGRAM_STREAM=0` -> `all`; `-ngr` followed by a
  positional that is not a size -> the size error, not a silent swallow; CLI overrides env;
  `-der` lands in the draft config. `tests/test_moe_expert_policy_.py` style (ast-extracted
  `register_layer` with fake specs): 11 V4.1-shaped layers under `expert_ram = 40GiB` ->
  `-mcl 11 keeps 52.4 GiB of routed experts in RAM, more than --expert_ram 40GiB; raise the cap or
  offload fewer layers` (DESIGN's reference parser printed 52.2 GiB counting trellis only; the
  implementation counts what `_HugeArena.rehome` stores, trellis plus fp16 suh/svh at 64 B
  alignment, 13,315,584 B per expert).

### T1.3 Placement: ram rule, hot= on stream and cpu, and the dict, JSON and file forms

- **Files:** new `exllamav3/model/placement_storage.py` (ported from `disk-tier/placement_tiers.py`
  with `[0-9]` and `isascii()` digits); `model/placement.py` (10.1 pins);
  `modules/block_sparse_mlp_cpu.py:99-121` (`placement_plan` and `cpu_expert_mode` accept the
  mapped `split` from `hot=`); `tests/test_placement_.py:75-76`; docs.
- **Tests:** `tests/test_placement_tiers_.py` (new, torch-free): every DESIGN example that is
  runnable at this commit (T1, T2, T3, T3b, T3c, T4, T16, T18) parses, prints the canonical form
  and round-trips through canonical, dict and JSON forms; `@file` with comments and quotes; every
  DESIGN 3.10 parse-time text for the words present; the 29 refusal cases of
  `tests/test_placement_.py` still refused except the two that became words; `expert_plan`:
  `*=cuda:0 experts=cpu hot=32` with E = 128 -> `("split", "hybrid", 96)`,
  `*=cuda:1 experts=stream hot=64` with E = 384 -> `("split", "stream", 320)`,
  `experts=cache` refused as unknown (until T3.3).

### T1.4 DeepSeek-V4.1: engram scale tables resident under the n-gram budget (optional)

- **Files:** `modules/dsv41_engram.py:302-356, 514-556` (with `ngram_ram >= 2 x 2.86 GiB` the two
  E8M0 scale tables are read once into pinned RAM; the gather reads weight rows from disk and scale
  rows from RAM), `modules/ngram_embedding.py` (the fill order of DESIGN 3.6 `ngram=`).
- **Not needed by (a), (b), (c)** (engrams on the SSD); kept because `-ngr SIZE` must mean the
  same thing for V4.1 as for PLE models. May be dropped from the series if the window time is
  better spent elsewhere.
- **Tests:** CPU plan (which tables fit a budget: 5.72 GiB -> both scale tables; 3 GiB -> none,
  whole tables only); GPU: the engram forward is bitwise equal with scales from RAM and from disk
  (`tests/test_dsv41_engram_fwd_gpu_.py` pattern, both engram layers).

### T2.1 MoE: optional routing trace (EXL3_MOE_TRACE)

- **Files:** new `exllamav3/model/moe_trace.py` (async D2H of `selected_experts` into a pinned
  ring, writer thread, format: header + per call `(layer u16, rows u32, topk u8, ids u16[])`);
  hook `block_sparse_mlp.py:1066` (one line, off by default).
- **Why:** risk 1 (V4.1 locality unmeasured); feeds `sim_presets.py` replays for T6 and the
  kernel replay of T3.1 with real routing.
- **Tests:** CPU round trip of the format; GPU overhead < 1 % of decode time on the serving layout
  (window W-T1).

### T2.2 Loader: expert extent index and slot geometry

- **Files:** new `exllamav3/model/expert_extents.py`; reads `SafetensorsCollection.file_headers`
  and `get_tensor_handle` (`loader/safetensors.py:587-612`, collection level `:1188-1195`, override
  files through `find_stc`).
- **Classes:** `ExpertExtent(path, offset, length, off_g, off_u, off_d, trellis_bytes, contiguous)`;
  `SlotGeometry(vram_slot_bytes, proj_off, ram_slot_bytes)`; `build_extent_index(stc, module)`:
  the span covers every tensor of the expert (trellis, suh, svh, mul1, bias); contiguous iff one
  file and no foreign tensor inside the span; otherwise three trellis-only extents (compact RAM
  layout).
- **Tests (`tests/test_expert_extents_.py`, CPU):** synthetic shards written by the test (odd
  offsets, a 13-byte tensor before each expert, one expert with a foreign tensor inside its span,
  one expert whose down projection is in an override file) -> exact offsets, sub-offsets,
  `contiguous` flags; reading every extent through `disk_read_extents` into 4 KiB-aligned slots
  gives the trellis bytes byte-exact (all three backends); on the real V4.1 checkpoint (headers
  only, CPU): 15,360 decoder experts, every extent 13,315,596 B and contiguous, `vram_slot_bytes`
  13,271,040, `ram_slot_bytes` 13,320,192, the 384 `mtp.*` extents excluded from the text
  component.

### T2.3 MoE: expert tier policy core (Python reference)

- **Files:** new `exllamav3/model/expert_tier_policy.py` (3.1-3.6 verbatim).
- **Tests (`tests/test_expert_tier_policy_.py`, torch-free):**
  1. **Micro-scenarios S1-S9** (3.7): every record, counter and final state exactly.
  2. **Invariants** after every call on 200 random configurations x 5,000 calls (E 16-64, S 4-64,
     spare 1-8, R 0-128, every policy x demote x admit x evict, disk off where R allows): the
     `slot_of` / `slot_key` bijection over VALID and PINNED; FREE + VALID + RETIRING + PINNED = S_total;
     no RETIRING after the host (deterministic mode); exclusive: no `DUP`; lazy: `DUP` only for keys
     VALID in VRAM; inclusive: every VALID key has a RAM entry; disk off: every key VALID or `OWN`,
     SSD reads 0; `swap`: D2H <= admissions sourced from RAM; RAM entries <= R.
  3. **Against the simulator:** the 8 presets of DESIGN 2.3 on the 24 `sim_presets` cases
     (`gen_trace` seed 1234, 3,000 + 6,000 tokens; the case grid copied into
     `tests/expert_tier/sim_presets_ref.py`) with `spare = 0` and instant demotion: SSD reads, H2D
     and D2H per token within 3 % (or 0.2 per token, whichever is larger) of `sim_presets` for every
     case (the cores differ only in fixed-point heat, RNG and tie-breaks).
  4. **Adaptive rule:** with C = 4, R = 12 (pmax 0.25) and the scripted cache lives
     `(lv, lr)` = (1, 3), (1, 1), (3, 1), (9, 1), (1, 99), (1, 99), (1, 99), p goes
     0.25 -> 0.1875 -> 0.1875 -> 0.234375 -> 0.25 (clamped from 0.328125) -> 0.1275 -> 0.065025 ->
     0.05 (clamped from 0.03316275), exactly (doubles).

### T2.4 MoE: expert tier policy core (native), with its standalone test driver

- **Files:** new `exllamav3_ext/tier/tier_policy.h/.cpp`, `tier_ext.h/.cpp` (`tier_policy_replay`
  binding only at this commit), `tier_bc.h`; `bindings.cpp:73, 121` (two includes);
  `tests/expert_tier/tier_policy_test.cpp` + `build.sh` (release, ASan+UBSan, TSan, `-Werror`).
- **Tests:** `tests/test_expert_tier_native_.py`: C++ vs Python core on S1-S9 and on the 200 x
  5,000 random runs: identical records, counters, final directories and RAM tables (exact);
  standalone driver: the same S1-S9 expectations compiled in, 10^6-call soak under ASan/UBSan.

### T3.1 MoE: tier directory and lookup kernel

- **Files:** new `exllamav3_ext/tier/tier_kernels.cuh/.cu`, `tier_device.h/.cpp`; bindings for a
  test-only `TierDevice` driven from Python.
- **Tests (GPU, `tests/expert_tier/tier_kernel_gpu_.py`, window W-T1):**
  1. S1, S2, S6, S9 as device calls (deterministic mode, host side emulated by the C++ core through
     the binding): records and final device arrays equal the expectations of 3.7 exactly.
  2. Replays: Zipf 0.6 and 1.0, stationary and drifting (`gen_trace`), bsz 1, 2 and 8, S in
     {spare + 1, 64, 1,024, 6,000}, evict lru / lfu, admit always / heat / adaptive with pinned p;
     20,000 calls each: every record equals the C++ core's (exact), final `slot_of`, `slot_key`,
     `slot_stamp`, heat equal (exact); pointer tables of every selected expert point at the recorded
     slot + `proj_off`.
  3. Timing: median and p99 lookup time at S = 6,000 with 0, 1, 6 and 48 misses (target <= 10 us
     for <= 6 misses).
  4. T0.2's two-stream flag check; T0.3's DQ views check.

### T3.2 MoE: tier runtime (RAM arena, VRAM pool, cold fill, promotion, demotion)

- **Files:** new `exllamav3_ext/tier/tier_runtime.h/.cpp`; `exllamav3/model/expert_tier.py`
  (`TierDevice`, `ExpertTierSet` without the module hooks; a test constructor
  `ExpertTierSet.for_modules(config, modules, device, pool_slots, ram_slots, cfg)`).
- **Tests (CPU, `tests/test_expert_tier_runtime_.py`):** RAM arena layout (80 slots per chunk,
  4 KiB aligned), registration with the disk engine (indices), cold fill of synthetic shards into
  RAM slots byte-exact, slab state machine, error path with `EXL3_DISK_FAULT_EIO` (pread fallback
  succeeds; with both failing the error names layer, expert, file, errno).
- **Tests (GPU, window W-T2):** runtime on the synthetic model, deterministic mode, S1-S9 end to
  end (kernel + thread + real copies): records equal 3.7; after every call the VRAM slot of every
  VALID key equals the checkpoint trellis bytes (read back), every RAM `OWN` slot equals the
  checkpoint (extent or compact layout); promotion cost per expert (RAM -> VRAM, 3 sub-copies
  and 1 compact copy), misaligned vs aligned source rate (the unmeasured `--host-misalign`
  gate: pass if >= 0.9x, else compaction at cold fill becomes the default), D2H under promotions
  at 1:20 (expect ~0.95 retained, 1.1).

### T3.3 MoE: experts=cache over a RAM tier that holds every cached expert

- **Files (new):** `modules/block_sparse_mlp_tier.py`; completion of `model/expert_tier.py`;
  grammar words (10.1) in `placement.py` / `placement_storage.py`.
- **Hooks (shared files):**
  - `block_sparse_mlp.py:16, 115` (mixin import and base list), `:479` (`self._tier_init_state()`),
    `:800-806` (`load`: `self.tier_maybe_detach(device)` after the CPU hooks: removes the expert
    `Linear`s of an `experts=cache` layer from `self.modules`, as the split does at
    `block_sparse_mlp_cpu.py:633-640`), `:842-845` (`self.tier_attach_experts()` before
    `load_local`: sentinel-view `LinearEXL3` inners with resident suh / svh / mcg / mul1;
    `self.tier_register()` after `load_routing`), `:1067` (`tier_resolve`), `:1361`
    (`tier_after_compute`), `:1283-1290` (a tiered layer must never reach the per-expert graph or
    torch paths: `RuntimeError` naming the layer), `:967-969` (`tier_unload`: restores the expert
    `Linear`s into `self.modules`, drops their inners).
  - `block_sparse_mlp_cpu.py:139-155, 174-188` (a `cache` plan is neither `ram` nor `split`; no
    change needed beyond `_PLAN`, verified by test).
  - `model/model_ls.py:329` (`ExpertTierSet.finalize(device_budget, max_transient,
    autosplit_margin)` then cold fill, before `unset_memory_fraction`), `:349-350, :359, :368-369,
    :376` (`begin_pass(params)` / `end_pass()` next to the MoE hosts' `begin_pass`),
    `:300-307` (the placement OOM message mentions `experts=cache`).
  - `architecture/dsv41/pipeline.py:222, 350-352` (tiers of each half's device begin / end their
    pass in that half's thread).
  - `model/model.py:404-416` (`unload` shuts the tier down after the modules),
    `:419-445` (`_resolve_placement`: tier stage-1 plan and refusals).
  - `util/memory.py:372-489` (`tier` category in `vram_accounting`; `format_vram_report` row).
- **Tests:**
  - CPU (`tests/test_expert_tier_load_.py`, ast-extracted hooks against fake modules, as
    `tests/test_moe_expert_policy_.py` does): `cache` layers detach and restore exactly their
    expert `Linear`s; refusals of 9.3 (texts exact) for ineligible layers, two geometries, slow link
    (mocked probe), staging, spare, demote switch, inclusive, disk off too small; the load printout
    of DESIGN's example T5 with mocked sizes matches `doc/expert_tiers.md` verbatim.
  - GPU **bitwise, synthetic** (`tests/expert_tier/tier_bitwise_gpu_.py`, window W-T3): a tiny
    `qwen3_moe` EXL3 checkpoint written by the test (4 layers, 32 experts, top-4, hidden 512,
    intermediate 512, K = 3 mul1, trellis at odd offsets); each layer loaded once resident and once
    tiered; identical inputs; `torch.equal` on the outputs viewed as int32, for decode rows 1, 2,
    5, 8 and routed rows 9, 33, 200; default arithmetic and `EXL3_STABLE_ARITHMETIC=1`; tier
    configurations: pool of `spare + 1` slots (every call misses), pool of 64, RAM holding all,
    every `policy x demote x admit` (demote forced by disk off) — every output identical to the
    resident one and to each other.
  - GPU **bitwise, real V4.1 layers** (same script, `DSV41_MODEL_DIR`): layers 1 (hash-routed),
    12, 25 and 39; resident on the PRO vs tiered with pools of 16 and 512 slots; inputs `randn`
    scaled to RMS 1 and a skewed input (one row repeated with 1e-3 noise, forcing heavy experts);
    same rows and arithmetic modes as above; exact equality.
  - GPU **churn / stress** (`tests/expert_tier/tier_churn_gpu_.py`, window W-T3): 20,000 decode
    calls over 4 real V4.1 layers, pool 16, RAM = all, bsz random in 1..8, production mode (not
    deterministic), `EXL3_MOE_TIER_VERIFY=1`; a prefill of 257 rows every 500 calls; every output
    compared with the resident reference computed in the same window; at the end: no RETIRING slot,
    return ring balanced, slab idle, zero tickets live, stats consistent (D2H <= retires; hits +
    admits + transients = unique uses).
  - GPU **budget refusal** (window W-T3): `cache=100MiB` -> DESIGN 3.10 "cache below its minimum";
    `cache=` above room -> "does not fit"; stage 2 with a cgroup `MemoryMax` below the RAM tier ->
    the budgets message with the cgroup note; after each refusal the model unloads and the device's
    allocated bytes return to the pre-load value.

### T3.4 MoE: whole-layer double-buffered prefill for cached experts

- **Files:** `tier_runtime.cpp` (prefill planner: directory sync, staging copies one layer ahead,
  pointer upload ring), `block_sparse_mlp_tier.py` (mode choice 7.1, DQ views for the planner's
  addresses), hook `block_sparse_mlp.py:1320-1322` (a tiered layer calls
  `run_single_expert_dq_views`).
- **Tests (GPU, window W-T4):** synthetic and real V4.1 layers, rows 256, 1,024, 4,096 and a
  skewed 4,096 (at least one expert above the DQ threshold of 455 rows, asserted from the counts):
  tiered == resident bitwise, default and stable arithmetic, `double` and `single` staging; a
  multi-layer sequence (8 real layers in forward order) checks that half reuse waits on
  `consumed` (a deliberately slow compute via `EXL3_MOE_TIER_VERIFY=1` must not see its staging
  overwritten: outputs stay bitwise equal); decode after prefill finds the pre-prefill VRAM
  directory unchanged (no admission, no stamps); copy/compute overlap measured (prefill tok/s with
  `double` vs `single`).

### T4.1 MoE: the SSD tier (disk experts=model, RAM tiers smaller than all)

- **Files:** `tier_runtime.cpp` (class-1 demand fills into RAM or slab, RAM admission, refill
  class 3, slab sizing), `placement_storage.py` (`disk experts=model|<dir>`, `io=`, `ram
  experts=<size>|auto` for cache layers; `<dir>` verified header by header), `expert_tier.py`.
- **Tests:** CPU: resolve sizes for DESIGN 7.1's table with this plan's staging (exact slot
  counts for C/52, C/32, D/96, D/64 given fixed inputs); `disk experts=<dir>` with a copy whose
  header differs -> refusal naming the file. GPU (window W-T5): real layers with RAM 32 slots and
  pool 16 (constant SSD traffic): bitwise equality as T3.3; engram decode rows p99 with expert
  demand reads alongside (engine stats) within 1.5x of the no-bulk baseline; fault injection
  (`EXL3_DISK_FAULT_EIO`, `EXL3_DISK_FAULT_DELAY_US=2000`): no hang, error raised at the next
  pass, pread fallback counted.

### T4.2 MoE: prefill read-ahead of SSD-only experts (prefetch=layer)

- **Files:** `tier_runtime.cpp` (class-2 reads D layers ahead with deadlines from the previous
  pass's layer timings, promote at enqueue), grammar `prefetch=`.
- **Tests (GPU, window W-T5):** bitwise equality unchanged for D in {1, 2, 4}; SSD-bound layer
  timeline (engine trace) shows reads of L+D overlapping compute of L; prefill tok/s for (b) at 4K
  and 16K chunks vs DESIGN 7.3.

### T5.1 MoE: hot pins chosen between generations, and the heat file

- **Files:** `expert_tier.py` (quiescent hook: re-choose `hot` pins by heat, move bytes in both
  directions (D2H for the downward half), save the heat file atomically), hook
  `generator/generator.py:575-581` (one call next to `run_pending_swap_sweeps`); fill order from
  `EXL3_MOE_HEAT_FILE` at cold fill.
- **Tests:** CPU: heat file round trip (format: the `.exl3moe`-compatible JSON of per-layer counts
  or a plain versioned JSON; version and model fingerprint checked); GPU: pins change only in
  `on_queue_drained`, outputs bitwise equal before and after a re-choice.

### T6 MoE: expert tier defaults from the AI VM measurements

Last commit, after the B3/B4 runs (section 13): flips any default the measurements beat
(candidates: `spare`, `EXL3_MOE_TIER_STAGING`, `EXL3_MOE_TIER_PREFILL_ROWS`, `admit`, `demote`,
`EXL3_MOE_HEAT_HALFLIFE`, `EXL3_MOE_TIER_RAM_ADMIT`, `EXL3_DISK_WINDOW_EXPERT` if the tier
changes its best value), with the decision table (median of >= 3 idle-gated runs per value) in
the commit message and `doc/expert_tiers.md`. No default changes without a measured win beyond
run-to-run spread.

---

## 12. Where and how the tests run

- **Tree on the VM:** `/root/rnd/tier/src` (rsync of this worktree), builds and caches under
  `/root/rnd/tier/` (`TORCH_EXTENSIONS_DIR=/root/rnd/tier/torch_ext`, `TMPDIR=/root/rnd/tier/tmp`),
  Python `/root/rnd/release/venv/bin/python`. Keep it under ~20 GB; delete build directories and
  synthetic checkpoints after each window.
- **CPU runs:**
  `CUDA_VISIBLE_DEVICES= TORCH_CUDA_ARCH_LIST="8.0;12.0a" CUDA_HOME=/usr/local/cuda MAX_JOBS=8 taskset -c 16-23 nice -n 10 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY -m pytest -q tests/<file>`.
- **GPU runs** only through `/root/rnd/gpu_window.sh <label> <command>` (stops and restores
  `ds41-exl3.service`, holds `/tmp/dsv41_gpu.lock`, log under `/root/rnd/gpu-windows/`), pinned
  the same way, each window minutes long, tests batched per window:

| Window | After | Contents | Est. |
|---|---|---|---|
| W-T1 | T3.1 | kernel replays and timing, flag check (T0.2), DQ views (T0.3), trace overhead (T2.1) | 10 min |
| W-T2 | T3.2 | runtime S1-S9 with real copies, promotion cost, misaligned-source rate, duplex at 1:20 | 10 min |
| W-T3 | T3.3 | bitwise synthetic and real layers, churn, budget refusals | 20 min |
| W-T4 | T3.4 | prefill bitwise, overlap, directory unchanged by prefill | 15 min |
| W-T5 | T4.2 | SSD tier bitwise, engram p99 next to demand reads, faults, read-ahead | 20 min |
| W-T6 | T5.1 | pins and heat file | 10 min |

Device selection: PRO alone `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0` (cuda:0 = PRO);
PRO + CMP as served `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,0` (cuda:0 = CMP,
cuda:1 = PRO). Measurements are idle-gated (no queued request, foreign load < 0.5 core) and
report bytes per token (L2 is 128 MiB on sm_120).

Per D10, a test whose prerequisite is missing (no checkpoint, no GPU, extension import error)
fails with a message naming it, instead of skipping.

---

## 13. The three test configurations

Sizes are [E] until the first load prints them (DESIGN 7.1's pools minus this plan's staging).

| | (a) PRO + RAM for every non-VRAM expert | (b) PRO + RAM <= 128 GiB + SSD tier | (c) PRO + CMP resident + RAM <= 128 GiB, no expert SSD |
|---|---|---|---|
| Devices | `CUDA_VISIBLE_DEVICES=0` (PRO) | same | `1,0` (CMP = cuda:0, PRO = cuda:1) |
| Placement | `*=cuda:0 experts=cache; ram experts=all; disk experts=off` | `*=cuda:0 experts=cache; ram experts=96GiB` | `0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=all; disk experts=off` |
| Memory limit | none (160 GiB VM) | `systemd-run --scope -p MemoryMax=128G -p MemorySwapMax=0` | same as (b) |
| Pool (double staging) | ~65.5 GiB, ~5,300 slots | same | ~71.5 GiB, ~5,780 slots |
| RAM tier | ~10,070 slots, ~126 GiB pinned (needs T0.1) | 96 GiB, 7,680 slots | ~4,980 slots, ~63 GiB |
| SSD-only experts | 0 | ~2,380 (~29.6 GiB) | 0 |
| Engram | disk (`ram ngram=0`) | disk | disk |
| Exercises | promotion + swap demotion at full RAM, prefill of ~125 GiB/chunk from RAM, memory ceiling | demand SSD reads next to engram rows, refill, read-ahead | the daily recipe family; CMP excluded from the tier by the link rule |

If (a) does not fit (`MemAvailable` after the OS is below ~130 GiB), the stage-2 refusal says by
how much; `EXL3_MOE_TIER_STAGING=single` returns ~4.5 GiB of RAM tier (more pool), or (a) runs as
(b) with `ram experts=` at the refusal's suggested size.

**Correctness for B3** (every configuration and every tier combination the grammar allows:
`vram` / `stream` / `cpu` / `split` / `cache`, the three policies, demote off / swap / heat /
all, admit adaptive / heat / always):

- tier state never changes logits: within one device layout, full-model logits of a fixed prompt
  set are bitwise equal across pool sizes (auto, 8 GiB, `spare + 1` slots), RAM sizes, policies
  and demote modes, and between production and deterministic mode — default and stable
  arithmetic;
- quality vs the retained anchors (NLL, top-1, confident / certain) for each layout, where layouts
  differ in which GPU computes which layers (CMP vs PRO arithmetic differs, so only NLL-level
  agreement is expected across (a)/(b) vs (c)).

**Speed for B3/B4** (idle-gated, >= 3 runs, decode and prefill tok/s, bytes per token, tier stats
per run): the knobs the tier adds to the recipe search in family (c): `cuda:1 cache=` (auto vs
explicit), `EXL3_MOE_TIER_STAGING`, `spare` (8, 12, 24, 48), `admit`, `evict`, `policy`,
`demote` and `EXL3_MOE_TIER_DEMOTE`, `hot` (0, 8, 16, 32), `EXL3_MOE_HEAT_HALFLIFE`,
`EXL3_MOE_TIER_PREFILL_ROWS`, the generator chunk size, `EXL3_DSV41_PIPELINE`, stable vs default
arithmetic, `EXL3_MOE_FUSED_PREFILL`, and the disk engine windows (for (b)).

---

## 14. Risks and open questions

1. **V4.1 routing locality is unmeasured.** Every hit rate is synthetic. T2.1 lands early; the
   first GPU window records a trace on the serving layout and replays it through the cores
   before any default is argued.
2. **Staging costs 9.49 GiB of PRO VRAM** (~720 pool slots). If decode hit rates suffer more than
   prefill gains, `single` (4.75 GiB) or the deferred bounded / borrowed staging (behind a bitwise
   gate) replaces it. T3.4 measures both sides.
3. **Host round trip per missed layer.** The spinning tier thread shares 24 vCPUs with the disk
   engine, the engram hasher and other VMs' noise. Mitigation: affinity, the all-hit fast path
   without the host, `cudaMemcpyBatchAsync`. Gate: <= 0.35 ms per RAM-sourced promotion end to end.
4. **Misaligned-source H2D is unmeasured.** Trellis offsets are odd; if the copy engines lose more
   than 10 %, the cold fill compacts RAM slots in place (parallel `memmove`) and decode SSD fills
   stay 3-copy.
5. **Load time and registration.** Cold fill ~13-20 s plus `cudaHostRegister` (~0.2 s per GiB,
   overlapped); io_uring registration needs `RLIMIT_MEMLOCK` (root on the VM).
6. **Pinned memory is unreclaimable** (swap 0): the ledger reads the cgroup; the service must pin
   explicit sizes, not `auto`, on a box other workloads share.
7. **Adaptive admission depends on host timing** in production mode (never logits). Replays pin p.
8. **Numerics.** Any path that reads host-held per-expert tensors for a tiered layer computes
   from the sentinel: guarded by a `RuntimeError`, covered by the skewed prefill tests (DQ views),
   and by `EXL3_MOE_TIER_VERIFY`. A kernel added later that reads `gates[e]->trellis` would bypass
   the redirect: `doc/expert_tiers.md` states the contract (device pointer tables only).
9. **Threads and streams.** One caller thread per GPU; the V4.1 pipeline's stage-2 thread is that
   caller for the PRO. Two pipeline halves with `cache` layers on the same GPU are refused.
10. **Record ring overflow** loses heat only; counted; ring sized for ~36 tokens of 28 layers.
11. **Disk errors** can corrupt at most the token in flight; the next pass raises.
12. **Engram vs expert reads on one virtio queue.** Class windows and the hold rule apply;
    measured 750 us rows next to an 8M class-1 stream [M]; W-T5 re-measures with real traffic.
13. **Decode-to-prefill directory sync** stalls the host until the GPU reaches the last lookup
    (once per prefill forward).
14. **`inclusive` sizing (R7) and RAM admission default (R6)** deviate from the simulator; both are
    options, both documented.
15. **Upstream fit.** `placement.py` grows substantially (the grammar lives there by design); every
    other shared file gets hooks of a few lines (Appendix A). If PR 2's placement is not merged,
    `-mcm cache` (deferred) is the grammar-free path.
16. **Split swap sweep and tier quiescent work** both run in `on_queue_drained`; the tier runs after
    `run_pending_swap_sweeps` (they touch disjoint layers: `split` vs `cache`).
17. **TabbyAPI loads through `load_gen`**: finalization and cold fill run inside `load_gen` (end of
    the autosplit), not in `Model.load` (`model.py:716-730`), so both entry points get them.
18. **The CMP refusal** could surprise users with other slow links; the threshold is a knob and the
    message names it.
19. **MTP head.** Draft MoE layers are never tiered in this series; `-der` caps their static
    arenas only.

Open questions for the user (none blocks T0-T2):

1. Staging default `double` (9.49 GiB of PRO VRAM, full prefill overlap) vs `single`: decide
   after W-T4, or keep `double` until then?
2. RAM admission default: the simulated rule (R6) or DESIGN's heat-gated text?
3. T1.4 (engram scales resident under `-ngr SIZE`): keep in the series, or drop since all three
   configurations keep the engram on the SSD?

---

## Appendix A. Hook index (shared files; line numbers at `bfba0e0`)

| File:line | Commit | Change |
|---|---|---|
| `exllamav3/exllamav3_ext/disk/disk_internal.h:52` | T0.1 | `kMaxUserBuffers` 1024 |
| `exllamav3/exllamav3_ext/disk/disk_uring.cpp:477-560` | T0.1 | sorted registration table, binary search |
| `exllamav3/exllamav3_ext/cpu/moe_handoff.cu:108-142`, `moe_handoff.h:94-96` | T0.2 | `_on(stream)` variants |
| `exllamav3/exllamav3_ext/libtorch/blocksparse_mlp.cpp:481-536`, `.h:187-195`, `_bc.h:114-116` | T0.3 | `dq_impl`, `run_single_expert_dq_views` |
| `exllamav3/exllamav3_ext/bindings.cpp:73, 121` | T2.4 | two includes |
| `exllamav3/util/memory.py:583-615` | T1.1 | `check_host_memory` via the ledger |
| `exllamav3/util/memory.py:372-489`, `:498-580` | T3.3 | `tier` VRAM category and report row |
| `exllamav3/model_init.py:64, 69, 129, 227-251` | T1.2 | `-er`, `-ngr [SIZE]`, `-der`, mapping |
| `exllamav3/model/config.py:41-73` | T1.2 | `expert_ram`, `draft_expert_ram`, `ngram_ram` |
| `exllamav3/model/placement.py:43-53, 56-64, 95-146, 165-244, 247-262` | T1.3, T3.3, T4.1, T4.2 | grammar (10.1) |
| `exllamav3/model/moe_cpu_host.py:724-800` | T1.2 | cumulative `--expert_ram` guard |
| `exllamav3/modules/ngram_embedding.py:213-253` | T1.2 | budget-sized table residency |
| `exllamav3/modules/dsv41_engram.py:302-356, 514-556` | T1.4 | resident scale tables |
| `exllamav3/modules/block_sparse_mlp.py:1066` | T2.1 | routing trace (one line) |
| `exllamav3/modules/block_sparse_mlp.py:16, 115, 479` | T3.3 | mixin, init state |
| `exllamav3/modules/block_sparse_mlp.py:800-806, 842-845` | T3.3 | detach / attach / register |
| `exllamav3/modules/block_sparse_mlp.py:1067, 1361` | T3.3 | `tier_resolve`, `tier_after_compute` |
| `exllamav3/modules/block_sparse_mlp.py:1283-1290, 1320-1322` | T3.3, T3.4 | guard, DQ views |
| `exllamav3/modules/block_sparse_mlp.py:967-969` | T3.3 | `tier_unload` |
| `exllamav3/modules/block_sparse_mlp_cpu.py:99-121` | T1.3 | `hot=` mapped onto `split` |
| `exllamav3/model/model_ls.py:300-307, 329, 349-350, 359, 368-369, 376` | T3.3 | message, finalize + cold fill, passes |
| `exllamav3/model/model.py:404-416, 419-445, 583-585` | T1.2, T3.3 | stage-1 ledger, tier plan, shutdown |
| `exllamav3/architecture/dsv41/pipeline.py:222, 350-352` | T3.3 | per-half tier passes |
| `exllamav3/generator/generator.py:575-581` | T5.1 | quiescent hook |
| `tests/test_placement_.py:75-76` | T1.3 | two refusals become words |
| `doc/placement.md:23-72, 74, 147, 178, 234` | T1.3 - T4.2 | grammar documentation |
| `doc/env_vars.md:863, 1236, 1272, 1419-1553, 1555` | T1.2 - T4.2 | knobs, new "Expert tiers" section |

New files: `exllamav3/util/host_budget.py`, `exllamav3/model/placement_storage.py`,
`exllamav3/model/expert_extents.py`, `exllamav3/model/expert_tier_policy.py`,
`exllamav3/model/expert_tier.py`, `exllamav3/model/moe_trace.py`,
`exllamav3/modules/block_sparse_mlp_tier.py`, `exllamav3/exllamav3_ext/tier/*`,
`doc/expert_tiers.md`, `tests/test_host_budget_.py`, `tests/test_placement_tiers_.py`,
`tests/test_expert_extents_.py`, `tests/test_expert_tier_policy_.py`,
`tests/test_expert_tier_native_.py`, `tests/test_expert_tier_runtime_.py`,
`tests/test_expert_tier_load_.py`, `tests/expert_tier/*` (GPU scripts, simulator reference, C++
driver).

## Appendix B. What ran for this plan

| Run | Where | Result |
|---|---|---|
| `micro_scenarios.py` (scratch reference of section 3) | AI VM, `/root/rnd/tier/plan/`, CPU only, `taskset -c 16-23 nice -n 10` | S1-S9 as in 3.7; its first version stamped hits in call order and retired a hit of the same call (the FreeToken barrier bug), fixed by the two-phase rule of 3.1 |
| duplex and disk results read | AI VM, `/root/rnd/bench-window/` | 1.1 |

No GPU was used, the service and the disk were untouched, nothing was committed.
