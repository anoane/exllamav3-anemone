#pragma once

// The VRAM expert cache's device side (doc/expert_tiers.md, "The VRAM cache"): the directory of one
// GPU's slot pool in device memory, the mapped control page the tier thread shares with it, and the
// two kernels every MoE call of an experts=cache layer runs on the compute stream, right after
// routing:
//
//   lookup  (1 CTA)  dedupes the call's experts, adds their heat, stamps its hits (decode), decides
//                    every miss exactly as the policy core does (tier_policy.h,
//                    VramDirectory::lookup: admission, the lowest free slot, one victim per
//                    admission past S - spare, in-place replacement with spare=0, transients into
//                    staging), rewrites the layer's per-expert trellis pointer tables (the only thing
//                    the unchanged MoE kernels read) and publishes the call's record into a ring in
//                    mapped host memory. Slots the host freed come back through a return ring that
//                    the next lookup harvests.
//   fetch   (G + 1 CTAs, cooperative)  moves the call's bytes: when the call missed (or always, in
//                    deterministic mode) it executes the copy commands the tier thread appends to the
//                    plan ring for this call (promotions host to device, rescues device to device,
//                    demotions device to host), with the SMs, until the call's END command; an
//                    all-hit call returns at once.
//
// The tier thread never calls the CUDA API while the model runs: it reads records and writes
// commands, returned slots and completion checks with plain loads and stores on mapped memory. A
// thread that holds the driver's locks in a synchronous call (a pageable copy waiting for the stream)
// therefore cannot hold up the copies the stream is waiting for. SM copies from page-locked memory
// run at the copy engines' rate on the AI VM (52 GB/s host to device; 37 GB/s device to host).
//
// Plain C++ (no CUDA types but cudaStream_t): included by the kernels (tier_kernels.cu) and by the
// host runtime (tier_gpu.cpp).

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace exl3_tier
{

constexpr int kLookupThreads = 1024;
constexpr int kLookupMaxExperts = 512;          // MAX_EXPERTS of libtorch/blocksparse_mlp.h
constexpr int kFetchThreads = 256;
constexpr uint32_t kFetchChunk = 65536;         // bytes of one command a CTA copies at a time

// Record layout in the ring (u32 words). A record is a header and one entry per unique expert of
// the call, in the order of first occurrence (Record / Entry of tier_policy.h)
constexpr int kRecHeaderWords = 8;              // nbytes, seq, lc, mode | needs_host << 8, n, tokens lo, tokens hi, 0
constexpr int kRecEntryWords = 10;              // key, kind | inplace << 8 | starved << 9, cnt, dst, heat,
                                                // vkey, vslot, vstamp, vheat, 0
constexpr uint32_t kRecWrap = 0xFFFFFFFFu;      // in place of nbytes: the ring continues at its start

// Device counters (u64 each, device memory)
enum DevCounter : int
{
    kDcAccess = 0, kDcOccupied, kDcRetHead, kDcRecHead,
    kDcNeedsHost,           // the last lookup's call has work for the host (read by its fetch)
    kDcFetchStart,          // plan index where the next fetch starts
    kDcFetchRead,           // plan index the reader CTA has taken so far
    kDcFetchChunk,          // global chunk counter (spreads commands over the worker CTAs)
    kDcFetchTail,           // commands published to the workers (device ring)
    kDcFetchArrived,        // workers that read the start index (a grid barrier)
    kDcFetchAbort,          // the reader saw the host's abort
    kDcCount
};

// First error a kernel met (TierCtl.error); the runtime raises it at the next call
enum DevError : uint32_t
{
    kErrNone = 0,
    kErrExpertRange = 1,        // a selected expert outside [0, E)
    kErrStaging = 2,            // more transients than staging slots
    kErrRing = 3,               // the tier thread stopped consuming records (abort)
    kErrPlan = 4,               // a fetch met a command of another call, or waited too long
};

// The mapped control page (cudaHostAlloc mapped + portable): every field the two sides exchange on
// its own 64-byte line
struct TierCtl
{
    uint64_t plan_tail; uint64_t _p0[7];        // host: commands appended to the plan ring
    uint32_t abort;     uint32_t _p1[15];       // host: the tier failed; kernels stop waiting on it
    uint64_t rec_head;  uint64_t _p2[7];        // kernel: bytes published into the record ring
    uint64_t rec_tail;  uint64_t _p3[7];        // host: bytes consumed
    uint32_t ret_tail;  uint32_t _p4[15];       // host: slots pushed into the return ring
    uint64_t p_q32;     uint64_t _p5[7];        // host: threshold of admit=adaptive (2^32 = always)
    uint32_t error;     uint32_t _p6[15];       // kernel: first DevError
    uint32_t overflow;  uint32_t _p7[15];       // kernel: all-hit records dropped on a full ring
    uint64_t layer_done; uint64_t _p8[7];       // copy stream: the last layer-mode copy batch completed
};
static_assert(sizeof(TierCtl) <= 4096, "the control page's fields fit its first 4 KiB");
constexpr size_t kCtlBytes = 4096;              // then the return ring, the record ring, the plan ring,
                                                // the plan's completion flags

// One copy command (host plan ring, 40 bytes). bytes a multiple of 16, dst 16-byte aligned, src any
// alignment. dep: a token (command index + 1) that must complete first, 0 none. Tokens and the
// completion flags are 64-bit: a command index never wraps
enum PlanKind : uint32_t { kPlanCopy = 0, kPlanEnd = 1 };
struct PlanCmd
{
    uint64_t src;
    uint64_t dst;
    uint64_t dep;
    uint32_t bytes;
    uint32_t seq;
    uint32_t kind;
    uint32_t pad;
};
static_assert(sizeof(PlanCmd) == 40, "the fetch kernel reads a PlanCmd as ten 32-bit words");

// The same command as the reader CTA hands it to the workers (device ring), with its first global chunk
struct FetchCmd
{
    uint64_t src;
    uint64_t dst;
    uint64_t base;
    uint64_t dep;
    uint32_t bytes;
    uint32_t seq;
    uint32_t kind;
    uint32_t pad;
};

// Everything a lookup reads: device pointers (directory, tables) and mapped pointers (control page)
struct TierKParams
{
    uint8_t* state;                             // [S + pins] FREE / VALID / RETIRING / PINNED
    int32_t* key;                               // [S + pins] lc * E + e, or -1
    uint32_t* stamp;                            // [S + pins] seq of the last decode use
    const int64_t* addr;                        // [S + pins] slot addresses (pins after the pool)
    int32_t* slot_of;                           // [L * E] VALID / PINNED slot of a key, or -1
    uint32_t* heat_v;                           // [L * E] 16.16
    uint32_t* heat_ep;                          // [L * E] epoch of the last write
    uint32_t* free_bits;                        // [ceil(S / 32)] FREE pool slots
    uint64_t* dctr;                             // [kDcCount]
    const int64_t* staging;                     // [nstage] staging slot addresses (half 0)
    const int64_t* tables;                      // [L * P] device addresses of the layers' int64 pointer tables
    int64_t proj_off[3];                        // trellis offsets of the projections in a slot
    TierCtl* ctl;                               // mapped control page (device address)
    uint32_t* ret_ring;                         // [ret_n] mapped
    uint8_t* rec_ring;                          // [rec_bytes] mapped
    uint64_t rec_bytes;
    uint32_t ret_n;                             // power of two, >= S + pins
    int P, S, pins, spare, nstage, E, L;
    int admit, evict;                           // Admit, Evict of tier_policy.h
    uint64_t seed;
    uint32_t halflife;
    uint32_t prefill_inc;
    int deterministic;                          // every call's fetch waits for the host
};

// Everything a fetch reads
struct FetchParams
{
    TierCtl* ctl;                               // mapped
    const PlanCmd* plan;                        // [plan_n] mapped
    uint64_t* done_host;                        // [plan_n] mapped: command i complete -> i + 1
    FetchCmd* dring;                            // [plan_n] device
    uint64_t* ddone;                            // [plan_n] device: command i complete -> i + 1
    uint32_t* dcnt;                             // [plan_n] device: workers done with command i
    uint64_t* dctr;                             // the lookup's counters (kDcNeedsHost, kDcFetch*)
    uint32_t plan_n;                            // power of two
    int workers;                                // G
    int deterministic;
};

// Launch one lookup of cache layer lc on `stream`: ids = the call's selected experts (int64,
// device, row-major, n of them), mode 0 decode (stamps, admission), 1 routed (no stamps, misses
// become transients), 2 layer (heat and the record only: the host staged the layer and wrote its
// tables) or 3 predict (the record only: the experts a later layer's router picks, for the disk
// read-ahead of prefetch=router), seq the call's sequence number, tokens the decode tokens so far
// (heat epoch)
void tier_lookup_launch(const TierKParams& p, const int64_t* ids, int n, int lc, int mode, uint32_t seq,
                        uint64_t tokens, cudaStream_t stream);

// *flag = value on `stream` once everything before it there has run (a one-thread kernel)
void tier_flag_write_launch(uint64_t* flag, uint64_t value, cudaStream_t stream);

// Launch the call's fetch on `stream` right after its lookup (a cooperative launch of workers + 1 CTAs)
cudaError_t tier_fetch_launch(const FetchParams& f, uint32_t seq, cudaStream_t stream);

}  // namespace exl3_tier
