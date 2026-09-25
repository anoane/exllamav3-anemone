// The expert cache's lookup and fetch kernels; see tier_device.h. Every decision of the lookup
// reproduces VramDirectory::lookup (tier_policy.cpp) exactly: the same dedup order, heat, stamps,
// admissions, slots, victims and transients, so a record from here equals the policy core's for the
// same calls. The fetch executes the tier thread's copy commands for the call.

#include "tier_device.h"

#include <climits>
#include <cuda_runtime.h>

namespace exl3_tier
{

namespace
{

constexpr int T = kLookupThreads;
constexpr int NW = T / 32;

enum : uint8_t { S_FREE = 0, S_VALID = 1, S_RETIRING = 2, S_PINNED = 3 };
enum : uint8_t { K_HIT = 0, K_ADMIT = 1, K_TRANSIENT = 2, K_MISS = 255 };
enum : int { A_ADAPTIVE = 0, A_HEAT = 1, A_ALWAYS = 2 };
enum : int { E_LRU = 0, E_LFU = 1 };

constexpr uint64_t kGoldenD = 0x9E3779B97F4A7C15ull;

__device__ __forceinline__ uint32_t ld_acquire_u32(const uint32_t* p)
{
    uint32_t v;
    asm volatile("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint64_t ld_acquire_u64(const uint64_t* p)
{
    uint64_t v;
    asm volatile("ld.global.acquire.sys.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_release_u32(uint32_t* p, uint32_t v)
{
    asm volatile("st.global.release.sys.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st_release_u64(uint64_t* p, uint64_t v)
{
    asm volatile("st.global.release.sys.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ void st_volatile_u32(uint32_t* p, uint32_t v)
{
    *(volatile uint32_t*) p = v;
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t x)
{
    uint64_t z = x + kGoldenD;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Heat of key k at `epoch` (Heat::read)
__device__ __forceinline__ uint32_t heat_read(const TierKParams& p, int32_t k, uint32_t epoch)
{
    uint32_t v = p.heat_v[k];
    int64_t d = (int64_t) epoch - (int64_t) p.heat_ep[k];
    return d > 0 ? v >> (d < 31 ? (uint32_t) d : 31u) : v;
}

// Errors collect in shared memory and reach the (mapped) control page with plain stores from one
// thread: device atomics on host memory are not atomic on every platform
__device__ __forceinline__ void first_error(uint32_t* s_err, uint32_t e)
{
    atomicCAS(s_err, 0u, e);
}

// A victim candidate: lexicographic (a, b); LRU a = stamp << 32 | slot, LFU a = heat << 32 | stamp,
// b = slot
struct Cand
{
    uint64_t a;
    uint32_t b;
};

__device__ __forceinline__ bool cand_less(const Cand& x, const Cand& y)
{
    return x.a < y.a || (x.a == y.a && x.b < y.b);
}

__device__ __forceinline__ void cand_min_shfl(Cand& c, uint32_t& w, int o)
{
    Cand x;
    x.a = __shfl_down_sync(0xFFFFFFFFu, c.a, o);
    x.b = __shfl_down_sync(0xFFFFFFFFu, c.b, o);
    uint32_t y = __shfl_down_sync(0xFFFFFFFFu, w, o);
    if (cand_less(x, c)) c = x;
    if (y < w) w = y;
}

// Block-wide: the lowest FREE pool slot (need & 1) and the victim the eviction rule picks now
// (need & 2): the least (stamp, slot) of the VALID pool slots not stamped by this call (LRU), or the
// least (decayed heat, stamp, slot) of them (LFU). Every thread calls it; results land in *free /
// *victim (-1: none)
__device__ void pick(const TierKParams& p, int need, uint32_t seq, uint32_t epoch, int* s_free, int* s_victim,
                     Cand* s_c, uint32_t* s_w)
{
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    Cand best { ~0ull, ~0u };
    uint32_t wbest = ~0u;
    if (need & 2)
    {
        for (int s = tid; s < p.S; s += T)
        {
            if (p.state[s] != S_VALID) continue;
            uint32_t st = p.stamp[s];
            if (st >= seq) continue;
            Cand c;
            if (p.evict == E_LRU)
            {
                c.a = ((uint64_t) st << 32) | (uint32_t) s;
                c.b = 0;
            }
            else
            {
                c.a = ((uint64_t) heat_read(p, p.key[s], epoch) << 32) | st;
                c.b = (uint32_t) s;
            }
            if (cand_less(c, best)) best = c;
        }
    }
    if (need & 1)
    {
        const int nw = (p.S + 31) >> 5;
        for (int w = tid; w < nw; w += T)
            if (p.free_bits[w])
            {
                wbest = (uint32_t) w;
                break;
            }
    }
    for (int o = 16; o; o >>= 1) cand_min_shfl(best, wbest, o);
    if (lane == 0)
    {
        s_c[warp] = best;
        s_w[warp] = wbest;
    }
    __syncthreads();
    if (warp == 0)
    {
        best = s_c[lane];
        wbest = s_w[lane];
        for (int o = 16; o; o >>= 1) cand_min_shfl(best, wbest, o);
        if (lane == 0)
        {
            int v = -1;
            if (best.a != ~0ull) v = p.evict == E_LRU ? (int) (uint32_t) best.a : (int) best.b;
            *s_victim = v;
            *s_free = wbest == ~0u ? -1 : (int) (wbest * 32 + (uint32_t) (__ffs(p.free_bits[wbest]) - 1));
        }
    }
    __syncthreads();
}

// Inclusive warp scan
__device__ __forceinline__ int warp_scan(int v)
{
    const int lane = threadIdx.x & 31;
    for (int o = 1; o < 32; o <<= 1)
    {
        int x = __shfl_up_sync(0xFFFFFFFFu, v, o);
        if (lane >= o) v += x;
    }
    return v;
}

__global__ void __launch_bounds__(T) tier_lookup_kernel(const TierKParams p, const int64_t* __restrict__ ids,
                                                        const int n, const int lc, const int mode, const uint32_t seq,
                                                        const uint64_t tokens)
{
    __shared__ int s_first[kLookupMaxExperts];
    __shared__ uint32_t s_cnt[kLookupMaxExperts];
    __shared__ int16_t s_u[kLookupMaxExperts];         // unique experts, first occurrence order
    __shared__ int32_t s_slot[kLookupMaxExperts];      // hit: its slot; then the entry's dst
    __shared__ uint8_t s_kind[kLookupMaxExperts];
    __shared__ uint8_t s_flags[kLookupMaxExperts];     // 1 inplace, 2 starved
    __shared__ uint32_t s_heat[kLookupMaxExperts];
    __shared__ int32_t s_vkey[kLookupMaxExperts];
    __shared__ int32_t s_vslot[kLookupMaxExperts];
    __shared__ uint32_t s_vstamp[kLookupMaxExperts];
    __shared__ uint32_t s_vheat[kLookupMaxExperts];
    __shared__ int s_scan[NW];
    __shared__ Cand s_c[NW];
    __shared__ uint32_t s_w[NW];
    __shared__ uint32_t s_ret_head, s_ret_tail;
    __shared__ int s_nu, s_j, s_need, s_adm, s_free, s_victim, s_occ, s_stg, s_nh, s_ok;
    __shared__ uint64_t s_acc0, s_pos;
    __shared__ uint32_t s_err;

    const int tid = threadIdx.x;
    const int E = p.E;
    const uint32_t epoch = (uint32_t) (tokens / p.halflife);

    // Slots the host returned since the last call become FREE
    if (tid == 0)
    {
        s_ret_head = (uint32_t) p.dctr[kDcRetHead];
        s_ret_tail = ld_acquire_u32(&p.ctl->ret_tail);
        s_err = 0;
    }
    for (int e = tid; e < E; e += T)
    {
        s_first[e] = INT_MAX;
        s_cnt[e] = 0;
    }
    __syncthreads();
    {
        const uint32_t h = s_ret_head, cnt = s_ret_tail - s_ret_head;
        for (uint32_t i = tid; i < cnt; i += T)
        {
            int32_t s = (int32_t) ld_acquire_u32(p.ret_ring + ((h + i) & (p.ret_n - 1)));
            if (s < 0 || s >= p.S) continue;
            p.state[s] = S_FREE;
            p.key[s] = -1;
            p.stamp[s] = 0;
            atomicOr(p.free_bits + (s >> 5), 1u << (s & 31));
        }
    }

    // Dedup: count and first position of every expert
    for (int i = tid; i < n; i += T)
    {
        int64_t e = ids[i];
        if (e < 0 || e >= E)
        {
            first_error(&s_err, kErrExpertRange);
            continue;
        }
        atomicMin(&s_first[e], i);
        atomicAdd(&s_cnt[e], 1u);
    }
    __syncthreads();
    if (tid == 0) p.dctr[kDcRetHead] = s_ret_tail;

    // Unique experts in order of first occurrence: a block scan over the positions
    {
        const int chunk = (n + T - 1) / T;
        const int i0 = min(n, tid * chunk), i1 = min(n, i0 + chunk);
        int c = 0;
        for (int i = i0; i < i1; ++i)
        {
            int64_t e = ids[i];
            if (e >= 0 && e < E && s_first[e] == i) ++c;
        }
        int incl = warp_scan(c);
        if ((tid & 31) == 31) s_scan[tid >> 5] = incl;
        __syncthreads();
        if (tid < 32)
        {
            int v = s_scan[tid];
            v = warp_scan(v);
            s_scan[tid] = v;
        }
        __syncthreads();
        int o = (tid >= 32 ? s_scan[(tid >> 5) - 1] : 0) + incl - c;
        for (int i = i0; i < i1; ++i)
        {
            int64_t e = ids[i];
            if (e >= 0 && e < E && s_first[e] == i) s_u[o++] = (int16_t) e;
        }
        if (tid == T - 1) s_nu = o;
    }
    __syncthreads();
    const int nu = s_nu;

    // Heat, then the protection phase: every hit of a decode call is stamped before any miss looks
    // for a victim, so a call never evicts an expert it uses
    const uint32_t inc = mode == 0 ? (1u << 16) : p.prefill_inc;
    for (int j = tid; j < nu; j += T)
    {
        const int e = s_u[j];
        const int32_t k = lc * E + e;
        uint64_t x = (uint64_t) heat_read(p, k, epoch) + (uint64_t) s_cnt[e] * inc;
        uint32_t hv = x > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t) x;
        p.heat_v[k] = hv;
        p.heat_ep[k] = epoch;
        s_heat[j] = hv;
        int32_t s = p.slot_of[k];
        s_slot[j] = s;
        s_flags[j] = 0;
        s_vkey[j] = -1;
        s_vslot[j] = -1;
        s_vstamp[j] = 0;
        s_vheat[j] = 0;
        if (s >= 0)
        {
            s_kind[j] = K_HIT;
            if (mode == 0) p.stamp[s] = seq;
        }
        else s_kind[j] = K_MISS;
    }
    __syncthreads();

    if (mode != 0)
    {
        // Routed (prefill below the layer mode): no admission, misses stage as transients
        if (tid == 0)
        {
            int stg = 0;
            for (int j = 0; j < nu; ++j)
                if (s_kind[j] != K_HIT)
                {
                    s_kind[j] = K_TRANSIENT;
                    s_slot[j] = stg++;
                }
            s_nh = stg > 0;
            s_stg = stg;
            p.dctr[kDcAccess] += (uint64_t) nu;
        }
        __syncthreads();
    }
    else
    {
        // Decode: the misses in order, each with at most one block-wide pick
        const int C = p.S - p.spare;
        if (tid == 0)
        {
            s_j = 0;
            s_occ = (int) p.dctr[kDcOccupied];
            s_acc0 = p.dctr[kDcAccess];
            s_stg = 0;
            s_nh = 0;
        }
        __syncthreads();
        while (true)
        {
            if (tid == 0)
            {
                int j = s_j;
                while (j < nu && s_kind[j] == K_HIT) ++j;
                s_j = j;
                int need = 0;
                if (j < nu)
                {
                    const uint64_t access = s_acc0 + (uint64_t) j + 1;
                    int adm;
                    if (s_occ < C) adm = 1;
                    else if (p.admit == A_ALWAYS) adm = 1;
                    else if (p.admit == A_HEAT) adm = 2;
                    else
                    {
                        const uint64_t pq = *(volatile uint64_t*) &p.ctl->p_q32;
                        adm = (splitmix64(p.seed ^ (access * kGoldenD)) >> 32) < pq ? 1 : 0;
                    }
                    s_adm = adm;
                    if (adm) need = 1 | (s_occ >= C ? 2 : 0);
                }
                s_need = need;
            }
            __syncthreads();
            if (s_j >= nu) break;
            const int need = s_need;
            if (need) pick(p, need, seq, epoch, &s_free, &s_victim, s_c, s_w);
            if (tid == 0)
            {
                const int j = s_j;
                const int32_t k = lc * E + s_u[j];
                const int v = (need & 2) ? s_victim : -1;
                int adm = s_adm;
                if (adm == 2) adm = v >= 0 && s_heat[j] >= heat_read(p, p.key[v], epoch);
                bool placed = false;
                if (adm)
                {
                    const int f = (need & 1) ? s_free : -1;
                    if (f >= 0)
                    {
                        p.free_bits[f >> 5] &= ~(1u << (f & 31));
                        p.state[f] = S_VALID;
                        p.key[f] = k;
                        p.stamp[f] = seq;
                        p.slot_of[k] = f;
                        ++s_occ;
                        s_kind[j] = K_ADMIT;
                        s_slot[j] = f;
                        if (s_occ > C && v >= 0)
                        {
                            const int32_t vk = p.key[v];
                            s_vkey[j] = vk;
                            s_vslot[j] = v;
                            s_vstamp[j] = p.stamp[v];
                            s_vheat[j] = heat_read(p, vk, epoch);
                            p.state[v] = S_RETIRING;
                            p.slot_of[vk] = -1;
                            --s_occ;
                        }
                        placed = true;
                    }
                    else if (p.spare == 0 && v >= 0)
                    {
                        const int32_t vk = p.key[v];
                        s_kind[j] = K_ADMIT;
                        s_slot[j] = v;
                        s_flags[j] |= 1;
                        s_vkey[j] = vk;
                        s_vslot[j] = v;
                        s_vstamp[j] = p.stamp[v];
                        s_vheat[j] = heat_read(p, vk, epoch);
                        p.slot_of[vk] = -1;
                        p.key[v] = k;
                        p.stamp[v] = seq;
                        p.slot_of[k] = v;
                        placed = true;
                    }
                    if (!placed) s_flags[j] |= 2;
                }
                if (!placed)
                {
                    s_kind[j] = K_TRANSIENT;
                    s_slot[j] = s_stg++;
                }
                s_nh = 1;
                s_j = j + 1;
            }
            __syncthreads();
        }
        if (tid == 0)
        {
            p.dctr[kDcAccess] = s_acc0 + (uint64_t) nu;
            p.dctr[kDcOccupied] = (uint64_t) s_occ;
        }
    }
    if (tid == 0 && s_stg > p.nstage) first_error(&s_err, kErrStaging);
    __syncthreads();

    // The layer's pointer tables: every expert of the call at its slot (or staging slot)
    for (int j = tid; j < nu; j += T)
    {
        const int e = s_u[j];
        int64_t base;
        if (s_kind[j] == K_TRANSIENT) base = p.staging[s_slot[j] < p.nstage ? s_slot[j] : 0];
        else base = p.addr[s_slot[j]];
        for (int q = 0; q < p.P; ++q)
        {
            int64_t* tab = reinterpret_cast<int64_t*>(p.tables[lc * p.P + q]);
            tab[e] = base + p.proj_off[q];
        }
    }

    // The record: room in the ring (an all-hit record is dropped on a full ring, one with host work
    // waits for the tier thread), then the entries, then the head
    const uint32_t rec_bytes = (uint32_t) (kRecHeaderWords + kRecEntryWords * nu) * 4u;
    if (tid == 0)
    {
        const uint64_t head = p.dctr[kDcRecHead];
        const uint64_t off = head % p.rec_bytes;
        const uint64_t pos = (p.rec_bytes - off < rec_bytes) ? head + (p.rec_bytes - off) : head;
        int ok = 1;
        while (pos + rec_bytes - ld_acquire_u64(&p.ctl->rec_tail) > p.rec_bytes)
        {
            if (!s_nh && !p.deterministic)
            {
                ok = 0;
                break;
            }
            if (ld_acquire_u32(&p.ctl->abort))
            {
                first_error(&s_err, kErrRing);
                ok = 0;
                break;
            }
            __nanosleep(2000);
        }
        if (ok)
        {
            if (pos != head) st_volatile_u32(reinterpret_cast<uint32_t*>(p.rec_ring + off), kRecWrap);
            s_pos = pos;
        }
        else if (s_nh == 0)
        {
            volatile uint32_t* ov = &p.ctl->overflow;
            *ov = *ov + 1u;
        }
        s_ok = ok;
    }
    __syncthreads();
    if (s_ok)
    {
        uint32_t* w = reinterpret_cast<uint32_t*>(p.rec_ring + (s_pos % p.rec_bytes));
        for (int j = tid; j < nu; j += T)
        {
            uint32_t* x = w + kRecHeaderWords + j * kRecEntryWords;
            x[0] = (uint32_t) (lc * E + s_u[j]);
            x[1] = (uint32_t) s_kind[j] | ((uint32_t) (s_flags[j] & 1) << 8) | ((uint32_t) ((s_flags[j] >> 1) & 1) << 9);
            x[2] = s_cnt[s_u[j]];
            x[3] = (uint32_t) s_slot[j];
            x[4] = s_heat[j];
            x[5] = (uint32_t) s_vkey[j];
            x[6] = (uint32_t) s_vslot[j];
            x[7] = s_vstamp[j];
            x[8] = s_vheat[j];
            x[9] = 0;
        }
        if (tid == 0)
        {
            w[0] = rec_bytes;
            w[1] = seq;
            w[2] = (uint32_t) lc;
            w[3] = (uint32_t) mode | ((uint32_t) s_nh << 8);
            w[4] = (uint32_t) nu;
            w[5] = (uint32_t) tokens;
            w[6] = (uint32_t) (tokens >> 32);
            w[7] = 0;
        }
        __threadfence_system();
    }
    __syncthreads();
    if (tid == 0)
    {
        if (s_ok)
        {
            p.dctr[kDcRecHead] = s_pos + rec_bytes;
            st_release_u64(&p.ctl->rec_head, s_pos + rec_bytes);
        }
        if (s_err && !*(volatile uint32_t*) &p.ctl->error) st_volatile_u32(&p.ctl->error, s_err);
        // the call's fetch reads this (same stream, next kernel)
        p.dctr[kDcNeedsHost] = (uint64_t) (s_ok && s_nh);
    }
}

// ---------------------------------------------------------------------------------------- fetch

__device__ __forceinline__ uint4 ld_cv(const uint4* a)
{
    uint4 v;
    asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a));
    return v;
}

// 16 bytes starting `sh` bytes into the 32 bytes a, b (sh in 1..15)
__device__ __forceinline__ uint4 shift16(const uint4& a, const uint4& b, uint32_t sh)
{
    uint32_t w[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
    const uint32_t q = sh >> 2, r = (sh & 3) * 8;
    uint32_t o[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        uint32_t lo = 0, hi = 0;
        #pragma unroll
        for (int k = 0; k < 4; ++k)
            if (q == (uint32_t) k)
            {
                lo = w[k + j];
                hi = w[k + j + 1];
            }
        o[j] = r ? __funnelshift_r(lo, hi, r) : lo;
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}

// Copy n bytes (a multiple of 16) from src (any alignment; host memory is read uncached, so a slot
// the host rewrote is never served stale) to dst (16-byte aligned), by one CTA
__device__ void cta_copy(const uint8_t* src, uint8_t* dst, uint32_t n)
{
    const uint32_t mis = (uint32_t) ((uintptr_t) src & 15);
    uint4* d = reinterpret_cast<uint4*>(dst);
    if (!mis)
    {
        const uint4* s = reinterpret_cast<const uint4*>(src);
        for (uint32_t i = threadIdx.x; i < n / 16; i += blockDim.x) d[i] = ld_cv(s + i);
    }
    else
    {
        const uint4* a = reinterpret_cast<const uint4*>(src - mis);
        for (uint32_t i = threadIdx.x; i < n / 16; i += blockDim.x) d[i] = shift16(ld_cv(a + i), ld_cv(a + i + 1), mis);
    }
}

__device__ __forceinline__ uint64_t ld_volatile_u64(const uint64_t* p)
{
    return *(const volatile uint64_t*) p;
}

__device__ __forceinline__ uint32_t ld_volatile_u32(const uint32_t* p)
{
    return *(const volatile uint32_t*) p;
}

// Workers (blocks 0..G-1) execute the commands in order, each command's 64 KiB chunks spread over the
// workers by a global chunk counter; the reader (block G) copies commands from the host's plan ring
// to the device ring. Completion of command i: every worker with a chunk of it is done; the last one
// marks it in device memory (for dependent commands) and in host memory (for the tier thread)
__global__ void __launch_bounds__(kFetchThreads) tier_fetch_kernel(const FetchParams f, const uint32_t seq)
{
    __shared__ uint64_t s_idx;
    __shared__ uint64_t s_upto;                 // commands [.., s_upto) known complete (this CTA's view)
    __shared__ FetchCmd s_cmd;
    __shared__ int s_stop;
    const int G = f.workers;
    const uint32_t N = f.plan_n;
    if (!f.deterministic && !ld_volatile_u64(&f.dctr[kDcNeedsHost])) return;

    if ((int) blockIdx.x == G)
    {
        // the reader
        if (threadIdx.x != 0) return;
        uint64_t pos = f.dctr[kDcFetchRead], chunk = f.dctr[kDcFetchChunk];
        uint64_t spins = 0;
        bool found = false;
        while (!found)
        {
            const uint64_t tail = ld_acquire_u64(&f.ctl->plan_tail);
            if (pos == tail)
            {
                if (ld_acquire_u32(&f.ctl->abort))
                {
                    *(volatile uint64_t*) &f.dctr[kDcFetchAbort] = 1;
                    break;
                }
                if (++spins > (60ull << 20))       // ~a minute of 1 us naps with no command
                {
                    if (!ld_volatile_u32(&f.ctl->error)) st_volatile_u32(&f.ctl->error, kErrPlan);
                    *(volatile uint64_t*) &f.dctr[kDcFetchAbort] = 1;
                    break;
                }
                __nanosleep(1000);
                continue;
            }
            spins = 0;
            for (; pos < tail && !found; ++pos)
            {
                const volatile uint32_t* w = reinterpret_cast<const volatile uint32_t*>(f.plan + (pos & (N - 1)));
                FetchCmd c;
                c.src = (uint64_t) w[0] | ((uint64_t) w[1] << 32);
                c.dst = (uint64_t) w[2] | ((uint64_t) w[3] << 32);
                c.bytes = w[4];
                c.seq = w[5];
                c.dep = w[6];
                c.kind = w[7];
                c.base = chunk;
                chunk += (c.bytes + kFetchChunk - 1) / kFetchChunk;
                f.dring[pos & (N - 1)] = c;
                __threadfence();
                *(volatile uint64_t*) &f.dctr[kDcFetchTail] = pos + 1;
                if (c.kind == kPlanEnd)
                {
                    if (c.seq != seq && !ld_volatile_u32(&f.ctl->error)) st_volatile_u32(&f.ctl->error, kErrPlan);
                    found = true;
                }
            }
        }
        f.dctr[kDcFetchRead] = pos;
        f.dctr[kDcFetchChunk] = chunk;
        // every worker has read the start index: the next fetch may start where this one ended
        while (atomicAdd((unsigned long long*) &f.dctr[kDcFetchArrived], 0ull) < (unsigned long long) G)
            __nanosleep(200);
        f.dctr[kDcFetchArrived] = 0;
        f.dctr[kDcFetchStart] = pos;
        __threadfence();
        return;
    }

    // a worker
    if (threadIdx.x == 0)
    {
        s_idx = f.dctr[kDcFetchStart];
        s_upto = s_idx;                         // every command of the earlier fetches is complete
        __threadfence();
        atomicAdd((unsigned long long*) &f.dctr[kDcFetchArrived], 1ull);
    }
    __syncthreads();
    uint64_t idx = s_idx;
    while (true)
    {
        if (threadIdx.x == 0)
        {
            s_stop = 0;
            while (ld_volatile_u64(&f.dctr[kDcFetchTail]) <= idx)
            {
                if (ld_volatile_u64(&f.dctr[kDcFetchAbort]))
                {
                    s_stop = 1;
                    break;
                }
                __nanosleep(200);
            }
            if (!s_stop)
            {
                __threadfence();
                const volatile FetchCmd* v = &f.dring[idx & (N - 1)];
                s_cmd.src = v->src;
                s_cmd.dst = v->dst;
                s_cmd.base = v->base;
                s_cmd.bytes = v->bytes;
                s_cmd.seq = v->seq;
                s_cmd.dep = v->dep;
                s_cmd.kind = v->kind;
            }
        }
        __syncthreads();
        if (s_stop || s_cmd.kind == kPlanEnd) break;
        const FetchCmd c = s_cmd;
        const uint32_t nch = (c.bytes + kFetchChunk - 1) / kFetchChunk;
        const uint32_t j0 = (uint32_t) (((uint64_t) blockIdx.x + G - c.base % (uint64_t) G) % (uint64_t) G);
        if (j0 < nch)
        {
            // A dependency is on every command up to the one it names, as PlanCopier::done() takes a
            // token: a command's chunks are spread over the workers, so the named command being done
            // says nothing of the ones before it, and a promotion is three copies (one per
            // projection) that a demotion into the same RAM slot must not overtake
            if (c.dep && threadIdx.x == 0)
            {
                while (s_upto < (uint64_t) c.dep)
                {
                    const uint64_t k = s_upto;
                    if (ld_volatile_u32(&f.ddone[k & (N - 1)]) == (uint32_t) (k + 1) ||
                        reinterpret_cast<const volatile FetchCmd*>(f.dring)[k & (N - 1)].kind == kPlanEnd)
                    {
                        s_upto = k + 1;
                        continue;
                    }
                    if (ld_volatile_u64(&f.dctr[kDcFetchAbort])) break;
                    __nanosleep(100);
                }
                __threadfence();
            }
            __syncthreads();
            for (uint32_t j = j0; j < nch; j += (uint32_t) G)
            {
                const uint32_t off = j * kFetchChunk;
                const uint32_t n = min(kFetchChunk, c.bytes - off);
                cta_copy(reinterpret_cast<const uint8_t*>(c.src) + off, reinterpret_cast<uint8_t*>(c.dst) + off, n);
            }
            __threadfence_system();
            __syncthreads();
            if (threadIdx.x == 0)
            {
                const uint32_t parts = min(nch, (uint32_t) G);
                if (atomicAdd(&f.dcnt[idx & (N - 1)], 1u) == parts - 1)
                {
                    f.dcnt[idx & (N - 1)] = 0;
                    __threadfence();
                    st_volatile_u32(&f.ddone[idx & (N - 1)], (uint32_t) (idx + 1));
                    st_release_u32(&f.done_host[idx & (N - 1)], (uint32_t) (idx + 1));
                }
            }
        }
        __syncthreads();
        ++idx;
    }
}

}  // namespace

void tier_lookup_launch(const TierKParams& p, const int64_t* ids, int n, int lc, int mode, uint32_t seq,
                        uint64_t tokens, cudaStream_t stream)
{
    tier_lookup_kernel<<<1, T, 0, stream>>>(p, ids, n, lc, mode, seq, tokens);
}

cudaError_t tier_fetch_launch(const FetchParams& f, uint32_t seq, cudaStream_t stream)
{
    FetchParams fp = f;
    uint32_t sq = seq;
    void* args[] = { &fp, &sq };
    // cooperative: the workers wait on each other (dependencies) and on the reader, so all of them
    // must be resident at once
    return cudaLaunchCooperativeKernel((const void*) tier_fetch_kernel, dim3(f.workers + 1), dim3(kFetchThreads), args,
                                       0, stream);
}

}  // namespace exl3_tier
