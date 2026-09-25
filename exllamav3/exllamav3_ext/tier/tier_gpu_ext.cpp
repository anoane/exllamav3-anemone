// Python side of the VRAM expert cache: argument conversion and results as Python objects.

#include "tier_gpu_ext.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "tier_gpu.h"
#include "tier_py.h"

using namespace exl3_tier;
using namespace tier_py;

namespace
{

std::vector<uint64_t> u64_list(const py::dict& d, const char* key)
{
    if (!d.contains(key) || d[key].is_none()) return {};
    std::vector<uint64_t> out;
    for (const auto& h : d[key].cast<py::list>()) out.push_back(h.cast<uint64_t>());
    return out;
}

GpuConfig gpu_config_of(const py::dict& d)
{
    GpuConfig g;
    g.device = value<int>(d, "device", 0);
    g.pool = u64_list(d, "pool");
    g.staging = u64_list(d, "staging");
    g.staging_per_half = value<int64_t>(d, "staging_per_half", (int64_t) g.staging.size());
    g.tables = u64_list(d, "tables");
    std::vector<int64_t> po = value<std::vector<int64_t>>(d, "proj_off", {});
    if (po.size() < 2 || po.size() > 3) throw std::invalid_argument("expert tier: proj_off has 2 or 3 offsets");
    g.projections = (int) po.size();
    for (size_t i = 0; i < po.size(); ++i) g.proj_off[i] = po[i];
    g.deterministic = value<bool>(d, "deterministic", false);
    g.spin_us = value<int64_t>(d, "spin_us", -1);
    g.affinity = value<std::vector<int>>(d, "affinity", {});
    g.rec_ring_bytes = value<int64_t>(d, "rec_ring_bytes", g.rec_ring_bytes);
    g.trace = value<int64_t>(d, "trace", 0);
    return g;
}

}  // namespace

TierGpuHandle::TierGpuHandle(const py::dict& policy, int64_t layers, int64_t experts, int64_t pool_slots,
                             int64_t ram_slots, int64_t pins, const py::object& geometry,
                             const std::vector<std::string>& files, const py::list& extents, const py::dict& host,
                             const py::dict& gpu)
{
    PolicyConfig pc = config_of(policy);
    GpuConfig gc = gpu_config_of(gpu);
    device_ = gc.device;
    std::unique_ptr<TierHost> th;
    std::unique_ptr<CudaCopier> cp;
    if (extents.size())
    {
        Geometry g;
        py::dict gd = geometry.cast<py::dict>();
        std::vector<int64_t> pb = gd["proj_bytes"].cast<std::vector<int64_t>>();
        if ((int) pb.size() != gc.projections) throw std::invalid_argument("expert tier: projections disagree");
        g.projections = (int) pb.size();
        for (size_t p = 0; p < pb.size(); ++p) g.proj_bytes[p] = pb[p];
        g.ram_slot_bytes = gd["ram_slot_bytes"].cast<int64_t>();
        int64_t acc = 0;
        for (size_t p = 0; p < pb.size(); ++p)
        {
            if (gc.proj_off[p] != acc) throw std::invalid_argument("expert tier: proj_off is not the compact layout");
            acc += pb[p];
        }
        HostConfig hc = host_config_of(host);
        hc.deterministic = gc.deterministic;
        hc.staging_slots = (int) gc.staging_per_half;
        std::vector<ExpertExtent> ex = extents_of(extents);
        py::gil_scoped_release nogil;
        cp.reset(new CudaCopier(gc.device, gc.pool, gc.staging, gc.staging_per_half));
        th.reset(new TierHost(pc, layers, experts, (int) pool_slots, ram_slots, (int) pins, g, files, ex, hc, cp.get(),
                              !gc.deterministic));
        const Arena& slab = th->slab();
        for (size_t i = 0; i < slab.base.size(); ++i) cp->pin(slab.base[i], slab.bytes[i]);
    }
    try
    {
        py::gil_scoped_release nogil;
        gpu_.reset(new TierGpu(pc, layers, experts, (int) pool_slots, ram_slots, (int) pins, gc, std::move(th),
                               std::move(cp)));
    }
    catch (...)
    {
        if (cp) cp->unpin_all();
        throw;
    }
}

TierGpuHandle::~TierGpuHandle()
{
    py::gil_scoped_release nogil;
    gpu_.reset();
}

void TierGpuHandle::cold_fill(const std::vector<int32_t>& order, const std::vector<int32_t>& pins)
{
    py::gil_scoped_release nogil;
    gpu_->cold_fill(order, pins);
}

void TierGpuHandle::release_slab()
{
    TierHost* h = gpu_->host();
    if (!h) return;
    py::gil_scoped_release nogil;
    gpu_->copier()->sync();
    const Arena& slab = h->slab();
    for (uint8_t* p : slab.base) gpu_->copier()->unpin(p);
    h->release_slab();
}

void TierGpuHandle::start() { gpu_->start(); }

void TierGpuHandle::stop()
{
    py::gil_scoped_release nogil;
    gpu_->stop();
}

void TierGpuHandle::lookup(int64_t lc, const at::Tensor& ids, int64_t mode, int64_t seq, int64_t tokens)
{
    TORCH_CHECK(ids.scalar_type() == at::kLong && ids.is_cuda() && ids.is_contiguous(),
                "expert tier: selected experts must be a contiguous int64 CUDA tensor");
    TORCH_CHECK(ids.get_device() == device_, "expert tier: selected experts on the wrong device");
    TORCH_CHECK(mode == 0 || mode == 1, "expert tier: lookup mode 0 (decode) or 1 (routed)");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_).stream();
    gpu_->lookup((int) lc, ids.data_ptr<int64_t>(), (int) ids.numel(), (int) mode, (uint32_t) seq, (uint64_t) tokens,
                 stream);
}

int64_t TierGpuHandle::step(bool wait)
{
    py::gil_scoped_release nogil;
    return gpu_->step(wait);
}

void TierGpuHandle::drain()
{
    py::gil_scoped_release nogil;
    gpu_->drain();
}

void TierGpuHandle::verify()
{
    py::gil_scoped_release nogil;
    gpu_->verify();
}

py::list TierGpuHandle::records()
{
    std::deque<Record> recs;
    {
        std::lock_guard<std::mutex> lk(gpu_->trace_mu);
        recs.swap(gpu_->trace);
    }
    py::list out;
    for (const Record& rec : recs)
    {
        py::list ents;
        for (const Entry& x : rec.entries) ents.append(entry_tuple(x));
        out.append(py::make_tuple(rec.seq, rec.lc, rec.mode, ents));
    }
    return out;
}

py::dict TierGpuHandle::state() { return core_state(gpu_->core()); }
py::dict TierGpuHandle::counters() { return core_counters(gpu_->core()); }

py::dict TierGpuHandle::stats()
{
    py::dict d;
    const GpuStats& s = gpu_->stats;
    d["records"] = s.records;
    d["host_records"] = s.host_records;
    d["returns"] = s.returns;
    d["max_lag"] = s.max_lag;
    d["idle_sleeps"] = s.idle_sleeps;
    d["work_ns"] = s.work_ns;
    d["overflow"] = gpu_->overflow();
    d["device_error"] = gpu_->device_error();
    // the fetch kernel's copies (every call since the load)
    const PlanCopier* pl = gpu_->plan();
    d["h2d_bytes"] = pl->h2d_bytes;
    d["d2h_bytes"] = pl->d2h_bytes;
    d["d2d_bytes"] = pl->d2d_bytes;
    d["h2d_copies"] = pl->h2d_copies;
    d["d2h_copies"] = pl->d2h_copies;
    d["d2d_copies"] = pl->d2d_copies;
    d["plan_commands"] = pl->commands;
    d["plan_full_waits"] = pl->full_waits;
    d["plan_deps_done"] = pl->dep_dropped;
    // the copy engines of the load (cold fill) and the page-locked memory
    if (CudaCopier* c = gpu_->copier())
    {
        d["cold_h2d_bytes"] = c->h2d_bytes;
        d["pinned_bytes"] = c->pinned_bytes;
        d["pin_ns"] = c->pin_ns;
    }
    if (TierHost* h = gpu_->host())
    {
        const HostStats& hs = h->stats;
        d["ssd_reads"] = hs.ssd_reads;
        d["ssd_bytes"] = hs.ssd_bytes;
        d["pread_fallbacks"] = hs.pread_fallbacks;
        d["hazard_waits"] = hs.hazard_waits;
        d["slab_waits"] = hs.slab_waits;
        d["refill_reads"] = hs.refill_reads;
        d["cold_ram"] = hs.cold_ram;
        d["cold_vram"] = hs.cold_vram;
        d["cold_bytes"] = hs.cold_bytes;
        d["cold_ns"] = hs.cold_ns;
        d["arena_ns"] = hs.arena_ns;
        d["compacted"] = hs.compacted;
        d["compact_ns"] = hs.compact_ns;
        d["ram_chunks"] = (int64_t) h->arena().base.size();
    }
    return d;
}

py::dict TierGpuHandle::device_state()
{
    TierGpu& g = *gpu_;
    c10::cuda::CUDAGuard guard((c10::DeviceIndex) device_);
    auto opt = at::TensorOptions().device(at::kCUDA, device_);
    const int64_t n = g.S + g.pins;
    auto view = [&](void* p, int64_t count, at::ScalarType t)
    {
        return at::from_blob(p, { count }, opt.dtype(t)).clone().cpu();
    };
    py::dict d;
    d["state"] = view(g.dev.state, n, at::kByte);
    d["key"] = view(g.dev.key, n, at::kInt);
    d["stamp"] = view(g.dev.stamp, n, at::kInt);
    d["addr"] = view(g.dev.addr, n, at::kLong);
    d["slot_of"] = view(g.dev.slot_of, g.K, at::kInt);
    d["heat_v"] = view(g.dev.heat_v, g.K, at::kInt);
    d["heat_ep"] = view(g.dev.heat_ep, g.K, at::kInt);
    d["free_bits"] = view(g.dev.free_bits, g.W, at::kInt);
    d["dctr"] = view(g.dev.dctr, kDcCount, at::kLong);
    return d;
}

std::string TierGpuHandle::error() { return gpu_->error(); }

at::Tensor TierGpuHandle::ram_trellis(int64_t r)
{
    TierHost* h = gpu_->host();
    if (!h || r < 0 || r >= h->core.ram.R) throw std::out_of_range("expert tier: RAM slot");
    at::Tensor t = at::empty({ h->geo.vram_slot_bytes }, at::TensorOptions().dtype(at::kByte));
    py::gil_scoped_release nogil;
    h->ram_trellis((int32_t) r, t.data_ptr<uint8_t>());
    return t;
}

void TierGpuHandle::place(int64_t key, int64_t slot, int64_t stamp)
{
    gpu_->core().vram.place((int32_t) key, (int32_t) slot, (uint32_t) stamp);
}

void TierGpuHandle::ram_add(int64_t key, bool dup, int64_t slot, int64_t stamp)
{
    if (gpu_->host()) throw std::invalid_argument("expert tier: ram_add seeds the policy-only runtime");
    gpu_->core().ram.add((int32_t) key, dup, (uint32_t) stamp, (int32_t) slot, false);
}

void TierGpuHandle::heat_set(int64_t key, int64_t value)
{
    gpu_->core().heat.set(key, (uint64_t) value);
}

void TierGpuHandle::set_seq(int64_t seq) { gpu_->core().seq = (uint32_t) seq; }

void TierGpuHandle::upload()
{
    py::gil_scoped_release nogil;
    gpu_->upload();
}
