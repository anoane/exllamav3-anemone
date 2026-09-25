// Python side of the expert tier: argument conversion and results as Python objects.

#include "tier_ext.h"

#include <torch/extension.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "tier_policy.h"
#include "tier_host.h"
#include "tier_py.h"

using namespace exl3_tier;
using namespace tier_py;

namespace tier_py
{

int choice(const py::dict& d, const char* key, std::initializer_list<const char*> names, int dflt)
{
    if (!d.contains(key) || d[key].is_none()) return dflt;
    std::string v = py::str(d[key]);
    int i = 0;
    for (const char* n : names)
    {
        if (v == n) return i;
        ++i;
    }
    throw std::invalid_argument(std::string("expert tier: ") + key + "=" + v + " is not a known value");
}

PolicyConfig config_of(const py::dict& d)
{
    PolicyConfig c;
    c.policy = choice(d, "policy", { "lazy-exclusive", "exclusive", "inclusive" }, c.policy);
    c.demote = choice(d, "demote", { "swap", "heat", "all", "off" }, c.demote);
    c.admit = choice(d, "admit", { "adaptive", "heat", "always" }, c.admit);
    c.evict = choice(d, "evict", { "lru", "lfu" }, c.evict);
    c.ram_evict = choice(d, "ram_evict", { "lru", "lfu" }, c.ram_evict);
    c.ram_admit = choice(d, "ram_admit", { "always", "heat" }, c.ram_admit);
    c.demote_on = value<bool>(d, "demote_on", c.demote_on);
    c.disk_off = value<bool>(d, "disk_off", c.disk_off);
    c.spare = value<int>(d, "spare", c.spare);
    c.sample = value<int>(d, "sample", c.sample);
    c.seed = value<uint64_t>(d, "seed", c.seed);
    c.halflife = value<uint32_t>(d, "halflife", c.halflife);
    c.prefill_inc = value<uint32_t>(d, "prefill_inc", c.prefill_inc);
    c.adapt_every = value<int64_t>(d, "adapt_every", c.adapt_every);
    c.pmin = value<double>(d, "pmin", c.pmin);
    if (d.contains("p") && !d["p"].is_none())
    {
        c.p_pinned = true;
        c.p = d["p"].cast<double>();
    }
    c.refill = value<bool>(d, "refill", c.refill);
    c.refill_inflight = value<int>(d, "refill_inflight", c.refill_inflight);
    c.validate();
    return c;
}

ScriptOp op_of(const py::handle& h)
{
    py::tuple t = py::reinterpret_borrow<py::tuple>(h);
    std::string name = py::str(t[0]);
    ScriptOp o;
    auto ids = [&](size_t i) { return t[i].cast<std::vector<int32_t>>(); };
    if (name == "tick") { o.op = ScriptOp::kTick; o.a = t[1].cast<int64_t>(); }
    else if (name == "call")
    {
        o.op = ScriptOp::kCall;
        o.a = t[1].cast<int64_t>();
        o.ids = ids(2);
        o.b = t.size() > 3 ? t[3].cast<int64_t>() : (int64_t) kDecode;
    }
    else if (name == "layer") { o.op = ScriptOp::kLayer; o.a = t[1].cast<int64_t>(); o.ids = ids(2); }
    else if (name == "complete") o.op = ScriptOp::kComplete;
    else if (name == "place")
    {
        o.op = ScriptOp::kPlace;
        o.a = t[1].cast<int64_t>(); o.b = t[2].cast<int64_t>(); o.c = t[3].cast<int64_t>();
    }
    else if (name == "ram_add")
    {
        o.op = ScriptOp::kRamAdd;
        o.a = t[1].cast<int64_t>(); o.b = t[2].cast<int64_t>(); o.c = t[3].cast<int64_t>(); o.d = t[4].cast<int64_t>();
    }
    else if (name == "heat") { o.op = ScriptOp::kHeatSet; o.a = t[1].cast<int64_t>(); o.b = t[2].cast<int64_t>(); }
    else if (name == "seq") { o.op = ScriptOp::kSeq; o.a = t[1].cast<int64_t>(); }
    else if (name == "cold_fill") { o.op = ScriptOp::kColdFill; o.ids = ids(1); }
    else if (name == "check") { o.op = ScriptOp::kCheck; o.a = t[1].cast<int64_t>(); }
    else throw std::invalid_argument("expert tier replay: unknown operation " + name);
    return o;
}

py::tuple action_tuple(const Action& a)
{
    int n = action_arity(a.op);
    py::tuple t(1 + n);
    t[0] = py::str(action_name(a.op));
    const int32_t v[4] = { a.a, a.b, a.c, a.d };
    for (int i = 0; i < n; ++i) t[(size_t) (1 + i)] = py::int_(v[i]);
    return t;
}

py::tuple entry_tuple(const Entry& x)
{
    return py::make_tuple(x.key, (int) x.kind, x.dst, x.cnt, x.heat, x.vkey, x.vslot, x.vstamp, x.vheat,
                          (int) x.inplace);
}

py::dict core_state(const TierCore& core)
{
    const VramDirectory& v = core.vram;
    const RamTier& m = core.ram;
    py::dict st;
    st["vram_state"] = py::cast(std::vector<int>(v.state.begin(), v.state.end()));
    st["vram_key"] = py::cast(v.key);
    st["vram_stamp"] = py::cast(v.stamp);
    st["slot_of"] = py::cast(v.slot_of);
    st["vram_free"] = py::cast(std::vector<int32_t>(v.free.begin(), v.free.end()));
    st["occupied"] = v.occupied;
    st["access"] = v.access;
    st["ram_key"] = py::cast(m.key);
    st["ram_dup"] = py::cast(std::vector<int>(m.dup.begin(), m.dup.end()));
    st["ram_compact"] = py::cast(std::vector<int>(m.compact.begin(), m.compact.end()));
    st["ram_stamp"] = py::cast(m.stamp);
    st["ram_of"] = py::cast(m.of);
    st["ram_free"] = py::cast(std::vector<int32_t>(m.free.begin(), m.free.end()));
    st["own"] = py::cast(m.lists[0]);
    st["dups"] = py::cast(m.lists[1]);
    st["draws"] = m.draws;
    st["heat_v"] = py::cast(core.heat.v);
    st["heat_ep"] = py::cast(core.heat.ep);
    st["tokens"] = core.heat.tokens;
    st["p"] = core.adapt.p;
    st["seq"] = core.seq;
    st["pending"] = py::cast(core.pending);
    return st;
}

py::dict core_counters(const TierCore& core)
{
    py::dict cn;
    for (int i = 0; i < kNumCounters; ++i) cn[counter_name(i)] = core.c[i];
    return cn;
}

}  // namespace tier_py

py::dict tier_policy_replay(const py::dict& config, int64_t layers, int64_t experts, int64_t slots, int64_t ram_slots,
                            int64_t pins, bool defer_returns, const py::list& script)
{
    PolicyConfig cfg = config_of(config);
    std::vector<ScriptOp> ops;
    for (const auto& h : script) ops.push_back(op_of(h));
    TierCore core(cfg, layers, experts, (int) slots, ram_slots, (int) pins, defer_returns);
    ReplayOut out;
    std::string error;
    try
    {
        py::gil_scoped_release nogil;
        replay(core, ops, out);
    }
    catch (const TierError& e)
    {
        error = e.what();
    }
    py::dict r;
    py::list recs;
    for (const Record& rec : out.records)
    {
        py::list ents;
        for (const Entry& x : rec.entries) ents.append(entry_tuple(x));
        recs.append(py::make_tuple(rec.seq, rec.lc, rec.mode, ents));
    }
    r["records"] = recs;
    py::list acts;
    for (size_t i = 0; i < out.actions.size(); ++i)
    {
        py::list l;
        for (const Action& a : out.actions[i]) l.append(action_tuple(a));
        acts.append(py::make_tuple(out.action_ops[i], l));
    }
    r["actions"] = acts;
    r["counters"] = core_counters(core);
    py::dict st = core_state(core);
    r["state"] = st;
    r["error"] = error;
    return r;
}

// ---------------------------------------------------------------------------------------- host

namespace tier_py
{

at::Tensor bytes_tensor(const uint8_t* p, int64_t n)
{
    at::Tensor t = at::empty({ n }, at::TensorOptions().dtype(at::kByte));
    std::memcpy(t.data_ptr<uint8_t>(), p, (size_t) n);
    return t;
}

HostConfig host_config_of(const py::dict& d)
{
    HostConfig h;
    h.chunk_bytes = value<int64_t>(d, "chunk_bytes", h.chunk_bytes);
    h.hugepage = value<bool>(d, "hugepage", h.hugepage);
    h.slab_slots = value<int>(d, "slab_slots", h.slab_slots);
    h.fill_inflight = value<int>(d, "fill_inflight", h.fill_inflight);
    h.staging_slots = value<int>(d, "staging_slots", h.staging_slots);
    h.prefault_threads = value<int>(d, "prefault_threads", h.prefault_threads);
    h.deterministic = value<bool>(d, "deterministic", h.deterministic);
    h.compact_fill = value<bool>(d, "compact_fill", h.compact_fill);
    if (h.chunk_bytes <= 0 || h.slab_slots < 0 || h.fill_inflight < 1)
        throw std::invalid_argument("expert tier: chunk_bytes > 0, slab_slots >= 0, fill_inflight >= 1");
    return h;
}

std::vector<ExpertExtent> extents_of(const py::list& l)
{
    std::vector<ExpertExtent> out;
    out.reserve(l.size());
    for (const auto& h : l)
    {
        py::tuple t = py::reinterpret_borrow<py::tuple>(h);
        ExpertExtent x;
        x.name = py::str(t[0]);
        py::list pieces = t[1].cast<py::list>();
        py::list trel = t[2].cast<py::list>();
        if (pieces.size() < 1 || pieces.size() > 3 || trel.size() < 2 || trel.size() > 3)
            throw std::invalid_argument("expert tier: " + x.name + ": 1 to 3 pieces, 2 or 3 projections");
        x.npieces = (int) pieces.size();
        for (size_t i = 0; i < pieces.size(); ++i)
        {
            py::tuple pc = pieces[i].cast<py::tuple>();
            x.file[i] = pc[0].cast<int32_t>();
            x.offset[i] = pc[1].cast<int64_t>();
            x.length[i] = pc[2].cast<int64_t>();
        }
        for (size_t p = 0; p < trel.size(); ++p)
        {
            py::tuple tr = trel[p].cast<py::tuple>();
            x.tpiece[p] = tr[0].cast<int32_t>();
            x.trel[p] = tr[1].cast<int64_t>();
        }
        out.push_back(std::move(x));
    }
    return out;
}

Record record_of(const py::tuple& t)
{
    Record r;
    r.seq = t[0].cast<uint32_t>();
    r.lc = t[1].cast<int32_t>();
    r.mode = t[2].cast<int>();
    for (const auto& h : t[3].cast<py::list>())
    {
        py::tuple e = py::reinterpret_borrow<py::tuple>(h);
        Entry x;
        x.key = e[0].cast<int32_t>();
        x.kind = (uint8_t) e[1].cast<int>();
        x.dst = e[2].cast<int32_t>();
        x.cnt = e[3].cast<uint32_t>();
        x.heat = e[4].cast<uint32_t>();
        x.vkey = e[5].cast<int32_t>();
        x.vslot = e[6].cast<int32_t>();
        x.vstamp = e[7].cast<uint32_t>();
        x.vheat = e[8].cast<uint32_t>();
        x.inplace = (uint8_t) e[9].cast<int>();
        r.entries.push_back(x);
    }
    return r;
}

}  // namespace tier_py

TierHostHandle::TierHostHandle(const py::dict& policy, int64_t layers, int64_t experts, int64_t pool_slots,
                               int64_t ram_slots, int64_t pins, const py::dict& geometry,
                               const std::vector<std::string>& files, const py::list& extents, const py::dict& host,
                               bool defer_returns)
{
    PolicyConfig pc = config_of(policy);
    Geometry g;
    std::vector<int64_t> pb = geometry["proj_bytes"].cast<std::vector<int64_t>>();
    if (pb.size() < 2 || pb.size() > 3) throw std::invalid_argument("expert tier: 2 or 3 projections");
    g.projections = (int) pb.size();
    for (size_t p = 0; p < pb.size(); ++p) g.proj_bytes[p] = pb[p];
    g.ram_slot_bytes = geometry["ram_slot_bytes"].cast<int64_t>();
    HostConfig hc = host_config_of(host);
    std::vector<ExpertExtent> ex = extents_of(extents);
    int64_t vram = 0;
    for (int64_t b : pb) vram += b;
    int64_t staging = hc.staging_slots > 0 ? hc.staging_slots : experts;
    copier_.reset(new HostCopier(pool_slots + pins, staging, vram));
    py::gil_scoped_release nogil;
    host_.reset(new TierHost(pc, layers, experts, (int) pool_slots, ram_slots, (int) pins, g, files, ex, hc,
                             copier_.get(), defer_returns));
}

TierHostHandle::~TierHostHandle()
{
    py::gil_scoped_release nogil;
    host_.reset();
}

void TierHostHandle::cold_fill(const std::vector<int32_t>& order)
{
    py::gil_scoped_release nogil;
    host_->cold_fill(order);
}

py::list TierHostHandle::call(int64_t lc, const std::vector<int32_t>& ids, int64_t mode)
{
    Record rec;
    {
        py::gil_scoped_release nogil;
        rec = host_->call((int32_t) lc, ids.data(), (int64_t) ids.size(), (int) mode);
    }
    py::list out;
    for (const Entry& x : rec.entries) out.append(entry_tuple(x));
    return out;
}

void TierHostHandle::process(const py::tuple& record)
{
    Record rec = record_of(record);
    py::gil_scoped_release nogil;
    host_->process(rec);
}

void TierHostHandle::layer(int64_t lc, const std::vector<int32_t>& ids, int64_t half)
{
    py::gil_scoped_release nogil;
    host_->layer_call((int32_t) lc, ids.data(), (int64_t) ids.size(), (int) half);
}

void TierHostHandle::prefetch(int64_t lc, int64_t deadline_ns)
{
    py::gil_scoped_release nogil;
    host_->prefetch_layer((int32_t) lc, deadline_ns);
}

void TierHostHandle::promote(int64_t lc) { host_->promote_layer((int32_t) lc); }
void TierHostHandle::tick(int64_t tokens) { host_->tick((uint64_t) tokens); }

void TierHostHandle::complete()
{
    py::gil_scoped_release nogil;
    host_->complete();
}

void TierHostHandle::drain()
{
    py::gil_scoped_release nogil;
    host_->drain();
}

void TierHostHandle::poll() { host_->poll(); }

at::Tensor TierHostHandle::vram_slot(int64_t slot)
{
    if (slot < 0 || slot >= host_->core.vram.S + host_->core.vram.pins) throw std::out_of_range("expert tier: pool slot");
    return bytes_tensor(copier_->pool((int32_t) slot), host_->geo.vram_slot_bytes);
}

at::Tensor TierHostHandle::staging(int64_t half, int64_t i)
{
    if (half < 0 || half > 1 || i < 0 || i >= host_->hcfg.staging_slots) throw std::out_of_range("expert tier: staging slot");
    return bytes_tensor(copier_->staging((int) half, (int32_t) i), host_->geo.vram_slot_bytes);
}

at::Tensor TierHostHandle::ram_trellis(int64_t r)
{
    if (r < 0 || r >= host_->core.ram.R) throw std::out_of_range("expert tier: RAM slot");
    at::Tensor t = at::empty({ host_->geo.vram_slot_bytes }, at::TensorOptions().dtype(at::kByte));
    host_->ram_trellis((int32_t) r, t.data_ptr<uint8_t>());
    return t;
}

at::Tensor TierHostHandle::ram_slot(int64_t r)
{
    if (r < 0 || r >= host_->core.ram.R) throw std::out_of_range("expert tier: RAM slot");
    return bytes_tensor(host_->ram_slot((int32_t) r), host_->geo.ram_slot_bytes);
}

int64_t TierHostHandle::ram_offset(int64_t r, int64_t p)
{
    if (r < 0 || r >= host_->core.ram.R || p < 0 || p >= host_->geo.projections) throw std::out_of_range("expert tier");
    return host_->ram_slot_offset((int32_t) r, (int) p);
}

py::list TierHostHandle::last_actions()
{
    py::list l;
    for (const Action& a : host_->last_actions()) l.append(action_tuple(a));
    return l;
}

py::dict TierHostHandle::state() { return core_state(host_->core); }
py::dict TierHostHandle::counters() { return core_counters(host_->core); }

py::dict TierHostHandle::stats()
{
    const HostStats& s = host_->stats;
    py::dict d;
    d["ssd_reads"] = s.ssd_reads;
    d["ssd_bytes"] = s.ssd_bytes;
    d["pread_fallbacks"] = s.pread_fallbacks;
    d["h2d"] = s.h2d;
    d["d2h"] = s.d2h;
    d["d2d"] = s.d2d;
    d["staged"] = s.staged;
    d["prefetch_issued"] = s.prefetch_issued;
    d["prefetch_used"] = s.prefetch_used;
    d["prefetch_starved"] = s.prefetch_starved;
    d["prefetch_dropped"] = s.prefetch_dropped;
    d["refill_reads"] = s.refill_reads;
    d["hazard_waits"] = s.hazard_waits;
    d["slab_waits"] = s.slab_waits;
    d["cold_ram"] = s.cold_ram;
    d["cold_vram"] = s.cold_vram;
    d["cold_bytes"] = s.cold_bytes;
    d["cold_ns"] = s.cold_ns;
    d["arena_ns"] = s.arena_ns;
    d["h2d_bytes"] = copier_->h2d_bytes;
    d["d2h_bytes"] = copier_->d2h_bytes;
    d["d2d_bytes"] = copier_->d2d_bytes;
    d["h2d_copies"] = copier_->h2d_copies;
    d["d2h_copies"] = copier_->d2h_copies;
    d["d2d_copies"] = copier_->d2d_copies;
    return d;
}

py::dict TierHostHandle::arena()
{
    auto info = [](const Arena& a)
    {
        py::dict d;
        d["slots"] = a.slots;
        d["slot_bytes"] = a.slot_bytes;
        d["chunk_slots"] = a.chunk_slots;
        py::list base, bytes;
        for (uint8_t* p : a.base) base.append((uint64_t) reinterpret_cast<uintptr_t>(p));
        for (size_t n : a.bytes) bytes.append(n);
        d["base"] = base;
        d["bytes"] = bytes;
        d["registered"] = py::cast(a.reg);
        return d;
    };
    py::dict d;
    d["ram"] = info(host_->arena());
    d["slab"] = info(host_->slab());
    d["vram_slot_bytes"] = host_->geo.vram_slot_bytes;
    d["ram_slot_bytes"] = host_->geo.ram_slot_bytes;
    d["staging_slots"] = host_->hcfg.staging_slots;
    return d;
}

std::string TierHostHandle::error() { return host_->error(); }
