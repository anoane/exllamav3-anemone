// Python side of the expert tier: argument conversion and results as Python objects.

#include "tier_ext.h"

#include <torch/extension.h>
#include <pybind11/stl.h>

#include <cmath>

#include "tier_policy.h"

using namespace exl3_tier;

namespace
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

template <typename T>
T value(const py::dict& d, const char* key, T dflt)
{
    if (!d.contains(key) || d[key].is_none()) return dflt;
    return d[key].cast<T>();
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

}  // namespace

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
    py::dict cn;
    for (int i = 0; i < kNumCounters; ++i) cn[counter_name(i)] = core.c[i];
    r["counters"] = cn;
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
    r["state"] = st;
    r["error"] = error;
    return r;
}
