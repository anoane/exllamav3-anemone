// Python side of the disk engine: argument checks, tensor lifetimes, GIL handling, errors.

#include "disk_ext.h"

#include <torch/extension.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstring>
#include <mutex>

#if defined(__linux__)
#include <pthread.h>
#endif

using exl3_disk::Engine;
using exl3_disk::Error;

namespace
{

int64_t mono_ns()
{
    return (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Per-thread timing of the last binding call (disk_profile): enter, submitted, completed, exit
thread_local bool t_prof = false;
thread_local int64_t t_times[4] = { 0, 0, 0, 0 };

inline void stamp(int i)
{
    if (t_prof) t_times[i] = mono_ns();
}

std::string errtext(int e)
{
    return std::string(std::strerror(e)) + " (errno " + std::to_string(e) + ")";
}

[[noreturn]] void raise_timeout(const std::string& msg)
{
    PyErr_SetString(PyExc_TimeoutError, msg.c_str());
    throw py::error_already_set();
}

uint32_t* flag_ptr(const c10::optional<at::Tensor>& flag, int64_t index, int64_t value,
                   std::vector<at::Tensor>* keep)
{
    TORCH_CHECK(value >= 0 && value <= (int64_t) UINT32_MAX,
                "disk engine: flag_value must fit in 32 bits");
    if (!flag.has_value()) return nullptr;
    const at::Tensor& f = flag.value();
    TORCH_CHECK(f.device().is_cpu() && f.scalar_type() == at::kInt && f.is_contiguous(),
                "disk engine: flag must be a contiguous int32 CPU tensor (pinned and mapped for "
                "a GPU wait)");
    TORCH_CHECK(index >= 0 && index < f.numel(), "disk engine: flag_index out of range");
    keep->push_back(f);
    return reinterpret_cast<uint32_t*>(f.data_ptr<int32_t>()) + index;
}

void check_cpu_bytes(const at::Tensor& t, const char* what)
{
    TORCH_CHECK(t.device().is_cpu() && t.is_contiguous(), "disk engine: ", what,
                " must be a contiguous CPU tensor");
}

void check_i64_vec(const at::Tensor& t, int64_t n, const char* what)
{
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == at::kLong && t.is_contiguous() &&
                t.dim() == 1 && (n < 0 || t.numel() == n),
                "disk engine: ", what, " must be a contiguous 1-D int64 CPU tensor",
                n >= 0 ? " with one entry per extent" : "");
}

int checked_fd(int64_t fd)
{
    TORCH_CHECK(fd >= 0 && fd <= INT_MAX, "disk engine: bad file descriptor ", fd);
    return (int) fd;
}

// Registered buffers belong to the engine that registered them; the tensor stays referenced
// here until unregistered, so its memory cannot be freed and reused while the kernel still
// holds the pages (a READ_FIXED would land in the old pages)
struct RegBuf
{
    std::weak_ptr<Engine> eng;
    int idx;
    at::Tensor t;
};
std::mutex g_reg_mx;
std::vector<RegBuf>* g_reg = new std::vector<RegBuf>();     // leaked: tensors outlive exit

void install_reg_fork_handler()
{
#if defined(__linux__)
    // a forked child must not inherit g_reg_mx locked by a thread it does not have
    static std::once_flag once;
    std::call_once(once, []
    {
        (void) pthread_atfork(+[] { g_reg_mx.lock(); }, +[] { g_reg_mx.unlock(); },
                              +[] { g_reg_mx.unlock(); });
    });
#endif
}

// Entries whose engine is gone (replaced by disk_engine_configure, shut down, or released by
// its last ticket): its ring was closed, so the kernel no longer holds their pages. Returns
// the tensors, to be dropped by the caller with the GIL held and g_reg_mx released (a tensor's
// deleter may need Python)
std::vector<at::Tensor> prune_registered_locked()
{
    std::vector<at::Tensor> drop;
    auto keep_end = std::stable_partition(g_reg->begin(), g_reg->end(),
                                          [](const RegBuf& b) { return !b.eng.expired(); });
    for (auto it = keep_end; it != g_reg->end(); ++it) drop.push_back(std::move(it->t));
    g_reg->erase(keep_end, g_reg->end());
    return drop;
}

void prune_registered()
{
    install_reg_fork_handler();
    std::vector<at::Tensor> drop;
    {
        std::lock_guard<std::mutex> lk(g_reg_mx);
        drop = prune_registered_locked();
    }
}

bool finalizing()
{
#if PY_VERSION_HEX >= 0x030D0000
    return Py_IsFinalizing();
#else
    return _Py_IsFinalizing();
#endif
}

}  // namespace

// ---- DiskTicket -------------------------------------------------------------------------------

DiskTicket::DiskTicket(std::shared_ptr<Engine> eng, uint64_t id, std::vector<at::Tensor> keep)
    : eng_(std::move(eng)), id_(id), keep_(std::move(keep))
{
}

DiskTicket::~DiskTicket()
{
    try
    {
        release();
    }
    catch (...)
    {
        // a destructor must not throw; release() only fails if the engine is unusable, in which
        // case nothing of this ticket can still be in flight in this process
    }
}

void DiskTicket::check() const
{
    TORCH_CHECK(!released_, "disk engine: ticket already released");
}

void DiskTicket::wait(py::object timeout)
{
    check();
    int64_t tns = -1;
    if (!timeout.is_none())
    {
        double s = timeout.cast<double>();
        TORCH_CHECK(s >= 0.0 && s < 1e9, "disk engine: bad timeout");
        tns = (int64_t) (s * 1e9);
    }
    // Another Python thread may release this ticket (or drop it) while this one waits without
    // the GIL: wait on an engine reference of our own. The engine keeps the ticket alive until
    // this wait returns; a release that finished first makes the ticket unknown
    std::shared_ptr<Engine> eng = eng_;
    int r = 0;
    try
    {
        py::gil_scoped_release nogil;
        r = eng->wait(id_, tns);
    }
    catch (const Error& e)
    {
        if (e.code() == ENOENT) throw std::runtime_error("disk engine: ticket already released");
        throw;
    }
    if (r == -ETIMEDOUT) raise_timeout("disk engine: ticket not complete after the timeout");
    if (r == -ECANCELED) throw std::runtime_error("disk engine: ticket was cancelled");
    if (r < 0) throw std::runtime_error("disk engine: read failed: " + errtext(-r));
}

bool DiskTicket::done()
{
    check();
    return eng_->done(id_);
}

void DiskTicket::cancel()
{
    check();
    eng_->cancel(id_);
}

void DiskTicket::promote(int64_t cls)
{
    check();
    TORCH_CHECK(cls >= 0 && cls < exl3_disk::kClasses, "disk engine: class out of range");
    eng_->promote(id_, (int) cls);
}

py::dict DiskTicket::times()
{
    check();
    exl3_disk::TicketTimes t = eng_->times(id_);
    py::dict d;
    d["t_submit"] = t.t_submit;
    d["t_first_issue"] = t.t_first_issue;
    d["t_done"] = t.t_done;
    d["ops"] = t.ops;
    return d;
}

int64_t DiskTicket::error()
{
    check();
    TORCH_CHECK(eng_->done(id_), "disk engine: ticket not complete");
    return eng_->wait(id_, 0);
}

void DiskTicket::release()
{
    if (released_) return;
    released_ = true;
    if (eng_)
    {
        if (Py_IsInitialized() && !finalizing() && PyGILState_Check())
        {
            py::gil_scoped_release nogil;
            eng_->release(id_);
        }
        else eng_->release(id_);
    }
    keep_.clear();          // with the GIL held again: a tensor's deleter may touch Python
    eng_.reset();
}

// ---- submissions --------------------------------------------------------------------------------

std::unique_ptr<DiskTicket> disk_gather_rows
(
    const at::Tensor& uids,
    int64_t uid_base,
    const std::vector<int64_t>& fds,
    const std::vector<int64_t>& offsets,
    const std::vector<int64_t>& row_bytes,
    const std::vector<at::Tensor>& outs,
    int64_t cls,
    bool hold,
    bool wait,
    const c10::optional<at::Tensor>& flag,
    int64_t flag_index,
    int64_t flag_value,
    int64_t deadline_ns,
    int64_t direct
)
{
    stamp(0);
    TORCH_CHECK(direct >= -1 && direct <= 1, "disk_gather_rows: direct must be -1, 0 or 1");
    TORCH_CHECK(uids.device().is_cpu() && uids.scalar_type() == at::kLong &&
                uids.is_contiguous() && uids.dim() == 1,
                "disk_gather_rows: uids must be a contiguous 1-D int64 CPU tensor");
    size_t nt = fds.size();
    TORCH_CHECK(nt >= 1 && nt <= 64 && offsets.size() == nt && row_bytes.size() == nt &&
                outs.size() == nt,
                "disk_gather_rows: fds, offsets, row_bytes and outs need one entry per table "
                "(1 to 64 tables)");
    std::vector<exl3_disk::RowTable> tabs(nt);
    std::vector<at::Tensor> keep;
    for (size_t k = 0; k < nt; ++k)
    {
        check_cpu_bytes(outs[k], "every out");
        tabs[k] = { checked_fd(fds[k]), offsets[k], row_bytes[k],
                    static_cast<uint8_t*>(outs[k].data_ptr()),
                    outs[k].numel() * (int64_t) outs[k].element_size() };
        if (!wait) keep.push_back(outs[k]);
    }
    exl3_disk::Options o;
    o.cls = cls < 0 ? exl3_disk::thread_class() : (int) std::min<int64_t>(cls, INT_MAX);
    o.hold = hold;
    o.deadline_ns = deadline_ns;
    o.flag = flag_ptr(flag, flag_index, flag_value, &keep);
    o.flag_value = (uint32_t) flag_value;
    o.direct = (int) direct;
    const int64_t* up = uids.data_ptr<int64_t>();
    int64_t n = uids.numel();

    std::shared_ptr<Engine> eng;
    uint64_t id = 0;
    int r = 0;
    {
        py::gil_scoped_release nogil;
        eng = exl3_disk::default_engine();
        if (wait)
        {
            // the engine's synchronous path: page-cache prepass (pools), inline submission
            // (io_uring, class 0), release included
            int64_t t_sub = 0;
            r = eng->gather_rows(up, n, uid_base, tabs.data(), (int) nt, o, &t_sub);
            if (t_prof)
            {
                t_times[1] = t_sub;
                t_times[2] = mono_ns();
            }
        }
        else
        {
            id = eng->submit_rows(up, n, uid_base, tabs.data(), (int) nt, o);
            stamp(1);
        }
    }
    if (wait)
    {
        stamp(3);
        if (r < 0) throw std::runtime_error("disk_gather_rows: read failed: " + errtext(-r));
        return nullptr;
    }
    std::unique_ptr<DiskTicket> t(new DiskTicket(eng, id, std::move(keep)));
    stamp(3);
    return t;
}

std::unique_ptr<DiskTicket> disk_read_extents
(
    const at::Tensor& fds,
    const at::Tensor& offsets,
    const at::Tensor& lengths,
    const at::Tensor& dst,
    const at::Tensor& dst_offsets,
    int64_t slot_bytes,
    const c10::optional<at::Tensor>& payload_offsets,
    int64_t cls,
    bool hold,
    bool wait,
    const c10::optional<at::Tensor>& flag,
    int64_t flag_index,
    int64_t flag_value,
    int64_t deadline_ns
)
{
    stamp(0);
    check_i64_vec(fds, -1, "fds");
    int64_t n = fds.numel();
    check_i64_vec(offsets, n, "offsets");
    check_i64_vec(lengths, n, "lengths");
    check_i64_vec(dst_offsets, n, "dst_offsets");
    check_cpu_bytes(dst, "dst");
    int64_t* pay = nullptr;
    if (payload_offsets.has_value())
    {
        check_i64_vec(payload_offsets.value(), n, "payload_offsets");
        pay = payload_offsets.value().data_ptr<int64_t>();
    }
    TORCH_CHECK(slot_bytes > 0, "disk_read_extents: slot_bytes must be positive");
    int64_t dst_bytes = dst.numel() * (int64_t) dst.element_size();
    uint8_t* base = static_cast<uint8_t*>(dst.data_ptr());
    const int64_t* fp = fds.data_ptr<int64_t>();
    const int64_t* op = offsets.data_ptr<int64_t>();
    const int64_t* lp = lengths.data_ptr<int64_t>();
    const int64_t* dp = dst_offsets.data_ptr<int64_t>();
    std::vector<exl3_disk::ExtentReq> ex((size_t) n);
    for (int64_t i = 0; i < n; ++i)
    {
        TORCH_CHECK(dp[i] >= 0 && dp[i] <= dst_bytes && slot_bytes <= dst_bytes - dp[i],
                    "disk_read_extents: slot ", i, " at ", dp[i], " (+", slot_bytes,
                    ") is outside dst (", dst_bytes, " bytes)");
        ex[(size_t) i] = { checked_fd(fp[i]), op[i], lp[i], base + dp[i], slot_bytes };
    }
    std::vector<at::Tensor> keep;
    if (!wait) keep.push_back(dst);
    exl3_disk::Options o;
    o.cls = cls < 0 ? exl3_disk::thread_class() : (int) std::min<int64_t>(cls, INT_MAX);
    o.hold = hold;
    o.deadline_ns = deadline_ns;
    o.flag = flag_ptr(flag, flag_index, flag_value, &keep);
    o.flag_value = (uint32_t) flag_value;

    std::shared_ptr<Engine> eng;
    uint64_t id = 0;
    int r = 0;
    {
        py::gil_scoped_release nogil;
        eng = exl3_disk::default_engine();
        if (wait)
        {
            int64_t t_sub = 0;
            r = eng->read_extents(ex.data(), n, o, pay, &t_sub);
            if (t_prof)
            {
                t_times[1] = t_sub;
                t_times[2] = mono_ns();
            }
        }
        else
        {
            id = eng->submit_extents(ex.data(), n, o, pay);
            stamp(1);
        }
    }
    if (wait)
    {
        stamp(3);
        if (r < 0) throw std::runtime_error("disk_read_extents: read failed: " + errtext(-r));
        return nullptr;
    }
    std::unique_ptr<DiskTicket> t(new DiskTicket(eng, id, std::move(keep)));
    stamp(3);
    return t;
}

std::tuple<int64_t, int64_t> disk_extent_geometry(int64_t fd, int64_t offset, int64_t length)
{
    std::shared_ptr<Engine> eng = exl3_disk::default_engine();
    exl3_disk::ExtentGeom g = eng->extent_geometry(checked_fd(fd), offset, length);
    return std::make_tuple(g.payload_offset, g.span_bytes);
}

void disk_arm_hold()
{
    exl3_disk::default_engine()->arm_hold();
}

// ---- introspection ------------------------------------------------------------------------------

py::dict disk_stats(bool reset)
{
    exl3_disk::Stats s;
    {
        std::shared_ptr<Engine> eng = exl3_disk::default_engine();
        py::gil_scoped_release nogil;
        s = eng->stats(reset);
    }
    py::list classes;
    for (int c = 0; c < exl3_disk::kClasses; ++c)
    {
        const exl3_disk::ClassStats& x = s.cls[c];
        py::dict d;
        d["tickets"] = x.tickets;
        d["ops"] = x.ops;
        d["op_errors"] = x.op_errors;
        d["ops_cancelled"] = x.ops_cancelled;
        d["bytes"] = x.bytes;
        d["held"] = x.held;
        d["aged"] = x.aged;
        d["max_inflight"] = x.max_inflight;
        d["max_inflight_bytes"] = x.max_inflight_bytes;
        d["hist_queue"] = x.hist_queue;
        d["hist_service"] = x.hist_service;
        d["hist_ticket"] = x.hist_ticket;
        classes.append(d);
    }
    std::vector<int64_t> lower((size_t) exl3_disk::kHistBuckets);
    for (int i = 0; i < exl3_disk::kHistBuckets; ++i) lower[(size_t) i] = exl3_disk::hist_bucket_lower(i);
    py::dict d;
    d["classes"] = classes;
    d["hist_lower_ns"] = lower;
    d["inline_reaped"] = s.inline_reaped;
    d["reaper_reaped"] = s.reaper_reaped;
    d["enters"] = s.enters;
    d["resubmits"] = s.resubmits;
    d["fixed_plain"] = s.fixed_plain;
    d["stray_cqes"] = s.stray_cqes;
    d["keepalive_reads"] = s.keepalive_reads;
    d["inflight_now"] = s.inflight_now;
    d["queued_ops_now"] = s.queued_ops_now;
    d["tickets_live"] = s.tickets_live;
    d["files_open"] = s.files_open;
    return d;
}

static py::dict info_of(const std::shared_ptr<Engine>& eng)
{
    const exl3_disk::Config& c = eng->config();
    py::dict d;
    d["backend"] = std::string(exl3_disk::backend_name(c.backend));
    d["backend_named"] = c.backend_named;
    d["ngram_engine"] = c.ngram_engine;
    d["auto_backend"] = std::string(exl3_disk::backend_name(exl3_disk::auto_backend()));
    d["auto_ngram_route"] = std::string(exl3_disk::auto_ngram_engine() ? "engine" : "original");
    d["auto_keepalive_ms"] = exl3_disk::auto_keepalive_ms();
    d["keepalive_ms"] = c.keepalive_ms;
    d["keepalive_idle_s"] = c.keepalive_idle_s;
    d["direct_rows"] = c.direct_rows;
    d["direct_extents"] = c.direct_extents;
    d["threads"] = c.threads;
    d["qd"] = c.qd;
    d["reserve0"] = c.reserve0;
    d["engram_qd"] = c.engram_qd;
    py::list hi, lo;
    for (int k = 0; k < exl3_disk::kClasses; ++k)
    {
        hi.append(c.window_hi[k]);
        lo.append(c.window_lo[k]);
    }
    d["window_hi"] = hi;
    d["window_lo"] = lo;
    d["chunk_bytes"] = c.chunk_bytes;
    d["align"] = c.align;
    d["sqpoll"] = c.sqpoll;
    d["iopoll"] = c.iopoll;
    d["reg_files"] = c.reg_files;
    d["reg_buffers"] = c.reg_buffers;
    d["describe"] = eng->describe();
    d["fallbacks"] = eng->fallbacks();
    d["notes"] = c.notes;
    d["ngram_route"] = disk_ngram_route();
    return d;
}

py::dict disk_engine_info()
{
    return info_of(exl3_disk::default_engine());
}

py::dict disk_engine_configure(const py::dict& overrides)
{
    exl3_disk::Overrides ov;
    for (auto item : overrides)
    {
        std::string k = py::str(item.first);
        py::handle v = item.second;
        std::string s;
        if (py::isinstance<py::bool_>(v)) s = v.cast<bool>() ? "1" : "0";
        else s = py::str(v);
        ov[k] = s;
    }
    std::shared_ptr<Engine> eng;
    {
        py::gil_scoped_release nogil;
        eng = exl3_disk::configure_default(ov);
    }
    prune_registered();         // buffers of the replaced engine, if it is gone
    return info_of(eng);
}

void disk_engine_shutdown()
{
    {
        py::gil_scoped_release nogil;
        exl3_disk::shutdown_default();
    }
    prune_registered();
}

int64_t disk_engine_forget(const std::string& path)
{
    std::shared_ptr<Engine> eng = exl3_disk::default_engine_if_created();
    if (!eng) return 0;                     // no engine: nothing of the file is open
    py::gil_scoped_release nogil;
    return eng->forget_path(path);
}

int64_t disk_set_thread_class(int64_t cls)
{
    TORCH_CHECK(cls >= 0 && cls < exl3_disk::kClasses, "disk engine: class out of range");
    return exl3_disk::set_thread_class((int) cls);
}

int64_t disk_set_thread_direct(int64_t direct)
{
    TORCH_CHECK(direct >= -1 && direct <= 1, "disk engine: direct must be -1, 0 or 1");
    return exl3_disk::set_thread_direct((int) direct);
}

std::string disk_ngram_route()
{
    return exl3_disk::ngram_route_engine() ? "engine" : "original";
}

int64_t disk_register_buffer(const at::Tensor& t)
{
    check_cpu_bytes(t, "the buffer");
    prune_registered();
    std::shared_ptr<Engine> eng = exl3_disk::default_engine();
    int idx = eng->register_buffer(t.data_ptr(), (size_t) (t.numel() * (int64_t) t.element_size()));
    if (idx < 0) return -1;
    std::lock_guard<std::mutex> lk(g_reg_mx);
    g_reg->push_back({ eng, idx, t });
    return idx;
}

void disk_unregister_buffer(int64_t idx)
{
    install_reg_fork_handler();
    std::shared_ptr<Engine> eng = exl3_disk::default_engine();
    at::Tensor keep;
    std::vector<at::Tensor> drop;           // dropped after g_reg_mx, with the GIL held
    {
        std::lock_guard<std::mutex> lk(g_reg_mx);
        drop = prune_registered_locked();
        auto it = std::find_if(g_reg->begin(), g_reg->end(), [&](const RegBuf& b)
                               { return b.idx == idx && b.eng.lock() == eng; });
        TORCH_CHECK(it != g_reg->end(), "disk engine: no registered buffer ", idx);
        eng->unregister_buffer((int) idx);          // throws EBUSY while reads target it
        keep = it->t;
        g_reg->erase(it);
    }
}

void disk_profile(bool enable)
{
    t_prof = enable;
}

std::tuple<int64_t, int64_t, int64_t, int64_t> disk_profile_last()
{
    return std::make_tuple(t_times[0], t_times[1], t_times[2], t_times[3]);
}

at::Tensor disk_trace(bool clear)
{
    std::vector<exl3_disk::TraceRec> tr = exl3_disk::default_engine()->trace(clear);
    at::Tensor t = at::empty({ (int64_t) tr.size(), 8 }, at::kLong);
    int64_t* p = t.data_ptr<int64_t>();
    for (size_t i = 0; i < tr.size(); ++i)
    {
        const exl3_disk::TraceRec& r = tr[i];
        int64_t* q = p + i * 8;
        q[0] = (int64_t) r.ticket;
        q[1] = r.cls;
        q[2] = r.res;
        q[3] = r.dev_off;
        q[4] = r.dev_len;
        q[5] = r.t_submit;
        q[6] = r.t_issue;
        q[7] = r.t_done;
    }
    return t;
}
