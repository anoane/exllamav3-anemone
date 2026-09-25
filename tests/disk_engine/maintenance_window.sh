#!/usr/bin/env bash
# The disk engine's full benchmark on a model's own files, and the decision that sets what
# EXL3_DISK_BACKEND=auto means on this host. For a maintenance window: it reads tens of GB from
# the disk the model lives on, so nothing else may use that disk meanwhile. It stops nothing: it
# checks that the service is down and the host idle, and refuses otherwise.
#
#   nohup tests/disk_engine/maintenance_window.sh --model DIR --service UNIT \
#       --env-label "how the disk reaches this machine" > window.log 2>&1 &
#
# Options:
#   --model DIR         the checkpoint (DeepSeek-V4.1 EXL3: engram tables and expert extents)
#   --out DIR           results (default: disk-window-<date> in the current directory)
#   --build-dir DIR     where the drivers are built (default: <out>/build; removed at the end
#                       unless --keep-build)
#   --budget-gb G       bytes read from the device, all runs together (default 100: the
#                       window preset's upper estimate is ~86 GB); the plan and its estimate
#                       are printed first; blocks run in order, so a smaller budget drops the
#                       last sweeps first
#   --scale F           multiply every iteration count (default 1; 0.5 halves time and reads)
#   --repeats R         decision runs per backend (default 3)
#   --service UNIT      a systemd unit that must be inactive (repeatable)
#   --env-label TEXT    the storage path, e.g. "NVMe passthrough" or "virtio-scsi, iothread"
#   --env-notes TEXT    anything else worth knowing about the host
#   --python-exe PY     Python with torch and numpy for the Python-share runs (default python3)
#   --python off        skip the Python runs (then the n-gram route is not decided)
#   --max-idle-cores C  other load allowed before starting (default 1.0 cores over 5 s)
#   --lock FILE         hold this flock(1) lock for the whole run and refuse to start while
#                       another job holds it, e.g. the lock the host's GPU jobs take (default:
#                       no lock)
#   --apply             write the decision into exllamav3_ext/disk/disk_auto.h of this tree,
#                       rebuild, and run the configuration tests against it; the patch is left
#                       in <out>/disk_auto.patch for the commit
#   --no-tests          skip the release test driver run before benchmarking
#   --keep-build        keep the build directory
#
# Preconditions it checks: every --service inactive; no other process has the engram files
# open; the --lock file free (held for the whole run); the host idle; room for the results.
# It never drops caches (cold reads drop exactly the pages they read, and every page the
# benchmark touched is put back as it found it), never changes read_ahead_kb, never kills
# anything.

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
model=""
out=""
build=""
budget=100
scale=1
repeats=3
services=()
label=""
notes=""
python_exe=python3
python_mode=mini
max_idle=1.0
apply=0
tests=1
keep_build=0
lock=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model) model=$2; shift 2 ;;
        --out) out=$2; shift 2 ;;
        --build-dir) build=$2; shift 2 ;;
        --budget-gb) budget=$2; shift 2 ;;
        --scale) scale=$2; shift 2 ;;
        --repeats) repeats=$2; shift 2 ;;
        --service) services+=("$2"); shift 2 ;;
        --env-label) label=$2; shift 2 ;;
        --env-notes) notes=$2; shift 2 ;;
        --python-exe) python_exe=$2; shift 2 ;;
        --python) python_mode=$2; shift 2 ;;
        --max-idle-cores) max_idle=$2; shift 2 ;;
        --lock) lock=$2; shift 2 ;;
        --apply) apply=1; shift ;;
        --no-tests) tests=0; shift ;;
        --keep-build) keep_build=1; shift ;;
        *) echo "unknown argument $1 (see the header of $0)" >&2; exit 2 ;;
    esac
done
die() { echo "maintenance_window: $*" >&2; exit 1; }
say() { echo "[$(date +%H:%M:%S)] $*"; }

[ -n "$model" ] && [ -f "$model/model.safetensors.index.json" ] || die "--model DIR with model.safetensors.index.json is required"
# absolute: the test extension is built from another directory, and relative paths would
# land there
abspath() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
[ -n "$out" ] || out="disk-window-$(date +%Y%m%d-%H%M)"
out=$(abspath "$out")
[ -n "$build" ] || build="$out/build"
build=$(abspath "$build")
[ -e "$out/native.jsonl" ] && die "$out already holds results"
mkdir -p "$out"
[ -n "$label" ] || say "WARNING: no --env-label: the report cannot say how the disk reaches this machine"

# ---- preconditions -------------------------------------------------------------------------------
if [ -n "$lock" ]; then
    exec 9> "$lock"
    flock -n 9 || die "$lock is held by another job"
fi

for u in "${services[@]}"; do
    if systemctl is-active --quiet "$u"; then die "service $u is active: stop it first (this script stops nothing)"; fi
done

# no other process may read the engram files meanwhile (a server that is still up, a loader)
mapfile -t engram_files < <("$python_exe" - "$model" <<'EOF'
import json, os, sys
d = sys.argv[1]
m = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
for f in sorted({v for k, v in m.items() if ".engram.embed." in k}):
    print(os.path.realpath(os.path.join(d, f)))
EOF
)
[ "${#engram_files[@]}" -gt 0 ] || die "$model has no engram tables"
for f in "${engram_files[@]}"; do
    holders=$(find /proc/[0-9]*/fd -lname "$f" 2> /dev/null | cut -d/ -f3 | sort -u | tr '\n' ' ' || true)
    [ -z "$holders" ] || die "$f is open in process(es) $holders: stop them first"
done

# idle: other load over 5 s
busy=$("$python_exe" - <<'EOF'
import time
def j():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:9]]
    return sum(v) - v[3] - v[4]
a = j(); time.sleep(5); b = j()
import os
print(f"{(b - a) / os.sysconf('SC_CLK_TCK') / 5:.2f}")
EOF
)
awk -v b="$busy" -v m="$max_idle" 'BEGIN { exit !(b <= m) }' || die "host not idle: $busy cores busy over 5 s (limit $max_idle)"
avail_kb=$(df -Pk "$out" | awk 'NR == 2 { print $4 }')
[ "$avail_kb" -gt 1048576 ] || die "less than 1 GB free under $out"
say "preconditions met: services down, engram files closed, $busy cores busy${lock:+, $lock held}"

# ---- build ---------------------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0;12.0a"}
export EXL3_DISK_MINI_BUILD_DIR="$build/mini_ext"
modes=release
say "building into $build"
nice -n 10 "$here/build.sh" --build-dir "$build" --modes "$modes" --no-run
if [ "$python_mode" != off ]; then
    # fail now, not an hour into the window: the Python-share runs decide the n-gram route
    "$python_exe" -c "import torch, numpy" 2> /dev/null \
        || die "$python_exe cannot import torch and numpy (needed for the Python runs; --python off skips them)"
    if [ "$python_mode" = mini ]; then
        [ -x "$CUDA_HOME/bin/nvcc" ] || die "no nvcc in $CUDA_HOME/bin (the test extension builds ngram.cu); set CUDA_HOME"
        say "building the test extension into $EXL3_DISK_MINI_BUILD_DIR"
        (cd "$here" && MAX_JOBS=8 nice -n 10 "$python_exe" -c "from mini_ext import load_ext; load_ext()") \
            > "$out/mini_build.log" 2>&1 || die "the test extension did not build: $out/mini_build.log"
    fi
fi
if [ "$tests" = 1 ]; then
    say "release test driver (quick) as a sanity gate"
    nice -n 10 "$here/build.sh" --build-dir "$build" --modes release --scratch "$build/scratch" -- --quick \
        > "$out/tests_release.log" 2>&1 || die "the test driver failed: $out/tests_release.log"
    tail -1 "$out/tests_release.log"
fi

# ---- benchmark -----------------------------------------------------------------------------------
"$python_exe" "$here/bench_suite.py" plan --preset window --scale "$scale"
say "running (budget $budget GB)"
"$python_exe" "$here/bench_suite.py" run --preset window --scale "$scale" --repeats "$repeats" \
    --model "$model" --out "$out" --driver "$build/disk_engine_bench" --budget-gb "$budget" \
    --python "$python_mode" --python-exe "$python_exe" --env-label "$label" --env-notes "$notes"

# ---- decision ------------------------------------------------------------------------------------
decide_rc=0
if [ "$apply" = 1 ]; then
    "$python_exe" "$here/bench_suite.py" decide "$out" --apply "$root" || decide_rc=$?
    if [ "$decide_rc" = 0 ]; then
        say "rebuilding with the new auto choice and running the configuration tests"
        nice -n 10 "$here/build.sh" --build-dir "$build" --modes release --scratch "$build/scratch" -- --only config \
            > "$out/tests_auto.log" 2>&1 || die "configuration tests failed with the new disk_auto.h: $out/tests_auto.log"
        nice -n 10 "$here/build.sh" --build-dir "$build" --modes release --scratch "$build/scratch" -- --only default-engine \
            >> "$out/tests_auto.log" 2>&1 || die "default-engine tests failed with the new disk_auto.h: $out/tests_auto.log"
        grep -E "ALL PASSED|FAILED" "$out/tests_auto.log" || true
    fi
else
    "$python_exe" "$here/bench_suite.py" decide "$out" --src "$root" || decide_rc=$?
fi

[ "$keep_build" = 1 ] || rm -rf "${build:?}"
say "report:   $out/report.md"
say "decision: $out/decision.md"
[ "$apply" = 1 ] && [ "$decide_rc" = 0 ] && say "patch:    $out/disk_auto.patch (commit it with the report's numbers)"
exit "$decide_rc"
