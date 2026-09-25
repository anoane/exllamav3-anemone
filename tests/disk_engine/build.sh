#!/usr/bin/env bash
# Build the disk engine's standalone test driver and benchmark driver (disk_engine_bench) in
# three variants, then run the test driver in each:
#   release   -O2, -Werror, _FORTIFY_SOURCE, _GLIBCXX_ASSERTIONS
#   asan      AddressSanitizer + UndefinedBehaviorSanitizer (leaks checked, first error fatal)
#   tsan      ThreadSanitizer (first report fatal)
#
# usage: tests/disk_engine/build.sh [--build-dir DIR] [--scratch DIR] [--modes "release asan tsan"]
#                                   [--no-run] [-- driver arguments, e.g. --quick]
#
# The scratch directory must be on the file system to test (the driver creates ~50 MB of
# files there and removes them). No torch, no CUDA: a C++17 compiler and libc only, g++ unless
# CXX names another (CXX=clang++ builds with clang, without GCC's two extra warnings).

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
ext="$root/exllamav3/exllamav3_ext"
out="$here/build"
scratch=""
modes="release asan tsan"
run=1
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --build-dir) out=$2; shift 2 ;;
        --scratch) scratch=$2; shift 2 ;;
        --modes) modes=$2; shift 2 ;;
        --no-run) run=0; shift ;;
        --) shift; args=("$@"); break ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -n "$scratch" ] || scratch="$out/scratch"
mkdir -p "$out" "$scratch"

CXX=${CXX:-g++}
warn="-Wall -Wextra -Wshadow -Wformat=2 -Wcast-qual -Wnon-virtual-dtor -Woverloaded-virtual"
warn="$warn -Wnull-dereference -Wconversion -Wsign-conversion -Werror"
# GCC only: clang rejects warning options it does not know under -Werror
# shellcheck disable=SC2086
if ! $CXX --version 2> /dev/null | grep -qi clang; then warn="$warn -Wduplicated-cond -Wlogical-op"; fi
src="$ext/disk/disk_config.cpp $ext/disk/disk_engine.cpp $ext/disk/disk_uring.cpp"
src="$src $here/disk_engine_test.cpp"
common="-std=c++17 -g -pthread -I$ext $warn -D_GLIBCXX_ASSERTIONS"

flags_for() {
    case "$1" in
        release) echo "-O2 -D_FORTIFY_SOURCE=2" ;;
        asan) echo "-O1 -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all" ;;
        tsan) echo "-O1 -fno-omit-frame-pointer -fsanitize=thread" ;;
        *) echo "unknown mode $1" >&2; exit 2 ;;
    esac
}

pids=()
bench_src="$ext/disk/disk_config.cpp $ext/disk/disk_engine.cpp $ext/disk/disk_uring.cpp"
bench_src="$bench_src $here/disk_engine_bench.cpp"
for m in $modes; do
    # shellcheck disable=SC2046
    $CXX $common $(flags_for "$m") $src -o "$out/disk_engine_test_$m" &
    pids+=($!)
    # the benchmark driver: disk_engine_bench (release), disk_engine_bench_asan, ..._tsan
    suffix="_$m"
    [ "$m" = release ] && suffix=""
    # shellcheck disable=SC2046,SC2086
    $CXX $common $(flags_for "$m") $bench_src -o "$out/disk_engine_bench$suffix" &
    pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
[ "$status" = 0 ] || { echo "build FAILED" >&2; exit 1; }
echo "built: disk_engine_test and disk_engine_bench, modes: $modes -> $out"
[ "$run" = 1 ] || exit 0

# Sanitizer runtimes can fail to map their shadow memory under the high mmap randomization of
# recent kernels; setarch -R disables randomization for the test process only.
norand=""
if command -v setarch > /dev/null; then norand="setarch $(uname -m) -R"; fi

status=0
for m in $modes; do
    echo "=== $m"
    case "$m" in
        release) env=() ;;
        asan) env=(ASAN_OPTIONS=detect_leaks=1:abort_on_error=1:detect_stack_use_after_return=1:check_initialization_order=1
                   UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1) ;;
        tsan) env=(TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1) ;;
    esac
    # shellcheck disable=SC2086
    if ! env "${env[@]}" $norand "$out/disk_engine_test_$m" --dir "$scratch/$m" "${args[@]}"; then
        echo "=== $m FAILED"
        status=1
    fi
    rm -rf "${scratch:?}/$m"
done
exit $status
