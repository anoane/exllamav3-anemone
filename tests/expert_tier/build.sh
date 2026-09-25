#!/usr/bin/env bash
# Build the expert tier's standalone test drivers in three variants and run them:
#   release   -O2, -Werror, _FORTIFY_SOURCE, _GLIBCXX_ASSERTIONS
#   asan      AddressSanitizer + UndefinedBehaviorSanitizer (leaks checked, first error fatal)
#   tsan      ThreadSanitizer (first report fatal)
#
#   tier_policy_test   the policy core alone (exllamav3_ext/tier/tier_policy.cpp)
#   tier_host_test     the RAM tier host with the disk engine (tier_host.cpp, disk/*.cpp): synthetic
#                      shards in the scratch directory, every byte checked after every call
#
# usage: tests/expert_tier/build.sh [--build-dir DIR] [--scratch DIR] [--modes "release asan tsan"]
#                                   [--no-run] [-- driver arguments, e.g. --quick]
#
# The scratch directory must be on a real file system (O_DIRECT). No torch, no CUDA: g++ and libc only.

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
warn="$warn -Wduplicated-cond -Wlogical-op -Wnull-dereference -Wconversion -Wsign-conversion"
warn="$warn -Werror"
common="-std=c++17 -g -pthread -I$ext $warn -D_GLIBCXX_ASSERTIONS"

flags_for() {
    case "$1" in
        release) echo "-O2 -D_FORTIFY_SOURCE=2" ;;
        asan) echo "-O1 -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all" ;;
        tsan) echo "-O1 -fno-omit-frame-pointer -fsanitize=thread" ;;
        *) echo "unknown mode $1" >&2; exit 2 ;;
    esac
}

policy_src="$ext/tier/tier_policy.cpp $here/tier_policy_test.cpp"
host_src="$ext/tier/tier_policy.cpp $ext/tier/tier_host.cpp $ext/disk/disk_config.cpp $ext/disk/disk_engine.cpp"
host_src="$host_src $ext/disk/disk_uring.cpp $here/tier_host_test.cpp"
pids=()
for m in $modes; do
    # shellcheck disable=SC2046,SC2086
    $CXX $common $(flags_for "$m") $policy_src -o "$out/tier_policy_test_$m" &
    pids+=($!)
    # shellcheck disable=SC2046,SC2086
    $CXX $common $(flags_for "$m") $host_src -o "$out/tier_host_test_$m" &
    pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
[ "$status" = 0 ] || { echo "build FAILED" >&2; exit 1; }
echo "built: tier_policy_test and tier_host_test, modes: $modes -> $out"
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
    if ! env "${env[@]}" $norand "$out/tier_policy_test_$m" "${args[@]}"; then
        echo "=== $m FAILED (tier_policy_test)"
        status=1
    fi
    # shellcheck disable=SC2086
    if ! env "${env[@]}" $norand "$out/tier_host_test_$m" --dir "$scratch/$m" "${args[@]}"; then
        echo "=== $m FAILED (tier_host_test)"
        status=1
    fi
    rm -rf "${scratch:?}/$m"
done
exit $status
