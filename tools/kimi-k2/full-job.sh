#!/usr/bin/env bash
# Run only inside the canonical two-host hardware lease; retain it through cleanup.
set -euo pipefail
if [[ $# != 1 && $# != 2 ]]; then
    echo "usage: hardware_lock.py run ... -- bash $0 /mnt/storage/.../NEW_CAPTURE_DIRECTORY" >&2
    exit 2
fi
run_dir=$1
case "$run_dir" in /mnt/storage/*) ;; *) echo "captures must use shared /mnt/storage" >&2; exit 2 ;; esac
uv run --no-project --with pydantic python - "$PPID" <<'PY'
import socket, sys
from pathlib import Path
sys.path.insert(0, "/mnt/code/projects/Strata/tools")
from hardware_lock import read_owner
records = [read_owner(Path("/mnt/storage/strata-hardware/locks") / key)
           for key in ("lucy.gpu0", "astor.gpu0")]
if any(r.owner != "kimi-k2" or r.thread != "d7f95b1d-9358-49ce-8e05-b44115b911b6"
       or r.host != socket.gethostname() or r.pid != int(sys.argv[1]) for r in records):
    raise SystemExit("full-job.sh requires its own canonical Kimi two-host lease")
if records[0].token != records[1].token:
    raise SystemExit("host reservation tokens differ")
PY

model_root=/mnt/storage/models/Kimi-K2-Instruct-UD-IQ3_XXS-23bf90d8196d
model=$model_root/UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00001-of-00009.gguf
cuda_root=/mnt/work/kimi-k2/build-cuda
hip_split=/var/tmp/strata-kimi2-split-short-hip
hip_worker=/var/tmp/strata-kimi2-worker-timeout-hip
profile=$run_dir/profile
mkdir "$run_dir"
cp -a "${2:-$model_root/profile}" "$profile"
chmod u+w "$profile/suffix-binary.identity"
cp /mnt/work/kimi-k2/k7/run_native.py "$run_dir/run_native.py"
cp /mnt/storage/models/strata-kimi-k2-evidence/k7-run7/sample_native.py "$run_dir/sample_native.py"
on_astor() { local command; printf -v command '%q ' "$@"; ssh -o BatchMode=yes astor "$command"; }
on_astor sha256sum "$hip_split" | cut -d ' ' -f 1 > "$profile/suffix-binary.identity"

# Cleanup uses only this fresh capture directory's native PIDs and matching argv.
stop_code=$(cat <<'PY'
from pathlib import Path
import json, os, signal, sys, time
root = Path(sys.argv[1])
for role in sys.argv[2:]:
    record = root / (role + ".process.json")
    if not record.exists():
        continue
    pid = json.loads(record.read_text())["pid"]
    proc = Path("/proc") / str(pid)
    def owned():
        try:
            return any(v.startswith(str(root).encode() + b"/")
                       for v in (proc / "cmdline").read_bytes().split(b"\0"))
        except FileNotFoundError:
            return False
    if owned():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        while owned():
            time.sleep(0.2)
PY
)
children=()
cleanup() {
    local result=$?
    trap - EXIT INT TERM
    until python3 -c "$stop_code" "$run_dir" prefix lucy-expert; do
        echo "Local cleanup not verified; retaining hardware lease and retrying." >&2
        sleep 5
    done
    until on_astor python3 -c "$stop_code" "$run_dir" suffix astor-expert; do
        echo "Remote cleanup not verified; retaining hardware lease and retrying." >&2
        sleep 5
    done
    for child in "${children[@]}"; do
        if wait "$child"; then :; else
            child_result=$?
            echo "Owned job $child exited $child_result during cleanup." >&2
            if ((result == 0)); then result=$child_result; fi
        fi
    done
    exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_ready() {
    local role=$1 pattern=$2
    for ((attempt=0; attempt<600; attempt++)); do
        if [[ -f $run_dir/$role.log ]] && rg -q "$pattern" "$run_dir/$role.log"; then return; fi
        if [[ -f $run_dir/$role.exit.json ]]; then cat "$run_dir/$role.log" >&2; return 1; fi
        sleep 0.25
    done
    echo "Timed out waiting for $role: $pattern" >&2
    return 1
}

python3 "$run_dir/run_native.py" "$run_dir/lucy-expert" "$cuda_root/strata-kimi2-expert-worker" \
    --kimi2-model "$model" --kimi2-identity "$profile/fixture.identity" \
    --bind 192.168.60.155:19472 --timeout-ms 3600000 --max-sessions 1 --verbose &
children+=("$!"); lucy_worker_job=$!
on_astor python3 "$run_dir/run_native.py" "$run_dir/astor-expert" env GPU_MAX_HW_QUEUES=1 "$hip_worker" \
    --kimi2-model "$model" --kimi2-identity "$profile/fixture.identity" \
    --bind 192.168.63.177:19473 --timeout-ms 3600000 --max-sessions 1 --verbose &
children+=("$!"); astor_worker_job=$!
wait_ready lucy-expert WORKER_LISTENING
wait_ready astor-expert WORKER_LISTENING
on_astor python3 "$run_dir/run_native.py" "$run_dir/suffix" env GPU_MAX_HW_QUEUES=1 "$hip_split" \
    worker "$profile" 192.168.63.177:19475 192.168.60.155:19472 "$run_dir/suffix" --model "$model" &
children+=("$!"); suffix_job=$!
wait_ready suffix K2_SUFFIX_LISTENING
python3 "$run_dir/run_native.py" "$run_dir/prefix" "$cuda_root/strata-kimi2-split" \
    coordinator "$profile" 192.168.63.177:19475 192.168.63.177:19473 "$run_dir/prefix" --model "$model" &
children+=("$!"); prefix_job=$!
while [[ ! -f $run_dir/prefix.process.json ]]; do kill -0 "$prefix_job"; sleep 0.1; done
runtime_code=$(cat <<'PY'
from pathlib import Path
import hashlib, json, os, sys, time
root = Path(sys.argv[1])
for role in sys.argv[2:]:
    pid = json.loads((root / (role + ".process.json")).read_text())["pid"]
    proc = Path("/proc") / str(pid)
    with (proc / "exe").open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    environment = dict(v.split(b"=", 1) for v in (proc / "environ").read_bytes().split(b"\0") if b"=" in v)
    record = {"time": time.time(), "pid": pid, "exe": os.readlink(proc / "exe"), "binary_sha256": digest,
              "argv": (proc / "cmdline").read_bytes().decode().split("\0")[:-1],
              "GPU_MAX_HW_QUEUES": environment.get(b"GPU_MAX_HW_QUEUES", b"").decode()}
    (root / (role + ".runtime.json")).write_text(json.dumps(record, indent=2) + "\n")
PY
)
python3 -c "$runtime_code" "$run_dir" prefix lucy-expert
on_astor python3 -c "$runtime_code" "$run_dir" suffix astor-expert
python3 "$run_dir/sample_native.py" "$run_dir" prefix cuda &
children+=("$!")
on_astor python3 "$run_dir/sample_native.py" "$run_dir" suffix hip &
children+=("$!")
wait "$prefix_job"
wait "$suffix_job"
wait "$lucy_worker_job"
wait "$astor_worker_job"
