# Shared hardware ownership for agent runs

Lucy and Astor are shared by the DeepSeek and Kimi worktrees. Every model run,
GPU/operator probe, or heavy CPU/RAM/storage model test must reserve its hosts
before starting. Source editing and ordinary builds can continue independently.
On these single-GPU hosts, use the same keys for CPU model tests so they do not
compete with an inference measurement's RAM and storage traffic.

The canonical CLI is `/mnt/code/projects/Strata/tools/hardware_lock.py`, including
when working in another checkout. Reservations live at
`/mnt/storage/strata-hardware/locks/{lucy.gpu0,astor.gpu0}/owner.json`.
Atomic directory creation arbitrates ownership on the shared NFS server; this
does not depend on client-local flock behavior or a private `nolock` mount.

Check ownership:

```sh
uv run --no-project --with pydantic python /mnt/code/projects/Strata/tools/hardware_lock.py status
```

Wrap the complete job, including service preparation and restoration:

```sh
uv run --no-project --with pydantic python /mnt/code/projects/Strata/tools/hardware_lock.py run \
  --resource lucy.gpu0 --resource astor.gpu0 \
  --owner kimi-k2 --thread YOUR_THREAD_ID -- bash /absolute/path/run-job.sh
```

A busy resource reports `HARDWARE_BUSY`, its owner/thread/job, and exit 75. The
command does not start. Multi-host acquisition rolls back its own partial locks
if any requested host is busy. Normal command success/failure releases the locks.
Reserve only the hosts used by the job. Do not wrap a launcher that returns before
its job finishes, such as `systemctl start` or a detached shell command.

For an already-running **owned** user service, `hold` records the reservation
until that service becomes inactive or failed, including its `ExecStopPost`:

```sh
uv run --no-project --with pydantic python /mnt/code/projects/Strata/tools/hardware_lock.py hold \
  --resource lucy.gpu0 --resource astor.gpu0 \
  --owner deepseek-q3-retry5 --thread YOUR_THREAD_ID --unit YOUR_EXISTING_UNIT.service
```

Q3 retry5 used `strata-q3-hardware-lease-20261111.service` to hold its reservation
through cleanup and restoration of Astor's generation service. That job has
finished and released its reservations. Always read `status` for current ownership.

Records include a unique ownership token, controller host/PID, UTC acquisition
time, thread, resources and command or service. There is **no automatic expiry
or lock stealing**. A crashed holder can leave a persistent reservation. Inspect
the owner and job, establish that all its local/remote processes and cleanup have
finished, then release explicitly using the recorded token:

```sh
uv run --no-project --with pydantic python /mnt/code/projects/Strata/tools/hardware_lock.py release \
  --resource lucy.gpu0 --resource astor.gpu0 --token RECORDED_TOKEN
```

Never delete or release an active job's reservation to obtain capacity. Coordinate
with its owning thread. Do not change another job's services or process state.
These are cooperative reservations: they gate participating agent launchers,
not arbitrary desktop applications or manually launched GPU processes.
