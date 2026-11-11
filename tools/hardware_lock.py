"""Serialize agent GPU jobs across hosts using shared, atomic NFS directories."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

DEFAULT_ROOT = Path("/mnt/storage/strata-hardware/locks")
Resource = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")]


class Owner(BaseModel):
    """One job's identity, replicated into each exclusively reserved resource."""

    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    token: str
    owner: str
    thread: str
    host: str
    pid: int
    acquired_utc: str
    resources: list[Resource]
    command: list[str]
    unit: str | None = None


def read_owner(directory: Path) -> Owner:
    """Read published ownership; absent/malformed metadata never means free."""
    return Owner.model_validate_json((directory / "owner.json").read_bytes())


class Lease:
    """Acquire all requested resources or roll back this acquisition only."""

    def __init__(self, root: Path, record: Owner):
        """Bind a shared directory and the same owner token for every resource."""
        self.root, self.record = root, record
        self.held: list[Path] = []

    def acquire(self) -> None:
        """Use server-side mkdir exclusion, including on NFS mounts without flock."""
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            for resource in sorted(set(self.record.resources)):
                directory = self.root / resource
                try:
                    directory.mkdir()
                except FileExistsError as error:
                    path = directory / "owner.json"
                    detail = (
                        path.read_text()
                        if path.exists()
                        else "owner metadata initializing or unresolved"
                    )
                    raise RuntimeError(f"HARDWARE_BUSY {resource}: {detail}") from error
                self.held.append(directory)
                temporary = directory / "owner.tmp"
                temporary.write_text(self.record.model_dump_json(indent=2) + "\n")
                temporary.replace(directory / "owner.json")
        except BaseException:
            self.release()
            raise
        print("HARDWARE_ACQUIRED " + self.record.model_dump_json(), flush=True)

    def release(self) -> None:
        """Release only this token's directories; never remove another owner."""
        for directory in reversed(self.held):
            path = directory / "owner.json"
            if path.exists():
                if read_owner(directory).token != self.record.token:
                    raise RuntimeError(f"ownership changed: {directory}")
                path.unlink()
            (directory / "owner.tmp").unlink(missing_ok=True)
            directory.rmdir()
        self.held.clear()


def status(root: Path) -> None:
    """Print live reservations; persistent records require explicit reconciliation."""
    records = {}
    if root.exists():
        for directory in sorted(root.iterdir()):
            if directory.is_dir():
                path = directory / "owner.json"
                records[directory.name] = (
                    read_owner(directory).model_dump()
                    if path.exists()
                    else {"state": "initializing_or_unresolved"}
                )
    print(json.dumps(records, indent=2))


def run(lease: Lease, command: list[str]) -> int:
    """Hold ownership until the wrapped command and its ordinary cleanup exit."""
    lease.acquire()
    child = None
    handlers = {}
    try:
        child = subprocess.Popen(command, start_new_session=True)

        def forward(signum, _frame):
            if child.poll() is None:
                os.killpg(child.pid, signum)

        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, forward)
        code = child.wait()
        return code if code >= 0 else 128 - code
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        lease.release()


def unit_state(unit: str) -> str:
    """Read an existing user service's lifecycle without controlling it."""
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "ActiveState", "--value"],
        check=True,
        capture_output=True,
        text=True,
    )
    state = result.stdout.strip()
    if state not in {
        "active",
        "activating",
        "deactivating",
        "reloading",
        "inactive",
        "failed",
    }:
        raise RuntimeError(f"unresolved unit state for {unit}: {state!r}")
    return state


def hold(lease: Lease, unit: str) -> int:
    """Adopt an already running job, releasing after its ExecStopPost finishes.

    A monitor error leaves ownership published for manual reconciliation, rather
    than announcing free hardware while the job might still be running.
    """
    if unit_state(unit) not in {"active", "activating", "reloading"}:
        raise RuntimeError(f"cannot adopt a stopped job: {unit}")
    lease.acquire()
    while unit_state(unit) not in {"inactive", "failed"}:
        time.sleep(2)
    lease.release()
    return 0


def main() -> int:
    """Expose run, current-job adoption, status and explicit stale release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("status")
    for action in ("run", "hold"):
        sub = actions.add_parser(action)
        sub.add_argument("--resource", action="append", required=True)
        sub.add_argument("--owner", required=True)
        sub.add_argument("--thread", required=True)
        if action == "run":
            sub.add_argument("command", nargs=argparse.REMAINDER)
        else:
            sub.add_argument("--unit", required=True)
    release = actions.add_parser("release")
    release.add_argument("--resource", action="append", required=True)
    release.add_argument("--token", required=True)
    args = parser.parse_args()
    if args.action == "status":
        status(args.root)
        return 0
    args.resource = sorted(
        set(TypeAdapter(list[Resource]).validate_python(args.resource))
    )
    if args.action == "release":
        for resource in args.resource:
            record = read_owner(args.root / resource)
            if record.token != args.token:
                raise RuntimeError(f"not the owner of {resource}")
        for resource in args.resource:
            directory = args.root / resource
            (directory / "owner.json").unlink()
            directory.rmdir()
        return 0
    command = args.command if args.action == "run" else []
    if command[:1] == ["--"]:
        command = command[1:]
    if args.action == "run" and not command:
        parser.error("run requires a command after --")
    record = Owner(
        token=str(uuid.uuid4()),
        owner=args.owner,
        thread=args.thread,
        host=socket.gethostname(),
        pid=os.getpid(),
        acquired_utc=datetime.now(UTC).isoformat(),
        resources=args.resource,
        command=command,
        unit=getattr(args, "unit", None),
    )
    lease = Lease(args.root, record)
    return run(lease, command) if args.action == "run" else hold(lease, args.unit)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(75) from error
