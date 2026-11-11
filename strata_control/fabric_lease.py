"""Durable fabric ownership fences, including unresolved cleanup and recovery.

An unlocked file is not proof that the previous installation is gone. Reserved
or quarantined epochs require explicit reconciliation before a newer start.
"""

import fcntl
import hashlib
import os
import socket
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from serve.tokenizer_assets import Digest, Record
from strata_control import atomic, procs
from strata_control.fabric_bundle import Installation
from strata_control.fabric_plan import NodeId, Positive


def boot_id() -> str:
    """Read Linux's boot identity; recovery never compares PID births across boots."""
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


class EpochRecord(Record):
    """One consumed epoch and its writer/parent lifetime fences, never a reset."""

    node: NodeId
    epoch: Positive
    plan_sha256: Digest
    installation_sha256: Digest
    boot_id: Annotated[
        str, Field(pattern=r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
    ]
    writer_pid: Positive
    writer_start_token: Annotated[str, Field(pattern=r"^l:[0-9]+$")]
    parent_pid: Positive
    parent_start_token: Annotated[str, Field(pattern=r"^l:[0-9]+$")]
    phase: Literal["reserved", "closed", "quarantined"]
    reason: Annotated[str, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def quarantine_reason(self):
        """An unresolved cleanup retains a concrete reason for its refusal."""
        if self.phase == "quarantined" and self.reason is None:
            raise ValueError("quarantined epoch requires its failure reason")
        return self


def _locked(directory: Path, node: str):
    """Acquire one node's advisory lock; errors never leave a live lock handle."""
    if not directory.is_absolute():
        raise ValueError("fabric state directory must be absolute")
    directory = directory / node
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = (directory / "owner.lock").open("a+b")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    return lock, directory / "epoch.json"


def _save(path: Path, record: EpochRecord) -> None:
    """Publish and fsync the ledger and directory before exposing a new state."""
    atomic.write_text(path, record.model_dump_json() + "\n")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def binding_sha256(spec: Installation) -> str:
    """Bind the complete canonical paths/endpoints/epoch as well as logical plan."""
    return hashlib.sha256(spec.model_dump_json().encode()).hexdigest()


def require_quiet(spec: Installation, node: str) -> None:
    """Refuse any surviving planned process, including the external HTTP prefix.

    This is a local /proc census, not a name-based kill. Exact generated argv
    identifies the installation's intents even if a crash preceded its PID
    journal. Unreadable same-user processes and matching binaries with an empty
    exec-transition cmdline are unresolved rather than silently absent.
    """
    if not Path("/proc/self/stat").is_file():
        raise RuntimeError("fabric quiescence checks require Linux /proc")
    plan = spec.document.plan
    place = spec.location(node)
    role = "prefix" if node == plan.prefix.node else "suffix"
    expected = {(str(place.layer_binary), "--serve", *spec.layer_arguments(role))}
    worker = next((worker for worker in plan.workers if worker.node == node), None)
    if worker is not None:
        expected.add((str(place.worker_binary), *spec.worker_arguments(worker)))
    binaries = {row[0] for row in expected}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] == "Z":
                continue
            raw = (path / "cmdline").read_bytes()
            argv = (
                tuple(os.fsdecode(part) for part in raw.rstrip(b"\0").split(b"\0"))
                if raw
                else ()
            )
            if argv in expected:
                raise RuntimeError(
                    f"unresolved native process {path.name} still belongs to fabric epoch {spec.epoch}"
                )
            if (
                not argv
                and os.readlink(path / "exe").removesuffix(" (deleted)") in binaries
            ):
                raise RuntimeError(
                    f"unresolved native exec transition for process {path.name}"
                )
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (PermissionError, ValueError, IndexError) as error:
            raise RuntimeError(
                f"cannot establish fabric quiescence for process {path.name}"
            ) from error


class NodeLease:
    """One live node owner, with a durable unresolved state after failed cleanup."""

    def __init__(self, directory: Path, spec: Installation, node: str):
        """Lock and validate the ledger before constructing a new installation."""
        if node not in {item.id for item in spec.document.plan.nodes}:
            raise ValueError("lease node is absent from installation")
        self.spec = spec
        self.node = node
        self.reserved = False
        self.lock, self.path = _locked(directory, node)
        try:
            self.check_newer()
        except BaseException:
            self.lock.close()
            raise

    def check_newer(self) -> None:
        """Malformed, stale and unresolved prior states never become free capacity."""
        if self.lock.closed:
            raise RuntimeError("fabric node lease lock is closed")
        if self.path.exists():
            previous = EpochRecord.model_validate_json(self.path.read_bytes())
            if previous.node != self.node or previous.epoch >= self.spec.epoch:
                raise ValueError(
                    "fabric installation epoch is stale or belongs to another node"
                )
            if previous.phase != "closed":
                raise RuntimeError(
                    "unresolved fabric ownership requires explicit reconciliation"
                )

    def reserve(self) -> None:
        """Durably consume this epoch before native startup, recording both lifetimes."""
        self.check_newer()
        writer, parent = os.getpid(), os.getppid()
        writer_token, parent_token = (
            procs.start_token(writer),
            procs.start_token(parent),
        )
        if writer_token is None or parent_token is None:
            raise RuntimeError("fabric ownership lifetime fences are unavailable")
        self.record = EpochRecord(
            node=self.node,
            epoch=self.spec.epoch,
            plan_sha256=self.spec.document.plan_sha256,
            installation_sha256=binding_sha256(self.spec),
            boot_id=boot_id(),
            writer_pid=writer,
            writer_start_token=writer_token,
            parent_pid=parent,
            parent_start_token=parent_token,
            phase="reserved",
        )
        self.reserved = True
        _save(self.path, self.record)

    def finish(self) -> None:
        """Mark closed only after all planned native processes are actually absent."""
        if self.lock.closed:
            raise RuntimeError("fabric node lease lock is closed")
        require_quiet(self.spec, self.node)
        self.record = EpochRecord.model_validate(
            {**self.record.model_dump(), "phase": "closed", "reason": None}
        )
        _save(self.path, self.record)

    def quarantine(self, reason: str) -> None:
        """Retain unresolved ownership while still holding the node lock."""
        if self.lock.closed:
            raise RuntimeError("fabric node lease lock is closed")
        self.record = EpochRecord.model_validate(
            {**self.record.model_dump(), "phase": "quarantined", "reason": reason}
        )
        _save(self.path, self.record)

    def close(self) -> None:
        """Release only the lock; this never implies successful cleanup."""
        self.lock.close()


def reconcile(directory: Path, spec: Installation, node: str) -> EpochRecord:
    """Explicitly close an abandoned epoch after writer, parent and native census.

    No process is signalled and the epoch is never reduced. On the same boot,
    both original lifetimes must have ended so neither can launch another child
    after the census. A changed boot eliminates those old lifetime fences, but
    the exact native-process census is still required.
    """
    if node != socket.gethostname():
        raise ValueError("reconciliation must run on the actual node")
    lock, path = _locked(directory, node)
    try:
        previous = EpochRecord.model_validate_json(path.read_bytes())
        if (previous.node, previous.epoch, previous.installation_sha256) != (
            node,
            spec.epoch,
            binding_sha256(spec),
        ):
            raise ValueError("reconciliation requires the exact previous installation")
        if previous.phase != "closed" and previous.boot_id == boot_id():
            for pid, token in (
                (previous.writer_pid, previous.writer_start_token),
                (previous.parent_pid, previous.parent_start_token),
            ):
                if procs.start_token(pid) == token:
                    raise RuntimeError(
                        "unresolved prior fabric writer or parent is still alive"
                    )
                if Path(f"/proc/{pid}").exists() and procs.start_token(pid) is None:
                    stat = (
                        Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                    )
                    if stat[0] != "Z":
                        raise RuntimeError("prior fabric lifetime is unreadable")
        require_quiet(spec, node)
        closed = EpochRecord.model_validate(
            {
                **previous.model_dump(),
                "phase": "closed",
                "reason": "explicit reconciliation: planned native processes absent",
            }
        )
        _save(path, closed)
        return closed
    finally:
        lock.close()
