"""Local node actuation for the experimental fabric's transactional installer.

This verifies/stages a declared node and owns its worker/suffix processes. It
never opens HTTP or claims whole-plan admission, quality or a resource reservation.
"""

import fcntl
import os
import re
import socket
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from serve.deepseek_layer import LayerReceipt
from serve.tokenizer_assets import Digest, Record, load_deepseek
from strata_control import atomic
from strata_control.fabric_bundle import Installation
from strata_control.fabric_plan import (
    FilePin,
    NodeId,
    Positive,
    check_inventory,
    check_resources,
)
from strata_control.fabric_process import NativeProcess, ProcessIdentity
from tools.network_fabric_inventory import observe


class EpochRecord(Record):
    """Durably consumed node epoch; failed starts must not silently reuse it."""

    node: NodeId
    epoch: Positive
    plan_sha256: Digest


class NodeLease:
    """One active fabric owner per node, with an atomically persisted epoch."""

    def __init__(self, directory: Path, node: str, epoch: int, plan: str):
        """Lock a private node state directory and refuse stale/corrupt records."""
        if not directory.is_absolute():
            raise ValueError("fabric state directory must be absolute")
        self.record = EpochRecord(node=node, epoch=epoch, plan_sha256=plan)
        directory = directory / node
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lock = (directory / "owner.lock").open("a+b")
        self.path = directory / "epoch.json"
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.check_newer()
        except BaseException:
            self.lock.close()
            raise

    def check_newer(self) -> None:
        """Only an absent ledger means first installation; malformed state fails."""
        if self.path.exists():
            previous = EpochRecord.model_validate_json(self.path.read_bytes())
            if previous.node != self.record.node or previous.epoch >= self.record.epoch:
                raise ValueError(
                    "fabric installation epoch is stale or belongs to another node"
                )

    def reserve(self) -> None:
        """Persist the fence before any native process can start, including fsync."""
        self.check_newer()
        atomic.write_text(self.path, self.record.model_dump_json() + "\n")
        with self.path.open("rb") as stream:
            os.fsync(stream.fileno())
        directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def close(self) -> None:
        """Release this owner's lock without resetting the persisted epoch."""
        self.lock.close()


class LayerBoot(Record):
    """Actual suffix instance, runtime identity and resolved native profile."""

    process: ProcessIdentity
    native: LayerReceipt
    incarnation: Positive


class WorkerBoot(Record):
    """Actual worker listener; this is explicitly not ownership READY."""

    process: ProcessIdentity
    endpoint: str
    model_sha256: Digest
    effective_experts: Annotated[int, Field(ge=6, le=6)]
    arithmetic: Annotated[int, Field(ge=2, le=2)]


class WorkerOwnership(Record):
    """The worker's actual installed ownership and native tier allocation report."""

    owned: Positive
    gpu: Annotated[int, Field(ge=0)]
    ram: Annotated[int, Field(ge=0)]
    mapped: Annotated[int, Field(ge=0)]
    gpu_bytes: Annotated[int, Field(ge=0)]
    ram_bytes: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def complete_coverage(self):
        """Each owned expert has exactly one declared native execution tier."""
        if self.gpu + self.ram + self.mapped != self.owned:
            raise ValueError("worker tiers do not cover installed ownership")
        return self


class HostStatus(Record):
    """Node-local observations; neither a listener nor this record publishes HTTP."""

    node: NodeId
    plan_sha256: Digest
    epoch: Positive
    inputs_staged: bool
    layer: LayerBoot | None
    worker: WorkerBoot | None
    ownership: WorkerOwnership | None
    layer_alive: bool
    worker_alive: bool
    layer_exit: int | None
    worker_exit: int | None


def layer_artifacts(text: str, expected: tuple[FilePin, ...]) -> None:
    """Match all six native post-load file hashes, sizes and mapping order."""
    rows = re.findall(
        r"^ds4-layer-artifact: index=(\d+) file=(\S+) bytes=(\d+) sha256=([0-9a-f]{64})$",
        text,
        re.MULTILINE,
    )
    if len(rows) != len(expected):
        raise ValueError("missing or duplicate native mapped-file identity")
    for index, (row, pin) in enumerate(zip(rows, expected, strict=True)):
        if row != (str(index), pin.name, str(pin.bytes), pin.sha256):
            raise ValueError("actual native mapped files differ from plan pins")


class NodeHost:
    """Stage and supervise one node; global publication belongs to the coordinator."""

    def __init__(self, spec: Installation, node: str, evidence: Path, state: Path):
        """Bind the real host and acquire its lease before any model work."""
        self.spec = Installation.model_validate_json(spec.model_dump_json())
        if node != socket.gethostname():
            raise ValueError("requested fabric node is not this host")
        self.node = next(
            item for item in self.spec.document.plan.nodes if item.id == node
        )
        self.place = self.spec.location(node)
        self.evidence = evidence / node
        self.owned = ExitStack()
        self.lease = NodeLease(state, node, spec.epoch, spec.document.plan_sha256)
        self.owned.callback(self.lease.close)
        self.verified = False
        self.worker = None
        self.suffix = None
        self.worker_boot = None
        self.layer_boot = None
        self.ownership = None
        self._exit_recorded = False

    def verify(self) -> None:
        """Verify all files, re-observe resources and stage fresh runtime inputs.

        Large files are streamed, not copied into another cache. These evidence
        hashes do not by themselves attest mathematical/quality qualification.
        Native loaded identities must still match before whole-plan admission.
        """
        if self.verified:
            raise RuntimeError("fabric node already verified/staged")
        if self.place.runtime_dir.exists():
            raise FileExistsError(self.place.runtime_dir)
        plan = self.spec.document.plan
        check_inventory(
            self.node, self.evidence / "inventory" / self.node.inventory.name
        )
        self.node.capability_evidence.read(
            self.evidence / "capability" / self.node.capability_evidence.name
        )
        self.node.requirement.evidence.read(
            self.evidence / "resource" / self.node.requirement.evidence.name
        )
        for pin in (*plan.model.mapped_files, plan.model.tokenizer_gguf):
            pin.verify(self.place.model_dir / pin.name)
        for pin in plan.model.pack_files:
            pin.verify(self.place.pack_dir / pin.name)
        self.node.layer_binary.verify(self.place.layer_binary)
        worker = next(
            (item for item in plan.workers if item.node == self.node.id), None
        )
        if worker is not None:
            self.node.worker_binary.verify(self.place.worker_binary)
        load_deepseek(self.place.tokenizer_dir, plan.model.tokenizer)
        fresh = check_resources(
            self.node,
            observe(
                self.place.layer_binary,
                [self.place.model_dir, self.place.pack_dir, self.place.tokenizer_dir],
            ),
        )
        self.lease.reserve()
        self.place.runtime_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        (self.place.runtime_dir / "installation.json").write_text(
            self.spec.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (self.place.runtime_dir / "runtime-inventory.json").write_text(
            fresh.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        role = "prefix" if plan.prefix.node == self.node.id else "suffix"
        remote = next((item for item in plan.workers if item.source == role), None)
        if remote is not None:
            (self.place.runtime_dir / "remote.plan").write_text(
                remote.native_placement(), encoding="utf-8"
            )
        self.verified = True

    def _start(self, kind: str, argv: list[str], pin: FilePin) -> NativeProcess:
        """Register ownership before waiting for any native startup report."""
        if not self.verified:
            raise RuntimeError("fabric node must verify before starting processes")
        child = NativeProcess(argv, pin, self.place.runtime_dir / (kind + ".log"))
        self.owned.callback(child.close)
        atomic.write_new_text(
            self.place.runtime_dir / (kind + "-process.json"),
            child.identity().model_dump_json(indent=2) + "\n",
        )
        return child

    def start_worker(self) -> WorkerBoot:
        """Start the declared worker once, requiring its actual model and listener."""
        if self.worker is not None:
            raise RuntimeError("fabric worker already started")
        worker = next(
            (
                item
                for item in self.spec.document.plan.workers
                if item.node == self.node.id
            ),
            None,
        )
        if worker is None:
            raise ValueError("no worker is assigned to this node")
        self.worker = self._start(
            "worker",
            [str(self.place.worker_binary), *self.spec.worker_arguments(worker)],
            self.node.worker_binary,
        )

        def receipt(text):
            listen = re.search(r"^WORKER_LISTENING (\S+)\n", text, re.MULTILINE)
            if listen is None:
                return None
            identity = re.findall(
                r"^ds4-network-identity: role=expert-worker effective_experts=(\d+) engine=\S+ arithmetic=(\d+) model_sha256=([0-9a-f]{64})$",
                text,
                re.MULTILINE,
            )
            if listen[1] != self.place.expert_endpoint.native() or len(identity) != 1:
                raise ValueError("actual worker endpoint/identity differs from plan")
            experts, arithmetic, model = identity[0]
            if (experts, arithmetic, model) != (
                "6",
                "2",
                self.spec.document.plan.model.expert_model_sha256,
            ):
                raise ValueError(
                    "actual worker six-expert arithmetic/model differs from plan"
                )
            return WorkerBoot(
                process=self.worker.identity(),
                endpoint=listen[1],
                model_sha256=model,
                effective_experts=6,
                arithmetic=2,
            )

        self.worker_boot = self.worker.wait_ready(
            receipt, self.spec.document.plan.deadlines.startup_s
        )
        return self.worker_boot

    def start_suffix(self) -> LayerBoot:
        """Start this node's suffix and require all actual artifact/epoch receipts."""
        if (
            self.node.id != self.spec.document.plan.suffix.node
            or self.suffix is not None
        ):
            raise ValueError("suffix is not assigned here or already started")
        self.suffix = self._start(
            "suffix",
            [
                str(self.place.layer_binary),
                "--serve",
                *self.spec.layer_arguments("suffix"),
            ],
            self.node.layer_binary,
        )

        def receipt(text):
            ready = re.search(
                r"^LAYER_READY endpoint=(\S+) incarnation=(\d+)\n", text, re.MULTILINE
            )
            if ready is None:
                return None
            values = dict(
                part.split("=", 1)
                for line in text.splitlines()
                if line.startswith("INFO ")
                for part in line[5:].split()
                if "=" in part
            )
            actual = LayerReceipt.model_validate_strings(
                values, strict=True, extra="ignore"
            )
            if (
                actual != self.spec.layer_receipt("suffix")
                or ready[1] != self.spec.layer_endpoint.native()
            ):
                raise ValueError(
                    "actual suffix profile/epoch/endpoint differs from plan"
                )
            layer_artifacts(text, self.spec.document.plan.model.mapped_files)
            return LayerBoot(
                process=self.suffix.identity(), native=actual, incarnation=int(ready[2])
            )

        self.layer_boot = self.suffix.wait_ready(
            receipt, self.spec.document.plan.deadlines.startup_s
        )
        return self.layer_boot

    def worker_ownership(self) -> WorkerOwnership:
        """Require actual ownership READY, not just the earlier listener receipt."""
        if self.worker is None or self.worker_boot is None:
            raise RuntimeError("worker has not started")
        expected = next(
            item
            for item in self.spec.document.plan.workers
            if item.node == self.node.id
        )

        def receipt(text):
            owned = re.findall(r"strata-expert-worker: READY owned (\d+) experts", text)
            if not owned:
                return None
            tiers = re.findall(
                r"^strata-worker-tiers: owned=(\d+) gpu=(\d+) ram=(\d+) mapped=(\d+) gpu_bytes=(\d+) ram_bytes=(\d+)$",
                text,
                re.MULTILINE,
            )
            if len(owned) != 1 or len(tiers) != 1:
                raise ValueError("missing/duplicate worker ownership report")
            actual = WorkerOwnership.model_validate_strings(
                dict(zip(WorkerOwnership.model_fields, tiers[0], strict=True)),
                strict=True,
            )
            if (
                owned[0] != str(actual.owned)
                or actual.owned != len(expected.pairs)
                or actual.gpu_bytes > expected.expert_vram_mib * 1024**2
                or actual.ram_bytes > expected.resident_ram_mib * 1024**2
            ):
                raise ValueError(
                    "actual worker ownership/tier budgets differ from plan"
                )
            return actual

        self.ownership = self.worker.wait_ready(
            receipt, self.spec.document.plan.deadlines.startup_s
        )
        return self.ownership

    def status(self) -> HostStatus:
        """Read actual child liveness; preserve degraded worker exit information."""
        layer_exit = None if self.suffix is None else self.suffix.proc.poll()
        worker_exit = None if self.worker is None else self.worker.proc.poll()
        return HostStatus(
            node=self.node.id,
            plan_sha256=self.spec.document.plan_sha256,
            epoch=self.spec.epoch,
            inputs_staged=self.verified,
            layer=self.layer_boot,
            worker=self.worker_boot,
            ownership=self.ownership,
            layer_alive=self.suffix is not None and layer_exit is None,
            worker_alive=self.worker is not None and worker_exit is None,
            layer_exit=layer_exit,
            worker_exit=worker_exit,
        )

    def close(self) -> None:
        """Stop all owned children before releasing the node's installation lease."""
        try:
            self.owned.close()
        finally:
            if self.verified and not self._exit_recorded:
                atomic.write_new_text(
                    self.place.runtime_dir / "node-exit.json",
                    self.status().model_dump_json(indent=2) + "\n",
                )
                self._exit_recorded = True

    def __enter__(self):
        """Make partial installations unwind every registered native resource."""
        return self

    def __exit__(self, *_):
        """Release this node even when startup or a caller fails."""
        self.close()
