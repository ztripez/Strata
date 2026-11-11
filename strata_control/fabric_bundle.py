"""Prepare native launch bundles from canonical static fabric plans.

Preparation starts no process and publishes no serving endpoint. The separate
installer must verify artifacts, current resources and all runtime READY receipts.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from serve.deepseek_layer import Launch, LayerReceipt, PrefixReceipt
from serve.tokenizer_assets import Record
from strata_control.fabric_plan import (
    ExpertOwner,
    NodeId,
    PlanDocument,
    Positive,
    Role,
    check_inventory,
)


class Endpoint(Record):
    """Numeric IPv4 endpoint accepted by the existing native socket parser."""

    address: Annotated[str, Field(pattern=r"^[0-9.]+$")]
    port: Annotated[int, Field(ge=1, le=65535)]

    @field_validator("address")
    @classmethod
    def numeric_address(cls, value: str) -> str:
        """Refuse DNS and wildcard listeners in a materialized cluster binding."""
        import ipaddress

        address = ipaddress.IPv4Address(value)
        if address.is_unspecified or address.is_multicast:
            raise ValueError("fabric endpoint must be a unicast IPv4 address")
        return str(address)

    def native(self) -> str:
        """Render the native IP:port spelling without shell interpretation."""
        return f"{self.address}:{self.port}"


class Location(Record):
    """Per-node locators; equal paths never substitute for pinned content."""

    node: NodeId
    model_dir: Path
    pack_dir: Path
    tokenizer_dir: Path
    layer_binary: Path
    worker_binary: Path | None = None
    runtime_dir: Path
    expert_endpoint: Endpoint | None = None

    @field_validator(
        "model_dir",
        "pack_dir",
        "tokenizer_dir",
        "layer_binary",
        "worker_binary",
        "runtime_dir",
    )
    @classmethod
    def absolute(cls, value: Path | None) -> Path | None:
        """All runtime locations are absolute and independent of SSH's cwd."""
        if value is not None and not value.is_absolute():
            raise ValueError("fabric locations must be absolute")
        return value


class Installation(Record):
    """One explicit epoch and transport/path binding for a logical static plan."""

    document: PlanDocument
    epoch: Positive
    locations: Annotated[tuple[Location, ...], Field(min_length=2, max_length=2)]
    layer_endpoint: Endpoint
    http_port: Annotated[int, Field(ge=1, le=65535)] = 8098

    @model_validator(mode="after")
    def matching_locations(self):
        """Require one exact location per node and non-overlapping listeners."""
        nodes = {node.id for node in self.document.plan.nodes}
        if (
            len({place.node for place in self.locations}) != 2
            or {place.node for place in self.locations} != nodes
        ):
            raise ValueError("installation must locate each declared node exactly once")
        endpoints = [self.layer_endpoint.native()]
        for worker in self.document.plan.workers:
            place = self.location(worker.node)
            if place.worker_binary is None or place.expert_endpoint is None:
                raise ValueError("worker executable/endpoint location is missing")
            endpoints.append(place.expert_endpoint.native())
            if (
                worker.node == self.document.plan.prefix.node
                and place.expert_endpoint.address == "127.0.0.1"
                and place.expert_endpoint.port == self.http_port
            ):
                raise ValueError("prefix expert listener collides with loopback HTTP")
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("fabric endpoints collide")
        return self

    def location(self, node: str) -> Location:
        """Resolve a declared node; unknown IDs are always an error."""
        return next(place for place in self.locations if place.node == node)

    def layer_arguments(self, role: Role) -> list[str]:
        """Derive native argv from one canonical placement and explicit epoch."""
        if role not in ("prefix", "suffix"):
            raise ValueError("unknown layer role")
        plan = self.document.plan
        owner = plan.prefix if role == "prefix" else plan.suffix
        place = self.location(owner.node)
        args = [
            "--role",
            role,
            "--pack",
            str(place.pack_dir),
            "--cut",
            str(plan.cut),
            "--endpoint",
            self.layer_endpoint.native(),
            "--model-sha256",
            plan.model.model_sha256,
            "--pack-sha256",
            plan.model.pack_sha256,
            "--plan-id",
            self.document.plan_sha256,
            "--plan-epoch",
            str(self.epoch),
            "--experts",
            "6",
            "--expert-vram-mib",
            str(owner.expert_vram_mib),
            "--cpu-threads",
            str(owner.cpu_threads),
            "--rpc-timeout-ms",
            str(plan.deadlines.layer_rpc_ms),
        ]
        if role == "prefix":
            peer = next(node for node in plan.nodes if node.id == plan.suffix.node)
            args += [
                "--peer-binary-sha256",
                peer.layer_binary.sha256,
                "--request-timeout-ms",
                str(plan.deadlines.request_ms),
            ]
        else:
            args += ["--idle-timeout-ms", str(plan.deadlines.idle_ms)]
        worker = next(
            (worker for worker in plan.workers if worker.source == role), None
        )
        if worker is not None:
            target = self.location(worker.node)
            args += [
                "--expert-worker",
                target.expert_endpoint.native(),
                "--expert-worker-model",
                str(place.model_dir / plan.model.mapped_files[0].name),
                "--expert-worker-placement",
                str(place.runtime_dir / "remote.plan"),
                "--expert-worker-timeout-ms",
                str(plan.deadlines.expert_rpc_ms),
            ]
        return args

    def worker_arguments(self, worker: ExpertOwner) -> list[str]:
        """Use native immutable ownership and its existing VRAM/RAM/mapped tiers."""
        if worker not in self.document.plan.workers:
            raise ValueError("worker is not part of this plan")
        place = self.location(worker.node)
        return [
            "--model",
            str(place.model_dir / self.document.plan.model.mapped_files[0].name),
            "--bind",
            place.expert_endpoint.native(),
            "--device",
            "0",
            "--timeout-ms",
            str(self.document.plan.deadlines.expert_rpc_ms),
            "--idle-timeout-ms",
            str(self.document.plan.deadlines.idle_ms),
            "--max-sessions",
            "1",
            "--expert-vram-mib",
            str(worker.expert_vram_mib),
            "--resident-budget-mib",
            str(worker.resident_ram_mib),
            "--verbose",
        ]

    def layer_receipt(self, role: Role) -> LayerReceipt:
        """Expected runtime identity derived from the same canonical native argv."""
        if role not in ("prefix", "suffix"):
            raise ValueError("unknown layer role")
        plan = self.document.plan
        owner = plan.prefix if role == "prefix" else plan.suffix
        node = next(node for node in plan.nodes if node.id == owner.node)
        return LayerReceipt(
            arch="deepseek41",
            fabric=1,
            role=role,
            context=2048,
            effective_experts=6,
            arithmetic=2,
            model_sha256=plan.model.model_sha256,
            pack_sha256=plan.model.pack_sha256,
            plan_sha256=self.document.plan_sha256,
            plan_epoch=self.epoch,
            binary_sha256=node.layer_binary.sha256,
            cpu_threads_limit=owner.cpu_threads,
            expert_vram_mib=owner.expert_vram_mib,
            request_timeout_ms=plan.deadlines.request_ms if role == "prefix" else 0,
            rpc_timeout_ms=plan.deadlines.layer_rpc_ms,
            idle_timeout_ms=0 if role == "prefix" else plan.deadlines.idle_ms,
        )

    def http_launch(self) -> Launch:
        """Build the existing prefix adapter binding; this does not start HTTP."""
        plan = self.document.plan
        place = self.location(plan.prefix.node)
        return Launch(
            executable=place.layer_binary,
            native_args=self.layer_arguments("prefix"),
            expected=PrefixReceipt.model_validate(
                self.layer_receipt("prefix").model_dump()
            ),
            tokenizer_dir=place.tokenizer_dir,
            tokenizer=plan.model.tokenizer,
            engine_log=place.runtime_dir / "layer.log",
            engine_ready_s=plan.deadlines.startup_s,
            port=self.http_port,
        )


class ProcessLaunch(Record):
    """A generated native argv/environment, inspected before any process starts."""

    node: NodeId
    kind: Literal["layer", "expert-worker"]
    argv: Annotated[tuple[str, ...], Field(min_length=2)]
    environment: dict[str, str]


def materialize(spec: Installation, output: Path, evidence_directory: Path) -> None:
    """Write a fresh inspectable bundle; no process starts or READY is published.

    The final manifest is written last. An interrupted/partial directory is
    retained and must be refused by any installer, rather than silently reused.
    """
    # Re-parse the bytes so mutation of nested dictionaries cannot bypass the
    # sealed content digest or cross-field validation at this IO boundary.
    spec = Installation.model_validate_json(spec.model_dump_json())
    evidence = {}
    for node in spec.document.plan.nodes:
        path = evidence_directory / node.id / "inventory" / node.inventory.name
        check_inventory(node, path)
        for kind, pin in (
            ("inventory", node.inventory),
            ("capability", node.capability_evidence),
            ("resource", node.requirement.evidence),
        ):
            evidence[f"evidence/{node.id}/{kind}/{pin.name}"] = pin.read(
                evidence_directory / node.id / kind / pin.name
            )
    output.mkdir(parents=True, exist_ok=False)
    files = {}

    def save(name: str, text: str) -> None:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    save("installation.json", spec.model_dump_json(indent=2) + "\n")
    for name, raw in evidence.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        files[name] = hashlib.sha256(raw).hexdigest()
    for role in ("prefix", "suffix"):
        plan = spec.document.plan
        owner = plan.prefix if role == "prefix" else plan.suffix
        place = spec.location(owner.node)
        command = ProcessLaunch(
            node=owner.node,
            kind="layer",
            argv=(str(place.layer_binary), "--serve", *spec.layer_arguments(role)),
            environment={"DS4_IQ_Q8K": "1", "DS4_PLACE_HASH": "1"},
        )
        save(
            f"{owner.node}/layer-command.json", command.model_dump_json(indent=2) + "\n"
        )
        worker = next(
            (worker for worker in plan.workers if worker.source == role), None
        )
        if worker is not None:
            save(f"{owner.node}/remote.plan", worker.native_placement())
            target = spec.location(worker.node)
            command = ProcessLaunch(
                node=worker.node,
                kind="expert-worker",
                argv=(str(target.worker_binary), *spec.worker_arguments(worker)),
                environment={"DS4_IQ_Q8K": "1", "DS4_PLACE_HASH": "1"},
            )
            save(
                f"{worker.node}/worker-command.json",
                command.model_dump_json(indent=2) + "\n",
            )
    save(
        f"{spec.document.plan.prefix.node}/http-launch.json",
        spec.http_launch().model_dump_json(indent=2) + "\n",
    )
    save(
        "bundle.json",
        json.dumps(
            {
                "status": "PREPARED ONLY; no process, reservation or READY",
                "plan_sha256": spec.document.plan_sha256,
                "epoch": spec.epoch,
                "sha256": files,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )


def main() -> None:
    """Materialize an explicit installation binding without executing any model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = Installation.model_validate_json(args.installation.read_bytes())
    materialize(spec, args.output, args.evidence_dir)
    print(
        "PREPARED ONLY; native admission and transactional installation remain required"
    )


if __name__ == "__main__":
    main()
