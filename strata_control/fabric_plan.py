"""Content-addressed executable plans for the experimental two-node DeepSeek fabric.

These static assignments are materialized by ``strata_control.fabric_bundle``.
No placement ranking, reservation or runtime admission is implied by validation.
"""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from serve.tokenizer_assets import Digest, Record, TokenizerPins
from strata_control.components import Artifact
from tools.network_fabric_inventory import Inventory

Positive = Annotated[int, Field(gt=0, le=2**64 - 1)]
Mib = Annotated[int, Field(ge=0, le=1048576)]
NodeId = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")]
Role = Literal["prefix", "suffix"]


class FilePin(Record):
    """One immutable file; directories and machine-specific locations are separate."""

    name: Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,255}$")]
    sha256: Digest
    bytes: Positive

    def read(self, path: Path) -> bytes:
        """Verify and return the same buffer that a caller will parse."""
        raw = path.read_bytes()
        Artifact(self.name, None, self.sha256, self.bytes, format="file").verify_bytes(
            raw
        )
        return raw

    def verify(self, path: Path) -> None:
        """Stream large backing files through Control's existing verifier."""
        Artifact(self.name, None, self.sha256, self.bytes, format="file").verify_file(
            path
        )


class Model(Record):
    """Native model identities and the complete immutable backing/pack inventory."""

    family: Literal["deepseek41"] = "deepseek41"
    revision: Annotated[str, Field(min_length=1, max_length=256)]
    model_sha256: Digest
    pack_sha256: Digest
    expert_model_sha256: Digest
    mapped_files: Annotated[tuple[FilePin, ...], Field(min_length=6, max_length=6)]
    tokenizer_gguf: FilePin
    pack_files: Annotated[tuple[FilePin, ...], Field(min_length=3)]
    tokenizer: TokenizerPins
    effective_experts: Annotated[int, Field(ge=6, le=6)] = 6
    arithmetic: Annotated[int, Field(ge=2, le=2)] = 2
    context: Annotated[int, Field(ge=2048, le=2048)] = 2048
    engram: Literal["native-assigned-range"] = "native-assigned-range"

    @field_validator("pack_files")
    @classmethod
    def ordered_pack(cls, value: tuple[FilePin, ...]) -> tuple[FilePin, ...]:
        """Pack source-file enumeration has no native mapping-order meaning."""
        return tuple(sorted(value, key=lambda pin: pin.name))

    @model_validator(mode="after")
    def complete_files(self):
        """Keep all six native mappings and all required pack metadata explicit."""
        names = [file.name for file in (*self.mapped_files, self.tokenizer_gguf)]
        if len(set(names)) != len(names):
            raise ValueError("duplicate model backing file")
        pack = [file.name for file in self.pack_files]
        if len(set(pack)) != len(pack) or not {
            "ds4.txt",
            "tensors.txt",
            "engram.bin",
        } <= set(pack):
            raise ValueError(
                "pack files must uniquely include ds4.txt, tensors.txt and engram.bin"
            )
        return self


class Budget(Record):
    """Explicit node limits; RAM includes pinned allocations, not OS page cache."""

    vram_mib: Annotated[int, Field(gt=0, le=1048576)]
    ram_mib: Annotated[int, Field(gt=0, le=1048576)]
    pinned_mib: Mib

    @model_validator(mode="after")
    def pinned_is_ram(self):
        """Pinned memory cannot exceed the enclosing RAM budget."""
        if self.pinned_mib > self.ram_mib:
            raise ValueError("pinned budget exceeds RAM budget")
        return self


class Requirement(Record):
    """Declared whole-node peaks/reserve, bound to their supporting receipt.

    This is a proposal input, not attestation that a receipt is accepted or that
    sampled usage proves an allocation peak. The installer must check evidence.
    """

    vram_mib: Annotated[int, Field(gt=0, le=1048576)]
    ram_mib: Annotated[int, Field(gt=0, le=1048576)]
    pinned_mib: Mib
    evidence: FilePin


class Node(Record):
    """One backend's exact binaries, inventory and evidence-bound resource limits."""

    id: NodeId
    backend: Literal["cuda", "hip"]
    source_sha256: Digest
    layer_binary: FilePin
    worker_binary: FilePin | None = None
    inventory: FilePin
    capability_evidence: FilePin
    budget: Budget
    requirement: Requirement

    @model_validator(mode="after")
    def declared_fit(self):
        """Reject contradictory declared peaks before preparing any commands."""
        for kind in ("vram_mib", "ram_mib", "pinned_mib"):
            if getattr(self.requirement, kind) > getattr(self.budget, kind):
                raise ValueError(f"declared {kind} requirement exceeds node budget")
        if self.requirement.pinned_mib > self.requirement.ram_mib:
            raise ValueError("pinned requirement exceeds RAM requirement")
        return self


class LayerOwner(Record):
    """The prefix/suffix role derives its layer range from the plan's cut."""

    node: NodeId
    cpu_threads: Annotated[int, Field(ge=1, le=64)]
    expert_vram_mib: Annotated[int, Field(ge=0, le=65536)]


class Pair(Record):
    """One routed expert identity; execution may fall back on its layer owner."""

    layer: Annotated[int, Field(ge=0, lt=40)]
    expert: Annotated[int, Field(ge=0, lt=384)]


class ExpertOwner(Record):
    """One immutable opposite-node worker session for one layer owner."""

    source: Role
    node: NodeId
    pairs: Annotated[tuple[Pair, ...], Field(min_length=1, max_length=15360)]
    expert_vram_mib: Mib
    resident_ram_mib: Mib
    fallback: Literal["native-local-recompute"] = "native-local-recompute"

    @field_validator("pairs")
    @classmethod
    def unique_pairs(cls, value: tuple[Pair, ...]) -> tuple[Pair, ...]:
        """Preserve native GPU/RAM fill priority while rejecting duplicates."""
        keys = [(pair.layer, pair.expert) for pair in value]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate expert ownership")
        return value

    def native_placement(self) -> str:
        """Use the existing native layer/expert text grammar, without a new parser."""
        return "".join(f"{pair.layer} {pair.expert}\n" for pair in self.pairs)


class Deadlines(Record):
    """Explicit startup, transaction and request bounds in the native units."""

    startup_s: Annotated[float, Field(gt=0, le=7200, allow_inf_nan=False)]
    layer_rpc_ms: Annotated[int, Field(ge=1, le=600000)]
    expert_rpc_ms: Annotated[int, Field(ge=1, le=600000)]
    request_ms: Annotated[int, Field(ge=1, le=7200000)]
    idle_ms: Annotated[int, Field(ge=1, le=7200000)]

    @model_validator(mode="after")
    def request_covers_rpc(self):
        """A request cannot expire before one declared layer/expert transaction."""
        if self.request_ms < max(self.layer_rpc_ms, self.expert_rpc_ms):
            raise ValueError("request timeout must cover layer and expert RPC")
        return self


class Plan(Record):
    """Logical static plan; installation epochs, addresses and paths are excluded."""

    schema_version: Literal["strata-fabric-plan/1"] = "strata-fabric-plan/1"
    selection: Literal["static"] = "static"
    model: Model
    workload: FilePin
    nodes: Annotated[tuple[Node, ...], Field(min_length=2, max_length=2)]
    prefix: LayerOwner
    suffix: LayerOwner
    cut: Annotated[int, Field(ge=20, le=24)]
    workers: Annotated[tuple[ExpertOwner, ...], Field(max_length=2)] = ()
    deadlines: Deadlines
    layer_protocol: Annotated[int, Field(ge=1, le=1)] = 1
    expert_protocol: Annotated[int, Field(ge=2, le=2)] = 2

    @field_validator("cut")
    @classmethod
    def supported_cut(cls, value: int) -> int:
        """Only cuts implemented by the current native state-transfer adapter."""
        if value not in (20, 24):
            raise ValueError("native layer cut must be 20 or 24")
        return value

    @field_validator("nodes", "workers")
    @classmethod
    def stable_order(cls, value):
        """Unordered node/worker input cannot change the logical plan identity."""
        return tuple(
            sorted(
                value,
                key=lambda item: item.id if isinstance(item, Node) else item.source,
            )
        )

    @model_validator(mode="after")
    def valid_ownership_and_budgets(self):
        """Reject aliases, unsupported ranges and impossible declared arenas."""
        nodes = {node.id: node for node in self.nodes}
        if len(nodes) != 2 or {self.prefix.node, self.suffix.node} != set(nodes):
            raise ValueError("prefix and suffix must own distinct declared nodes")
        sources = [worker.source for worker in self.workers]
        if len(set(sources)) != len(sources):
            raise ValueError("each layer owner may install only one worker session")
        gpu = {node: 0 for node in nodes}
        ram = {node: 0 for node in nodes}
        for owner in (self.prefix, self.suffix):
            gpu[owner.node] += owner.expert_vram_mib
        for worker in self.workers:
            owner = self.prefix if worker.source == "prefix" else self.suffix
            if worker.node not in nodes or worker.node == owner.node:
                raise ValueError("expert worker must use the opposite declared node")
            if nodes[worker.node].worker_binary is None:
                raise ValueError("expert worker requires a pinned worker binary")
            begin, end = (0, self.cut) if worker.source == "prefix" else (self.cut, 40)
            if any(not begin <= pair.layer < end for pair in worker.pairs):
                raise ValueError(
                    "expert assignment lies outside its source layer range"
                )
            gpu[worker.node] += worker.expert_vram_mib
            ram[worker.node] += worker.resident_ram_mib
        for node in self.nodes:
            if (
                gpu[node.id] > node.requirement.vram_mib
                or ram[node.id] > node.requirement.ram_mib
            ):
                raise ValueError(
                    "declared node requirement omits configured expert arenas"
                )
        return self

    def identity(self) -> str:
        """Hash the validated logical snapshot with an explicit domain separator."""
        raw = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(
            b"strata-fabric-plan/1\0" + raw.encode("ascii")
        ).hexdigest()


class PlanDocument(Record):
    """A saved logical plan whose claimed content digest must be reproducible."""

    plan: Plan
    plan_sha256: Digest

    @model_validator(mode="after")
    def matching_identity(self):
        """A changed placement or inventory cannot retain an earlier plan ID."""
        if self.plan.identity() != self.plan_sha256:
            raise ValueError("plan content differs from claimed plan ID")
        return self


def check_inventory(node: Node, path: Path) -> Inventory:
    """Check pinned Control observations against declarations; reserve nothing."""
    observed = Inventory.model_validate_json(node.inventory.read(path))
    return check_resources(node, observed)


def check_resources(node: Node, observed: Inventory) -> Inventory:
    """Apply the same fit rules to a pinned snapshot or a fresh host observation."""
    if (
        observed.node != node.id
        or observed.executable_sha256 != node.layer_binary.sha256
    ):
        raise ValueError("inventory node/binary differs from plan")
    card = next(
        (
            card
            for card in observed.gpus
            if card.index == 0
            and card.vendor == ("nvidia" if node.backend == "cuda" else "amd")
        ),
        None,
    )
    if card is None or card.vram_total_mib is None or card.vram_used_mib is None:
        raise ValueError("known backend device-0 VRAM observations are required")
    if (
        node.budget.vram_mib > card.vram_total_mib
        or node.requirement.vram_mib > card.vram_total_mib - card.vram_used_mib
    ):
        raise ValueError("declared VRAM does not fit the pinned inventory snapshot")
    if observed.memory.total_mib is None or observed.memory.available_mib is None:
        raise ValueError("known RAM observations are required")
    if (
        node.budget.ram_mib > observed.memory.total_mib
        or node.requirement.ram_mib > observed.memory.available_mib
    ):
        raise ValueError("declared RAM does not fit the pinned inventory snapshot")
    return observed
