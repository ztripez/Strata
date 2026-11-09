"""Record read-only resources for experimental network placement, without ranking plans.

Run from the checkout with ``uv run --no-project --with pydantic==2.14.2
python -m tools.network_fabric_inventory --help``. Hardware observations reuse
Strata Control. Missing probes remain unknown; no measured cost is invented.
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from strata_control import tune_machine, updater

Nonnegative = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(gt=0)]


class Record(BaseModel):
    """Strict observation boundary; never coerce a failed probe to a resource."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Memory(Record):
    """Control's RAM observation in MiB; None means unreadable, zero means zero."""

    total_mib: Positive | None
    available_mib: Nonnegative | None

    @model_validator(mode="after")
    def consistent(self):
        """Reject contradictory readings rather than silently clamping them."""
        if (
            self.total_mib is not None
            and self.available_mib is not None
            and self.available_mib > self.total_mib
        ):
            raise ValueError("available RAM exceeds total RAM")
        return self


class Gpu(Record):
    """Control's device reading; this does not attest kernel/format support."""

    vendor: Literal["nvidia", "amd"]
    index: Nonnegative
    name: str
    vram_total_mib: Positive | None
    vram_used_mib: Nonnegative | None
    driver: str | None

    @model_validator(mode="after")
    def consistent(self):
        """Reject impossible memory readings before a future fit decision."""
        if (
            self.vram_total_mib is not None
            and self.vram_used_mib is not None
            and self.vram_used_mib > self.vram_total_mib
        ):
            raise ValueError("used VRAM exceeds total VRAM")
        return self


class Cpu(Record):
    """Control's CPU observation; physical cores are not guessed from threads."""

    model: str
    logical: Positive
    physical: Positive | None
    hybrid: bool


class Filesystem(Record):
    """One filesystem capacity reading, shared by all listed paths on that device."""

    device: Nonnegative
    paths: Annotated[list[str], Field(min_length=1)]
    target: str
    source: str
    fstype: str
    options: str
    total_bytes: Nonnegative
    available_bytes: Nonnegative

    @model_validator(mode="after")
    def consistent(self):
        """Reject contradictory filesystem capacity readings."""
        if self.available_bytes > self.total_bytes:
            raise ValueError("available filesystem bytes exceed total")
        return self


class Inventory(Record):
    """An observed node, not a placement admission or performance estimate."""

    schema_version: Literal["strata-network-inventory/1"] = "strata-network-inventory/1"
    node: str
    observed_unix_ns: Positive
    memory: Memory
    cpu: Cpu
    gpus: list[Gpu]
    backend_probe: dict[str, JsonValue]
    filesystems: list[Filesystem]
    executable: str
    executable_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    planning_status: Literal["UNCALIBRATED; no plan ranking or READY admission"] = (
        "UNCALIBRATED; no plan ranking or READY admission"
    )
    limits: str = (
        "Point-in-time observations, not reservations. Missing probes remain unknown. "
        "Backend detection does not attest this executable's model/format support. "
        "No compute, I/O or network cost is measured here. NFS export capacity "
        "reported by multiple nodes must not be added as independent storage."
    )


def filesystem_observations(paths: list[Path]) -> list[Filesystem]:
    """Group requested paths by actual device and record bounded findmnt/statvfs facts."""
    groups: dict[int, list[Path]] = {}
    for path in paths:
        resolved = path.resolve(strict=True)
        groups.setdefault(resolved.stat().st_dev, []).append(resolved)
    observations = []
    for device, members in sorted(groups.items()):
        result = subprocess.run(
            [
                "findmnt",
                "--json",
                "--target",
                str(members[0]),
                "--output",
                "TARGET,SOURCE,FSTYPE,OPTIONS",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        mounts = json.loads(result.stdout)["filesystems"]
        if len(mounts) != 1:
            raise ValueError("expected one backing filesystem per requested path")
        capacity = os.statvfs(members[0])
        observations.append(
            Filesystem.model_validate(
                {
                    "device": device,
                    "paths": sorted({str(path) for path in members}),
                    **mounts[0],
                    "total_bytes": capacity.f_blocks * capacity.f_frsize,
                    "available_bytes": capacity.f_bavail * capacity.f_frsize,
                }
            )
        )
    return observations


def observe(executable: Path, storage: list[Path]) -> Inventory:
    """Read current Control probes and explicit paths; never load or execute a model."""
    if platform.system() != "Linux":
        raise ValueError("the experimental network inventory currently requires Linux")
    binary = executable.resolve(strict=True)
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return Inventory(
        node=socket.gethostname(),
        observed_unix_ns=time.time_ns(),
        memory=Memory.model_validate(tune_machine.memory()),
        cpu=Cpu.model_validate(tune_machine.cpu()),
        gpus=[Gpu.model_validate(card) for card in tune_machine.gpus()],
        backend_probe=updater.detect_hardware(),
        filesystems=filesystem_observations(storage),
        executable=str(binary),
        executable_sha256=digest,
    )


def main() -> None:
    """Save a new observation; existing receipts are never replaced."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--storage", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    receipt = observe(arguments.engine, arguments.storage)
    with arguments.output.open("x") as stream:
        stream.write(receipt.model_dump_json(indent=2) + "\n")
    print(receipt.planning_status)


if __name__ == "__main__":
    main()
