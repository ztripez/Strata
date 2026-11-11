"""Join opt-in DeepSeek native window timings without inventing placement costs.

Use uv and pydantic==2.14.2. This reads completed trace records and pinned
inventories. It does not execute a model, rank plans or establish E6 acceptance.
"""

import argparse
import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from tools.network_fabric_inventory import Inventory, Record

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Counter = Annotated[int, Field(ge=0, le=2**64 - 1)]
Positive = Annotated[int, Field(gt=0, le=2**64 - 1)]
Finite = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Participant(Record):
    """Explicit node snapshot and native process profile used by this trace."""

    node: Annotated[str, Field(min_length=1)]
    inventory_sha256: Digest
    binary_sha256: Digest
    cpu_threads_limit: Annotated[int, Field(ge=1, le=64)]
    expert_vram_mib: Annotated[int, Field(ge=0, le=65536)]


class Binding(Record):
    """Declared workload and exact runtime identities; not a placement manifest."""

    schema_version: Literal["strata-window-calibration-binding/1"]
    model_sha256: Digest
    pack_sha256: Digest
    plan_sha256: Digest
    epoch: Positive
    incarnation: Positive
    cut: Literal[20, 24]
    workload_sha256: Digest
    expected_windows: Annotated[int, Field(gt=0)]
    prefix: Participant
    suffix: Participant


class Transaction(Record):
    """Caller-clock spans and framed bytes of one ACKED layer transaction."""

    total_us: Counter
    execute_us: Counter
    ack_us: Counter
    tx_bytes: Positive
    rx_bytes: Positive

    @model_validator(mode="after")
    def contained(self):
        """EXECUTE and ACK are disjoint pieces of the complete transaction."""
        if self.execute_us + self.ack_us > self.total_us:
            raise ValueError("transaction spans exceed the measured total")
        return self


class Window(Record):
    """Actual native boundary identity, with a pre-consumption position."""

    schema_version: Literal["strata-native-window-cost/1"] = Field(alias="schema")
    model_sha256: Digest
    pack_sha256: Digest
    plan_sha256: Digest
    binary_sha256: Digest
    epoch: Positive
    incarnation: Positive
    session: Annotated[int, Field(ge=2, le=2**64 - 1)]
    sequence: Positive
    position: Counter
    rows: Annotated[int, Field(ge=1, le=16)]
    cut: Literal[20, 24]
    cpu_threads_limit: Annotated[int, Field(ge=1, le=64)]
    expert_vram_mib: Annotated[int, Field(ge=0, le=65536)]

    @model_validator(mode="after")
    def context(self):
        """The initial native window profile has a fixed 2K context."""
        if self.position + self.rows > 2048:
            raise ValueError("native window exceeds the 2K profile")
        return self


class PrefixWindow(Window):
    """One successful feed, split around its acknowledged boundary callback."""

    role: Literal["prefix"]
    feed_us: Counter
    before_rpc_us: Counter
    rpc_us: Counter
    after_rpc_us: Counter
    transaction: Transaction

    @model_validator(mode="after")
    def partition(self):
        """Three separately rounded microsecond spans may lose at most two us."""
        subtotal = self.before_rpc_us + self.rpc_us + self.after_rpc_us
        if not subtotal <= self.feed_us <= subtotal + 2:
            raise ValueError("prefix spans do not partition the measured feed")
        if self.transaction.total_us > self.rpc_us:
            raise ValueError("transaction exceeds its enclosing callback")
        return self


class SuffixWindow(Window):
    """Remote native replay and host-logit readback, already inside prefix RPC."""

    role: Literal["suffix"]
    native_us: Counter


Trace = TypeAdapter(Annotated[PrefixWindow | SuffixWindow, Field(discriminator="role")])


class Pair(Record):
    """Two observations of the same acknowledged boundary, without clock sync."""

    prefix: PrefixWindow
    suffix: SuffixWindow
    rpc_unattributed_us: int

    @model_validator(mode="after")
    def same_window(self):
        """Reject mismatched windows and an impossible cross-clock residual."""
        fields = (
            "model_sha256",
            "pack_sha256",
            "plan_sha256",
            "epoch",
            "incarnation",
            "session",
            "sequence",
            "position",
            "rows",
            "cut",
        )
        if any(
            getattr(self.prefix, key) != getattr(self.suffix, key) for key in fields
        ):
            raise ValueError("prefix/suffix identities differ")
        residual = self.prefix.rpc_us - self.suffix.native_us
        if residual < 0:
            raise ValueError("remote duration exceeds caller RPC; inspect clocks/trace")
        if self.suffix.native_us > self.prefix.transaction.execute_us:
            raise ValueError(
                "remote duration exceeds EXECUTE exchange; inspect clocks/trace"
            )
        if self.rpc_unattributed_us != residual:
            raise ValueError("RPC residual differs from the paired observations")
        return self


class Distribution(Record):
    """Empirical nearest-rank p95; units come from the containing field name."""

    count: Positive
    minimum: Counter
    median: Finite
    mean: Finite
    p95: Counter
    maximum: Counter


class Group(Record):
    """Rows-per-window grouping; a one-row window is not necessarily decode."""

    rows: Annotated[int, Field(ge=1, le=16)]
    feed_us: Distribution
    rpc_us: Distribution
    suffix_native_us: Distribution
    rpc_unattributed_us: Distribution
    tx_bytes: Distribution
    rx_bytes: Distribution


class Observations(Record):
    """Provenance and paired spans; deliberately cannot assert E6 acceptance."""

    schema_version: Literal["strata-window-observations/1"] = (
        "strata-window-observations/1"
    )
    status: Literal["observations only; no plan ranking or READY admission"] = (
        "observations only; no plan ranking or READY admission"
    )
    binding: Binding
    source_sha256: dict[str, Digest]
    prefix_inventory: Inventory
    suffix_inventory: Inventory
    windows: list[Pair]
    by_rows: list[Group]
    limits: str = (
        "RPC includes remote computation; do not add suffix_native_us to it. "
        "The paired residual includes serialization, validation, protocol, ACK, "
        "trace I/O and clock-rate error; it is not pure network latency. "
        "No absolute clocks are subtracted. Local prefix trace writing is outside "
        "feed_us; remote trace writing is inside RPC. The workload hash binds "
        "declared inputs, not proof that those token contents were consumed. "
        "Expert coverage/fallback and cache state need separate runtime receipts. "
        "Per-layer dense/expert costs, worker-tier isolation, cold backing reads, "
        "held-out prediction accuracy and transactional installation remain open. "
        "These observations do not qualify model quality, capacity or speed."
    )

    @model_validator(mode="after")
    def consistent(self):
        """Persisted summaries must agree with their bound underlying windows."""
        if set(self.source_sha256) != {
            "binding",
            "prefix_log",
            "suffix_log",
            "prefix_inventory",
            "suffix_inventory",
            "workload",
        }:
            raise ValueError("observation source identities are incomplete")
        if self.source_sha256["workload"] != self.binding.workload_sha256:
            raise ValueError("declared workload bytes differ from binding")
        for role in ("prefix", "suffix"):
            participant = getattr(self.binding, role)
            inventory = getattr(self, role + "_inventory")
            if (
                self.source_sha256[role + "_inventory"] != participant.inventory_sha256
                or inventory.node != participant.node
                or inventory.executable_sha256 != participant.binary_sha256
            ):
                raise ValueError("inventory identity differs: " + role)
            indexed_windows(
                [getattr(pair, role) for pair in self.windows],
                self.binding,
                participant,
            )
        if self.by_rows != summarize(self.windows):
            raise ValueError("summary differs from its paired raw observations")
        return self


def distribution(values: list[int]) -> Distribution:
    """Keep raw measured totals; no fabricated overlap or missing-value fill."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty timing group")
    return Distribution(
        count=len(ordered),
        minimum=ordered[0],
        maximum=ordered[-1],
        median=float(statistics.median(ordered)),
        mean=float(statistics.mean(ordered)),
        p95=ordered[math.ceil(0.95 * len(ordered)) - 1],
    )


def records(raw: bytes, role: str) -> list[Window]:
    """Read complete native trace lines, keeping unrelated startup logs separate."""
    result = []
    for line in raw.splitlines(keepends=True):
        if not line.startswith(b"FABRIC_WINDOW "):
            continue
        if not line.endswith(b"\n"):
            raise ValueError("incomplete native trace record")
        record = Trace.validate_json(line[len(b"FABRIC_WINDOW ") :])
        if record.role != role:
            raise ValueError("wrong role in native process trace")
        result.append(record)
    return result


def indexed_windows(
    windows: list[Window], binding: Binding, participant: Participant
) -> dict[tuple[int, int], Window]:
    """Validate each native profile and one strictly ordered sequence per session."""
    if len(windows) != binding.expected_windows:
        raise ValueError("trace window count differs")
    indexed, previous = {}, {}
    latest_session = 0
    for record in windows:
        if record.session < latest_session:
            raise ValueError("stale or interleaved native session")
        latest_session = record.session
        for key in (
            "model_sha256",
            "pack_sha256",
            "plan_sha256",
            "epoch",
            "incarnation",
            "cut",
        ):
            if getattr(record, key) != getattr(binding, key):
                raise ValueError("trace binding differs: " + key)
        for key in ("binary_sha256", "cpu_threads_limit", "expert_vram_mib"):
            if getattr(record, key) != getattr(participant, key):
                raise ValueError("native profile differs: " + key)
        key = (record.session, record.sequence)
        if key in indexed:
            raise ValueError("duplicate native window")
        # A new fenced session starts at sequence 1 and position zero.
        sequence, position = previous.get(record.session, (0, 0))
        if record.sequence != sequence + 1 or record.position != position:
            raise ValueError("native window sequence or position gap")
        previous[record.session] = (record.sequence, record.position + record.rows)
        indexed[key] = record
    return indexed


def summarize(paired: list[Pair]) -> list[Group]:
    """Aggregate paired residuals, rather than subtracting marginal percentiles."""
    by_rows = defaultdict(list)
    for pair in paired:
        by_rows[pair.prefix.rows].append(pair)
    return [
        Group(
            rows=rows,
            feed_us=distribution([p.prefix.feed_us for p in values]),
            rpc_us=distribution([p.prefix.rpc_us for p in values]),
            suffix_native_us=distribution([p.suffix.native_us for p in values]),
            rpc_unattributed_us=distribution([p.rpc_unattributed_us for p in values]),
            tx_bytes=distribution([p.prefix.transaction.tx_bytes for p in values]),
            rx_bytes=distribution([p.prefix.transaction.rx_bytes for p in values]),
        )
        for rows, values in sorted(by_rows.items())
    ]


def collect(
    binding_file: Path,
    prefix_log: Path,
    suffix_log: Path,
    prefix_inventory: Path,
    suffix_inventory: Path,
    workload: Path,
) -> Observations:
    """Validate and hash the same buffers; reject unpaired or stale observations."""
    buffers = {
        key: path.read_bytes()
        for key, path in {
            "binding": binding_file,
            "prefix_log": prefix_log,
            "suffix_log": suffix_log,
            "prefix_inventory": prefix_inventory,
            "suffix_inventory": suffix_inventory,
            "workload": workload,
        }.items()
    }
    hashes = {key: hashlib.sha256(raw).hexdigest() for key, raw in buffers.items()}
    binding = Binding.model_validate_json(buffers["binding"])
    inventories = {
        role: Inventory.model_validate_json(buffers[role + "_inventory"])
        for role in ("prefix", "suffix")
    }
    observed = {
        role: indexed_windows(
            records(buffers[role + "_log"], role), binding, getattr(binding, role)
        )
        for role in ("prefix", "suffix")
    }
    if observed["prefix"].keys() != observed["suffix"].keys():
        raise ValueError("unpaired or unacknowledged native windows")
    paired = [
        Pair(
            prefix=prefix,
            suffix=observed["suffix"][key],
            rpc_unattributed_us=prefix.rpc_us - observed["suffix"][key].native_us,
        )
        for key, prefix in observed["prefix"].items()
    ]
    return Observations(
        binding=binding,
        source_sha256=hashes,
        prefix_inventory=inventories["prefix"],
        suffix_inventory=inventories["suffix"],
        windows=paired,
        by_rows=summarize(paired),
    )


def main() -> None:
    """Write a new observation receipt, preserving existing experiment files."""
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "binding",
        "prefix-log",
        "suffix-log",
        "prefix-inventory",
        "suffix-inventory",
        "workload",
        "output",
    ):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = collect(
        args.binding,
        args.prefix_log,
        args.suffix_log,
        args.prefix_inventory,
        args.suffix_inventory,
        args.workload,
    )
    with args.output.open("x") as output:
        output.write(result.model_dump_json(indent=2, by_alias=True) + "\n")
    print(result.status)


if __name__ == "__main__":
    main()
