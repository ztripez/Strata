"""Owned, single-flight stdio transport for the experimental fabric node agent.

The local process and SSH channel have the coordinator's lifetime. A dropped
remote channel is not a receipt that remote cleanup has completed; the remote
node's durable lease must still fence any subsequent installation.
"""

import ipaddress
import os
import selectors
import shlex
import subprocess
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from serve import lifeline
from serve.tokenizer_assets import Record
from strata_control import atomic, procs
from strata_control.fabric_bundle import Installation
from strata_control.fabric_host import HostStatus
from strata_control.fabric_lease import binding_sha256
from strata_control.fabric_node import Command, Reply
from strata_control.fabric_plan import NodeId, Positive


class AgentRoute(Record):
    """Explicit trusted code/input locations and optional numeric SSH route."""

    node: NodeId
    python: Path
    repository: Path
    installation: Path
    evidence_dir: Path
    state_dir: Path
    ssh_address: str | None = None
    ssh_host_alias: NodeId | None = None
    ssh_user: Annotated[str, Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_-]*$")] | None = None

    @field_validator(
        "python", "repository", "installation", "evidence_dir", "state_dir"
    )
    @classmethod
    def absolute(cls, value: Path) -> Path:
        """Remote execution never depends on a login shell's cwd or PATH."""
        if not value.is_absolute():
            raise ValueError("fabric agent locations must be absolute")
        return value

    @model_validator(mode="after")
    def ssh_binding(self):
        """SSH requires an explicit unicast address and known-host identity."""
        if self.ssh_address is None:
            if self.ssh_host_alias is not None or self.ssh_user is not None:
                raise ValueError("local agent cannot specify SSH identity")
        else:
            address = ipaddress.IPv4Address(self.ssh_address)
            if address.is_unspecified or address.is_multicast:
                raise ValueError("agent SSH address must be unicast IPv4")
            if self.ssh_host_alias is None:
                raise ValueError("SSH agent requires an explicit host-key alias")
        return self

    def argv(self, installation_sha256: str) -> list[str]:
        """Render only the fixed agent CLI; SSH shell words are individually quoted."""
        args = [
            str(self.python),
            "-m",
            "strata_control.fabric_node",
            "--installation",
            str(self.installation),
            "--installation-sha256",
            installation_sha256,
            "--node",
            self.node,
            "--evidence-dir",
            str(self.evidence_dir),
            "--state-dir",
            str(self.state_dir),
        ]
        if self.ssh_address is None:
            return args
        destination = (
            self.node if self.ssh_user is None else f"{self.ssh_user}@{self.node}"
        )
        return [
            "ssh",
            "-T",
            "-oBatchMode=yes",
            "-oConnectTimeout=10",
            "-oServerAliveInterval=5",
            "-oServerAliveCountMax=2",
            "-oStrictHostKeyChecking=yes",
            f"-oHostName={self.ssh_address}",
            f"-oHostKeyAlias={self.ssh_host_alias}",
            "--",
            destination,
            shlex.join(["env", f"PYTHONPATH={self.repository}", *args]),
        ]


class AgentExit(Record):
    """Raw owned-channel exit; a clean stop reply also confirms node cleanup."""

    node: NodeId
    pid: Positive
    start_token: str | None
    argv: tuple[str, ...]
    returncode: int | None
    stop_confirmed: bool
    error: str | None


class AgentClient:
    """Bounded serial RPC channel; malformed or timed-out replies poison it.

    Construction starts the fixed agent only. The caller must explicitly verify
    inputs and request native startup. No retry or remote process restart occurs.
    """

    def __init__(self, route: AgentRoute, spec: Installation, log_dir: Path):
        """Own a fresh channel and retain its stderr and complete reply frames."""
        self.route = route
        self.spec = spec
        self.log_dir = log_dir
        self.proc = None
        self.start_token = None
        self.closed = False
        self.poisoned = False
        self.stop_confirmed = False
        self.failure = None
        self.number = 0
        self.pending = bytearray()
        self.serial = threading.Lock()
        self.selector = selectors.DefaultSelector()
        self.stderr = None
        self.frames = None
        digest = binding_sha256(spec)
        try:
            self.stderr = (log_dir / f"{route.node}.stderr.log").open("xb")
            self.frames = (log_dir / f"{route.node}.replies.jsonl").open("xb")
            self.proc = lifeline.popen(
                route.argv(digest),
                role="engine",
                start_new_session=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr,
                bufsize=0,
                env=dict(os.environ, PYTHONPATH=str(route.repository)),
            )
            identity = procs.identify(self.proc.pid)
            deadline = time.monotonic() + 1
            while (
                identity is None
                and self.proc.poll() is None
                and time.monotonic() < deadline
            ):
                # An exec transition can briefly expose an empty /proc cmdline.
                # Observe this same owned process; never start another one.
                time.sleep(0.01)
                identity = procs.identify(self.proc.pid)
            if identity is None:
                raise RuntimeError("fabric agent exited before identity observation")
            self.start_token = identity.token
            os.set_blocking(self.proc.stdout.fileno(), False)
            os.set_blocking(self.proc.stdin.fileno(), False)
            self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        except BaseException as error:
            self.poisoned = True
            self.failure = f"{type(error).__name__}: {error}"
            self.close()
            raise

    def request(self, action: str, timeout_s: float) -> HostStatus:
        """Wait for one matching typed reply, including node/plan/epoch identity."""
        if not self.serial.acquire(blocking=False):
            raise RuntimeError("fabric agent already has an in-flight operation")
        try:
            if self.closed or self.poisoned or self.proc.poll() is not None:
                raise RuntimeError("fabric agent channel is unavailable")
            if self.pending or self.selector.select(0):
                raise ValueError("unsolicited fabric agent output")
            if not 0 < timeout_s <= 14400:
                raise ValueError(
                    "fabric control deadline must be in (0, 14400] seconds"
                )
            self.number += 1
            command = Command(id=self.number, action=action)
            frame = (command.model_dump_json() + "\n").encode()
            # Frames are far smaller than PIPE_BUF; a partial write is fatal.
            if os.write(self.proc.stdin.fileno(), frame) != len(frame):
                raise OSError("incomplete fabric control command write")
            deadline = time.monotonic() + timeout_s
            while b"\n" not in self.pending:
                left = deadline - time.monotonic()
                if left <= 0 or not self.selector.select(left):
                    raise TimeoutError(
                        f"fabric {self.route.node} {action} reply timed out"
                    )
                chunk = os.read(self.proc.stdout.fileno(), 65536)
                if not chunk:
                    raise EOFError(
                        f"fabric {self.route.node} closed before {action} reply"
                    )
                self.pending.extend(chunk)
                if len(self.pending) > 4 * 1024**2:
                    raise ValueError("fabric agent reply exceeds 4 MiB")
            raw, _, extra = self.pending.partition(b"\n")
            self.pending = bytearray(extra)
            self.frames.write(raw + b"\n")
            self.frames.flush()
            reply = Reply.model_validate_json(raw)
            if reply.id != command.id or self.pending:
                raise ValueError("fabric agent reply sequence/framing mismatch")
            if reply.error is not None:
                raise RuntimeError(f"fabric {self.route.node}: {reply.error}")
            status = reply.status
            if (status.node, status.plan_sha256, status.epoch) != (
                self.route.node,
                self.spec.document.plan_sha256,
                self.spec.epoch,
            ):
                raise ValueError(
                    "fabric agent replied with a different node/plan/epoch"
                )
            if action == "stop":
                if status.layer_alive or status.worker_alive:
                    raise ValueError("fabric agent stop left native children running")
                self.stop_confirmed = True
            return status
        except BaseException as error:
            self.poisoned = True
            self.failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.serial.release()

    def close(self) -> None:
        """Request owned cleanup, then bound channel exit and preserve raw outcome.

        After a protocol error, closing stdin activates the agent's owner guard
        instead of sending another ambiguous operation. No PID-name cleanup is used.
        """
        if self.closed:
            return
        failure = None
        try:
            if self.proc is not None:
                if self.proc.poll() is None and not self.poisoned:
                    try:
                        self.request("stop", 30)
                    except Exception as error:  # noqa: BLE001 - re-raised after owned cleanup
                        failure = error
                if self.proc.stdin is not None:
                    self.proc.stdin.close()
                try:
                    self.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait(timeout=5)
                if not self.poisoned and self.proc.returncode != 0:
                    failure = RuntimeError("fabric agent failed after stop reply")
        finally:
            self.closed = True
            self.selector.close()
            with ExitStack() as closing:
                for stream in (self.stderr, self.frames):
                    if stream is not None:
                        closing.callback(stream.close)
                if self.proc is not None:
                    closing.callback(self.proc.stdout.close)
                    if self.proc.poll() is not None:
                        closing.callback(lifeline.release, self.proc)
                    outcome = AgentExit(
                        node=self.route.node,
                        pid=self.proc.pid,
                        start_token=self.start_token,
                        argv=tuple(self.proc.args),
                        returncode=self.proc.poll(),
                        stop_confirmed=self.stop_confirmed,
                        error=self.failure
                        if failure is None
                        else f"{type(failure).__name__}: {failure}",
                    )
                    atomic.write_new_text(
                        self.log_dir / f"{self.route.node}.exit.json",
                        outcome.model_dump_json(indent=2) + "\n",
                    )
        if failure is not None:
            raise failure
