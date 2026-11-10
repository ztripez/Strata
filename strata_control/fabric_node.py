"""Supervise one fabric node over an owned stdio control channel.

Run in a dedicated Linux process. EOF ends the node and its owned children;
commands do not carry arbitrary executable arguments or per-token inference.
This node agent never opens the user-facing HTTP endpoint.
"""

import argparse
import os
import queue
import signal
import sys
import threading
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from serve import owner_guard
from serve.tokenizer_assets import Record
from strata_control.fabric_bundle import Installation
from strata_control.fabric_host import HostStatus, NodeHost
from strata_control.fabric_plan import Positive


class Command(Record):
    """Bounded serial control operation; native processes come only from the plan."""

    id: Positive
    action: Literal[
        "verify", "start-worker", "start-suffix", "ownership", "status", "stop"
    ]


class Reply(Record):
    """One command result carrying actual node status, or an explicit fatal error."""

    id: Positive
    status: HostStatus | None = None
    error: Annotated[str, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def one_result(self):
        """A reply is either an observed status or a failure, never ambiguous."""
        if (self.status is None) == (self.error is None):
            raise ValueError("fabric reply requires exactly one status or error")
        return self


def control(node: NodeHost) -> None:
    """Read commands off-thread so parent EOF also interrupts blocked startup.

    The existing owner guard supplies emergency process-group termination.
    Lifeline ownership separately binds isolated native children to this agent.
    Kernel-blocked I/O may still require quarantine rather than immediate exit.
    """
    if os.name != "posix":
        raise RuntimeError("fabric node control requires POSIX ownership")
    if os.getpgrp() != os.getpid():
        os.setsid()
    owner_read, owner_write = os.pipe()
    requests = queue.Queue(maxsize=1)
    inflight = threading.Event()
    owner_guard.start(owner_read, grace=10.0)

    def read():
        clean_stop = False
        try:
            while raw := sys.stdin.buffer.readline(4 * 1024**2 + 1):
                if len(raw) > 4 * 1024**2 or not raw.endswith(b"\n"):
                    raise ValueError("fabric control command exceeds framing bound")
                command = Command.model_validate_json(raw)
                if inflight.is_set():
                    raise ValueError(
                        "fabric control permits only one in-flight command"
                    )
                inflight.set()
                requests.put(command, block=False)
                if command.action == "stop":
                    clean_stop = True
                    return
        finally:
            if not clean_stop:
                os.close(owner_write)

    reader = threading.Thread(target=read, name="fabric-control-reader", daemon=True)
    reader.start()
    previous = 0
    while True:
        command = requests.get()
        try:
            if command.id <= previous:
                raise ValueError("fabric control command IDs must increase")
            previous = command.id
            if command.action == "verify":
                node.verify()
            elif command.action == "start-worker":
                node.start_worker()
            elif command.action == "start-suffix":
                node.start_suffix()
            elif command.action == "ownership":
                node.worker_ownership()
            elif command.action == "stop":
                node.close()
            reply = Reply(id=command.id, status=node.status())
        except Exception as error:
            print(
                Reply(
                    id=command.id, error=f"{type(error).__name__}: {error}"
                ).model_dump_json(),
                flush=True,
            )
            raise
        inflight.clear()
        print(reply.model_dump_json(), flush=True)
        if command.action == "stop":
            reader.join(timeout=1)
            if reader.is_alive():
                raise RuntimeError("fabric control reader did not end on stop")
            return


def main() -> None:
    """Bind one actual hostname and close all native children on any exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installation", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()

    def stop(*_):
        raise SystemExit(1)

    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, stop)
    spec = Installation.model_validate_json(args.installation.read_bytes())
    with NodeHost(spec, args.node, args.evidence_dir, args.state_dir) as node:
        control(node)


if __name__ == "__main__":
    main()
