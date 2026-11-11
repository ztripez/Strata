"""Install a trusted two-node native fabric and publish the existing HTTP API.

Run on the prefix owner in a dedicated process. Inputs must already be staged
at the explicit routes; this module does not copy weights or generate placement
costs. Node verification and native READY precede the global publication barrier.
"""

import argparse
import signal
import socket
import threading
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from serve import deepseek_layer, server
from serve.tokenizer_assets import Digest, Record
from strata_control import atomic, procs
from strata_control.fabric_bundle import Installation
from strata_control.fabric_client import AgentClient, AgentRoute
from strata_control.fabric_host import HostStatus, layer_artifacts
from strata_control.fabric_plan import NodeId, Positive
from strata_control.fabric_process import ProcessIdentity


class CoordinatorSpec(Record):
    """One installation and its explicit local-prefix/SSH-suffix control routes."""

    installation: Installation
    routes: Annotated[tuple[AgentRoute, ...], Field(min_length=2, max_length=2)]
    journal_dir: Path
    health_interval_s: Annotated[float, Field(ge=0.1, le=10, allow_inf_nan=False)] = 1.0
    health_timeout_s: Annotated[float, Field(gt=0, le=10, allow_inf_nan=False)] = 5.0
    drain_s: Annotated[float, Field(ge=0, le=300, allow_inf_nan=False)] = 30.0

    @model_validator(mode="after")
    def exact_routes(self):
        """Require both declared nodes, with no implicit routing or restart policy."""
        spec = self.installation
        plan = spec.document.plan
        if {route.node for route in self.routes} != {node.id for node in plan.nodes}:
            raise ValueError("coordinator must route both plan nodes exactly once")
        if not self.journal_dir.is_absolute():
            raise ValueError("coordinator journal directory must be absolute")
        for route in self.routes:
            local = route.ssh_address is None
            if local != (route.node == plan.prefix.node):
                raise ValueError(
                    "prefix agent must be local and suffix agent must use SSH"
                )
            runtime = spec.location(route.node).runtime_dir
            if route.state_dir == runtime or runtime in route.state_dir.parents:
                raise ValueError(
                    "persistent node state must be outside fresh runtime directory"
                )
        return self


class Activation(Record):
    """Completed global startup barrier; no numerical or performance acceptance."""

    status: Literal["startup_barrier_pass"] = "startup_barrier_pass"
    plan_sha256: Digest
    epoch: Positive
    prefix: ProcessIdentity
    nodes: tuple[HostStatus, ...]
    endpoint: str


class CoordinatorExit(Record):
    """Installation outcome and cleanup limits, retained on failed startup too."""

    activated: bool
    failure: str | None
    cleanup_errors: tuple[str, ...]
    prefix_returncode: int | None
    unconfirmed_nodes: tuple[NodeId, ...]


def validate_status(spec: Installation, status: HostStatus, *, admitted: bool) -> None:
    """Check complete native receipts, allowing only post-publication expert loss."""
    plan = spec.document.plan
    if (
        status.node not in {node.id for node in plan.nodes}
        or status.plan_sha256 != spec.document.plan_sha256
        or status.epoch != spec.epoch
        or not status.inputs_staged
    ):
        raise ValueError("fabric node is not staged for this plan and epoch")
    node = next(node for node in plan.nodes if node.id == status.node)
    worker = next(
        (worker for worker in plan.workers if worker.node == status.node), None
    )
    if worker is not None:
        actual = status.worker
        owned = status.ownership
        if (
            actual is None
            or owned is None
            or actual.endpoint != spec.location(node.id).expert_endpoint.native()
            or actual.model_sha256 != plan.model.expert_model_sha256
            or actual.process.binary_sha256 != node.worker_binary.sha256
            or actual.process.argv
            != (
                str(spec.location(node.id).worker_binary),
                *spec.worker_arguments(worker),
            )
            or owned.owned != len(worker.pairs)
            or owned.gpu_bytes > worker.expert_vram_mib * 1024**2
            or owned.ram_bytes > worker.resident_ram_mib * 1024**2
        ):
            raise ValueError(
                "fabric expert listener/ownership differs from installation"
            )
        if status.worker_alive and status.worker_exit is not None:
            raise ValueError("fabric worker liveness contradicts its raw exit")
        if not status.worker_alive and (not admitted or status.worker_exit is None):
            raise RuntimeError(
                "mandatory expert worker is not alive at startup barrier"
            )
    elif (
        status.worker is not None or status.worker_alive or status.ownership is not None
    ):
        raise ValueError("fabric node reports an unassigned expert worker")
    if status.node == plan.suffix.node:
        actual = status.layer
        if (
            actual is None
            or actual.native != spec.layer_receipt("suffix")
            or actual.process.binary_sha256 != node.layer_binary.sha256
            or actual.process.argv
            != (
                str(spec.location(node.id).layer_binary),
                "--serve",
                *spec.layer_arguments("suffix"),
            )
        ):
            raise ValueError("fabric suffix runtime receipt differs from installation")
        if not status.layer_alive or status.layer_exit is not None:
            raise RuntimeError(
                "fabric suffix has exited; layer sessions cannot be replayed"
            )
    elif status.layer is not None or status.layer_alive:
        raise ValueError("prefix node agent must not own a second layer process")


class Coordinator:
    """Own both agents, one native prefix and one loopback HTTP publication.

    Control calls occur at startup and between periodic health checks, never per
    token. Expert-process loss after publication retains native recomputation;
    agent or layer loss closes this installation without automatic replay.
    """

    def __init__(self, config: CoordinatorSpec):
        """Bind the actual prefix host before creating any journal or process."""
        self.config = CoordinatorSpec.model_validate_json(config.model_dump_json())
        self.spec = self.config.installation
        if socket.gethostname() != self.spec.document.plan.prefix.node:
            raise ValueError("fabric coordinator must run on the actual prefix owner")
        self.clients = {}
        self.svc = None
        self.httpd = None
        self.prefix_proc = None
        self.prefix_boot = None
        self.activation = None
        self.failure = None
        self.closed = False
        self.started = False
        self.journal_created = False

    def _prefix_identity(self) -> ProcessIdentity:
        """Require the actual live prefix executable and exact derived argv."""
        if self.prefix_proc is None or self.prefix_proc.poll() is not None:
            raise RuntimeError("fabric prefix is unavailable; no automatic restart")
        actual = procs.identify(self.prefix_proc.pid)
        launch = self.spec.http_launch()
        if actual is None or actual.cmd != (
            str(launch.executable),
            "--serve",
            *launch.native_args,
        ):
            raise ValueError("actual fabric prefix argv differs from installation")
        node = next(
            node
            for node in self.spec.document.plan.nodes
            if node.id == self.spec.document.plan.prefix.node
        )
        observed = ProcessIdentity(
            pid=actual.pid,
            start_token=actual.token,
            argv=actual.cmd,
            binary_sha256=node.layer_binary.sha256,
        )
        if self.prefix_boot is None:
            node.layer_binary.verify(Path(f"/proc/{actual.pid}/exe"))
            self.prefix_boot = observed
        elif observed != self.prefix_boot:
            raise ValueError("fabric prefix process incarnation changed")
        return observed

    def health(self) -> tuple[HostStatus, ...]:
        """Observe required agents/layers; expert exits remain explicit degradation."""
        self._prefix_identity()
        observed = []
        for route in self.config.routes:
            status = self.clients[route.node].request(
                "status", self.config.health_timeout_s
            )
            validate_status(self.spec, status, admitted=self.activation is not None)
            if self.activation is not None:
                previous = next(
                    node for node in self.activation.nodes if node.node == status.node
                )
                if (status.layer, status.worker, status.ownership) != (
                    previous.layer,
                    previous.worker,
                    previous.ownership,
                ):
                    raise ValueError(
                        "fabric native process/ownership incarnation changed"
                    )
            observed.append(status)
        return tuple(observed)

    def start(self) -> Activation:
        """Prepare both nodes, require all native receipts, then publish HTTP.

        Any failure rolls back owned startup. The accepted startup-barrier receipt
        is stored separately from publication; neither advances E5/E6 acceptance.
        """
        if self.started or self.closed:
            raise RuntimeError("fabric coordinator cannot start twice")
        self.started = True
        root = self.config.journal_dir
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.journal_created = True
        atomic.write_new_text(
            root / "coordinator.json", self.config.model_dump_json(indent=2) + "\n"
        )
        timeout = self.spec.document.plan.deadlines.startup_s
        try:
            for route in self.config.routes:
                client = AgentClient(route, self.spec, root)
                self.clients[route.node] = client
                status = client.request("verify", timeout)
                if not status.inputs_staged:
                    raise ValueError("fabric agent did not stage verified inputs")
            for worker in self.spec.document.plan.workers:
                self.clients[worker.node].request("start-worker", timeout)
            suffix_node = self.spec.document.plan.suffix.node
            self.clients[suffix_node].request("start-suffix", timeout)
            self.svc = deepseek_layer.create_service(self.spec.http_launch())
            self.prefix_proc = self.svc.engine.proc
            log = self.spec.http_launch().engine_log
            if log.stat().st_size > 64 * 1024**2:
                raise ValueError("prefix startup log exceeds 64 MiB")
            layer_artifacts(log.read_text(), self.spec.document.plan.model.mapped_files)
            for worker in self.spec.document.plan.workers:
                self.clients[worker.node].request("ownership", timeout)
            statuses = self.health()
            barrier = Activation(
                plan_sha256=self.spec.document.plan_sha256,
                epoch=self.spec.epoch,
                prefix=self._prefix_identity(),
                nodes=statuses,
                endpoint=f"http://127.0.0.1:{self.spec.http_port}",
            )
            atomic.write_new_text(
                root / "startup-barrier.json", barrier.model_dump_json(indent=2) + "\n"
            )
            self.svc.begin_drain()  # Refuse inference until publication is recorded.
            self.httpd = server.serve(
                self.svc, host="127.0.0.1", port=self.spec.http_port
            )
            # Active HTTP handlers have bounded model cancellation, but socket clients
            # must not make socketserver's default thread join block node cleanup.
            self.httpd.block_on_close = False
            atomic.write_new_text(
                root / "published.json", barrier.model_dump_json(indent=2) + "\n"
            )
            self.activation = barrier
            with self.svc.admission_changed:
                self.svc.draining = False
                self.svc.admission_changed.notify_all()
            return barrier
        except BaseException as error:
            self.failure = f"{type(error).__name__}: {error}"
            self.close()
            raise

    def close(self) -> None:
        """Drain admission, cancel remaining calls, then stop prefix before agents.

        Every cleanup is attempted and failures are journalled and raised. Kernel
        blocked children or an unreachable remote host require external quarantine;
        a local SSH exit alone never certifies remote quiescence.
        """
        if self.closed:
            return
        self.closed = True
        errors = []

        def attempt(operation):
            try:
                operation()
            except Exception as error:  # noqa: BLE001 - aggregated and raised after cleanup
                errors.append(error)

        if self.svc is not None:
            self.svc.begin_drain()
            if self.httpd is not None:
                attempt(self.httpd.shutdown)
                attempt(self.httpd.server_close)
            if self.svc.wait_for_drain(self.config.drain_s):
                self.svc.cancel_owned_requests()
            attempt(self.svc.engine.close)
            if self.svc.wait_for_drain(5):
                errors.append(
                    RuntimeError("HTTP requests remained after prefix shutdown")
                )
            if self.httpd is None and getattr(self.svc, "telemetry", None) is not None:
                attempt(self.svc.telemetry.close)
        for client in reversed(tuple(self.clients.values())):
            attempt(client.close)
        root = self.config.journal_dir
        if self.journal_created:
            receipt = CoordinatorExit(
                activated=self.activation is not None,
                failure=self.failure,
                cleanup_errors=tuple(
                    f"{type(error).__name__}: {error}" for error in errors
                ),
                prefix_returncode=None
                if self.prefix_proc is None
                else self.prefix_proc.poll(),
                unconfirmed_nodes=tuple(
                    node
                    for node, client in self.clients.items()
                    if not client.stop_confirmed
                ),
            )
            attempt(
                lambda: atomic.write_new_text(
                    root / "coordinator-exit.json",
                    receipt.model_dump_json(indent=2) + "\n",
                )
            )
        if errors:
            raise ExceptionGroup("fabric installation cleanup failed", errors)

    def run(self) -> None:
        """Serve until interrupted or mandatory health fails; never reinstall here."""
        try:
            self.start()
            print(self.activation.model_dump_json(), flush=True)
            wake = threading.Event()
            while not wake.wait(self.config.health_interval_s):
                self.health()
        except KeyboardInterrupt:
            pass
        except BaseException as error:
            self.failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.close()


def main() -> None:
    """Run an explicit prepared binding without changing the ordinary server path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = CoordinatorSpec.model_validate_json(args.config.read_bytes())

    def stop(*_):
        raise KeyboardInterrupt

    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, stop)
    Coordinator(config).run()


if __name__ == "__main__":
    main()
