"""Node verification/ownership controls using a clearly synthetic native ELF."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from serve.deepseek_layer import LayerReceipt
from strata_control import fabric_host as host
from strata_control import procs
from strata_control.components import ComponentError
from strata_control.fabric_bundle import Installation
from strata_control.fabric_node import Reply
from strata_control.fabric_process import NativeProcess, ProcessExit
from tools.network_fabric_inventory import Inventory
from tools.test_network_fabric_plan import installation, parse, pin, proposal


class LeaseTests(unittest.TestCase):
    """Failed starts consume a fence; malformed state is never reset implicitly."""

    def test_active_owner_stale_epoch_and_corrupt_record_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = host.NodeLease(root, "a", 42, "a" * 64)
            try:
                first.reserve()
                with self.assertRaises(BlockingIOError):
                    host.NodeLease(root, "a", 43, "a" * 64)
            finally:
                first.close()
            with self.assertRaises(ValueError):
                host.NodeLease(root, "a", 42, "a" * 64)
            second = host.NodeLease(root, "a", 43, "a" * 64)
            second.reserve()
            second.close()
            self.assertEqual(
                host.EpochRecord.model_validate_json(
                    (root / "a/epoch.json").read_bytes()
                ).epoch,
                43,
            )
            (root / "a/epoch.json").write_text("corrupt")
            with self.assertRaises(ValidationError):
                host.NodeLease(root, "a", 44, "a" * 64)
            self.assertEqual((root / "a/epoch.json").read_text(), "corrupt")

    def test_suffix_profile_is_not_admitted_as_prefix(self):
        spec = installation(parse(proposal()[0]), Path("/synthetic"))
        suffix = spec.layer_receipt("suffix")
        self.assertEqual(suffix.role, "suffix")
        for change in [
            {"request_timeout_ms": 1},
            {"idle_timeout_ms": 0},
            {"plan_epoch": 0},
        ]:
            with self.assertRaises(ValidationError):
                LayerReceipt.model_validate({**suffix.model_dump(), **change})


@unittest.skipUnless(
    shutil.which("c++"), "C++ compiler required for synthetic ELF controls"
)
class HostTests(unittest.TestCase):
    """Exercise real local children, pidfds and files without loading a model."""

    @classmethod
    def setUpClass(cls):
        cls.compilation = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.compilation.name) / "synthetic-native"
        source = Path(__file__).parent / "fixtures/fabric_native_control.cpp"
        subprocess.run(
            ["c++", "-std=c++20", str(source), "-lcrypto", "-o", str(cls.binary)],
            check=True,
            capture_output=True,
            timeout=60,
        )

    @classmethod
    def tearDownClass(cls):
        cls.compilation.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        data, inventories = proposal()
        data["deadlines"]["startup_s"] = 0.5
        binary_pin = pin("native", self.binary.read_bytes())
        for node in data["nodes"]:
            node["layer_binary"] = binary_pin
            node["worker_binary"] = binary_pin
            observed = json.loads(inventories[node["id"]])
            observed["executable_sha256"] = binary_pin["sha256"]
            raw = json.dumps(observed).encode()
            inventories[node["id"]] = raw
            node["inventory"] = pin(node["id"] + ".inventory.json", raw)
        self.spec = installation(parse(data), self.root)
        self.place = self.spec.location("b")
        for directory in [
            self.place.model_dir,
            self.place.pack_dir,
            self.place.tokenizer_dir,
        ]:
            directory.mkdir(parents=True)
        for item in [
            *self.spec.document.plan.model.mapped_files,
            self.spec.document.plan.model.tokenizer_gguf,
        ]:
            (self.place.model_dir / item.name).write_bytes(b"synthetic artifact")
        for item in self.spec.document.plan.model.pack_files:
            (self.place.pack_dir / item.name).write_bytes(b"synthetic artifact")
        shutil.copy2(self.binary, self.place.layer_binary)
        shutil.copy2(self.binary, self.place.worker_binary)
        self.evidence = self.root / "evidence"
        self.node = next(
            node for node in self.spec.document.plan.nodes if node.id == "b"
        )
        for kind, item in [
            ("inventory", self.node.inventory),
            ("capability", self.node.capability_evidence),
            ("resource", self.node.requirement.evidence),
        ]:
            path = self.evidence / "b" / kind / item.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                inventories["b"] if kind == "inventory" else b"synthetic artifact"
            )
        self.fresh = Inventory.model_validate_json(inventories["b"])
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(host.socket, "gethostname", return_value="b").start()
        mock.patch.object(host, "load_deepseek").start()
        mock.patch.object(host, "observe", return_value=self.fresh).start()
        artifacts = "".join(
            f"ds4-layer-artifact: index={i} file={item.name} bytes={item.bytes} sha256={item.sha256}\n"
            for i, item in enumerate(self.spec.document.plan.model.mapped_files)
        )
        worker = next(
            item for item in self.spec.document.plan.workers if item.node == "b"
        )
        mock.patch.dict(
            os.environ,
            {
                "STRATA_FIXTURE_MODE": "good",
                "STRATA_FIXTURE_EXPERT_SHA": self.spec.document.plan.model.expert_model_sha256,
                "STRATA_FIXTURE_OWNED": str(len(worker.pairs)),
                "STRATA_FIXTURE_ARTIFACTS": artifacts,
            },
        ).start()

    def open(self):
        value = host.NodeHost(self.spec, "b", self.evidence, self.root / "state")
        self.addCleanup(value.close)
        return value

    def test_complete_verification_suffix_and_worker_receipts_and_cleanup(self):
        with self.open() as value:
            with self.assertRaises(RuntimeError):
                value.start_suffix()
            value.verify()
            worker = value.start_worker()
            self.assertEqual(
                worker.model_sha256, self.spec.document.plan.model.expert_model_sha256
            )
            owned = value.worker_ownership()
            self.assertEqual(owned.owned, 2)
            self.assertEqual(owned.mapped, 2)
            suffix = value.start_suffix()
            self.assertEqual(suffix.native.plan_epoch, 42)
            self.assertEqual(suffix.native, self.spec.layer_receipt("suffix"))
            self.assertTrue(
                procs.is_same(
                    suffix.process.pid, suffix.process.start_token, suffix.process.argv
                )
            )
            children = [value.worker.proc, value.suffix.proc]
        self.assertTrue(all(child.poll() is not None for child in children))
        with self.assertRaises(ValueError):
            self.open()

    def test_changed_backing_file_refuses_before_native_start_or_epoch_reservation(
        self,
    ):
        (
            self.place.model_dir / self.spec.document.plan.model.mapped_files[0].name
        ).write_bytes(b"changed")
        with (
            self.open() as value,
            mock.patch("strata_control.fabric_host.NativeProcess") as start,
        ):
            with self.assertRaises(ComponentError):
                value.verify()
            start.assert_not_called()
            self.assertFalse(self.place.runtime_dir.exists())
            self.assertFalse(value.lease.path.exists())

    def test_fresh_resource_exhaustion_refuses_before_staging(self):
        raw = self.fresh.model_dump(mode="json")
        raw["memory"]["available_mib"] = 0
        with (
            self.open() as value,
            mock.patch.object(
                host,
                "observe",
                return_value=Inventory.model_validate_json(json.dumps(raw)),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "RAM"):
                value.verify()
            self.assertFalse(self.place.runtime_dir.exists())
            self.assertFalse(value.lease.path.exists())

    def test_existing_runtime_data_is_preserved(self):
        self.place.runtime_dir.mkdir()
        file = self.place.runtime_dir / "user-data"
        file.write_text("preserve")
        with self.open() as value, self.assertRaises(FileExistsError):
            value.verify()
        self.assertEqual(file.read_text(), "preserve")

    def test_listening_is_not_ownership_ready(self):
        with (
            self.open() as value,
            mock.patch.dict(os.environ, STRATA_FIXTURE_MODE="listen-only"),
        ):
            value.verify()
            value.start_worker()
            with self.assertRaises(TimeoutError):
                value.worker_ownership()
        self.assertIsNotNone(value.worker.proc.poll())

    def test_bad_epoch_artifact_report_and_worker_coverage_fail_and_reap(self):
        # Each failed installation gets a fresh directory/state, not a reused epoch.
        for mode, operation in [
            ("wrong-epoch", "suffix"),
            ("bad-artifacts", "suffix"),
            ("bad-coverage", "worker"),
        ]:
            changed = self.spec.model_dump(mode="json")
            changed["epoch"] += len(mode)
            changed["locations"][1]["runtime_dir"] = str(self.root / mode)
            spec = Installation.model_validate_json(json.dumps(changed))
            with (
                host.NodeHost(
                    spec, "b", self.evidence, self.root / "states" / mode
                ) as value,
                mock.patch.dict(os.environ, STRATA_FIXTURE_MODE=mode),
            ):
                value.verify()
                with self.assertRaises(ValueError):
                    if operation == "suffix":
                        value.start_suffix()
                    else:
                        value.start_worker()
                        value.worker_ownership()
                child = value.suffix if operation == "suffix" else value.worker
            self.assertIsNotNone(child.proc.poll())

    def test_pidfd_cleanup_does_not_end_another_process_with_same_binary(self):
        self.root.joinpath("logs").mkdir()
        args = [str(self.binary), "--role", "suffix"]
        with mock.patch.dict(os.environ, STRATA_FIXTURE_MODE="hold"):
            left = NativeProcess(
                args, self.node.layer_binary, self.root / "logs/left.log"
            )
            right = NativeProcess(
                args, self.node.layer_binary, self.root / "logs/right.log"
            )
        try:
            left.close()
            self.assertIsNone(right.proc.poll())
            self.assertEqual(
                right.identity().binary_sha256, self.node.layer_binary.sha256
            )
        finally:
            left.close()
            right.close()

    def agent(self, new_session=True):
        """Run the real control main with synthetic files/probes in a child process."""
        wrapper = self.root / "synthetic-agent.py"
        observed = self.root / "fresh.json"
        observed.write_text(self.fresh.model_dump_json())
        wrapper.write_text(
            "from pathlib import Path\nfrom unittest import mock\n"
            "from strata_control import fabric_host, fabric_node\n"
            "from tools.network_fabric_inventory import Inventory\n"
            f"fresh=Inventory.model_validate_json(Path({str(observed)!r}).read_bytes())\n"
            "with mock.patch.object(fabric_host.socket,'gethostname',return_value='b'), mock.patch.object(fabric_host,'load_deepseek'), mock.patch.object(fabric_host,'observe',return_value=fresh):\n"
            " fabric_node.main()\n"
        )
        specfile = self.root / "installation.json"
        specfile.write_text(self.spec.model_dump_json())
        errorlog = (self.root / "agent.stderr").open("wb")
        self.addCleanup(errorlog.close)
        process = subprocess.Popen(
            [
                sys.executable,
                str(wrapper),
                "--installation",
                str(specfile),
                "--node",
                "b",
                "--evidence-dir",
                str(self.evidence),
                "--state-dir",
                str(self.root / "agent-state"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errorlog,
            text=True,
            start_new_session=new_session,
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])),
        )

        def end():
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
            if not process.stdin.closed:
                process.stdin.close()

        self.addCleanup(end)
        return process

    def call(self, process, number, action):
        """Use bounded select/read on the real single-flight JSON channel."""
        import select

        process.stdin.write(json.dumps({"id": number, "action": action}) + "\n")
        process.stdin.flush()
        self.assertTrue(
            select.select([process.stdout], [], [], 5)[0], "agent reply timed out"
        )
        reply = Reply.model_validate_json(process.stdout.readline())
        self.assertEqual(reply.id, number)
        self.assertIsNone(reply.error, reply.error)
        return reply.status

    def test_agent_stop_replies_after_cleanup_and_exits_zero(self):
        process = self.agent()
        self.assertTrue(self.call(process, 1, "verify").inputs_staged)
        worker = self.call(process, 2, "start-worker").worker
        self.assertIsNotNone(worker)
        status = self.call(process, 3, "stop")
        self.assertFalse(status.worker_alive)
        self.assertEqual(process.wait(timeout=5), 0)
        self.assertFalse(
            procs.is_same(
                worker.process.pid, worker.process.start_token, worker.process.argv
            )
        )
        self.assertEqual(
            host.HostStatus.model_validate_json(
                (self.place.runtime_dir / "node-exit.json").read_bytes()
            ).worker_exit,
            -15,
        )

    def test_agent_parent_eof_during_startup_reaps_the_owned_native_child(self):
        with mock.patch.dict(os.environ, STRATA_FIXTURE_MODE="hold"):
            process = self.agent(new_session=False)
        self.call(process, 1, "verify")
        process.stdin.write(json.dumps({"id": 2, "action": "start-suffix"}) + "\n")
        process.stdin.flush()
        receipt = self.place.runtime_dir / "suffix-process.json"
        deadline = time.monotonic() + 5
        while (
            not receipt.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertTrue(receipt.exists(), "native child was not started")
        identity = host.ProcessIdentity.model_validate_json(receipt.read_bytes())
        process.stdin.close()
        self.assertNotEqual(process.wait(timeout=8), 0)
        self.assertFalse(
            procs.is_same(identity.pid, identity.start_token, identity.argv)
        )

    def test_partial_native_start_preserves_raw_exit(self):
        with (
            self.open() as value,
            mock.patch.dict(os.environ, STRATA_FIXTURE_MODE="exit"),
        ):
            value.verify()
            with self.assertRaises((OSError, RuntimeError)):
                value.start_suffix()
        exit_record = ProcessExit.model_validate_json(
            (self.place.runtime_dir / "suffix.exit.json").read_bytes()
        )
        self.assertEqual(exit_record.returncode, 42)

    def test_atomic_new_record_refuses_to_replace_user_evidence(self):
        from strata_control.atomic import write_new_text

        target = self.root / "record.json"
        write_new_text(target, "original")
        with self.assertRaises(FileExistsError):
            write_new_text(target, "replacement")
        self.assertEqual(target.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
