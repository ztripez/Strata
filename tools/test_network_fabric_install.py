"""Real pipes/ELF/HTTP lifecycle controls, with synthetic models and SSH routing.

Agents use the real node host, process owner, receipt parsers and coordinator.
Their hostname/resource/tokenizer probes are explicitly synthetic. SSH is mapped
to local exec here; these tests do not claim two-machine model or network proof.
"""

import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from serve import deepseek_layer, frontend
from serve import fabric_install as install
from serve.test_dsv41 import TEMPLATE, mini_tokenizer
from strata_control import fabric_client, procs
from strata_control.fabric_client import AgentClient, AgentExit, AgentRoute
from tools.test_network_fabric_plan import installation, parse, pin, proposal


@unittest.skipUnless(shutil.which("c++"), "synthetic ELF compiler required")
class InstallTests(unittest.TestCase):
    """Exercise actual owned subprocesses and HTTP without any model math."""

    @classmethod
    def setUpClass(cls):
        cls.compilation = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.compilation.name) / "synthetic-native"
        subprocess.run(
            [
                "c++",
                "-std=c++20",
                str(Path(__file__).parent / "fixtures/fabric_native_control.cpp"),
                "-lcrypto",
                "-o",
                str(cls.binary),
            ],
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
        data["deadlines"]["startup_s"] = 2.0
        self.tokenizer = mini_tokenizer()
        data["model"]["tokenizer"]["assets_sha256"]["chat_template.jinja"] = (
            hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
        )
        binary_pin = pin("native", self.binary.read_bytes())
        for node in data["nodes"]:
            node["layer_binary"] = binary_pin
            node["worker_binary"] = binary_pin
            inventory = json.loads(inventories[node["id"]])
            inventory["executable_sha256"] = binary_pin["sha256"]
            raw = json.dumps(inventory).encode()
            inventories[node["id"]] = raw
            node["inventory"] = pin(node["id"] + ".inventory.json", raw)
        spec = installation(parse(data), self.root)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.spec = type(spec).model_validate_json(
            json.dumps({**spec.model_dump(mode="json"), "http_port": port})
        )
        self.binding = self.root / "installation.json"
        self.binding.write_text(self.spec.model_dump_json(indent=2))
        self.evidence = self.root / "evidence"
        for node in self.spec.document.plan.nodes:
            place = self.spec.location(node.id)
            for directory in (place.model_dir, place.pack_dir, place.tokenizer_dir):
                directory.mkdir(parents=True)
            for item in (
                *self.spec.document.plan.model.mapped_files,
                self.spec.document.plan.model.tokenizer_gguf,
            ):
                (place.model_dir / item.name).write_bytes(b"synthetic artifact")
            for item in self.spec.document.plan.model.pack_files:
                (place.pack_dir / item.name).write_bytes(b"synthetic artifact")
            (place.tokenizer_dir / "chat_template.jinja").write_bytes(
                TEMPLATE.read_bytes()
            )
            shutil.copy2(self.binary, place.layer_binary)
            shutil.copy2(self.binary, place.worker_binary)
            for kind, item in (
                ("inventory", node.inventory),
                ("capability", node.capability_evidence),
                ("resource", node.requirement.evidence),
            ):
                path = self.evidence / node.id / kind / item.name
                path.parent.mkdir(parents=True)
                path.write_bytes(
                    inventories[node.id]
                    if kind == "inventory"
                    else b"synthetic artifact"
                )
        # Only these spawned test interpreters load the synthetic probe overrides.
        self.code = self.root / "synthetic code ' with spaces"
        self.code.mkdir()
        self.code.joinpath("sitecustomize.py").write_text(
            "import os,sys\nfrom pathlib import Path\n"
            f"sys.path.insert(0,{str(Path(__file__).resolve().parents[1])!r})\n"
            "if '--node' in sys.argv:\n"
            " from strata_control import fabric_host\n"
            " from strata_control.fabric_bundle import Installation\n"
            " from tools.network_fabric_inventory import Inventory\n"
            " node=sys.argv[sys.argv.index('--node')+1]\n"
            " binding=Installation.model_validate_json(Path(sys.argv[sys.argv.index('--installation')+1]).read_bytes())\n"
            f" fresh=Inventory.model_validate_json((Path({str(self.evidence)!r})/node/'inventory'/(node+'.inventory.json')).read_bytes())\n"
            " fabric_host.socket.gethostname=lambda:node\n"
            " fabric_host.observe=lambda *a,**k:fresh\n"
            " fabric_host.load_deepseek=lambda *a,**k:None\n"
            " worker=next(w for w in binding.document.plan.workers if w.node==node)\n"
            " os.environ['STRATA_FIXTURE_OWNED']=str(len(worker.pairs))\n"
            " os.environ['STRATA_FIXTURE_MODE']=os.environ.get('STRATA_FIXTURE_MODE_'+node,'good')\n"
            " if os.environ.get('STRATA_FIXTURE_CONTROL_MODE')=='wrong-node':\n"
            "  original=fabric_host.NodeHost.status\n"
            "  def wrong_status(self):\n"
            "   data=original(self).model_dump(mode='json');data['node']='unexpected'\n"
            "   import json\n"
            "   return fabric_host.HostStatus.model_validate_json(json.dumps(data))\n"
            "  fabric_host.NodeHost.status=wrong_status\n"
        )
        routes = []
        for node in ("a", "b"):
            routes.append(
                AgentRoute(
                    node=node,
                    python=Path(sys.executable),
                    repository=self.code,
                    installation=self.binding,
                    evidence_dir=self.evidence,
                    state_dir=self.root / "state",
                    ssh_address="127.0.0.1" if node == "b" else None,
                    ssh_host_alias="synthetic-host" if node == "b" else None,
                )
            )
        self.config = install.CoordinatorSpec(
            installation=self.spec,
            routes=tuple(routes),
            journal_dir=self.root / "journal",
            drain_s=0.0,
            health_interval_s=0.1,
            health_timeout_s=1.0,
        )
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(install.socket, "gethostname", return_value="a").start()
        mock.patch.object(
            deepseek_layer, "load_deepseek", return_value=self.tokenizer
        ).start()
        self.addCleanup(frontend.use_family, None)
        saved_timeout = deepseek_layer.server.ENGINE_READY_S
        self.addCleanup(setattr, deepseek_layer.server, "ENGINE_READY_S", saved_timeout)
        artifacts = "".join(
            f"ds4-layer-artifact: index={index} file={item.name} bytes={item.bytes} sha256={item.sha256}\n"
            for index, item in enumerate(self.spec.document.plan.model.mapped_files)
        )
        mock.patch.dict(
            os.environ,
            {
                "STRATA_FIXTURE_MODE": "good",
                "STRATA_FIXTURE_EXPERT_SHA": self.spec.document.plan.model.expert_model_sha256,
                "STRATA_FIXTURE_ARTIFACTS": artifacts,
                "STRATA_FIXTURE_OUTPUT": " ".join(
                    map(str, self.tokenizer.encode("OK", parse_special=True) + [1])
                ),
            },
        ).start()
        real_popen = fabric_client.lifeline.popen

        def synthetic_ssh(argv, *args, **kwargs):
            if argv[0] == "ssh":
                argv = shlex.split(argv[-1])
            return real_popen(argv, *args, **kwargs)

        mock.patch.object(
            fabric_client.lifeline, "popen", side_effect=synthetic_ssh
        ).start()

    def open(self):
        value = install.Coordinator(self.config)
        self.addCleanup(value.close)
        return value

    def assert_children_gone(self, value):
        for client in value.clients.values():
            self.assertIsNotNone(client.proc.poll())
        for place in self.spec.locations:
            for file in place.runtime_dir.glob("*-process.json"):
                identity = install.ProcessIdentity.model_validate_json(
                    file.read_bytes()
                )
                self.assertFalse(
                    procs.is_same(identity.pid, identity.start_token, identity.argv)
                )
        if value.prefix_proc is not None:
            self.assertIsNotNone(value.prefix_proc.poll())

    def post(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.spec.http_port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "deepseek41",
                    "messages": [{"role": "user", "content": "Say OK"}],
                    "max_tokens": 8,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.load(response)

    def test_actual_agents_prefix_and_http_publish_only_after_all_receipts(self):
        value = self.open()
        accepted = value.start()
        self.assertEqual(len(accepted.nodes), 2)
        self.assertTrue(all(node.ownership is not None for node in accepted.nodes))
        self.assertEqual(self.post()["choices"][0]["message"]["content"], "OK")
        self.assertEqual(value.health(), accepted.nodes)
        value.close()
        self.assert_children_gone(value)
        for node in ("a", "b"):
            exited = AgentExit.model_validate_json(
                (self.config.journal_dir / f"{node}.exit.json").read_bytes()
            )
            self.assertTrue(exited.stop_confirmed)
            self.assertEqual(exited.returncode, 0)

    def test_ownership_missing_after_listen_rolls_back_without_http(self):
        with mock.patch.dict(os.environ, STRATA_FIXTURE_MODE_b="listen-only"):
            value = self.open()
            with mock.patch.object(
                install.server, "serve", wraps=install.server.serve
            ) as publish:
                with self.assertRaises((RuntimeError, TimeoutError)):
                    value.start()
                publish.assert_not_called()
        self.assertFalse((self.config.journal_dir / "published.json").exists())
        self.assert_children_gone(value)
        self.assertIn(
            "ownership", (self.config.journal_dir / "coordinator-exit.json").read_text()
        )

    def test_wrong_suffix_epoch_rolls_back_before_prefix_or_http(self):
        with mock.patch.dict(os.environ, STRATA_FIXTURE_MODE_b="wrong-epoch"):
            value = self.open()
            with mock.patch.object(deepseek_layer, "create_service") as prefix:
                with self.assertRaises(RuntimeError):
                    value.start()
                prefix.assert_not_called()
        self.assert_children_gone(value)

    def test_publication_record_failure_closes_bound_endpoint_and_owned_children(self):
        value = self.open()
        write = install.atomic.write_new_text

        def fail_publication(path, text):
            if path.name == "published.json":
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    self.post()
                self.assertEqual(denied.exception.code, 503)
                raise OSError("synthetic publication disk failure")
            return write(path, text)

        with (
            mock.patch.object(
                install.atomic, "write_new_text", side_effect=fail_publication
            ),
            self.assertRaisesRegex(OSError, "publication disk"),
        ):
            value.start()
        self.assert_children_gone(value)
        self.assertTrue((self.config.journal_dir / "startup-barrier.json").exists())
        self.assertFalse((self.config.journal_dir / "published.json").exists())
        with self.assertRaises(OSError):
            self.post()

    def test_expert_exit_after_publication_is_degraded_not_layer_restart(self):
        value = self.open()
        accepted = value.start()
        worker = next(node.worker for node in accepted.nodes if node.node == "b")
        self.assertTrue(
            procs.is_same(
                worker.process.pid, worker.process.start_token, worker.process.argv
            )
        )
        os.kill(worker.process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            observed = value.health()
            stopped = next(node for node in observed if node.node == "b")
            if not stopped.worker_alive:
                break
            time.sleep(0.01)
        self.assertEqual(stopped.worker_exit, -15)
        self.assertEqual(self.post()["choices"][0]["message"]["content"], "OK")
        self.assertEqual(value._prefix_identity(), accepted.prefix)

    def test_lost_suffix_supervisor_closes_http_without_replay(self):
        value = self.open()
        failures = []

        def supervise():
            try:
                value.run()
            except Exception as error:  # noqa: BLE001 - asserted below
                failures.append(error)

        thread = threading.Thread(target=supervise)
        thread.start()
        deadline = time.monotonic() + 8
        while (
            value.activation is None
            and thread.is_alive()
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertIsNotNone(value.activation, failures)
        suffix = next(
            node.layer for node in value.activation.nodes if node.layer is not None
        )
        self.assertTrue(
            procs.is_same(
                suffix.process.pid, suffix.process.start_token, suffix.process.argv
            )
        )
        os.kill(suffix.process.pid, signal.SIGTERM)
        thread.join(timeout=8)
        self.assertFalse(thread.is_alive())
        self.assertTrue(failures)
        self.assert_children_gone(value)
        with self.assertRaises(OSError):
            self.post()
        self.assertIn(
            "suffix has exited",
            (self.config.journal_dir / "coordinator-exit.json").read_text(),
        )

    def test_staged_binding_mismatch_fails_before_native_start(self):
        changed = json.loads(self.binding.read_text())
        changed["http_port"] += 1
        self.binding.write_text(json.dumps(changed))
        value = self.open()
        with self.assertRaises((ValueError, EOFError)):
            value.start()
        self.assertFalse(
            any(place.runtime_dir.exists() for place in self.spec.locations)
        )
        self.assert_children_gone(value)

    def test_existing_journal_is_untouched(self):
        root = self.config.journal_dir
        root.mkdir()
        (root / "keep").write_text("original")
        value = self.open()
        with self.assertRaises(FileExistsError):
            value.start()
        value.close()
        self.assertEqual(list(root.iterdir()), [root / "keep"])

    def test_control_timeout_reaps_child_and_poisoned_channel_is_not_reused(self):
        root = self.config.journal_dir
        root.mkdir()
        route = self.config.routes[1]
        with mock.patch.dict(os.environ, STRATA_FIXTURE_MODE_b="hold"):
            client = AgentClient(route, self.spec, root)
        self.addCleanup(client.close)
        client.request("verify", 3)
        with self.assertRaises(TimeoutError):
            client.request("start-suffix", 0.2)
        with self.assertRaises(RuntimeError):
            client.request("status", 1)
        client.close()
        identity_file = self.spec.location("b").runtime_dir / "suffix-process.json"
        self.assertTrue(identity_file.exists())
        identity = install.ProcessIdentity.model_validate_json(
            identity_file.read_bytes()
        )
        self.assertFalse(
            procs.is_same(identity.pid, identity.start_token, identity.argv)
        )

    def test_explicit_route_validation_and_shell_quoting(self):
        remote = self.config.routes[1]
        args = remote.argv("a" * 64)
        self.assertIn("-oStrictHostKeyChecking=yes", args)
        words = shlex.split(args[-1])
        self.assertEqual(words[1], f"PYTHONPATH={remote.repository}")
        self.assertIn(str(remote.installation), words)
        raw = self.config.model_dump(mode="json")
        raw["routes"][1]["ssh_address"] = None
        raw["routes"][1]["ssh_host_alias"] = None
        with self.assertRaises(ValidationError):
            install.CoordinatorSpec.model_validate_json(json.dumps(raw))
        raw = remote.model_dump(mode="json")
        raw["ssh_host_alias"] = None
        with self.assertRaises(ValidationError):
            AgentRoute.model_validate_json(json.dumps(raw))

    def test_wrong_node_reply_poisoned_and_owned_agent_closed(self):
        self.config.journal_dir.mkdir()
        with mock.patch.dict(os.environ, STRATA_FIXTURE_CONTROL_MODE="wrong-node"):
            client = AgentClient(
                self.config.routes[0], self.spec, self.config.journal_dir
            )
        self.addCleanup(client.close)
        with self.assertRaisesRegex(ValueError, "different node/plan/epoch"):
            client.request("status", 3)
        client.close()
        outcome = AgentExit.model_validate_json(
            (self.config.journal_dir / "a.exit.json").read_bytes()
        )
        self.assertFalse(outcome.stop_confirmed)
        self.assertIn("different node/plan/epoch", outcome.error)

    def test_cleanup_failure_is_recorded_after_other_owned_resources_close(self):
        value = self.open()
        value.start()
        client = value.clients["b"]
        close = client.close

        def fail_after_cleanup():
            close()
            raise OSError("synthetic remote cleanup receipt failure")

        with (
            mock.patch.object(client, "close", side_effect=fail_after_cleanup),
            self.assertRaises(ExceptionGroup),
        ):
            value.close()
        self.assert_children_gone(value)
        outcome = install.CoordinatorExit.model_validate_json(
            (self.config.journal_dir / "coordinator-exit.json").read_bytes()
        )
        self.assertEqual(len(outcome.cleanup_errors), 1)
        self.assertIn("receipt failure", outcome.cleanup_errors[0])

    def test_exit_journal_failure_still_reaps_channel_and_closes_all_pipes(self):
        self.config.journal_dir.mkdir()
        client = AgentClient(self.config.routes[0], self.spec, self.config.journal_dir)
        self.addCleanup(client.close)
        client.request("status", 3)
        with (
            mock.patch.object(
                fabric_client.atomic,
                "write_new_text",
                side_effect=OSError("synthetic disk full"),
            ),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            client.close()
        self.assertIsNotNone(client.proc.poll())
        self.assertTrue(client.proc.stdout.closed)
        self.assertTrue(client.stderr.closed)
        self.assertTrue(client.frames.closed)

    def test_transient_identity_observation_waits_for_same_owned_process(self):
        self.config.journal_dir.mkdir()
        identify = procs.identify
        calls = []

        def observe(pid):
            calls.append(pid)
            return None if len(calls) == 1 else identify(pid)

        with mock.patch.object(fabric_client.procs, "identify", side_effect=observe):
            client = AgentClient(
                self.config.routes[0], self.spec, self.config.journal_dir
            )
        self.addCleanup(client.close)
        self.assertEqual(set(calls), {client.proc.pid})
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(fabric_client.lifeline.popen.call_count, 1)
        self.assertFalse(client.request("status", 3).inputs_staged)
        client.close()


if __name__ == "__main__":
    unittest.main()
