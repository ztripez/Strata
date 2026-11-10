"""Static manifest/bundle controls, with optional real native syntax preflight."""

import copy
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from serve.tokenizer_assets import ASSETS
from strata_control import fabric_bundle as bundle
from strata_control import fabric_plan as fabric
from strata_control.components import ComponentError


def pin(name, raw=b"synthetic artifact"):
    """Pin known synthetic bytes rather than claiming a model artifact identity."""
    return {"name": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def proposal():
    """Two synthetic nodes and the real static ownership geometry, without costs."""
    nodes = []
    inventories = {}
    for name, backend, vendor in (("a", "hip", "amd"), ("b", "cuda", "nvidia")):
        observed = {
            "node": name,
            "observed_unix_ns": 123,
            "memory": {"total_mib": 65536, "available_mib": 50000},
            "cpu": {
                "model": "synthetic CPU",
                "logical": 24,
                "physical": 12,
                "hybrid": False,
            },
            "gpus": [
                {
                    "vendor": vendor,
                    "index": 0,
                    "name": "synthetic GPU",
                    "vram_total_mib": 16000,
                    "vram_used_mib": 1000,
                    "driver": None,
                }
            ],
            "backend_probe": {},
            "filesystems": [],
            "executable": "/synthetic/layer",
            "executable_sha256": pin("layer")["sha256"],
        }
        raw = json.dumps(observed).encode()
        inventories[name] = raw
        nodes.append(
            {
                "id": name,
                "backend": backend,
                "source_sha256": "a" * 64,
                "layer_binary": pin("layer"),
                "worker_binary": pin("worker"),
                "inventory": pin(name + ".inventory.json", raw),
                "capability_evidence": pin("capability.json"),
                "budget": {"vram_mib": 9000, "ram_mib": 48000, "pinned_mib": 0},
                "requirement": {
                    "vram_mib": 6000,
                    "ram_mib": 10000,
                    "pinned_mib": 0,
                    "evidence": pin("resource.json"),
                },
            }
        )
    data = {
        "model": {
            "revision": "synthetic-revision",
            "model_sha256": "a" * 64,
            "pack_sha256": "b" * 64,
            "expert_model_sha256": "c" * 64,
            "mapped_files": [
                pin(name)
                for name in (
                    "main0.gguf",
                    "main1.gguf",
                    "main2.gguf",
                    "main3.gguf",
                    "engram.gguf",
                    "dspark.gguf",
                )
            ],
            "tokenizer_gguf": pin("tokenizer.gguf"),
            "pack_files": [
                pin(name) for name in ("ds4.txt", "tensors.txt", "engram.bin")
            ],
            "tokenizer": {
                "assets_sha256": {name: "a" * 64 for name in ASSETS},
                "source_sha256": "d" * 64,
            },
        },
        "workload": pin("workload.json"),
        "nodes": nodes,
        "prefix": {"node": "a", "cpu_threads": 8, "expert_vram_mib": 384},
        "suffix": {"node": "b", "cpu_threads": 8, "expert_vram_mib": 384},
        "cut": 20,
        "workers": [
            {
                "source": "prefix",
                "node": "b",
                "pairs": [{"layer": 0, "expert": 255}, {"layer": 0, "expert": 128}],
                "expert_vram_mib": 20,
                "resident_ram_mib": 20,
            },
            {
                "source": "suffix",
                "node": "a",
                "pairs": [{"layer": 20, "expert": 129}],
                "expert_vram_mib": 20,
                "resident_ram_mib": 20,
            },
        ],
        "deadlines": {
            "startup_s": 2700.0,
            "layer_rpc_ms": 600000,
            "expert_rpc_ms": 30000,
            "request_ms": 1800000,
            "idle_ms": 7200000,
        },
    }
    return data, inventories


def parse(data):
    """Use the actual strict JSON boundary that a saved manifest crosses."""
    return fabric.Plan.model_validate_json(json.dumps(data))


def installation(plan, root):
    """Supply explicit addresses and per-host locations outside logical identity."""
    return bundle.Installation.model_validate_json(
        json.dumps(
            {
                "document": {
                    "plan": plan.model_dump(mode="json"),
                    "plan_sha256": plan.identity(),
                },
                "epoch": 42,
                "layer_endpoint": {"address": "127.0.0.1", "port": 19192},
                "locations": [
                    {
                        "node": node,
                        "model_dir": str(root / node / "weights"),
                        "pack_dir": str(root / node / "pack"),
                        "tokenizer_dir": str(root / node / "tokenizer"),
                        "layer_binary": str(root / node / "layer"),
                        "worker_binary": str(root / node / "worker"),
                        "runtime_dir": str(root / node / "runtime"),
                        "expert_endpoint": {"address": "127.0.0.1", "port": port + 1},
                    }
                    for node, port in (("a", 19190), ("b", 19192))
                ],
            }
        )
    )


class PlanTests(unittest.TestCase):
    """Reject contradictions without claiming hardware or quality admission."""

    def setUp(self):
        self.data, self.inventories = proposal()
        self.plan = parse(self.data)

    def test_identity_ignores_only_unordered_enumeration(self):
        changed = copy.deepcopy(self.data)
        changed["nodes"].reverse()
        changed["workers"].reverse()
        for worker in changed["workers"]:
            worker["pairs"].reverse()
        changed["model"]["pack_files"].reverse()
        self.assertEqual(parse(changed).identity(), self.plan.identity())
        changed["model"]["mapped_files"].reverse()
        self.assertNotEqual(parse(changed).identity(), self.plan.identity())

    def test_changed_bound_content_cannot_keep_plan_id(self):
        for key, value in (("workload", pin("other-workload")), ("cut", 24)):
            changed = copy.deepcopy(self.data)
            changed[key] = value
            # Layer 20 must also move ownership when the cut moves.
            if key == "cut":
                changed["workers"][1]["pairs"][0]["layer"] = 24
            plan = parse(changed)
            self.assertNotEqual(plan.identity(), self.plan.identity())
            with self.assertRaises(ValidationError):
                fabric.PlanDocument(plan=plan, plan_sha256=self.plan.identity())

    def test_invalid_ownership_and_declared_budgets(self):
        mutations = [
            lambda d: d["suffix"].update(node="a"),
            lambda d: d["workers"][0].update(node="a"),
            lambda d: d["workers"][0]["pairs"][0].update(layer=20),
            lambda d: d["workers"][0]["pairs"].append(d["workers"][0]["pairs"][0]),
            lambda d: d["workers"][0]["pairs"][0].update(expert=384),
            lambda d: d["nodes"][1].update(worker_binary=None),
            lambda d: d["nodes"][0]["requirement"].update(vram_mib=100),
            lambda d: d["nodes"][0]["budget"].update(vram_mib=100),
            lambda d: d["nodes"][0]["requirement"].update(vram_mib=0),
            lambda d: d["nodes"][0]["requirement"].update(pinned_mib=10001),
            lambda d: d["deadlines"].update(request_ms=1),
            lambda d: d.update(selection="measured"),
            lambda d: d.update(layer_protocol=True),
            lambda d: d.update(cut=20.0),
            lambda d: d["model"].update(effective_experts=6.0),
            lambda d: d["model"]["mapped_files"].pop(),
            lambda d: d["model"]["pack_files"].pop(),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(self.data)
            mutate(changed)
            with self.subTest(data=changed), self.assertRaises(ValidationError):
                parse(changed)

    def test_verified_snapshot_and_unknown_or_insufficient_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inventory.json"
            node = self.plan.nodes[0]
            path.write_bytes(self.inventories[node.id])
            self.assertEqual(fabric.check_inventory(node, path).node, node.id)
            for change in ("unknown", "exhausted", "wrong_binary", "wrong_node"):
                value = json.loads(self.inventories[node.id])
                if change == "unknown":
                    value["gpus"][0]["vram_used_mib"] = None
                if change == "exhausted":
                    value["memory"]["available_mib"] = 0
                if change == "wrong_binary":
                    value["executable_sha256"] = "f" * 64
                if change == "wrong_node":
                    value["node"] = "other"
                raw = json.dumps(value).encode()
                path.write_bytes(raw)
                changed = fabric.Node.model_validate_json(
                    json.dumps(
                        {
                            **node.model_dump(mode="json"),
                            "inventory": pin("inventory.json", raw),
                        }
                    )
                )
                with self.subTest(change=change), self.assertRaises(ValueError):
                    fabric.check_inventory(changed, path)
            path.write_bytes(b"changed buffer")
            with self.assertRaises(ComponentError):
                fabric.check_inventory(node, path)


class BundleTests(unittest.TestCase):
    """Materialization produces inspectable native inputs without starting them."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        data, inventories = proposal()
        self.spec = installation(parse(data), self.root)
        self.evidence = self.root / "evidence"
        for node in self.spec.document.plan.nodes:
            for kind, item in (
                ("inventory", node.inventory),
                ("capability", node.capability_evidence),
                ("resource", node.requirement.evidence),
            ):
                path = self.evidence / node.id / kind / item.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(
                    inventories[node.id]
                    if kind == "inventory"
                    else b"synthetic artifact"
                )

    def test_epoch_and_locator_changes_keep_logical_identity(self):
        changed = self.spec.model_dump(mode="json")
        changed["epoch"] = 43
        changed["locations"][0]["pack_dir"] = "/new/pack"
        other = bundle.Installation.model_validate_json(json.dumps(changed))
        self.assertEqual(other.document.plan_sha256, self.spec.document.plan_sha256)
        self.assertEqual(other.http_launch().expected.plan_epoch, 43)
        self.assertIn("/new/pack", other.layer_arguments("prefix"))

    def test_complete_bundle_and_existing_destination_preserved(self):
        output = self.root / "bundle"
        bundle.materialize(self.spec, output, self.evidence)
        summary = json.loads((output / "bundle.json").read_bytes())
        self.assertIn("PREPARED ONLY", summary["status"])
        for name, expected in summary["sha256"].items():
            self.assertEqual(
                hashlib.sha256((output / name).read_bytes()).hexdigest(), expected
            )
        for node in ("a", "b"):
            for kind in ("layer", "worker"):
                command = bundle.ProcessLaunch.model_validate_json(
                    (output / node / (kind + "-command.json")).read_bytes()
                )
                self.assertEqual(command.node, node)
                self.assertEqual(command.environment["DS4_IQ_Q8K"], "1")
        self.assertEqual((output / "a/remote.plan").read_text(), "0 128\n0 255\n")
        self.assertEqual((output / "b/remote.plan").read_text(), "20 129\n")
        saved = (output / "installation.json").read_bytes()
        with self.assertRaises(FileExistsError):
            bundle.materialize(self.spec, output, self.evidence)
        self.assertEqual((output / "installation.json").read_bytes(), saved)

    def test_nested_mutation_cannot_bypass_sealed_plan_at_materialization(self):
        self.spec.document.plan.model.tokenizer.assets_sha256["vocab.json"] = "f" * 64
        output = self.root / "refused"
        with self.assertRaises(ValidationError):
            bundle.materialize(self.spec, output, self.evidence)
        self.assertFalse(output.exists())

    def test_materialization_never_starts_any_declared_process(self):
        with mock.patch(
            "subprocess.Popen", side_effect=AssertionError("must not start")
        ):
            bundle.materialize(self.spec, self.root / "prepared-only", self.evidence)

    def test_layer_only_bundle_requires_no_worker_locations(self):
        data = proposal()[0]
        data["workers"] = []
        for node in data["nodes"]:
            node["worker_binary"] = None
        spec = installation(parse(data), self.root).model_dump(mode="json")
        for place in spec["locations"]:
            place["expert_endpoint"] = None
            place["worker_binary"] = None
        spec = bundle.Installation.model_validate_json(json.dumps(spec))
        output = self.root / "layer-only"
        bundle.materialize(spec, output, self.evidence)
        self.assertFalse(list(output.glob("*/worker-command.json")))
        self.assertFalse(list(output.glob("*/remote.plan")))
        for role in ("prefix", "suffix"):
            self.assertNotIn("--expert-worker", spec.layer_arguments(role))

    def test_changed_evidence_refused_before_creating_output(self):
        node = self.spec.document.plan.nodes[0]
        path = self.evidence / node.id / "resource" / node.requirement.evidence.name
        path.write_bytes(b"changed resource evidence")
        output = self.root / "refused"
        with self.assertRaises(ComponentError):
            bundle.materialize(self.spec, output, self.evidence)
        self.assertFalse(output.exists())

    def test_invalid_bindings_refused(self):
        changes = [
            lambda d: d.update(epoch=0),
            lambda d: d.update(epoch=2**64),
            lambda d: d["locations"][0].update(model_dir="relative"),
            lambda d: d["locations"][0].update(worker_binary=None),
            lambda d: d["locations"][0].update(node="b"),
            lambda d: d["layer_endpoint"].update(address="0.0.0.0"),
            lambda d: d["locations"][0].update(expert_endpoint=None),
            lambda d: d.update(http_port=d["locations"][0]["expert_endpoint"]["port"]),
            lambda d: d["locations"][0].update(
                expert_endpoint=d["locations"][1]["expert_endpoint"]
            ),
        ]
        for change in changes:
            data = self.spec.model_dump(mode="json")
            change(data)
            with self.subTest(data=data), self.assertRaises(ValidationError):
                bundle.Installation.model_validate_json(json.dumps(data))

    @unittest.skipUnless(
        os.environ.get("STRATA_TEST_LAYER_BINARY"), "native syntax binary not supplied"
    )
    def test_generated_argv_passes_real_native_syntax_only(self):
        for role in ("prefix", "suffix"):
            result = subprocess.run(
                [
                    os.environ["STRATA_TEST_LAYER_BINARY"],
                    "--serve",
                    *self.spec.layer_arguments(role),
                    "--check-config",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                result.stdout.strip(),
                "CONFIG_CHECKED syntax_only; model/device/quality NOT VERIFIED",
            )


if __name__ == "__main__":
    unittest.main()
