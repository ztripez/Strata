"""Synthetic calibration fixtures are not native model or measured speed proof."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from tools import network_fabric_calibration as calibration
from tools.network_fabric_inventory import Cpu, Inventory, Memory


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class CalibrationTests(unittest.TestCase):
    """Pair actual identities before computing any residual or percentile."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paths = {
            name: self.root / name
            for name in (
                "binding",
                "prefix_log",
                "suffix_log",
                "prefix_inventory",
                "suffix_inventory",
                "workload",
            )
        }
        self.paths["workload"].write_bytes(b"explicit synthetic control inputs")
        participants = {}
        for role, binary in [("prefix", "d" * 64), ("suffix", "e" * 64)]:
            inventory = Inventory(
                node=role,
                observed_unix_ns=1,
                memory=Memory(total_mib=None, available_mib=None),
                cpu=Cpu(model="synthetic", logical=2, physical=None, hybrid=False),
                gpus=[],
                backend_probe={},
                filesystems=[],
                executable="/synthetic",
                executable_sha256=binary,
            )
            raw = (inventory.model_dump_json() + "\n").encode()
            self.paths[role + "_inventory"].write_bytes(raw)
            participants[role] = {
                "node": role,
                "inventory_sha256": digest(raw),
                "binary_sha256": binary,
                "cpu_threads_limit": 8,
                "expert_vram_mib": 384,
            }
        self.binding = {
            "schema_version": "strata-window-calibration-binding/1",
            "model_sha256": "a" * 64,
            "pack_sha256": "b" * 64,
            "plan_sha256": "c" * 64,
            "epoch": 1,
            "incarnation": 1234,
            "cut": 20,
            "workload_sha256": digest(self.paths["workload"].read_bytes()),
            "expected_windows": 3,
            **participants,
        }
        self.paths["binding"].write_text(json.dumps(self.binding))
        self.prefix = []
        self.suffix = []
        for index, (rpc, native) in enumerate([(101, 100), (200, 1), (1000, 999)], 1):
            common = {
                key: self.binding[key]
                for key in (
                    "model_sha256",
                    "pack_sha256",
                    "plan_sha256",
                    "epoch",
                    "incarnation",
                    "cut",
                )
            }
            common.update(
                {
                    "schema": "strata-native-window-cost/1",
                    "session": 2,
                    "sequence": index,
                    "position": index - 1,
                    "rows": 1,
                    "cpu_threads_limit": 8,
                    "expert_vram_mib": 384,
                }
            )
            self.prefix.append(
                {
                    **common,
                    "role": "prefix",
                    "binary_sha256": "d" * 64,
                    "feed_us": rpc + 8,
                    "before_rpc_us": 5,
                    "rpc_us": rpc,
                    "after_rpc_us": 3,
                    "transaction": {
                        "total_us": rpc,
                        "execute_us": rpc - 1,
                        "ack_us": 1,
                        "tx_bytes": 123,
                        "rx_bytes": 456,
                    },
                }
            )
            self.suffix.append(
                {
                    **common,
                    "role": "suffix",
                    "binary_sha256": "e" * 64,
                    "native_us": native,
                }
            )
        self.write_logs()

    def write_logs(self):
        for role, values in [("prefix", self.prefix), ("suffix", self.suffix)]:
            self.paths[role + "_log"].write_text(
                "unrelated startup log\n"
                + "".join("FABRIC_WINDOW " + json.dumps(v) + "\n" for v in values)
            )

    def collect(self):
        return calibration.collect(
            **{
                ("binding_file" if key == "binding" else key): path
                for key, path in self.paths.items()
            }
        )

    def test_paired_residual_then_percentile_not_difference_of_percentiles(self):
        result = self.collect()
        self.assertEqual([p.rpc_unattributed_us for p in result.windows], [1, 199, 1])
        self.assertEqual(result.by_rows[0].rpc_unattributed_us.median, 1.0)
        self.assertEqual(result.by_rows[0].rpc_us.median, 200.0)
        self.assertEqual(result.by_rows[0].suffix_native_us.median, 100.0)
        self.assertEqual(result.by_rows[0].feed_us.mean, (109 + 208 + 1008) / 3)
        self.assertEqual(result.by_rows[0].feed_us.count, 3)
        self.assertIn("no plan ranking", result.status)
        self.assertIsNone(result.prefix_inventory.memory.available_mib)
        calibration.Observations.model_validate_json(
            result.model_dump_json(by_alias=True)
        )

    def test_binding_profile_and_incarnation_mismatch_refused(self):
        for field, value in [
            ("model_sha256", "f" * 64),
            ("plan_sha256", "f" * 64),
            ("binary_sha256", "f" * 64),
            ("cut", 24),
            ("incarnation", 1235),
            ("epoch", 2),
            ("cpu_threads_limit", 4),
            ("expert_vram_mib", 0),
        ]:
            with self.subTest(field=field):
                original = self.suffix[0][field]
                self.suffix[0][field] = value
                self.write_logs()
                with self.assertRaises(ValueError):
                    self.collect()
                self.suffix[0][field] = original

    def test_counts_duplicates_gaps_and_orphan_execution_refused(self):
        original = self.suffix.copy()
        for values in [
            original[:-1],
            original + [original[-1]],
            [original[0], original[0], original[2]],
            [original[0], original[2], original[1]],
        ]:
            with self.subTest(values=values):
                self.suffix = values
                self.write_logs()
                with self.assertRaises(ValueError):
                    self.collect()
        self.suffix = original
        self.suffix[2] = {**original[2], "session": 3, "sequence": 1, "position": 0}
        self.write_logs()
        with self.assertRaisesRegex(ValueError, "unpaired"):
            self.collect()

    def test_truncation_noninteger_negative_and_unexplained_fields_refused(self):
        for change in [
            {"rows": True},
            {"native_us": -1},
            {"native_us": 1.5},
            {"native_us": float("inf")},
            {"unknown": 1},
        ]:
            with self.subTest(change=change):
                old = self.suffix[0]
                self.suffix[0] = {**old, **change}
                self.write_logs()
                with self.assertRaises(ValueError):
                    self.collect()
                self.suffix[0] = old
        self.write_logs()
        raw = self.paths["suffix_log"].read_bytes()
        self.paths["suffix_log"].write_bytes(raw[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.collect()

    def test_impossible_spans_are_not_clamped_or_double_counted(self):
        for change in [
            {"feed_us": 999},
            {"transaction": {**self.prefix[0]["transaction"], "total_us": 999}},
            {"transaction": {**self.prefix[0]["transaction"], "ack_us": 999}},
        ]:
            with self.subTest(change=change):
                old = self.prefix[0]
                self.prefix[0] = {**old, **change}
                self.write_logs()
                with self.assertRaises(ValidationError):
                    self.collect()
                self.prefix[0] = old
        self.suffix[0]["native_us"] = self.prefix[0]["rpc_us"] + 1
        self.write_logs()
        with self.assertRaisesRegex(ValueError, "inspect clocks"):
            self.collect()

    def test_inventory_and_workload_bytes_are_verified(self):
        for name in ("prefix_inventory", "suffix_inventory", "workload"):
            with self.subTest(name=name):
                raw = self.paths[name].read_bytes()
                self.paths[name].write_bytes(raw + b" ")
                with self.assertRaises(ValueError):
                    self.collect()
                self.paths[name].write_bytes(raw)

    def test_session_reordering_and_exceeded_context_refused(self):
        self.prefix[0]["position"] = 2048
        self.write_logs()
        with self.assertRaises(ValidationError):
            self.collect()
        self.prefix[0]["position"] = 0
        self.prefix[0]["session"] = 3
        self.write_logs()
        with self.assertRaisesRegex(ValueError, "interleaved"):
            self.collect()

    def test_existing_output_is_never_replaced(self):
        output = self.root / "output"
        output.write_text("original evidence")
        argv = [
            "calibration",
            *[
                item
                for key, path in self.paths.items()
                for item in ["--" + key.replace("_", "-"), str(path)]
            ],
            "--output",
            str(output),
        ]
        with mock.patch("sys.argv", argv), self.assertRaises(FileExistsError):
            calibration.main()
        self.assertEqual(output.read_text(), "original evidence")

    def test_serialized_observation_cannot_change_summary_count_or_binding(self):
        original = self.collect().model_dump(mode="json", by_alias=True)
        for change in ("summary", "count", "binding", "source", "status"):
            with self.subTest(change=change):
                value = json.loads(json.dumps(original))
                if change == "summary":
                    value["by_rows"][0]["feed_us"]["mean"] = 0.0
                if change == "count":
                    value["windows"].pop()
                if change == "binding":
                    value["binding"]["prefix"]["cpu_threads_limit"] = 4
                if change == "source":
                    value["source_sha256"].pop("suffix_log")
                if change == "status":
                    value["status"] = "E6 passed"
                with self.assertRaises(ValueError):
                    calibration.Observations.model_validate(value)

    def test_remote_time_must_fit_execute_not_just_the_enclosing_rpc(self):
        self.suffix[0]["native_us"] = self.prefix[0]["rpc_us"]
        self.write_logs()
        with self.assertRaisesRegex(ValueError, "EXECUTE"):
            self.collect()

    def test_cli_writes_the_validated_new_receipt(self):
        output = self.root / "new-observations.json"
        argv = [
            "calibration",
            *[
                item
                for key, path in self.paths.items()
                for item in ["--" + key.replace("_", "-"), str(path)]
            ],
            "--output",
            str(output),
        ]
        with mock.patch("sys.argv", argv), mock.patch("builtins.print"):
            calibration.main()
        report = calibration.Observations.model_validate_json(output.read_bytes())
        self.assertEqual(len(report.windows), 3)
        self.assertEqual(report.by_rows[0].rpc_unattributed_us.median, 1.0)


if __name__ == "__main__":
    unittest.main()
