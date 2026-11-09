"""Check unknown resources, impossible readings and shared-filesystem accounting."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from tools.network_fabric_inventory import Gpu, Memory, filesystem_observations, observe


class InventoryTests(unittest.TestCase):
    """Inventory validity does not depend on having a model or GPU available."""

    def test_unknown_and_exhausted_memory_remain_distinct(self):
        unknown = Memory(total_mib=None, available_mib=None)
        exhausted = Memory(total_mib=1024, available_mib=0)
        self.assertIsNone(unknown.model_dump()["available_mib"])
        self.assertEqual(exhausted.available_mib, 0)
        for values in (
            {"total_mib": 0, "available_mib": 0},
            {"total_mib": 1, "available_mib": 2},
            {"total_mib": 1, "available_mib": True},
            {"total_mib": 1, "available_mib": -1},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Memory.model_validate(values)

    def test_gpu_observations_do_not_guess_free_memory(self):
        values = {
            "vendor": "amd",
            "index": 0,
            "name": "fixture",
            "vram_total_mib": 16,
            "vram_used_mib": None,
            "driver": None,
        }
        self.assertIsNone(Gpu.model_validate(values).vram_used_mib)
        for change in ({"vram_used_mib": 17}, {"index": False}, {"vendor": "unknown"}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                Gpu.model_validate(values | change)

    def test_paths_on_one_filesystem_do_not_multiply_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            file = root / "weights"
            file.write_bytes(b"abc")
            observations = filesystem_observations([root, file, root])
            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0].paths, sorted([str(root), str(file)]))
            self.assertLessEqual(
                observations[0].available_bytes, observations[0].total_bytes
            )
            with self.assertRaises(FileNotFoundError):
                filesystem_observations([root / "missing"])

    def test_missing_probes_stay_uncalibrated_and_binary_is_hashed(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "tools.network_fabric_inventory.tune_machine.memory",
                return_value={
                    "total_mib": None,
                    "available_mib": None,
                },
            ),
            patch(
                "tools.network_fabric_inventory.tune_machine.cpu",
                return_value={
                    "model": "fixture",
                    "logical": 2,
                    "physical": None,
                    "hybrid": False,
                },
            ),
            patch("tools.network_fabric_inventory.tune_machine.gpus", return_value=[]),
            patch(
                "tools.network_fabric_inventory.updater.detect_hardware",
                return_value={
                    "backend": None,
                    "gfx": [],
                    "cuda_archs": [],
                },
            ),
        ):
            root = Path(temporary)
            binary = root / "engine"
            binary.write_bytes(b"abc")
            receipt = observe(binary, [root])
            self.assertEqual(
                receipt.executable_sha256, hashlib.sha256(b"abc").hexdigest()
            )
            self.assertEqual(receipt.gpus, [])
            self.assertIsNone(receipt.cpu.physical)
            self.assertIsNone(receipt.memory.available_mib)
            self.assertEqual(
                receipt.planning_status,
                "UNCALIBRATED; no plan ranking or READY admission",
            )
            self.assertNotIn("measured_costs", receipt.model_dump())


if __name__ == "__main__":
    unittest.main()
