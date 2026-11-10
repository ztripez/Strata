"""HTTP/stdio adapter tests with a synthetic subprocess, not native model math."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from serve import deepseek_layer as fabric
from serve import frontend, server
from serve.test_dsv41 import TEMPLATE, mini_tokenizer
from serve.tokenizer_assets import ASSETS, TokenizerPins, load_deepseek
from tools.strata_tokenizer import Tokenizer


class LayerHttpTests(unittest.TestCase):
    """Verify actual pipes/HTTP and refusals without using a GPU or model file."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tok = mini_tokenizer()
        self.assets = self.root / "tokenizer"
        self.assets.mkdir()
        (self.assets / "chat_template.jinja").write_bytes(TEMPLATE.read_bytes())
        self.addCleanup(frontend.use_family, None)
        self.saved_timeout = server.ENGINE_READY_S
        self.addCleanup(setattr, server, "ENGINE_READY_S", self.saved_timeout)
        self.environment = mock.patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        # patch.dict restores changes made inside the scope, including runtime policy.
        self.executable = self.root / "synthetic-prefix"
        self.commands = self.root / "commands.jsonl"
        self.pid = self.root / "prefix.pid"
        self.expected = fabric.PrefixReceipt(
            arch="deepseek41",
            fabric=1,
            role="prefix",
            context=2048,
            effective_experts=6,
            arithmetic=2,
            model_sha256="0" + "1" * 63,
            pack_sha256="b" * 64,
            plan_sha256="c" * 64,
            binary_sha256="d" * 64,
            cpu_threads_limit=8,
            expert_vram_mib=384,
            request_timeout_ms=30000,
            rpc_timeout_ms=1000,
            idle_timeout_ms=0,
        )

    def binding(self, wrong_plan=False, delay_ready=False, fail_start=False):
        """Create a pinned synthetic stdio peer with known token output and INFO."""
        info = self.expected.model_dump()
        info.pop("binary_sha256")
        if wrong_plan:
            info["plan_sha256"] = "e" * 64
        response = self.tok.encode("OK", parse_special=True) + [1]
        source = f"""#!/usr/bin/env python3
import hashlib,json,os,sys,time
from pathlib import Path
if '--check-config' in sys.argv:
 print('CONFIG_CHECKED syntax_only; model/device/quality NOT VERIFIED');sys.exit(0)
Path({str(self.pid)!r}).write_text(str(os.getpid()))
if {fail_start!r}:sys.exit(42)
if {delay_ready!r}:time.sleep(60)
info={info!r}
info['binary_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
print('INFO '+' '.join(k+'='+str(v) for k,v in info.items()),flush=True)
print('READY 2048 stop',flush=True)
for line in sys.stdin:
 if line.strip()=='QUIT':break
 if not line.startswith('GEN '):continue
 with Path({str(self.commands)!r}).open('a') as f:f.write(json.dumps(line.strip())+'\\n')
 count=int(line.split()[1]);ids=line.split()[-1].split(',')
 output={response!r}[:count]
 for token in output:print('T '+str(token),flush=True)
 print('DONE '+str(len(output))+' '+str(len(ids))+' 1.0 2.0 stop 0 0 0 0 0',flush=True)
"""
        self.executable.write_text(source)
        self.executable.chmod(0o755)
        expected = fabric.PrefixReceipt.model_validate(
            {
                **self.expected.model_dump(),
                "binary_sha256": hashlib.sha256(
                    self.executable.read_bytes()
                ).hexdigest(),
            }
        )
        pins = TokenizerPins(
            assets_sha256={
                name: hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
                if name == "chat_template.jinja"
                else "a" * 64
                for name in ASSETS
            },
            source_sha256="a" * 64,
        )
        return fabric.Launch(
            executable=self.executable,
            native_args=[
                "--role",
                "prefix",
                "--pack",
                "/synthetic-only",
                "--cut",
                "20",
                "--endpoint",
                "127.0.0.1:19190",
                "--model-sha256",
                expected.model_sha256,
                "--pack-sha256",
                expected.pack_sha256,
                "--plan-id",
                expected.plan_sha256,
                "--peer-binary-sha256",
                "d" * 64,
                "--experts",
                "6",
                "--expert-vram-mib",
                "384",
                "--cpu-threads",
                "8",
                "--rpc-timeout-ms",
                "1000",
                "--request-timeout-ms",
                "30000",
            ],
            expected=expected,
            tokenizer_dir=self.assets,
            tokenizer=pins,
            engine_log=self.root / "engine.log",
            engine_ready_s=3.0,
            port=0,
        )

    def service(self, spec=None):
        """Use real StrataEngine pipes, mocking only separately tested asset loading."""
        spec = spec or self.binding()
        with mock.patch.object(fabric, "load_deepseek", return_value=self.tok):
            svc = fabric.create_service(spec)
        self.addCleanup(svc.engine.close)
        return svc

    def test_http_response_and_greedy_defaults_without_restore_keys(self):
        svc = self.service()
        httpd = server.serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        body = {
            "model": svc.model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "logprobs": False,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.load(response)
        self.assertEqual(result["choices"][0]["message"]["content"], "OK")
        command = json.loads(self.commands.read_text().splitlines()[0])
        for unsupported in ("temperature=1", "marks=", "tail=", "shared=", "pin="):
            self.assertNotIn(unsupported, command)
        self.assertEqual(svc.engine.info["model_sha256"], "0" + "1" * 63)
        self.assertEqual(svc.engine.info["context"], 2048)
        body["stream"] = True
        req = urllib.request.Request(
            req.full_url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            events = response.read().decode().splitlines()
        self.assertIn("data: [DONE]", events)
        chunks = [
            json.loads(line[6:])
            for line in events
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        self.assertEqual(
            "".join(
                chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
            ),
            "OK",
        )

    def test_stream_transport_repeats_cleanly_and_validates_before_native_write(self):
        svc = self.service()
        for _ in range(2):
            self.assertEqual(
                list(
                    svc.engine.generate(
                        [42], 16, {"temperature": 0.0}, threading.Event()
                    )
                ),
                self.tok.encode("OK", parse_special=True) + [1],
            )
        before = self.commands.read_bytes()
        for sampling in (
            {"temperature": 0.1},
            {"top_p": 0.9},
            {"logprobs": True},
            {"strata_prefix": {"tokens": 1}},
        ):
            with self.subTest(sampling=sampling), self.assertRaises(ValueError):
                list(svc.engine.generate([42], 16, sampling, threading.Event()))
        with self.assertRaises(ValidationError):
            list(svc.engine.generate([42] * 1900, 149, {}, threading.Event()))
        self.assertEqual(self.commands.read_bytes(), before)

    def test_wrong_image_refused_before_tokenizer_or_process(self):
        spec = self.binding()
        changed = fabric.Launch.model_validate(
            {
                **spec.model_dump(),
                "expected": {**spec.expected.model_dump(), "binary_sha256": "f" * 64},
            }
        )
        with (
            mock.patch.object(fabric, "load_deepseek") as loader,
            self.assertRaisesRegex(ValueError, "pinned binary"),
        ):
            fabric.create_service(changed)
        loader.assert_not_called()
        self.assertFalse(self.pid.exists())

    def test_wrong_ready_closes_only_the_owned_prefix(self):
        spec = self.binding(wrong_plan=True)
        with (
            mock.patch.object(fabric, "load_deepseek", return_value=self.tok),
            self.assertRaisesRegex(ValueError, "differ from pinned launch"),
        ):
            fabric.create_service(spec)
        self.assertFalse((Path("/proc") / self.pid.read_text()).exists())

    def test_failed_start_never_uses_shared_transient_retry(self):
        spec = self.binding(fail_start=True)
        with (
            mock.patch.object(fabric, "load_deepseek", return_value=self.tok),
            mock.patch.object(
                server, "transient_start_failure", return_value=True
            ) as retry,
            self.assertRaises(server.EngineDied),
        ):
            fabric.create_service(spec)
        retry.assert_not_called()
        self.assertFalse((Path("/proc") / self.pid.read_text()).exists())
        self.assertIn("engine started:", spec.engine_log.read_text())

    def test_existing_log_preserved_and_dead_engine_never_replayed(self):
        svc = self.service()
        svc.engine.close()
        with self.assertRaisesRegex(server.EngineDied, "explicitly reinstall"):
            svc.ensure_loaded()
        self.assertFalse(self.commands.exists())
        spec = self.binding()
        original = spec.engine_log.read_bytes()
        with (
            mock.patch.object(fabric, "load_deepseek", return_value=self.tok),
            self.assertRaises(FileExistsError),
        ):
            fabric.create_service(spec)
        self.assertEqual(spec.engine_log.read_bytes(), original)

    def test_sigterm_during_startup_and_after_ready_reaps_owned_child(self):
        for delay_ready in (True, False):
            with self.subTest(delay_ready=delay_ready):
                spec = self.binding(delay_ready=delay_ready)
                # Distinct preserved logs and PID receipts for the two launches.
                spec = fabric.Launch.model_validate(
                    {
                        **spec.model_dump(),
                        "engine_log": self.root / f"engine-{delay_ready}.log",
                    }
                )
                self.pid.unlink(missing_ok=True)
                config = self.root / "launch.json"
                config.write_text(spec.model_dump_json())
                output = self.root / f"launcher-{delay_ready}.log"
                with output.open("w") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            (
                                "from unittest.mock import patch; "
                                "from serve.test_dsv41 import mini_tokenizer; "
                                "from serve import deepseek_layer as fabric; "
                                "fabric.server.ENGINE_QUIT_WAIT_S=0.1; "
                                "patch.object(fabric, 'load_deepseek', return_value=mini_tokenizer()).start(); "
                                "fabric.main()"
                            ),
                            str(config),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                    )
                    try:
                        deadline = time.monotonic() + 8
                        while not self.pid.exists() or (
                            not delay_ready
                            and "Fabric HTTP listening" not in output.read_text()
                        ):
                            if (
                                process.poll() is not None
                                or time.monotonic() >= deadline
                            ):
                                self.fail(
                                    f"synthetic launcher did not start: {output.read_text()}"
                                )
                            time.sleep(0.02)
                        native_pid = int(self.pid.read_text())
                        process.send_signal(signal.SIGTERM)
                        self.assertEqual(
                            process.wait(timeout=10), 0, output.read_text()
                        )
                        self.assertFalse((Path("/proc") / str(native_pid)).exists())
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait(timeout=5)

    def test_launch_and_receipt_boundaries_are_strict(self):
        spec = self.binding()
        for change in (
            {"unexpected": True},
            {"executable": Path("relative")},
            {"native_args": ["--role", "suffix"]},
            {"native_args": ["--role", "prefix", "--serve"]},
        ):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                fabric.Launch.model_validate({**spec.model_dump(), **change})
        for change in (
            {"fabric": True},
            {"context": 4096},
            {"effective_experts": 5},
            {"request_timeout_ms": 999},
        ):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                fabric.PrefixReceipt.model_validate(
                    {**spec.expected.model_dump(), **change}
                )


class AssetTests(unittest.TestCase):
    """Verify buffer identity and tokenizer shape using synthetic exported assets."""

    def test_complete_pins_shape_and_changed_source_or_asset_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "vocab.json").write_text(
                json.dumps({f"T{i}": i for i in range(129280)})
            )
            (root / "token_type.json").write_text(json.dumps([1] * 129280))
            (root / "tokenizer.json").write_text(
                json.dumps({"pre": "deepseek3", "special_ids": {"bos": 0, "eos": 1}})
            )
            (root / "merges.txt").write_text("T2 T3")
            (root / "chat_template.jinja").write_text("synthetic template")
            import sys

            source = Path(sys.modules[Tokenizer.__module__].__file__)
            pins = TokenizerPins(
                assets_sha256={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in root.iterdir()
                },
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            tokenizer = load_deepseek(root, pins)
            self.assertEqual(len(tokenizer.tokens), 129280)
            with self.assertRaisesRegex(ValueError, "pinned source"):
                load_deepseek(
                    root,
                    TokenizerPins.model_validate(
                        {**pins.model_dump(), "source_sha256": "0" * 64}
                    ),
                )
            (root / "merges.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "pinned assets"):
                load_deepseek(root, pins)
            with self.assertRaises(ValidationError):
                TokenizerPins(
                    assets_sha256={"vocab.json": "0" * 64}, source_sha256="0" * 64
                )


if __name__ == "__main__":
    unittest.main()
