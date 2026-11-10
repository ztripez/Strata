"""Serve a pinned experimental native prefix through Strata's existing HTTP API.

Run in a dedicated process with uv, pydantic, regex and jinja2. This launches one
already specified prefix; it does not discover nodes, install a plan or start its
remote suffix/expert workers. Model quality and E6 installation remain separate.
"""

import argparse
import hashlib
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from serve import server
from serve.frontend import ChatTemplate
from serve.tokenizer_assets import Digest, Record, TokenizerPins, load_deepseek


class PrefixReceipt(Record):
    """Actual native prefix identity and resolved limits required before HTTP."""

    arch: Literal["deepseek41"]
    fabric: Annotated[int, Field(ge=1, le=1)]
    role: Literal["prefix"]
    context: Annotated[int, Field(ge=2048, le=2048)]
    effective_experts: Annotated[int, Field(ge=6, le=6)]
    arithmetic: Annotated[int, Field(ge=2, le=2)]
    model_sha256: Digest
    pack_sha256: Digest
    plan_sha256: Digest
    binary_sha256: Digest
    cpu_threads_limit: Annotated[int, Field(ge=1, le=64)]
    expert_vram_mib: Annotated[int, Field(ge=0, le=65536)]
    request_timeout_ms: Annotated[int, Field(ge=1, le=7200000)]
    rpc_timeout_ms: Annotated[int, Field(ge=1, le=600000)]
    idle_timeout_ms: Annotated[int, Field(ge=0, le=0)]

    @model_validator(mode="after")
    def matching_deadlines(self):
        """The request allowance must cover the configured layer transaction."""
        if self.request_timeout_ms < self.rpc_timeout_ms:
            raise ValueError("request timeout does not cover the layer RPC timeout")
        return self


class Launch(Record):
    """Static prefix launch binding; this is not a calibrated cluster plan."""

    schema_version: Literal["strata-layer-http-launch/1"] = "strata-layer-http-launch/1"
    executable: Path
    native_args: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=4096)]],
        Field(min_length=2, max_length=128),
    ]
    expected: PrefixReceipt
    tokenizer_dir: Path
    tokenizer: TokenizerPins
    engine_log: Path
    engine_ready_s: Annotated[float, Field(gt=0, le=7200, allow_inf_nan=False)]
    port: Annotated[int, Field(ge=0, le=65535)] = 8098
    max_queue: Annotated[int, Field(ge=0, le=32)] = 0
    queue_timeout_s: Annotated[float, Field(gt=0, le=3600, allow_inf_nan=False)] = 30.0
    model_name: Annotated[str, Field(min_length=1, max_length=128)] = (
        "deepseek-v4.1-flash-fabric"
    )

    @field_validator("executable", "tokenizer_dir", "engine_log")
    @classmethod
    def absolute_path(cls, value: Path) -> Path:
        """Launch locations must not depend on the caller's working directory."""
        if not value.is_absolute():
            raise ValueError("launch paths must be absolute")
        return value

    @field_validator("native_args")
    @classmethod
    def prefix_arguments(cls, value: list[str]) -> list[str]:
        """The native parser owns other flags; this launcher owns the prefix role."""
        if value[:2] != ["--role", "prefix"]:
            raise ValueError("native_args must begin with --role prefix")
        if any(flag in value for flag in ("--serve", "--check-config")):
            raise ValueError(
                "native_args must omit injected --serve and check-only mode"
            )
        return value


class Greedy(Record):
    """Sampling controls representable by the qualified greedy window path."""

    temperature: Annotated[float, Field(le=0, allow_inf_nan=False)] = 0.0
    top_p: Annotated[float, Field(ge=1, le=1, allow_inf_nan=False)] = 1.0
    top_k: Annotated[int, Field(ge=0, le=64)] = 64
    min_p: Annotated[float, Field(ge=0, le=0, allow_inf_nan=False)] = 0.0
    repetition_penalty: Annotated[float, Field(ge=1, le=1, allow_inf_nan=False)] = 1.0
    frequency_penalty: Annotated[float, Field(ge=0, le=0, allow_inf_nan=False)] = 0.0
    presence_penalty: Annotated[float, Field(ge=0, le=0, allow_inf_nan=False)] = 0.0
    penalty_last_n: Annotated[int, Field(ge=0, le=0)] = 0
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)] = 1

    @field_validator("top_k")
    @classmethod
    def neutral_top_k(cls, value: int) -> int:
        """Only the native protocol's two neutral spellings are admitted."""
        if value not in (0, 64):
            raise ValueError("fabric greedy profile requires top_k 0 or 64")
        return value


class Generation(Record):
    """Native request bounds checked before writing GEN to a live engine."""

    ids: Annotated[
        list[Annotated[int, Field(ge=0, lt=129280)]],
        Field(min_length=1, max_length=1900),
    ]
    maximum: Annotated[int, Field(ge=1, le=1024)]

    @model_validator(mode="after")
    def context_fits(self):
        """Both the prompt and complete requested continuation must fit 2K."""
        if len(self.ids) + self.maximum > 2048:
            raise ValueError("fabric request exceeds the 2048-token context")
        return self


class PinnedLayerEngine(server.StrataEngine):
    """Existing stdio engine with explicit fabric admission and no automatic reload."""

    def __init__(self, *args, **kwargs):
        """Start once and release this child even if admission is interrupted."""
        self.proc, self.pump, self.log = None, None, None
        try:
            # The shared engine normally retries transient starts. A fabric
            # launch must fail with its original receipt and explicit ownership.
            super().__init__(*args, **kwargs, _attempt=server.START_RETRIES)
        except BaseException:
            self.close()
            if self.log not in (None, subprocess.DEVNULL):
                self.log.close()
            raise

    def restart(self, tries: int = 3):
        """A failed layer process requires a fresh explicit installation/start."""
        raise server.EngineDied(
            "fabric layer process stopped; explicitly reinstall/restart the plan"
        )

    def generate(
        self, ids, max_new, sampling, cancel, embeddings=None, prefill_only=False
    ):
        """Refuse unsupported controls before the existing token-stream transport."""
        request = Generation(ids=ids, maximum=max_new)
        sampling = sampling or {}
        Greedy.model_validate(
            {
                key: sampling[key]
                for key in Greedy.model_fields
                if sampling.get(key) is not None
            }
        )
        unsupported = (
            "_restore",
            "strata_prefix",
            "strata_tune",
            "allowed_token_ids",
            "experimental_speed_projection",
        )
        if (
            embeddings is not None
            or prefill_only
            or server.top_logprobs(sampling) is not None
            or sampling.get("logit_bias")
            or any(sampling.get(key) is not None for key in unsupported)
        ):
            raise ValueError(
                "fabric profile supports text greedy generation without checkpoints, tuning or logit controls"
            )
        yield from super().generate(request.ids, request.maximum, sampling, cancel)


def prepare_runtime(ready_seconds: float) -> None:
    """Configure this dedicated server process without changing other services."""
    os.environ["STRATA_KEEP_LEFTOVERS"] = "1"
    os.environ["STRATA_DS4_RESTORE_POINTS"] = "0"
    os.environ["STRATA_DS4_AUTO_PIN"] = "0"
    server.apply_ready_timeout({"engine_ready_s": ready_seconds})


def create_service(spec: Launch) -> server.Service:
    """Verify artifacts and actual native READY, closing owned resources on failure."""
    with spec.executable.open("rb") as image:
        actual = hashlib.file_digest(image, "sha256").hexdigest()
    if actual != spec.expected.binary_sha256:
        raise ValueError("prefix executable differs from pinned binary")
    tokenizer = load_deepseek(spec.tokenizer_dir, spec.tokenizer)
    template = ChatTemplate(spec.tokenizer_dir / "chat_template.jinja")
    if (
        hashlib.sha256(template.source.encode("utf-8")).hexdigest()
        != spec.tokenizer.assets_sha256["chat_template.jinja"]
    ):
        raise ValueError("loaded chat template differs from pinned bytes")
    prepare_runtime(spec.engine_ready_s)
    child_environment = dict(os.environ, DS4_IQ_Q8K="1", DS4_PLACE_HASH="1")
    checked = subprocess.run(
        [str(spec.executable), "--serve", *spec.native_args, "--check-config"],
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    if (
        checked.stdout.strip()
        != "CONFIG_CHECKED syntax_only; model/device/quality NOT VERIFIED"
    ):
        raise ValueError("prefix did not confirm syntax-only preflight")
    spec.engine_log.parent.mkdir(parents=True, exist_ok=True)
    with spec.engine_log.open("x"):
        pass
    engine = PinnedLayerEngine(
        str(spec.executable),
        spec.native_args,
        log=str(spec.engine_log),
        env=child_environment,
    )
    try:
        observed = PrefixReceipt.model_validate(engine.info, extra="ignore")
        if (
            observed != spec.expected
            or engine.max_context != 2048
            or not engine.can_stop
        ):
            raise ValueError(
                "actual prefix READY/identity/limits differ from pinned launch"
            )
        svc = server.Service(
            engine,
            tokenizer,
            template,
            model_name=spec.model_name,
            sampling_defaults={"temperature": 0.0, "top_p": 1.0},
            fit_max_tokens=False,
        )
        svc.model_arch = "deepseek41"
        svc.set_family("deepseek41")
        svc.config_effort = "none"
        svc.answer_room = False
        svc.offline = True
        svc.max_queue = spec.max_queue
        svc.queue_timeout_s = spec.queue_timeout_s
        return svc
    except BaseException:
        engine.close()
        raise


def start(spec: Launch):
    """Open only loopback HTTP after native admission; own just this prefix."""
    svc = create_service(spec)
    try:
        return server.serve(svc, host="127.0.0.1", port=spec.port)
    except BaseException:
        svc.engine.close()
        raise


def main() -> None:
    """Launch the explicit static binding and stop its owned engine on exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch", type=Path)
    args = parser.parse_args()
    spec = Launch.model_validate_json(args.launch.read_bytes())

    def interrupted(*_):
        raise KeyboardInterrupt

    previous = {
        number: signal.signal(number, interrupted)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    httpd = None
    try:
        httpd = start(spec)
        print(
            f"Fabric HTTP listening on http://127.0.0.1:{httpd.server_address[1]}; plan {spec.expected.plan_sha256}",
            flush=True,
        )
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if httpd is not None:
                httpd.svc.cancel_owned_requests()
                httpd.shutdown()
                httpd.server_close()
                httpd.svc.engine.close()
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


if __name__ == "__main__":
    main()
