"""Own one native fabric child using the existing lifeline and Linux pidfds."""

import ctypes
import errno
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from serve import lifeline
from serve.tokenizer_assets import Digest, Record
from strata_control import atomic, procs
from strata_control.fabric_plan import FilePin, Positive

T = TypeVar("T")


class ProcessIdentity(Record):
    """Actual live child identity; a PID alone never authorizes a signal."""

    pid: Positive
    start_token: str
    argv: tuple[str, ...]
    binary_sha256: Digest


class ProcessExit(Record):
    """Raw child outcome, including unknown exit/identity on partial startup."""

    pid: Positive
    argv: tuple[str, ...]
    returncode: int | None
    verified_identity: ProcessIdentity | None


class NativeProcess:
    """Launch a declared executable once and release just that owned process.

    Startup logs are preserved on every outcome. Native exit, timeout, changed
    executable and unavailable pidfds fail visibly; no restart is attempted.
    """

    def __init__(self, argv: list[str], pin: FilePin, log: Path):
        """Verify the executable and create a lifetime-bound child with a new log."""
        if not sys.platform.startswith("linux"):
            raise RuntimeError("native fabric process ownership requires Linux")
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.pidfd_open.argtypes = (ctypes.c_int, ctypes.c_uint)
        self._libc.pidfd_open.restype = ctypes.c_int
        self._libc.pidfd_send_signal.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint,
        )
        self._libc.pidfd_send_signal.restype = ctypes.c_int
        # Refuse unsupported kernels before starting any native process.
        probe = self._libc.pidfd_open(os.getpid(), 0)
        if probe < 0:
            raise OSError(ctypes.get_errno(), "fabric pidfd preflight failed")
        os.close(probe)
        self.proc = None
        self.pidfd = None
        self.stream = None
        self.log = log
        self.pin = pin
        self.verified_identity = None
        self._exit_recorded = False
        pin.verify(Path(argv[0]))
        try:
            self.stream = log.open("xb")
            self.proc = lifeline.popen(
                argv,
                role="engine",
                stdin=subprocess.DEVNULL,
                stdout=self.stream,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, DS4_IQ_Q8K="1", DS4_PLACE_HASH="1"),
            )
            fd = self._libc.pidfd_open(self.proc.pid, 0)
            if fd < 0:
                raise OSError(ctypes.get_errno(), "cannot pin native child lifetime")
            self.pidfd = fd
        except BaseException:
            self.close()
            raise

    def identity(self) -> ProcessIdentity:
        """Pin the running executable, birth token and argv after native startup."""
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("native fabric process has exited")
        actual = procs.identify(self.proc.pid)
        if actual is None or tuple(self.proc.args) != actual.cmd:
            raise RuntimeError("native child identity is unavailable or changed")
        self.pin.verify(Path(f"/proc/{self.proc.pid}/exe"))
        self.verified_identity = ProcessIdentity(
            pid=actual.pid,
            start_token=actual.token,
            argv=actual.cmd,
            binary_sha256=self.pin.sha256,
        )
        return self.verified_identity

    def text(self) -> str:
        """Read the preserved startup log without silently truncating evidence."""
        if self.log.stat().st_size > 64 * 1024**2:
            raise RuntimeError("native startup log exceeds 64 MiB")
        return self.log.read_text(encoding="utf-8")

    def wait_ready(self, parse: Callable[[str], T | None], timeout_s: float) -> T:
        """Wait for a validated native receipt while requiring a live process."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError(
                    f"native child exited before admission; see {self.log}"
                )
            result = parse(self.text())
            if result is not None:
                if time.monotonic() >= deadline:
                    break
                self.identity()
                return result
            time.sleep(0.05)
        raise TimeoutError(f"native startup deadline exceeded; see {self.log}")

    def _signal(self, number: int) -> None:
        """Signal the original process instance through its kernel-held pidfd."""
        if self.pidfd is None:
            raise RuntimeError("native child has no verified pidfd")
        if self._libc.pidfd_send_signal(self.pidfd, number, None, 0) != 0:
            error = ctypes.get_errno()
            if error != errno.ESRCH:
                raise OSError(error, "cannot signal owned native child")

    def close(self) -> None:
        """Terminate/reap this child, escalating once; retain logs and raw exits."""
        try:
            if self.proc is not None and self.proc.poll() is None:
                if self.pidfd is None:
                    # Only partial construction can reach this path. No
                    # runtime receipt is admitted without a pidfd.
                    self.proc.terminate()
                else:
                    self._signal(signal.SIGTERM)
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if self.pidfd is None:
                        self.proc.kill()
                    else:
                        self._signal(signal.SIGKILL)
                    self.proc.wait(timeout=5)
            if self.proc is not None:
                lifeline.release(self.proc)
        finally:
            if self.pidfd is not None:
                os.close(self.pidfd)
                self.pidfd = None
            if self.stream is not None:
                self.stream.close()
                self.stream = None
            if self.proc is not None and not self._exit_recorded:
                record = ProcessExit(
                    pid=self.proc.pid,
                    argv=tuple(self.proc.args),
                    returncode=self.proc.returncode,
                    verified_identity=self.verified_identity,
                )
                atomic.write_new_text(
                    self.log.with_suffix(".exit.json"),
                    record.model_dump_json(indent=2) + "\n",
                )
                self._exit_recorded = True
