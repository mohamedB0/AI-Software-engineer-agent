"""
Docker sandbox runner.

Every piece of agent-generated code is executed inside an ephemeral,
network-disabled, resource-limited Docker container. The container is always
removed after execution, even on failure.

Security properties:
- network_disabled=True: generated code cannot make outbound network calls.
- mem_limit / nano_cpus / pids_limit: bound resource usage to prevent
  runaway or malicious scripts from affecting the host.
- Non-root user (sandboxuser) baked into Dockerfile.sandbox.
- Container always removed in the finally block (no orphaned containers).
- File paths are sanitised before being written, so agent-supplied paths
  cannot escape the working directory (no '..', absolute, or empty paths).
- exec_run enforces a hard timeout so a hanging test cannot block a worker.
- No secrets are passed to the sandbox via environment variables.

The Docker client is initialised lazily so importing this module never
requires a running daemon (important for unit tests and app startup).
"""

import io
import logging
import re
import tarfile

import docker

from app.config import settings

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Return a lazily-initialised Docker client (no daemon I/O at import time)."""
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


def _safe_relative_path(path: str) -> str | None:
    """
    Coerce an agent-supplied path into a safe relative path.

    Returns None when the path is unsafe (absolute, contains a '..' segment,
    is empty, or has empty segments) so callers can skip it instead of
    extracting a file outside the sandbox working directory.
    """
    if path.startswith("/") or path.startswith("\\"):
        return None
    normalized = path.replace("\\", "/").strip("/")
    parts = normalized.split("/")
    if not parts:
        return None
    if any(part in ("", ".", "..") for part in parts):
        return None
    return normalized


def _build_tar(files: dict[str, str]) -> io.BytesIO:
    """Pack a dict of {path: content} into an in-memory tar archive.

    Unsafe paths are skipped with a warning so a single bad path cannot fail
    the whole sandbox run.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path, content in files.items():
            safe_path = _safe_relative_path(path)
            if safe_path is None:
                logger.warning("Skipping unsafe sandbox file path: %r", path)
                continue
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=safe_path)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def run_in_sandbox(
    code_files: dict[str, str],
    test_command: str,
    timeout: int = 60,
) -> dict:
    """
    Execute `test_command` against `code_files` inside a fresh, isolated,
    network-disabled, resource-limited container built from SANDBOX_IMAGE.

    Args:
        code_files: Mapping of relative file path -> source code content.
        test_command: Shell command to run inside the container (e.g. 'pytest -q').
        timeout: Maximum seconds the command is allowed to run.

    Returns:
        dict with keys: passed, total, failed, failures, logs.
    """
    container = _get_client().containers.create(
        image=settings.sandbox_image,
        # Keep the container alive; the actual command is injected via exec_run.
        command=["sleep", str(timeout + 5)],
        working_dir="/workspace",
        network_disabled=settings.sandbox_network_disabled,
        mem_limit="512m",
        nano_cpus=1_000_000_000,  # 1 CPU
        pids_limit=128,  # prevent fork bombs
        detach=True,
    )
    try:
        container.start()
        container.put_archive("/workspace", _build_tar(code_files))
        try:
            exit_code, output = container.exec_run(
                cmd=["bash", "-lc", test_command],
                workdir="/workspace",
                demux=False,
                timeout=timeout,
            )
            logs = output.decode("utf-8", errors="replace") if output else ""
            return _parse_pytest_output(exit_code, logs)
        except Exception as e:
            # e.g. ReadTimeout when the command exceeds `timeout`.
            logger.warning("Sandbox command failed or timed out: %s", e)
            return {
                "passed": False,
                "total": 0,
                "failed": 0,
                "failures": [],
                "logs": f"Sandbox execution failed or timed out after {timeout}s: {e}"[:4000],
            }
    finally:
        try:
            container.kill()
        except Exception:
            pass
        container.remove(force=True)


def _parse_pytest_output(exit_code: int, logs: str) -> dict:
    """Parse pytest's stdout/stderr into a structured result dict."""
    passed = exit_code == 0

    m = re.search(r"(\d+) passed", logs)
    total_passed = int(m.group(1)) if m else 0

    m = re.search(r"(\d+) failed", logs)
    total_failed = int(m.group(1)) if m else 0

    failures = re.findall(r"FAILED (\S+)", logs)

    return {
        "passed": passed,
        "total": total_passed + total_failed,
        "failed": total_failed,
        "failures": failures,
        # Cap log size so it does not bloat the LangGraph state checkpoint.
        "logs": logs[-4000:],
    }
