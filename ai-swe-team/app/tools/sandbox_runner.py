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
- No secrets are passed to the sandbox via environment variables.
"""

import io
import re
import tarfile

import docker

from app.config import settings

_client = docker.from_env()


def _build_tar(files: dict[str, str]) -> io.BytesIO:
    """Pack a dict of {path: content} into an in-memory tar archive."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=path)
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
    container = _client.containers.create(
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
        exit_code, output = container.exec_run(
            cmd=["bash", "-lc", test_command],
            workdir="/workspace",
            demux=False,
        )
        logs = output.decode("utf-8", errors="replace") if output else ""
        return _parse_pytest_output(exit_code, logs)
    finally:
        container.kill()
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
