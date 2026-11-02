"""Deployment scaffolding.

These tests pin the properties that make the generated image safe to deploy:
it installs the package rather than a dev requirements file, it runs as a
non-root user, it has a liveness check, and its command is exec-form so the
container receives signals. Each assertion fails if the template regresses.
"""
import subprocess
import sys

import pytest

from kssrag.services.deployment_service import DeploymentService


def _scaffold(tmp_path, **kwargs):
    return DeploymentService().scaffold(str(tmp_path), **kwargs)


def test_scaffold_writes_every_file(tmp_path):
    """All deployment files are created."""
    result = _scaffold(tmp_path)

    assert sorted(result.written) == [".dockerignore", ".env.example", "Dockerfile", "docker-compose.yml"]
    for name in result.written:
        assert (tmp_path / name).is_file()


def test_scaffold_does_not_overwrite_without_force(tmp_path):
    """Existing files are left alone, so a second run cannot destroy edits."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("my edited dockerfile\n", encoding="utf-8")

    result = _scaffold(tmp_path)

    assert "Dockerfile" in result.skipped
    assert dockerfile.read_text(encoding="utf-8") == "my edited dockerfile\n"


def test_scaffold_overwrites_with_force(tmp_path):
    """--force replaces the file."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("stale\n", encoding="utf-8")

    result = _scaffold(tmp_path, overwrite=True)

    assert "Dockerfile" in result.written
    assert "FROM python:" in dockerfile.read_text(encoding="utf-8")


def test_scaffold_rejects_missing_directory(tmp_path):
    """A bad path is reported, not silently created."""
    with pytest.raises(FileNotFoundError):
        DeploymentService().scaffold(str(tmp_path / "does-not-exist"))


def test_dockerfile_installs_the_package_not_a_requirements_file(tmp_path):
    """The image must install kssrag, not a dev requirements list."""
    _scaffold(tmp_path)
    dockerfile = (tmp_path / "Dockerfile").read_text(encoding="utf-8")

    assert "pip install --no-cache-dir kssrag" in dockerfile
    assert "requirements.txt" not in dockerfile


def test_dockerfile_runs_as_non_root(tmp_path):
    """A user is created and switched to."""
    _scaffold(tmp_path)
    dockerfile = (tmp_path / "Dockerfile").read_text(encoding="utf-8")

    assert "useradd" in dockerfile
    assert "USER appuser" in dockerfile
    assert dockerfile.index("USER appuser") > dockerfile.index("useradd")


def test_dockerfile_has_a_healthcheck(tmp_path):
    """The container reports liveness on the real endpoint."""
    _scaffold(tmp_path)
    dockerfile = (tmp_path / "Dockerfile").read_text(encoding="utf-8")

    assert "HEALTHCHECK" in dockerfile
    assert "/health" in dockerfile


def test_dockerfile_command_is_exec_form(tmp_path):
    """CMD is a JSON array, so SIGTERM reaches the server rather than a shell."""
    _scaffold(tmp_path)
    dockerfile = (tmp_path / "Dockerfile").read_text(encoding="utf-8")

    assert "CMD [" in dockerfile
    assert "bash -c" not in dockerfile
    assert "rm -rf" not in dockerfile


def test_dockerignore_excludes_secrets(tmp_path):
    """The environment file never reaches the build context."""
    _scaffold(tmp_path)
    dockerignore = (tmp_path / ".dockerignore").read_text(encoding="utf-8")

    assert ".env" in dockerignore.split()
    assert ".git" in dockerignore.split()


def test_deploy_command_runs_without_an_api_key(tmp_path):
    """`kssrag deploy` must work before the user has configured a provider."""
    result = subprocess.run(
        [sys.executable, "-m", "kssrag.cli", "deploy", "--dir", str(tmp_path)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "."},
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "Dockerfile").is_file()
    assert "[OK] wrote Dockerfile" in result.stdout
