"""Enforcement tests for arch-check.py.

These tests exist so the checker cannot silently stop checking. Each one builds
a throwaway repository tree, seeds one violation, and asserts the checker exits
non-zero and names the rule that was broken.

The checks run against a temporary root rather than the real source tree, so a
failing assertion here can never leave the repository modified. The last test
runs the checker against the real tree and asserts it is clean.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

ARCH_CHECK = Path(__file__).resolve().parent.parent / "arch-check.py"

ARCHITECTURE_FILE = """\
layers:
  entities: "models/entities"
  schemas: "schemas"
  crud: "crud"
  services: "services"
  routers: "routers"

banned_name_exemptions:
  - kssrag/utils
  - kssrag/utils/helpers.py
"""


def _build_tree(root: Path, files: dict) -> Path:
    """Create a repository root containing the given package files."""
    (root / ".architecture").write_text(ARCHITECTURE_FILE, encoding="utf-8")
    for relative_path, content in files.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(content), encoding="utf-8")
    return root


def _run_checker(root: Path) -> subprocess.CompletedProcess:
    """Run the checker against ``root``."""
    return subprocess.run(
        [sys.executable, str(ARCH_CHECK), "--root", str(root)],
        capture_output=True,
        text=True,
    )


def test_clean_tree_passes(tmp_path):
    """A tree whose imports only point inward must pass."""
    root = _build_tree(
        tmp_path,
        {
            "kssrag/core/errors.py": "class DomainError(Exception):\n    pass\n",
            "kssrag/services/rag_service.py": "from ..core.errors import DomainError\n",
            "kssrag/routers/query_router.py": "from ..services.rag_service import RAGService\n",
            "kssrag/services/rag_service_alias.py": "class RAGService:\n    pass\n",
        },
    )

    result = _run_checker(root)

    assert result.returncode == 0, result.stdout + result.stderr


def test_router_importing_crud_fails(tmp_path):
    """A router reaching past the service layer must fail."""
    root = _build_tree(
        tmp_path,
        {
            "kssrag/crud/session_crud.py": "class SessionCrud:\n    pass\n",
            "kssrag/routers/query_router.py": "from ..crud.session_crud import SessionCrud\n",
        },
    )

    result = _run_checker(root)

    assert result.returncode == 1
    assert "imports outward layer 'crud'" in result.stdout


def test_service_importing_sibling_service_fails(tmp_path):
    """Two services importing each other must fail."""
    root = _build_tree(
        tmp_path,
        {
            "kssrag/services/rag_service.py": "from .session_service import SessionService\n",
            "kssrag/services/session_service.py": "class SessionService:\n    pass\n",
        },
    )

    result = _run_checker(root)

    assert result.returncode == 1
    assert "imports sibling services" in result.stdout


def test_entity_importing_service_fails(tmp_path):
    """An entity must import nothing from this project."""
    root = _build_tree(
        tmp_path,
        {
            "kssrag/models/entities/document_model.py": "from ...services.rag_service import RAGService\n",
            "kssrag/services/rag_service.py": "class RAGService:\n    pass\n",
        },
    )

    result = _run_checker(root)

    assert result.returncode == 1
    assert "imports outward layer 'services'" in result.stdout


def test_emoji_in_source_fails(tmp_path):
    """A pictograph in a source file must fail."""
    root = _build_tree(tmp_path, {"kssrag/core/errors.py": "MESSAGE = 'launch \U0001F680'\n"})

    result = _run_checker(root)

    assert result.returncode == 1
    assert "Prohibited symbol" in result.stdout


def test_banned_directory_name_fails(tmp_path):
    """A banned standalone directory name must fail."""
    root = _build_tree(tmp_path, {"kssrag/tmp/loader.py": "VALUE = 1\n"})

    result = _run_checker(root)

    assert result.returncode == 1
    assert "Banned directory name 'tmp'" in result.stdout


def test_banned_name_exemption_differs_from_unexempted_use(tmp_path):
    """A recorded exemption passes while the same name elsewhere still fails."""
    root = _build_tree(
        tmp_path,
        {
            "kssrag/utils/helpers.py": "VALUE = 1\n",
            "kssrag/services/helpers.py": "VALUE = 2\n",
        },
    )

    result = _run_checker(root)

    assert result.returncode == 1
    assert "Banned file name 'helpers.py' at kssrag/services/helpers.py" in result.stdout
    assert "at kssrag/utils/helpers.py" not in result.stdout


def test_banned_user_facing_phrase_fails(tmp_path):
    """A banned phrase in a schema string must fail."""
    root = _build_tree(
        tmp_path,
        {"kssrag/schemas/query_schema.py": 'MESSAGE = "Something went wrong"\n'},
    )

    result = _run_checker(root)

    assert result.returncode == 1
    assert "Banned user-facing phrase" in result.stdout


def test_real_repository_is_clean():
    """The checked-in repository must satisfy every rule."""
    result = subprocess.run(
        [sys.executable, str(ARCH_CHECK)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout
    assert "[FAIL]" not in result.stdout
