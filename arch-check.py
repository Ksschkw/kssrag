#!/usr/bin/env python3
"""Architecture and standards enforcement for KSS RAG.

Machine checks required by house-standard.md:

  section 2  inward-only layer dependencies
  section 4  banned standalone names
  section 6  cross-layer and same-layer coupling
  section 8  no emoji or decorative symbols in source
  section 10 no banned user-facing phrases
  section 16 every check runs from the default test command

Exit code 0 when clean. Exit code 1 with one [FAIL] line per violation.

The previous version of this file checked almost nothing: the banned-name
table was defined and never referenced, layer detection matched a substring
anywhere in the path, and routers, middleware and core had no rules at all.
This version resolves imports to real modules and enforces every layer.
"""
import ast
import os
import sys
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.join(ROOT_DIR, "kssrag")

# Directories scanned for emoji and decorative symbols (section 8). Tests are
# excluded: fixtures may legitimately carry unicode sample data.
HYGIENE_SCAN_DIRECTORIES = ("kssrag", "examples")

# Layers, keyed by the path components below the repository root.
LAYER_DIRECTORIES: Dict[str, Tuple[str, ...]] = {
    "entities": ("kssrag", "models", "entities"),
    "providers": ("kssrag", "models"),
    "schemas": ("kssrag", "schemas"),
    "crud": ("kssrag", "crud"),
    "services": ("kssrag", "services"),
    "routers": ("kssrag", "routers"),
    "middleware": ("kssrag", "middleware"),
    "core": ("kssrag", "core"),
}

# Which layers each layer may import. Anything not listed is outward and fails.
ALLOWED_TARGET_LAYERS: Dict[str, Set[str]] = {
    "entities": set(),
    "providers": {"core", "entities"},
    "schemas": {"entities"},
    "crud": {"core", "entities"},
    "services": {"core", "crud", "entities", "schemas"},
    "routers": {"core", "schemas", "services"},
    "middleware": {"core"},
    "core": {"entities"},
}

# A service importing a sibling service, or a crud module importing another
# entity's crud module, is forbidden (section 2 hard rules 2 and 7). Package
# aggregators are exempt: __init__.py exists to re-export.
SAME_LAYER_FORBIDDEN = frozenset({"services", "crud"})

# Standalone names banned by section 4.
BANNED_NAMES = frozenset(
    {
        "common",
        "data",
        "do",
        "handle",
        "helper",
        "helpers",
        "info",
        "manager",
        "misc",
        "obj",
        "process",
        "temp",
        "thing",
        "tmp",
        "util",
        "utils",
        "val",
    }
)

# Banned user-facing phrases (section 10), matched case-insensitively against
# string literals in the transport and service layers.
BANNED_PHRASES = (
    "something went wrong",
    "an error occurred",
    "invalid input",
    "unknown error",
    "error 500",
    "error 404",
)

# Non-ASCII characters that are typography, not decoration: the section sign
# used in docstrings, and the em and en dashes. See .architecture.
ALLOWED_NON_ASCII = frozenset("§\u2014\u2013")


def _relative_parts(path: str) -> Tuple[str, ...]:
    """Path components of ``path`` relative to the repository root."""
    return tuple(os.path.relpath(path, ROOT_DIR).split(os.sep))


def _layer_of(parts: Sequence[str]) -> Optional[str]:
    """The layer a module path belongs to, or None for facade and root modules.

    Longest prefix wins so models/entities is matched before models.
    """
    for layer, prefix in sorted(LAYER_DIRECTORIES.items(), key=lambda item: -len(item[1])):
        if tuple(parts[: len(prefix)]) == prefix:
            return layer
    return None


def _resolve_import(
    file_parts: Sequence[str], module: Optional[str], level: int
) -> Tuple[str, ...]:
    """Resolve an import statement to absolute path components.

    ``level`` is 0 for an absolute import and 1 or more for a relative one,
    matching ast.ImportFrom.level. The result is dotted components starting at
    the repository root, e.g. ("kssrag", "crud", "session_crud").
    """
    package_parts = list(file_parts[:-1])
    if level == 0:
        return tuple((module or "").split("."))
    if level > 1:
        package_parts = package_parts[: len(package_parts) - (level - 1)]
    if module:
        package_parts.extend(module.split("."))
    return tuple(package_parts)


def _iter_source_files(directory: str) -> Iterable[str]:
    """Yield every .py file under ``directory``, skipping caches."""
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(root, name)


def _parse(path: str) -> Optional[ast.Module]:
    """Parse a file, returning None when it is not valid Python."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return ast.parse(handle.read(), filename=path)
    except (SyntaxError, UnicodeDecodeError):
        return None


def _load_architecture_settings() -> Dict[str, List[str]]:
    """Read the small key/list pairs this checker needs from .architecture.

    Parsed by hand rather than with PyYAML so the checker stays dependency free.
    """
    settings: Dict[str, List[str]] = {"banned_name_exemptions": []}
    path = os.path.join(ROOT_DIR, ".architecture")
    if not os.path.exists(path):
        return settings
    current_key: Optional[str] = None
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not line.startswith((" ", "\t", "-")):
                key = line.split(":", 1)[0].strip()
                current_key = key if key in settings else None
                continue
            if current_key and line.strip().startswith("- "):
                settings[current_key].append(line.strip()[2:].strip().strip('"'))
    return settings


def check_layer_imports() -> List[str]:
    """Enforce inward-only imports and same-layer decoupling."""
    violations: List[str] = []

    for path in _iter_source_files(PACKAGE_DIR):
        file_parts = _relative_parts(path)
        source_layer = _layer_of(file_parts)
        if source_layer is None:
            continue

        tree = _parse(path)
        if tree is None:
            violations.append(f"[FAIL] {os.path.relpath(path, ROOT_DIR)} is not valid Python")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [_resolve_import(file_parts, alias.name, 0) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                targets = [_resolve_import(file_parts, node.module, node.level)]
            else:
                continue

            for resolved in targets:
                target_layer = _layer_of(resolved)
                if target_layer is None or target_layer == source_layer:
                    if target_layer == source_layer and source_layer in SAME_LAYER_FORBIDDEN:
                        if os.path.basename(path) == "__init__.py":
                            continue
                        if resolved[-1] == os.path.splitext(os.path.basename(path))[0]:
                            continue
                        violations.append(
                            f"[FAIL] {os.path.relpath(path, ROOT_DIR)} imports sibling "
                            f"{source_layer} module '{'.'.join(resolved)}' "
                            f"(section 2 hard rule 7)"
                        )
                    continue

                if target_layer not in ALLOWED_TARGET_LAYERS[source_layer]:
                    violations.append(
                        f"[FAIL] {os.path.relpath(path, ROOT_DIR)} ({source_layer}) imports "
                        f"outward layer '{target_layer}': {'.'.join(resolved)}"
                    )
    return violations


def check_banned_names() -> List[str]:
    """Reject banned standalone file and directory names (section 4)."""
    violations: List[str] = []
    exemptions = {entry.rstrip("/") for entry in _load_architecture_settings()["banned_name_exemptions"]}

    for root, dirs, files in os.walk(PACKAGE_DIR):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for directory in dirs:
            if directory in BANNED_NAMES:
                relative = os.path.relpath(os.path.join(root, directory), ROOT_DIR)
                if relative not in exemptions:
                    violations.append(
                        f"[FAIL] Banned directory name '{directory}' at {relative} (section 4). "
                        f"Rename it, or record it in .architecture banned_name_exemptions."
                    )
        for name in files:
            stem = os.path.splitext(name)[0]
            if stem in BANNED_NAMES:
                relative = os.path.relpath(os.path.join(root, name), ROOT_DIR)
                if relative not in exemptions:
                    violations.append(
                        f"[FAIL] Banned file name '{name}' at {relative} (section 4). "
                        f"Rename it, or record it in .architecture banned_name_exemptions."
                    )
    return violations


def check_emoji_and_non_ascii() -> List[str]:
    """Reject emoji and decorative symbols in source (section 8)."""
    violations: List[str] = []
    for directory in HYGIENE_SCAN_DIRECTORIES:
        target = os.path.join(ROOT_DIR, directory)
        if not os.path.isdir(target):
            continue
        for path in _iter_source_files(target):
            relative = os.path.relpath(path, ROOT_DIR)
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                for line_number, line in enumerate(handle, start=1):
                    for character in line:
                        if ord(character) < 128 or character in ALLOWED_NON_ASCII:
                            continue
                        category = unicodedata.category(character)
                        is_symbol = category.startswith(("S", "C"))
                        is_pictograph = 0x1F000 <= ord(character) <= 0x1FAFF
                        if is_symbol or is_pictograph:
                            try:
                                name = unicodedata.name(character)
                            except ValueError:
                                name = "unnamed"
                            violations.append(
                                f"[FAIL] Prohibited symbol U+{ord(character):04X} "
                                f"({name}) in {relative}:{line_number} (section 8)"
                            )
    return violations


def check_banned_phrases() -> List[str]:
    """Reject banned user-facing phrases in transport and service strings."""
    violations: List[str] = []
    scanned = [os.path.join(PACKAGE_DIR, name) for name in ("routers", "schemas", "services")]

    for directory in scanned:
        if not os.path.isdir(directory):
            continue
        for path in _iter_source_files(directory):
            tree = _parse(path)
            if tree is None:
                continue
            relative = os.path.relpath(path, ROOT_DIR)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                lowered = node.value.lower()
                for phrase in BANNED_PHRASES:
                    if phrase in lowered:
                        violations.append(
                            f"[FAIL] Banned user-facing phrase '{phrase}' in {relative}:{node.lineno} "
                            f"(section 10)"
                        )
    return violations


def _configure_root(root: str) -> None:
    """Point the checks at a repository root other than this file's directory.

    The default is the repository this script lives in. The override exists so
    the test suite can run the rules against a throwaway tree instead of
    mutating the real source files.
    """
    global ROOT_DIR, PACKAGE_DIR
    ROOT_DIR = os.path.abspath(root)
    PACKAGE_DIR = os.path.join(ROOT_DIR, "kssrag")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run every check and report violations."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "--root":
        if len(arguments) < 2:
            print("[FAIL] --root requires a directory path")
            return 1
        _configure_root(arguments[1])

    print("Running architecture and standards enforcement check...")

    checks = (
        ("layer imports", check_layer_imports),
        ("banned names", check_banned_names),
        ("emoji and decorative symbols", check_emoji_and_non_ascii),
        ("banned user-facing phrases", check_banned_phrases),
    )

    total = 0
    for label, check in checks:
        violations = check()
        total += len(violations)
        if violations:
            print(f"[FAIL] {label}: {len(violations)} violation(s)")
            for violation in violations:
                print(f"  {violation}")
        else:
            print(f"[OK] {label}")

    if total:
        print(f"\n[FAIL] Architecture check failed with {total} violation(s).")
        return 1

    print("\n[OK] All house standard enforcement rules satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
