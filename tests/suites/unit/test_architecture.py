"""Executable dependency boundaries for framework and domain isolation."""

import ast
import subprocess
import sys
from pathlib import Path

PIPECAT_ROOT = Path("src/therapy/integrations/pipecat")
DOMAIN_PACKAGES = {"dialogue", "knowledge", "memory", "perception", "session", "speech"}


def _imports(path: Path) -> list[tuple[int, str]]:
    """Return direct imported module names with their source lines."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append((node.lineno, node.module))
    return imports


def test_pipecat_imports_are_confined_to_integration_package(repo_root: Path) -> None:
    violations: list[str] = []
    source_root = repo_root / "src" / "therapy"
    allowed_root = repo_root / PIPECAT_ROOT
    for path in sorted(source_root.rglob("*.py")):
        for line, module in _imports(path):
            if (
                module == "pipecat" or module.startswith("pipecat.")
            ) and not path.is_relative_to(allowed_root):
                violations.append(
                    f"{path.relative_to(repo_root)}:{line} imports {module}"
                )

    assert not violations, "Pipecat boundary violations:\n" + "\n".join(violations)


def test_domain_packages_do_not_import_infrastructure(repo_root: Path) -> None:
    violations: list[str] = []
    domain_roots = [repo_root / "src" / "therapy" / name for name in DOMAIN_PACKAGES]
    for domain_root in domain_roots:
        if not domain_root.exists():
            continue
        for path in sorted(domain_root.rglob("*.py")):
            for line, module in _imports(path):
                if module == "therapy.integrations" or module.startswith(
                    "therapy.integrations."
                ):
                    violations.append(
                        f"{path.relative_to(repo_root)}:{line} imports {module}"
                    )

    assert not violations, "Domain-to-infrastructure violations:\n" + "\n".join(
        violations
    )


def test_importing_fastapi_server_does_not_import_pipecat() -> None:
    script = """
import sys
import therapy.server.app
loaded = sorted(name for name in sys.modules if name == 'pipecat' or name.startswith('pipecat.'))
assert not loaded, loaded
"""

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
