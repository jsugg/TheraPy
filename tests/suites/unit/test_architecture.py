"""Executable dependency boundaries for framework and domain isolation."""

import ast
import subprocess
import sys
from pathlib import Path

PIPECAT_ROOT = Path("src/therapy/integrations/pipecat")
DOMAIN_PACKAGES = {"dialogue", "knowledge", "memory", "perception", "session", "speech"}

OBSERVABILITY_ROOT = Path("src/therapy/observability")

# Vendor stacks that domain modules must never import (plan §2): web framework,
# realtime framework, telemetry SDK, and every observability backend SDK.
FRAMEWORK_PREFIXES = ("fastapi", "starlette", "pipecat", "opentelemetry")
BACKEND_SDK_PREFIXES = ("phoenix", "arize", "mlflow", "langfuse", "langsmith", "grafana")


def _module_matches(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


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


def test_domain_packages_do_not_import_frameworks_or_backends(
    repo_root: Path,
) -> None:
    """Plan §2: domain modules stay free of FastAPI/Pipecat/OTel/backend SDKs."""
    violations: list[str] = []
    forbidden = FRAMEWORK_PREFIXES + BACKEND_SDK_PREFIXES
    domain_roots = [repo_root / "src" / "therapy" / name for name in DOMAIN_PACKAGES]
    for domain_root in domain_roots:
        if not domain_root.exists():
            continue
        for path in sorted(domain_root.rglob("*.py")):
            for line, module in _imports(path):
                if _module_matches(module, forbidden):
                    violations.append(
                        f"{path.relative_to(repo_root)}:{line} imports {module}"
                    )

    assert not violations, "Domain framework/backend violations:\n" + "\n".join(
        violations
    )


def test_observability_vendor_imports_are_confined(repo_root: Path) -> None:
    """Plan §3 dependency rule.

    - `observability.model` / `observability.context` / the package public API
      stay framework-free (no third-party vendor imports at all).
    - Only `observability.telemetry` may import the OTel SDK/exporters.
    - Only `observability.exporters` (and adapters below it) may import a
      selected backend SDK.
    - Pipecat types never appear inside `therapy/observability/`.
    """
    obs_root = repo_root / OBSERVABILITY_ROOT
    if not obs_root.exists():
        return
    violations: list[str] = []
    for path in sorted(obs_root.rglob("*.py")):
        rel = path.relative_to(obs_root).as_posix()
        for line, module in _imports(path):
            where = f"{path.relative_to(repo_root)}:{line} imports {module}"
            if _module_matches(module, ("pipecat", "fastapi", "starlette")):
                violations.append(where)
            elif _module_matches(module, ("opentelemetry",)) and rel != "telemetry.py":
                violations.append(where)
            elif _module_matches(module, BACKEND_SDK_PREFIXES) and not rel.startswith(
                "exporters"
            ):
                violations.append(where)

    assert not violations, "Observability vendor-import violations:\n" + "\n".join(
        violations
    )


def test_product_modules_do_not_import_backend_sdks(repo_root: Path) -> None:
    """Plan §3: no product/domain module imports a selected backend SDK."""
    violations: list[str] = []
    source_root = repo_root / "src" / "therapy"
    exporters_root = repo_root / OBSERVABILITY_ROOT
    for path in sorted(source_root.rglob("*.py")):
        if path.is_relative_to(exporters_root) and path.relative_to(
            exporters_root
        ).as_posix().startswith("exporters"):
            continue
        for line, module in _imports(path):
            if _module_matches(module, BACKEND_SDK_PREFIXES):
                violations.append(
                    f"{path.relative_to(repo_root)}:{line} imports {module}"
                )

    assert not violations, "Backend SDK import violations:\n" + "\n".join(violations)


def test_ser_package_does_not_import_therapy(repo_root: Path) -> None:
    """Plan §2: `therapy` may depend on `ser`; `ser` must never depend back."""
    ser_root = repo_root / "src" / "ser"
    if not ser_root.exists():
        return
    violations: list[str] = []
    for path in sorted(ser_root.rglob("*.py")):
        for line, module in _imports(path):
            if module == "therapy" or module.startswith("therapy."):
                violations.append(
                    f"{path.relative_to(repo_root)}:{line} imports {module}"
                )

    assert not violations, "ser->therapy violations:\n" + "\n".join(violations)


def test_pipecat_observability_adapter_is_the_only_pipecat_telemetry_seam(
    repo_root: Path,
) -> None:
    """Plan §3: only `integrations/pipecat/observability.py` converts Pipecat
    telemetry types; no other module may import both pipecat and
    therapy.observability internals beyond the public model/context API."""
    allowed_public = {
        "therapy.observability",
        "therapy.observability.model",
        "therapy.observability.context",
    }
    violations: list[str] = []
    pipecat_root = repo_root / PIPECAT_ROOT
    if not pipecat_root.exists():
        return
    for path in sorted(pipecat_root.rglob("*.py")):
        if path.name == "observability.py":
            continue
        for line, module in _imports(path):
            if (
                module.startswith("therapy.observability")
                and module not in allowed_public
            ):
                violations.append(
                    f"{path.relative_to(repo_root)}:{line} imports {module}"
                )

    assert not violations, "Pipecat telemetry seam violations:\n" + "\n".join(
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
