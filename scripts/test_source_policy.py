"""Source-policy checks for the Python tree."""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from importlib.util import resolve_name
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
NEW_SOURCE_ROOT = SOURCE_ROOT / "entlearn"
HELPER_SOURCE_ROOT = NEW_SOURCE_ROOT / "helpers"
ADAPTER_SOURCE_ROOT = NEW_SOURCE_ROOT / "scikit_adapter"
PLOTTING_SOURCE_ROOT = NEW_SOURCE_ROOT / "plotting"
DISPATCH_SOURCES = [NEW_SOURCE_ROOT / "network" / "dispatch.py", NEW_SOURCE_ROOT / "_warnings.py"]
DISPLAY_SOURCE = NEW_SOURCE_ROOT / "network" / "display.py"
PERSISTENCE_SOURCE_ROOT = NEW_SOURCE_ROOT / "network" / "persistence"
# A decoded description validates its own scalar domains. These calls validate the rest:
# block names and connection endpoints, and the prediction policy against its head.
DESCRIPTION_OWNERS = {"Recipe", "_resolve_predict_config", "validate_original_state"}
REQUIRED_HELPER_DEPENDENCIES = {"numpy"}
OPTIONAL_HELPER_DEPENDENCIES = {"joblib", "scipy", "sklearn"}
ADAPTER_DEPENDENCIES = {"narwhals", "numpy", "pandas", "sklearn"}

PROHIBITED_PROSE = (
    re.compile(r"(?:#\d+|\bissue\s+#?\d+\b)", re.IGNORECASE),
    re.compile(r"\b(?:design|implementation) plans?\b", re.IGNORECASE),
    re.compile(r"\b(?:PRs?\s+#?\d+|pull requests?)\b", re.IGNORECASE),
    re.compile(r"\bhistorical implementation\b", re.IGNORECASE),
    re.compile(r"\bconversations?\b", re.IGNORECASE),
)

ALIASING_TENSOR_METHODS = {
    "as_strided",
    "contiguous",
    "detach",
    "diagonal",
    "expand",
    "expand_as",
    "flatten",
    "movedim",
    "narrow",
    "permute",
    "reshape",
    "select",
    "squeeze",
    "swapaxes",
    "swapdims",
    "t",
    "transpose",
    "unfold",
    "unsqueeze",
    "view",
}


def _relative(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def _source_files() -> list[Path]:
    return sorted(SOURCE_ROOT.rglob("*.py"))


def _imported_roots(tree: ast.AST) -> set[str]:
    """Return the root names imported anywhere in a syntax tree."""
    imported = {
        alias.name.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported |= {
        node.module.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    return imported


def _function_local_nodes(tree: ast.AST) -> set[ast.AST]:
    """Return every node nested within a function."""
    return {
        nested
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        for nested in ast.walk(function)
    }


def _dependency_violations(
    paths: list[Path], allowed: set[str], *, type_only: frozenset[str] = frozenset()
) -> set[str]:
    """Return source files that import a dependency outside the allowed set."""
    violations = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        type_checking_nodes = {
            nested
            for guard in ast.walk(tree)
            if isinstance(guard, ast.If)
            and isinstance(guard.test, ast.Name)
            and guard.test.id == "TYPE_CHECKING"
            for statement in guard.body
            for nested in ast.walk(statement)
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                permitted = allowed | type_only if node in type_checking_nodes else allowed
                if _imported_roots(node) - permitted:
                    violations.add(_relative(path))
    return violations


def _adapter_boundary_violations(
    source: str, local_methods: set[str], package: str = "entlearn.scikit_adapter"
) -> list[str]:
    """Reject core implementation imports and private attribute access in adapters.

    Each package may import its own modules, the helpers and the primitives.
    """
    own = ".".join(package.split(".")[:2]) + "."
    violations = []
    tree = ast.parse(source)
    local_fields = {
        field.target.id
        for definition in ast.walk(tree)
        if isinstance(definition, ast.ClassDef)
        for field in definition.body
        if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = resolve_name("." * node.level + (node.module or ""), package)
            names = [alias.name for alias in node.names]
            if module == "entlearn._seeds" and names == ["_SEED_UPPER_BOUND"]:
                continue  # Shared scalar validation, not numerical implementation.
            if module == "entlearn.network.config" and names == ["_resolve_predict_config"]:
                continue  # The task default's one owner, not numerical implementation.
            if module == "entlearn.network.validation" and names == ["_stage_sample_weights"]:
                continue  # The sample-weight rule's one owner, not numerical implementation.
            if module == "entlearn._warnings" and names == ["_warn"]:
                continue  # Warning attribution, not numerical implementation.
            if module == "entlearn.network.queries" and names == ["_scores_available"]:
                continue  # Score availability's one owner, not numerical implementation.
            if module in ("entlearn", "entlearn.helpers"):
                if any(name.startswith("_") or name == "*" for name in names):
                    violations.append(ast.unparse(node))
                continue
            modules = [module]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                continue
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr in local_methods | local_fields
            ):
                continue
            violations.append(ast.unparse(node))
            continue
        else:
            continue
        for module in modules:
            if module.startswith("entlearn.") and not module.startswith(
                (own, "entlearn.helpers.", "entlearn.primitives.")
            ):
                violations.append(ast.unparse(node))
    return violations


def _tensor_parameter_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    arguments = (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
    return {
        argument.arg
        for argument in arguments
        if argument.annotation is not None and "Tensor" in ast.unparse(argument.annotation)
    }


def _tensor_argument_roots(expression: ast.AST, aliases: dict[str, str]) -> set[str]:
    if isinstance(expression, ast.Name):
        return {aliases[expression.id]} if expression.id in aliases else set()
    if isinstance(expression, (ast.Attribute, ast.Subscript)):
        return _tensor_argument_roots(expression.value, aliases)
    if isinstance(expression, (ast.List, ast.Tuple)):
        return set().union(
            *(_tensor_argument_roots(item, aliases) for item in expression.elts), set()
        )
    if isinstance(expression, ast.Call):
        out_roots = set().union(
            *(
                _tensor_argument_roots(keyword.value, aliases)
                for keyword in expression.keywords
                if keyword.arg == "out"
            ),
            set(),
        )
        if out_roots:
            return out_roots
        if (
            isinstance(expression.func, ast.Attribute)
            and expression.func.attr in ALIASING_TENSOR_METHODS
        ):
            return _tensor_argument_roots(expression.func.value, aliases)
    return set()


def _mutates_tensor_argument(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    tensor_parameters = _tensor_parameter_names(function)
    aliases = {name: name for name in tensor_parameters}
    nodes = list(ast.walk(function))

    changed = True
    while changed:
        changed = False
        for node in nodes:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is None:
                continue
            roots = _tensor_argument_roots(value, aliases)
            if len(roots) != 1:
                continue
            root = roots.pop()
            for target in targets:
                if isinstance(target, ast.Name) and aliases.get(target.id) != root:
                    aliases[target.id] = root
                    changed = True

    for node in nodes:
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr.endswith("_")
                and _tensor_argument_roots(node.func.value, aliases)
            ):
                return True
            if (
                isinstance(node.func, ast.Name)
                and node.func.id.endswith("_")
                and node.args
                and _tensor_argument_roots(node.args[0], aliases)
            ):
                return True
            if any(
                keyword.arg == "out" and _tensor_argument_roots(keyword.value, aliases)
                for keyword in node.keywords
            ):
                return True
        if isinstance(node, ast.AugAssign) and _tensor_argument_roots(node.target, aliases):
            return True
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Subscript) and _tensor_argument_roots(target, aliases)
            for target in node.targets
        ):
            return True
    return False


def _called_names(function: ast.AST) -> set[str]:
    return {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    }


def _unvalidated_description_callers(sources: list[str]) -> list[str]:
    """Return the outermost callers that receive a decoded description no owner validated.

    The search climbs from ``decode_description`` through its callers and stops at any
    function that calls one of ``DESCRIPTION_OWNERS``.
    """
    calls = {
        node.name: _called_names(node)
        for source in sources
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
    }
    escapes, pending, seen = [], ["decode_description"], {"decode_description"}
    while pending:
        decoder = pending.pop()
        callers = [name for name, called in calls.items() if decoder in called]
        if not callers:
            escapes.append(decoder)
        for caller in callers:
            if caller not in seen and not calls[caller] & DESCRIPTION_OWNERS:
                seen.add(caller)
                pending.append(caller)
    return sorted(escapes)


def _docstrings(tree: ast.AST) -> list[str]:
    prose = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node, clean=False)
            if docstring is not None:
                prose.append(docstring)
    return prose


def _comments(source: str) -> list[str]:
    return [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    ]


class TestSourcePolicy:
    def test_collected_tests_are_grouped_in_classes(self) -> None:
        violations = []
        for path in sorted((REPOSITORY_ROOT / "tests").rglob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            violations.extend(
                f"{_relative(path)}:{node.lineno}: {node.name}"
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            )
        assert not violations, "\n".join(violations)

    def test_tensor_mutators_have_trailing_underscores(self) -> None:
        violations = set()
        for path in _source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if any(
                _mutates_tensor_argument(node) and not node.name.endswith("_")
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ):
                violations.add(_relative(path))
        assert not violations, "\n".join(sorted(violations))

    def test_entlearn_core_uses_only_core_dependencies(self) -> None:
        paths = sorted(
            path
            for path in NEW_SOURCE_ROOT.rglob("*.py")
            if not path.is_relative_to(HELPER_SOURCE_ROOT)
            and not path.is_relative_to(ADAPTER_SOURCE_ROOT)
            and not path.is_relative_to(PLOTTING_SOURCE_ROOT)
            and path not in DISPATCH_SOURCES
            and path != DISPLAY_SOURCE
        )
        assert paths, "src/entlearn contains no Python files"
        allowed = set(sys.stdlib_module_names) | {"entlearn", "torch", "safetensors"}
        violations = _dependency_violations(paths, allowed, type_only=frozenset({"plotly", "rich"}))
        # Network's console hook loads the optional renderer, not the numerical core.
        display_allowed = set(sys.stdlib_module_names) | {"entlearn", "rich"}
        violations |= _dependency_violations([DISPLAY_SOURCE], display_allowed)
        assert not violations, sorted(violations)

    def test_plotting_uses_only_declared_optional_dependencies(self) -> None:
        paths = sorted(PLOTTING_SOURCE_ROOT.rglob("*.py"))
        allowed = set(sys.stdlib_module_names) | {"entlearn", "torch", "numpy", "plotly"}
        assert not _dependency_violations(paths, allowed)

    def test_generic_dispatch_uses_only_declared_dependencies(self) -> None:
        allowed = set(sys.stdlib_module_names) | {"entlearn", "joblib"}
        assert not _dependency_violations(DISPATCH_SOURCES, allowed)

    def test_optional_helpers_use_only_declared_dependencies(self) -> None:
        paths = sorted(HELPER_SOURCE_ROOT.rglob("*.py"))
        assert paths, f"{_relative(HELPER_SOURCE_ROOT)} contains no Python files"
        allowed = (
            set(sys.stdlib_module_names)
            | {"entlearn", "torch"}
            | REQUIRED_HELPER_DEPENDENCIES
            | OPTIONAL_HELPER_DEPENDENCIES
        )
        assert not _dependency_violations(paths, allowed)

    def test_optional_dependencies_are_call_time_only(self) -> None:
        violations = set()
        helper_paths = sorted(HELPER_SOURCE_ROOT.rglob("*.py"))
        assert helper_paths, f"{_relative(HELPER_SOURCE_ROOT)} contains no Python files"
        paths = [*helper_paths, *DISPATCH_SOURCES]
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            function_local = _function_local_nodes(tree)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                imported = _imported_roots(node) & OPTIONAL_HELPER_DEPENDENCIES
                if imported and node not in function_local:
                    violations.add(_relative(path))
        assert not violations, sorted(violations)

    def test_adapter_uses_only_declared_dependencies(self) -> None:
        paths = sorted(ADAPTER_SOURCE_ROOT.rglob("*.py"))
        assert paths, f"{_relative(ADAPTER_SOURCE_ROOT)} contains no Python files"
        allowed = set(sys.stdlib_module_names) | {"entlearn", "torch"} | ADAPTER_DEPENDENCIES
        assert not _dependency_violations(paths, allowed)

    def test_adapter_uses_public_core_operations_and_state(self) -> None:
        sources = {
            path: path.read_text(encoding="utf-8")
            for path in [
                *ADAPTER_SOURCE_ROOT.rglob("*.py"),
                HELPER_SOURCE_ROOT / "reporting.py",
                *PLOTTING_SOURCE_ROOT.rglob("*.py"),
            ]
        }
        local_methods = {
            method.name
            for source in sources.values()
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ClassDef)
            for method in node.body
            if isinstance(method, ast.FunctionDef)
        }
        violations = {
            _relative(path): errors
            for path, source in sources.items()
            if (
                errors := _adapter_boundary_violations(
                    source, local_methods, ".".join(path.relative_to(SOURCE_ROOT).parent.parts)
                )
            )
        }
        assert not violations, violations

    def test_source_prose_has_no_prohibited_references(self) -> None:
        violations = set()
        for path in _source_files():
            source = path.read_text(encoding="utf-8")
            prose = (*_docstrings(ast.parse(source, filename=str(path))), *_comments(source))
            if any(pattern.search(item) for pattern in PROHIBITED_PROSE for item in prose):
                violations.add(_relative(path))
        assert not violations, sorted(violations)

    def test_decoded_descriptions_reach_a_validating_owner(self) -> None:
        sources = [
            path.read_text(encoding="utf-8") for path in PERSISTENCE_SOURCE_ROOT.glob("*.py")
        ]
        assert sources, f"{_relative(PERSISTENCE_SOURCE_ROOT)} contains no Python files"
        assert not _unvalidated_description_callers(sources)


class TestDescriptionOwnerDetection:
    def test_a_description_that_bypasses_every_owner_is_reported(self) -> None:
        decoders = (
            "def decode_block(value):\n    return decode_description(value, Input)\n"
            "def decode_recipe(value):\n    return Recipe(blocks=(decode_block(value),))\n"
        )
        assert not _unvalidated_description_callers([decoders])
        escaping = decoders + "def load(value):\n    return decode_block(value)\n"
        assert _unvalidated_description_callers([escaping]) == ["load"]


class TestAdapterBoundaryDetection:
    def test_declared_adapter_state_does_not_permit_private_network_access(self) -> None:
        declaration = "class Adapter:\n    _target_was_vector: bool\n"
        assert not _adapter_boundary_violations(
            declaration + "    def predict(self):\n        return self._target_was_vector\n",
            set(),
        )
        for access in ("self.network_._target_was_vector", "network._target_was_vector"):
            assert _adapter_boundary_violations(
                declaration + f"    def predict(self):\n        return {access}\n", set()
            )

    def test_private_imports_and_attributes_are_rejected(self) -> None:
        for source in (
            "from entlearn.network.model import Network",
            "import entlearn.network.fit as solver",
            "from entlearn import _warnings",
            "from entlearn.network.dispatch import _map_seeds",
            "from entlearn.helpers import _hidden",
            "from ..network.fit import fit",
            "from .. import _seeds",
            "network._graph",
            "self.network_._fitted",
            "alias._publish()",
        ):
            assert _adapter_boundary_violations(source, set()), source

    def test_each_package_imports_only_its_own_modules(self) -> None:
        plotting = "from entlearn.plotting.network import plot_network"
        assert _adapter_boundary_violations(plotting, set())
        assert not _adapter_boundary_violations(plotting, set(), "entlearn.plotting")
        adapter = "from entlearn.scikit_adapter.base import _BaseEON"
        assert _adapter_boundary_violations(adapter, set(), "entlearn.plotting")

    def test_public_operations_and_shared_scalar_validation_are_allowed(self) -> None:
        source = (
            "from entlearn import Network\n"
            "from entlearn._seeds import _SEED_UPPER_BOUND\n"
            "from entlearn.network.config import _resolve_predict_config\n"
            "network = Network.fit(recipe, X, y)\n"
            "state = network.initial_state\n"
            "probabilities = network.predict(X)\n"
            "self._prepare_prediction_features(X)\n"
        )
        assert not _adapter_boundary_violations(source, {"_prepare_prediction_features"})


class TestMutatorDetection:
    def test_mutator_detection_distinguishes_tensor_views_from_copies(self) -> None:
        copied = ast.parse(
            "def copied(source: torch.Tensor):\n"
            "    result = source.clone()\n"
            "    result.add_(1)\n"
            "    return result\n"
        ).body[0]
        viewed = ast.parse(
            "def viewed(source: torch.Tensor):\n    result = source.view(-1)\n    result.add_(1)\n"
        ).body[0]
        wrapped = ast.parse(
            "def wrapped(source: torch.Tensor, scratch: torch.Tensor):\n"
            "    normalise_(source, 0, scratch)\n"
        ).body[0]
        copied_method = ast.parse(
            "def copied_method(source: torch.Tensor):\n"
            "    result = source.clone()\n"
            "    result.addmm_(source, source)\n"
            "    return result\n"
        ).body[0]
        copied_helper = ast.parse(
            "def copied_helper(source: torch.Tensor):\n"
            "    result = torch.empty_like(source)\n"
            "    floored_log_(result, source)\n"
            "    return result\n"
        ).body[0]

        assert isinstance(copied, ast.FunctionDef)
        assert isinstance(viewed, ast.FunctionDef)
        assert isinstance(wrapped, ast.FunctionDef)
        assert isinstance(copied_method, ast.FunctionDef)
        assert isinstance(copied_helper, ast.FunctionDef)
        assert not _mutates_tensor_argument(copied)
        assert _mutates_tensor_argument(viewed)
        assert _mutates_tensor_argument(wrapped)
        assert not _mutates_tensor_argument(copied_method)
        assert not _mutates_tensor_argument(copied_helper)
