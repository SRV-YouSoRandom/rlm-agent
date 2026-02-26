"""
Sandboxed Python REPL for RLM.
The LLM writes Python code to explore document text stored as variables.
Uses RestrictedPython when sandbox_mode=True, plain exec() otherwise.
"""
import time
import logging
import io
import sys
from typing import Any
from core.config import get_settings

logger = logging.getLogger("rlm_agent")

# Safe builtins allowed in restricted mode
SAFE_BUILTINS = {
    "len": len, "range": range, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool,
    "sorted": sorted, "reversed": reversed, "max": max, "min": min,
    "sum": sum, "abs": abs, "round": round, "print": print,
    "isinstance": isinstance, "type": type,
    "True": True, "False": False, "None": None,
}


class REPLEnvironment:
    """
    Stateful REPL environment. The 'context' variable is pre-loaded
    with document text. The LLM writes code to explore it.
    """

    def __init__(self, document_text: str = ""):
        self._globals: dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            "context": document_text,
            "results": [],
        }
        self.execution_history: list[dict] = []

    def load_context(self, text: str):
        """Load or replace document text in REPL environment."""
        self._globals["context"] = text
        logger.info(f"REPL context loaded: {len(text)} chars")

    def execute(self, code: str) -> dict:
        """
        Execute code in the sandboxed environment.
        Returns dict with output, error, success, execution_time_ms.
        """
        settings = get_settings()
        start = time.time()

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        result = {
            "code": code,
            "output": "",
            "error": None,
            "success": False,
            "execution_time_ms": 0.0,
        }

        try:
            if settings.rlm_sandbox_mode:
                self._exec_restricted(code)
            else:
                exec(code, self._globals)  # noqa: S102

            result["output"] = captured.getvalue()
            result["success"] = True

        except TimeoutError:
            result["error"] = f"Execution timed out after {settings.rlm_repl_timeout_seconds}s"
        except Exception as e:
            result["error"] = str(e)
        finally:
            sys.stdout = old_stdout
            result["execution_time_ms"] = (time.time() - start) * 1000
            self.execution_history.append(result)

        return result

    def get_variable(self, name: str) -> Any:
        """Read a variable from the REPL environment."""
        return self._globals.get(name)

    def set_variable(self, name: str, value: Any):
        """Inject a variable into the REPL environment."""
        self._globals[name] = value

    def _exec_restricted(self, code: str):
        """Execute with RestrictedPython if available, else plain exec."""
        try:
            from RestrictedPython import compile_restricted
            from RestrictedPython.Guards import safe_builtins, guarded_getiter, guarded_getattr
            from RestrictedPython.PrintCollector import PrintCollector

            byte_code = compile_restricted(code, "<string>", "exec")

            # PrintCollector handles print() in RestrictedPython
            _print_ = PrintCollector
            _getiter_ = guarded_getiter
            _getattr_ = guarded_getattr

            restricted_globals = {
                "__builtins__": safe_builtins,
                "_print_": PrintCollector,
                "_getiter_": guarded_getiter,
                "_getattr_": guarded_getattr,
                **{k: v for k, v in self._globals.items() if k != "__builtins__"},
            }

            exec(byte_code, restricted_globals)  # noqa: S102

            # Capture printed output via PrintCollector
            if "_print" in restricted_globals:
                printed = str(restricted_globals["_print"])
                if printed:
                    sys.stdout.write(printed)

            # Sync back any new variables set by the code
            for k, v in restricted_globals.items():
                if not k.startswith("_"):
                    self._globals[k] = v

        except ImportError:
            logger.warning("RestrictedPython not installed, falling back to plain exec")
            exec(code, self._globals)  # noqa: S102