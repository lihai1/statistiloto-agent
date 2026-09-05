"""Code editor tools — allow the admin agent to read and edit its own source.

Paths are always resolved relative to the configured code root (default /app).
Attempts to escape outside the code root are rejected.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from app.tools import ToolError

log = logging.getLogger(__name__)


def _get_code_root() -> Path:
    """Read the code root from the environment on every call (allows tests to override)."""
    return Path(os.environ.get("AGENT_CODE_ROOT", "/app")).resolve()


def _resolve(file_path: str | None) -> Path:
    """Resolve a requested path under the configured code root."""
    if not file_path:
        raise ValueError("file_path is required")
    code_root = _get_code_root()
    target = (code_root / file_path).resolve()
    if not target.is_relative_to(code_root):
        raise ValueError(f"Path {file_path} is outside the allowed code root")
    return target


def read_code(file_path: str) -> dict:
    """Read a source file and return its contents."""
    try:
        target = _resolve(file_path)
        if not target.exists():
            raise ToolError(f"File not found: {file_path}")
        if target.is_dir():
            raise ToolError(f"Path is a directory: {file_path}")
        # Refuse binary or very large files (> 1 MB).
        size = target.stat().st_size
        if size > 1_000_000:
            raise ToolError(f"File too large ({size} bytes)")
        content = target.read_text(encoding="utf-8")
        return {
            "path": str(target),
            "size": size,
            "lines": content.count("\n") + 1,
            "content": content,
        }
    except ToolError:
        raise
    except ValueError as e:
        raise ToolError(str(e))
    except Exception as e:
        log.warning("[read_code] failed: %s", e)
        raise ToolError(str(e))


def list_files(directory: str | None = None, max_results: int = 100) -> dict:
    """List files under a directory (relative to the configured code root)."""
    try:
        code_root = _get_code_root()
        target = _resolve(directory or ".")
        if not target.exists():
            raise ToolError(f"Directory not found: {directory}")
        if not target.is_dir():
            raise ToolError(f"Path is not a directory: {directory}")

        files = []
        for p in target.rglob("*"):
            if p.is_file():
                rel = p.relative_to(code_root).as_posix()
                files.append(rel)
            if len(files) >= max_results:
                break
        return {"directory": str(target.relative_to(code_root)), "files": files}
    except ToolError:
        raise
    except ValueError as e:
        raise ToolError(str(e))
    except Exception as e:
        log.warning("[list_files] failed: %s", e)
        raise ToolError(str(e))


def edit_file(file_path: str, old_string: str | None = None,
              new_string: str | None = None, content: str | None = None) -> dict:
    """Edit a source file.

    Two modes:
      1. Full overwrite: pass `content`.
      2. Surgical edit: pass `old_string` and `new_string`.

    HITL must approve any edit before this function is called.
    """
    if old_string == "":
        old_string = None
    if new_string == "":
        new_string = None
    if content is None and (old_string is None or new_string is None):
        raise ToolError("edit_file requires either 'content' or both 'old_string' and 'new_string'")

    try:
        target = _resolve(file_path)
        if not target.exists():
            raise ToolError(f"File not found: {file_path}")
        if not target.is_file():
            raise ToolError(f"Path is not a file: {file_path}")

        if content is not None:
            target.write_text(content, encoding="utf-8")
            return {"status": "edited", "path": str(target), "mode": "overwrite"}

        # Surgical replacement.
        text = target.read_text(encoding="utf-8")
        occurrences = text.count(old_string)
        if occurrences == 0:
            raise ToolError(f"old_string not found in {file_path}")
        if occurrences > 1:
            raise ToolError(f"old_string is not unique in {file_path} ({occurrences} matches)")
        new_text = text.replace(old_string, new_string, 1)
        target.write_text(new_text, encoding="utf-8")
        return {
            "status": "edited",
            "path": str(target),
            "mode": "replace",
            "old_string": old_string,
            "new_string": new_string,
        }
    except ToolError:
        raise
    except ValueError as e:
        raise ToolError(str(e))
    except Exception as e:
        log.warning("[edit_file] failed: %s", e)
        raise ToolError(str(e))
