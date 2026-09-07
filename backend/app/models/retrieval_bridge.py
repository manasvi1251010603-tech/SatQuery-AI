from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

RETRIEVAL_PYTHON = (
    PROJECT_ROOT
    / ".venv-retrieval"
    / "bin"
    / "python"
)

RETRIEVAL_MODULE = (
    "backend.app.models.semantic_retrieval"
)


# ============================================================
# HELPERS
# ============================================================

def _run_python(
    code: str,
) -> dict[str, Any]:

    if not RETRIEVAL_PYTHON.exists():

        raise FileNotFoundError(
            "Retrieval environment not found: "
            f"{RETRIEVAL_PYTHON}"
        )

    process = subprocess.run(
        [
            str(RETRIEVAL_PYTHON),
            "-c",
            code,
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )

    if process.returncode != 0:

        stdout = process.stdout.strip()
        stderr = process.stderr.strip()

        raise RuntimeError(
            "Semantic Retrieval process failed.\n"
            f"stdout:\n{stdout}\n\n"
            f"stderr:\n{stderr}"
        )

    output = process.stdout.strip()

    if not output:

        raise RuntimeError(
            "Semantic Retrieval returned no output."
        )

    # The retrieval module prints status messages.
    # The final line is reserved for JSON.
    lines = output.splitlines()

    json_line = None

    for line in reversed(lines):

        line = line.strip()

        if line.startswith("{"):

            json_line = line
            break

        if line.startswith("["):

            json_line = line
            break

    if json_line is None:

        raise RuntimeError(
            "Could not parse Semantic Retrieval output:\n"
            f"{output}"
        )

    try:

        return json.loads(
            json_line
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Invalid JSON returned by Semantic Retrieval:\n"
            f"{json_line}"
        ) from exc


# ============================================================
# RETRIEVE
# ============================================================

def run_semantic_retrieval(
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Execute RemoteCLIP semantic retrieval inside the
    dedicated retrieval environment.

    Main backend:
        FastAPI / main .venv
                ↓
        retrieval bridge
                ↓
        .venv-retrieval
                ↓
        RemoteCLIP
                ↓
        NumPy cosine similarity
    """

    if not query or not query.strip():

        raise ValueError(
            "Semantic retrieval query cannot be empty."
        )

    code = f"""
import json
from backend.app.models.semantic_retrieval import retrieve

result = retrieve(
    query={query!r},
    top_k={int(top_k)},
)

print(json.dumps(result))
"""

    results = _run_python(
        code
    )

    return {
        "query": query,
        "top_k": top_k,
        "results": results,
        "model": "RemoteCLIP ViT-B-32",
        "search_backend": "NumPy cosine similarity",
    }


# ============================================================
# STATUS
# ============================================================

def get_semantic_retrieval_status() -> dict[str, Any]:
    """
    Get retrieval model/index status.
    """

    code = """
import json
from backend.app.models.semantic_retrieval import retrieval_status

print(json.dumps(retrieval_status()))
"""

    return _run_python(
        code
    )
