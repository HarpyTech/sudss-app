"""
rca_agent.py – Root Cause Analysis Agent

Given a log message or exception string produced by the application's
structured logger (format: ``TIMESTAMP - LOGGER - LEVEL - [FILE:FUNC:LINE] - MESSAGE``),
this agent:

1. Parses the log entry to extract the source file, function name, and line
   number.
2. Reads the surrounding source-code context from the on-disk file.
3. Sends the log message + code context to Gemini so it can reason about
   what went wrong and suggest a fix.

Usage
-----
The module exposes a single public function::

    from agents.rca_agent import analyze

    result = analyze(log_message="<full log line or multi-line traceback>")
    # result is an RCAResult dict with keys:
    #   parsed_location, code_context, analysis, raw_log

Sample log messages / exceptions that trigger representative code paths:

    # 1. Normal structured log line
    LOG_SAMPLE_INFO = (
        "2026-03-06 12:00:00,000 - app.agents.fetch - INFO"
        " - [fetch.py:summarize:241] - Prepared context for inference."
    )

    # 2. Error log with exception detail
    LOG_SAMPLE_ERROR = (
        "2026-03-06 12:05:00,000 - app.main - ERROR"
        " - [main.py:diagnose:208] - Error during diagnosis:"
        " HTTPException(status_code=500, detail='Model not loaded')"
    )

    # 3. Multi-line Python traceback (e.g. from an unhandled exception)
    LOG_SAMPLE_TRACEBACK = (
        "Traceback (most recent call last):\\n"
        "  File \\"src/app/agents/gemma.py\\", line 239, in infer\\n"
        "    output = PIPE(text=messages, max_new_tokens=max_new_tokens)\\n"
        "RuntimeError: CUDA out of memory."
    )
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import google.generativeai as genai

from logger_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class ParsedLocation(TypedDict):
    filename: str
    function: str
    lineno: int


class RCAResult(TypedDict):
    raw_log: str
    parsed_location: Optional[ParsedLocation]
    code_context: str
    analysis: str


# ---------------------------------------------------------------------------
# Sample log messages / exceptions (for testing and documentation purposes)
# ---------------------------------------------------------------------------

#: A normal INFO-level structured log line.
LOG_SAMPLE_INFO = (
    "2026-03-06 12:00:00,000 - app.agents.fetch - INFO"
    " - [fetch.py:summarize:241] - Prepared context for inference."
)

#: An ERROR log line that includes an exception summary.
LOG_SAMPLE_ERROR = (
    "2026-03-06 12:05:00,000 - app.main - ERROR"
    " - [main.py:diagnose:208] - Error during diagnosis:"
    " HTTPException(status_code=500, detail='Model not loaded')"
)

#: A WARNING about a missing patient record.
LOG_SAMPLE_WARNING = (
    "2026-03-06 12:10:00,000 - app.main - WARNING"
    " - [main.py:diagnose:176] - No patient history found for patient_id='P-9999'."
    " Proceeding without patient context."
)

#: A multi-line Python traceback, the kind written to stderr (or captured by
#: a logging exception handler).
LOG_SAMPLE_TRACEBACK = (
    "Traceback (most recent call last):\n"
    "  File \"src/app/agents/gemma.py\", line 239, in infer\n"
    "    output = PIPE(text=messages, max_new_tokens=max_new_tokens)\n"
    "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB."
)

#: An exception raised inside the retrieval helper.
LOG_SAMPLE_RETRIEVAL_EXCEPTION = (
    "2026-03-06 12:15:00,000 - app.agents.fetch - ERROR"
    " - [fetch.py:retrive_topk_hybrid:147] - Retrieval failed:"
    " ValueError: operands could not be broadcast together with shapes (25000,768) (1,512)"
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Regex for the structured log format:
#   TIMESTAMP - LOGGER_NAME - LEVEL - [FILE:FUNC:LINE] - MESSAGE
_STRUCTURED_RE = re.compile(
    r"\[(?P<filename>[^\]]+\.py):(?P<function>[^\]]+):(?P<lineno>\d+)\]"
)

# Regex for a bare Python traceback line:
#   File "path/to/file.py", line N, in func_name
_TRACEBACK_RE = re.compile(
    r'File "(?P<filepath>[^"]+)", line (?P<lineno>\d+), in (?P<function>\S+)'
)

# How many lines of source context to include around the target line.
_CONTEXT_RADIUS = 15


def _parse_location(log_message: str) -> Optional[ParsedLocation]:
    """Extract the first (file, function, lineno) reference from *log_message*.

    Tries the structured log format first, then falls back to bare Python
    traceback syntax.  Returns *None* if no location can be found.
    """
    # 1 – structured log format
    m = _STRUCTURED_RE.search(log_message)
    if m:
        return ParsedLocation(
            filename=m.group("filename"),
            function=m.group("function"),
            lineno=int(m.group("lineno")),
        )

    # 2 – Python traceback format (use the *last* frame, closest to the error)
    matches = list(_TRACEBACK_RE.finditer(log_message))
    if matches:
        last = matches[-1]
        filepath = last.group("filepath")
        return ParsedLocation(
            filename=Path(filepath).name,
            function=last.group("function"),
            lineno=int(last.group("lineno")),
        )

    return None


def _find_source_file(filename: str, search_roots: List[Path]) -> Optional[Path]:
    """Walk *search_roots* looking for a file whose name matches *filename*.

    Returns the first match, or *None*.
    """
    for root in search_roots:
        for candidate in root.rglob(filename):
            if candidate.is_file():
                return candidate
    return None


def _read_code_context(filepath: Path, lineno: int, radius: int = _CONTEXT_RADIUS) -> str:
    """Return a snippet of *filepath* centred on *lineno* (1-based).

    The returned string includes line numbers so the LLM can orient itself.
    """
    try:
        lines = filepath.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning(
            "Could not read source file '%s': %s",
            filepath,
            exc,
            exc_info=True,
        )
        return f"<source not available: {exc}>"

    total = len(lines)
    start = max(0, lineno - radius - 1)
    end = min(total, lineno + radius)

    snippet_lines: List[str] = []
    for i, line in enumerate(lines[start:end], start=start + 1):
        marker = ">>>" if i == lineno else "   "
        snippet_lines.append(f"{marker} {i:4d} | {line}")

    return "\n".join(snippet_lines)


def _build_prompt(log_message: str, location: Optional[ParsedLocation], code_context: str) -> str:
    loc_summary = (
        f"File: `{location['filename']}`, function `{location['function']}`, line {location['lineno']}"
        if location
        else "Location could not be determined from the log message."
    )
    return textwrap.dedent(f"""
        You are a senior software engineer performing a Root Cause Analysis (RCA).

        ## Log Message / Exception
        ```
        {log_message.strip()}
        ```

        ## Source Location
        {loc_summary}

        ## Surrounding Code Context
        ```python
        {code_context}
        ```

        ## Task
        1. Identify the **root cause** of the error or warning shown in the log.
        2. Explain **why** the code at the indicated location produces this log entry.
        3. Suggest a **concrete fix** (code snippet if possible).
        4. List any **related areas** in the code that should also be reviewed.

        Be concise and precise. Focus on the specific code shown.
    """).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(
    log_message: str,
    source_roots: Optional[List[str]] = None,
) -> RCAResult:
    """Perform root-cause analysis on a log message or exception string.

    Args:
        log_message: A single structured log line, a Python traceback string,
            or any free-form error message.
        source_roots: Directories to search for source files.  Defaults to the
            ``src`` directory relative to this file.

    Returns:
        An :class:`RCAResult` dict with the following keys:

        - ``raw_log`` – the original *log_message* passed in.
        - ``parsed_location`` – parsed file / function / line info, or *None*.
        - ``code_context`` – the source code snippet around the error location.
        - ``analysis`` – the LLM-generated root cause analysis text.
    """
    logger.info(
        "RCA Agent: starting analysis. log_message=%r",
        log_message[:200],
    )

    # -- 1. Parse location --------------------------------------------------
    location = _parse_location(log_message)
    if location:
        logger.info(
            "RCA Agent: parsed location – file=%s function=%s line=%d",
            location["filename"],
            location["function"],
            location["lineno"],
        )
    else:
        logger.warning(
            "RCA Agent: could not parse a source location from the log message."
        )

    # -- 2. Code lookup -----------------------------------------------------
    if source_roots is None:
        # Default: walk from the ``src`` directory that sits two levels up from
        # this file (src/app/agents/rca_agent.py -> src/).
        this_dir = Path(__file__).resolve().parent
        default_root = this_dir.parent.parent  # …/src
        source_roots = [default_root]

    search_paths = [Path(r) for r in source_roots]

    code_context: str
    if location:
        source_file = _find_source_file(location["filename"], search_paths)
        if source_file:
            logger.info(
                "RCA Agent: found source file at '%s'",
                source_file,
            )
            code_context = _read_code_context(source_file, location["lineno"])
        else:
            logger.warning(
                "RCA Agent: source file '%s' not found under search roots %s",
                location["filename"],
                [str(p) for p in search_paths],
            )
            code_context = f"<source file '{location['filename']}' not found>"
    else:
        code_context = "<no location available for code lookup>"

    # -- 3. LLM analysis ----------------------------------------------------
    prompt = _build_prompt(log_message, location, code_context)

    analysis: str
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        analysis = response.text.strip()
        logger.info("RCA Agent: analysis generated successfully.")
    except Exception as exc:
        logger.error(
            "RCA Agent: Gemini call failed: %s",
            exc,
            exc_info=True,
        )
        analysis = (
            f"RCA analysis unavailable – Gemini API call failed: {exc}\n\n"
            f"Parsed location: {location}\n\n"
            f"Code context:\n{code_context}"
        )

    return RCAResult(
        raw_log=log_message,
        parsed_location=location,
        code_context=code_context,
        analysis=analysis,
    )
