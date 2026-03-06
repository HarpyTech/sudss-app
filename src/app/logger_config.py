"""
Centralized logging configuration for the Clinical Diagnosis AI Suite.

Provides a consistent log format that includes file, function, and line number
so that the RCA Agent can perform precise code-level lookups from log messages.
"""

import logging
import sys

LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s"
    " - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s"
)

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once per process.

    Call this at application startup.  Subsequent calls are no-ops so that the
    configuration is not duplicated when the module is imported multiple times.
    """
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any handlers that may have been added automatically before our
    # explicit configuration so we don't get duplicate entries.
    root.handlers.clear()
    root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger using the shared configuration.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
