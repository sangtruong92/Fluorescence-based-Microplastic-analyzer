"""
Centralized Logging Utility — fbma namespace

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.debug("Detailed diagnostic info")
    logger.info("Normal operational message")
    logger.warning("Something unexpected but handled")
    logger.error("Something failed", exc_info=True)

Configuration (via environment variables):
    LOG_LEVEL : Logging level name (DEBUG, INFO, WARNING, ERROR). Default: INFO.
    LOG_FILE  : Optional file path to write log file alongside console output.

Runtime control:
    from src.utils.logger import set_level
    set_level("DEBUG")   # Enable verbose debug logging at runtime
    set_level("INFO")    # Revert to standard operational level
"""

import logging
import os
import sys

# ── Root logger namespace ───────────────────────────────────────────────────
_ROOT_LOGGER_NAME = "fbma"

# ── Internal flag to ensure handler initialization runs only once ──────────
_configured = False


def _configure_root_logger() -> None:
    """Configure handlers and formatters for the root fbma logger."""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)  # Capture all logs; handlers filter by level

    # Avoid adding duplicate handlers if module is reloaded
    if root.handlers:
        return

    # ── Console Handler ──────────────────────────────────────────────────────
    console_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    console_level = getattr(logging, console_level_name, logging.INFO)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(console_handler)

    # ── Optional File Handler ────────────────────────────────────────────────
    log_file = os.environ.get("LOG_FILE", "").strip()
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)  # Always log debug level to file
            file_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root.addHandler(file_handler)
        except OSError as exc:
            # Fallback gracefully if log file destination is unwritable
            root.warning("Could not open LOG_FILE %r: %s", log_file, exc)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance prefixed with the 'fbma' root namespace.

    Args:
        name: Caller module identifier (typically __name__).
              If name starts with 'fbma.', it is used as-is;
              otherwise it is prefixed with 'fbma.'.

    Returns:
        logging.Logger instance.

    Example:
        logger = get_logger(__name__)
        # In src/core/image_processing.py -> fbma.src.core.image_processing
    """
    _configure_root_logger()

    # Normalize name to ensure hierarchy stays inside 'fbma' namespace
    if name == "__main__" or not name:
        full_name = _ROOT_LOGGER_NAME
    elif name.startswith(_ROOT_LOGGER_NAME + "."):
        full_name = name
    else:
        full_name = f"{_ROOT_LOGGER_NAME}.{name}"

    return logging.getLogger(full_name)


def set_level(level: str) -> None:
    """
    Dynamically update the console handler logging level at runtime.

    Args:
        level: Log level name ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Example:
        from src.utils.logger import set_level
        set_level('DEBUG')   # Enable verbose output for troubleshooting
        set_level('INFO')    # Silence debug messages
    """
    _configure_root_logger()
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level!r}")

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stderr:
            handler.setLevel(numeric_level)
            break
