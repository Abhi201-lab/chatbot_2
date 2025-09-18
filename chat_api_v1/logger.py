import logging
import sys
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    """Create a stdout logger with a consistent format and attach an
    uncaught-exception hook so unhandled exceptions are emitted to the
    same logger.
    """
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    lg.addHandler(h)

    # Install a global excepthook so background/unhandled exceptions are logged
    def _handle_uncaught(exc_type, exc_value, exc_tb):
        # Respect KeyboardInterrupt to allow graceful exits
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        lg.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    try:
        sys.excepthook = _handle_uncaught
    except Exception:
        # If we can't set it, still return the logger
        lg.warning("Failed to set global excepthook")

    return lg
