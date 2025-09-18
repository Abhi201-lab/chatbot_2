import logging
import sys


def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    lg.addHandler(h)

    def _handle_uncaught(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        lg.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    try:
        sys.excepthook = _handle_uncaught
    except Exception:
        lg.warning("Failed to set global excepthook")

    return lg
