import logging, sys

def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers: return log
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    log.addHandler(h)
    return log
