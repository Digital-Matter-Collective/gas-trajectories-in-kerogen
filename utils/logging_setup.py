import logging


def setup_logging(verbosity: int = 0) -> None:
    """Configure the root logger once, from a CLI script's main().

    verbosity < 0: WARNING and above; 0 (default): INFO; > 0: DEBUG.
    """
    level = logging.INFO
    if verbosity < 0:
        level = logging.WARNING
    elif verbosity > 0:
        level = logging.DEBUG
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
