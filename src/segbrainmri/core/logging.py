"""Configuration centralisée du logging."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(verbosity: int = 0) -> None:
    """Configure le logger racine du paquet.

    verbosity : 0 -> INFO, >=1 -> DEBUG (>=2 : aussi les bibliothèques), <0 -> WARNING.
    """
    if verbosity < 0:
        level = logging.WARNING
    elif verbosity == 0:
        level = logging.INFO
    else:
        level = logging.DEBUG
    logging.basicConfig(level=level, format=_FORMAT, datefmt="%H:%M:%S", force=True)
    logging.getLogger("segbrainmri").setLevel(level)
    # Bibliothèques très bavardes en INFO : seulement en mode debug
    for noisy in ("pynetdicom", "highdicom", "numexpr", "nnunetv2"):
        logging.getLogger(noisy).setLevel(level if verbosity >= 2 else logging.WARNING)
