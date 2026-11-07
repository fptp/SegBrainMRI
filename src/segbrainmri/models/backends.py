"""Interface commune des moteurs de segmentation de lésions (FLAIR -> masque)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import SimpleITK as sitk


@dataclass
class Analysis:
    """Masque des lésions + cerveau extrait (pour le recalage longitudinal), même grille."""

    mask: sitk.Image
    brain: sitk.Image | None = None


def analyze(backend, flair: sitk.Image) -> Analysis:
    """Utilise ``backend.analyze`` s'il existe, sinon ``segment`` (sans cerveau extrait)."""
    if hasattr(backend, "analyze"):
        return backend.analyze(flair)
    return Analysis(mask=backend.segment(flair))


@runtime_checkable
class LesionBackend(Protocol):
    """Prend un FLAIR natif et renvoie un masque binaire ``uint8`` sur la même grille."""

    name: str
    version: str

    def segment(self, flair: sitk.Image) -> sitk.Image: ...
