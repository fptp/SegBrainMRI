"""Interface commune des moteurs de segmentation de lésions (FLAIR -> masque)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import SimpleITK as sitk


@runtime_checkable
class LesionBackend(Protocol):
    """Prend un FLAIR natif et renvoie un masque binaire ``uint8`` sur la même grille."""

    name: str
    version: str

    def segment(self, flair: sitk.Image) -> sitk.Image: ...
