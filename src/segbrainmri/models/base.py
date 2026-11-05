"""Interface commune des modèles de segmentation.

Tout backend (UNet PyTorch maison, MONAI bundle, nnU-Net, LST-AI...) doit exposer
cette interface pour être utilisable par :mod:`segbrainmri.inference`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


class SegmentationModel(nn.Module, ABC):
    """Réseau ``(N, C_in, Z, Y, X) -> logits (N, C_out, Z, Y, X)``."""

    #: Modalités attendues en entrée, dans l'ordre des canaux.
    input_modalities: tuple[str, ...] = ("flair",)
    #: Nombre de classes en sortie (1 = binaire lésion / fond, activation sigmoïde).
    num_classes: int = 1

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def load_weights(self, path: str | Path, device: torch.device) -> None:
        """Charge un ``state_dict`` (ou un checkpoint ``{"state_dict": ...}``).

        ``map_location`` garantit qu'un checkpoint sauvegardé sur GPU se charge sur CPU.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Poids introuvables : {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.load_state_dict(state_dict)
        logger.info("Poids chargés depuis %s", path)
