"""Modèles de segmentation et registre associé."""

from segbrainmri.models.base import SegmentationModel
from segbrainmri.models.registry import available_models, build_model, register_model
from segbrainmri.models.unet3d import UNet3D  # noqa: F401  (enregistre "unet3d")

__all__ = ["SegmentationModel", "UNet3D", "available_models", "build_model", "register_model"]
