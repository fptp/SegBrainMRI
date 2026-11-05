"""Opérations sur les intensités : biais de champ, masque d'avant-plan, normalisation."""

from __future__ import annotations

import logging

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def n4_bias_correction(
    image: sitk.Image, mask: sitk.Image | None = None, shrink: int = 4
) -> sitk.Image:
    """Correction du biais de champ N4 (calculée sur une version sous-échantillonnée)."""
    image = sitk.Cast(image, sitk.sitkFloat32)
    if mask is None:
        mask = sitk.OtsuThreshold(image, 0, 1, 200)
    small = sitk.Shrink(image, [shrink] * image.GetDimension())
    small_mask = sitk.Shrink(mask, [shrink] * image.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(small, small_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(image)
    return image / sitk.Exp(log_bias)


def foreground_mask(image: sitk.Image) -> sitk.Image:
    """Masque binaire grossier de la tête par seuillage d'Otsu + plus grande composante.

    Ce n'est PAS une extraction du cerveau (skull-stripping). Il sert uniquement à
    calculer des statistiques d'intensité robustes en excluant le fond.
    """
    mask = sitk.OtsuThreshold(sitk.Cast(image, sitk.sitkFloat32), 0, 1, 200)
    components = sitk.ConnectedComponent(mask)
    largest = sitk.RelabelComponent(components, sortByObjectSize=True) == 1
    largest = sitk.BinaryMorphologicalClosing(largest, [3, 3, 3])
    return sitk.BinaryFillhole(largest)


def normalize_intensity(
    array: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    method: str = "zscore",
    clip_percentiles: tuple[float, float] | None = (0.5, 99.5),
) -> np.ndarray:
    """Rognage par percentiles puis normalisation, statistiques calculées dans ``mask``.

    Le fond (hors masque) est mis à 0 après normalisation.
    """
    array = array.astype(np.float32, copy=True)
    if mask is not None:
        mask = mask.astype(bool)
        if not mask.any():
            logger.warning("Masque d'avant-plan vide : normalisation sur tout le volume.")
            mask = None
    values = array[mask] if mask is not None else array.ravel()

    if clip_percentiles is not None:
        low, high = np.percentile(values, clip_percentiles)
        array = np.clip(array, low, high)
        values = np.clip(values, low, high)

    if method == "zscore":
        std = float(values.std())
        array = (array - float(values.mean())) / (std if std > 1e-8 else 1.0)
    elif method == "minmax":
        vmin, vmax = float(values.min()), float(values.max())
        array = (array - vmin) / (vmax - vmin if vmax - vmin > 1e-8 else 1.0)
    elif method != "none":
        raise ValueError(f"Méthode de normalisation inconnue : {method!r}")

    array = array.astype(np.float32, copy=False)
    if mask is not None:
        array[~mask] = 0.0
    return array
