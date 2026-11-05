"""Opérations géométriques : réorientation et rééchantillonnage."""

from __future__ import annotations

import SimpleITK as sitk


def reorient(image: sitk.Image, orientation: str | None) -> sitk.Image:
    """Réoriente le volume selon un code à 3 lettres (ex. "RAS", "LPS")."""
    if orientation is None:
        return image
    return sitk.DICOMOrient(image, orientation.upper())


def resample_to_spacing(
    image: sitk.Image,
    spacing: tuple[float, float, float] | None,
    *,
    is_label: bool = False,
) -> sitk.Image:
    """Rééchantillonne à l'espacement voulu en conservant l'étendue physique.

    Interpolation B-spline pour les intensités, plus proche voisin pour les masques.
    """
    if spacing is None:
        return image
    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    new_size = [
        max(1, round(sz * osp / nsp))
        for sz, osp, nsp in zip(old_size, old_spacing, spacing, strict=True)
    ]

    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline,
        image.GetOrigin(),
        tuple(spacing),
        image.GetDirection(),
        0,
        image.GetPixelID(),
    )


def resample_like(
    image: sitk.Image, reference: sitk.Image, *, is_label: bool = False
) -> sitk.Image:
    """Rééchantillonne ``image`` sur la grille de ``reference`` (même espace physique supposé).

    Sert à aligner les modalités secondaires (T1...) sur la référence (FLAIR) et à
    ramener le masque prédit dans l'espace natif du patient.
    """
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline,
        0,
        image.GetPixelID(),
    )
