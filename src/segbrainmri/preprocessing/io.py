"""Lecture / écriture de volumes NIfTI (.nii, .nii.gz)."""

from __future__ import annotations

import logging
from pathlib import Path

import SimpleITK as sitk

logger = logging.getLogger(__name__)

NIFTI_SUFFIXES = (".nii", ".nii.gz")


def is_nifti(path: str | Path) -> bool:
    return str(path).lower().endswith(NIFTI_SUFFIXES)


def load_nifti(path: str | Path, pixel_type: int = sitk.sitkFloat32) -> sitk.Image:
    """Charge un volume NIfTI 3D en conservant sa géométrie (origine, espacement, direction)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    if not is_nifti(path):
        raise ValueError(f"Format non supporté (attendu .nii ou .nii.gz) : {path.name}")

    image = sitk.ReadImage(str(path), pixel_type)
    if image.GetDimension() == 4 and image.GetSize()[3] == 1:
        # Certains convertisseurs DICOM->NIfTI produisent un 4D à un seul volume
        image = image[:, :, :, 0]
    if image.GetDimension() != 3:
        raise ValueError(f"Volume 3D attendu, reçu {image.GetDimension()}D : {path.name}")

    logger.debug(
        "Chargé %s : taille=%s, espacement=%s mm",
        path.name,
        image.GetSize(),
        tuple(round(s, 3) for s in image.GetSpacing()),
    )
    return image


def save_nifti(image: sitk.Image, path: str | Path) -> Path:
    """Écrit ``image`` en NIfTI (compression si le suffixe est .nii.gz)."""
    path = Path(path)
    if not is_nifti(path):
        raise ValueError(f"Le fichier de sortie doit finir par .nii ou .nii.gz : {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), useCompression=str(path).endswith(".gz"))
    logger.debug("Écrit %s", path)
    return path
