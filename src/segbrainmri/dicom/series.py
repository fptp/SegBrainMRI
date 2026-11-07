"""Lecture d'une série DICOM et sélection des séries FLAIR."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset

logger = logging.getLogger(__name__)

#: Valeur de Manufacturer écrite dans nos séries résultats (évite de les retraiter).
OUR_MANUFACTURER = "SegBrainMRI"
DEFAULT_FLAIR_PATTERNS = ("flair", "dark[ _-]?fluid", "t2[ _-]?fl", "spcir")


@dataclass
class DicomSeries:
    """Série chargée : volume SimpleITK + datasets dans l'ordre des coupes du volume."""

    image: sitk.Image
    datasets: list[Dataset]

    @property
    def first(self) -> Dataset:
        return self.datasets[0]


def _text(ds: Dataset, keyword: str) -> str:
    value = ds.get(keyword, "")
    if isinstance(value, pydicom.multival.MultiValue):
        value = " ".join(str(v) for v in value)
    return str(value)


def is_our_output(ds: Dataset) -> bool:
    return _text(ds, "Manufacturer") == OUR_MANUFACTURER


def is_flair(ds: Dataset, patterns: Iterable[str] = DEFAULT_FLAIR_PATTERNS) -> bool:
    """Heuristique : IRM originale dont la description / le protocole évoque une FLAIR."""
    if _text(ds, "Modality") != "MR" or is_our_output(ds):
        return False
    if "DERIVED" in _text(ds, "ImageType").upper():
        return False
    haystack = " ".join(
        _text(ds, kw) for kw in ("SeriesDescription", "ProtocolName", "SequenceName")
    ).lower()
    return any(re.search(p, haystack, flags=re.IGNORECASE) for p in patterns)


def check_slice_spacing(datasets: list[Dataset], tolerance: float = 0.2) -> None:
    """Refuse une série incomplète (coupe manquante) ou à espacement irrégulier.

    Une coupe manquante fausserait la géométrie du volume (et donc du résultat).
    ``tolerance`` : écart relatif maximal entre deux espacements consécutifs.
    """
    orientation = np.asarray(datasets[0].ImageOrientationPatient, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    positions = np.array(
        [np.dot(normal, np.asarray(ds.ImagePositionPatient, dtype=float)) for ds in datasets]
    )
    gaps = np.abs(np.diff(positions))
    median = float(np.median(gaps))
    if median <= 0 or np.any(gaps < 1e-3):
        raise ValueError("Série invalide : plusieurs images à la même position (doublons ?).")
    worst = float(np.max(np.abs(gaps - median)) / median)
    if worst > tolerance:
        missing = int(np.sum(np.round(gaps / median) - 1))
        raise ValueError(
            f"Série incomplète ou irrégulière : espacement entre coupes de {gaps.min():.2f} à "
            f"{gaps.max():.2f} mm (~{max(missing, 0)} coupe(s) manquante(s)). "
            "Renvoyer la série complète."
        )


def read_header(path: str | Path) -> Dataset:
    return pydicom.dcmread(path, stop_before_pixels=True)


def load_series(directory: str | Path) -> DicomSeries:
    """Charge une série mono-trame (une coupe par fichier) en volume 3D.

    L'ordre des coupes est celui de GDCM (position le long de la normale), et la liste
    ``datasets`` suit exactement ce même ordre : la coupe ``k`` du volume (axe Z de
    ``GetArrayFromImage``) correspond à ``datasets[k]``.
    """
    directory = Path(directory)
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory))
    if not series_ids:
        raise ValueError(f"Aucune série DICOM lisible dans {directory}")
    if len(series_ids) > 1:
        raise ValueError(f"Plusieurs séries dans {directory} : {series_ids}")
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), series_ids[0])
    if len(files) < 3:
        raise ValueError(f"Série trop courte ({len(files)} fichier(s)) : volume 3D attendu.")

    datasets = [pydicom.dcmread(f) for f in files]
    if int(datasets[0].get("NumberOfFrames", 1) or 1) > 1:
        raise ValueError(
            "Les séries DICOM multi-trames (Enhanced MR) ne sont pas encore prises "
            "en charge : exporter la série en mono-trame (classic)."
        )
    check_slice_spacing(datasets)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    if image.GetSize()[2] != len(datasets):
        raise ValueError("Incohérence entre le nombre de coupes et de fichiers.")
    logger.info(
        "Série chargée : %s (%d coupes, %s mm)",
        _text(datasets[0], "SeriesDescription") or "?",
        len(datasets),
        tuple(round(s, 2) for s in image.GetSpacing()),
    )
    return DicomSeries(image=image, datasets=datasets)
