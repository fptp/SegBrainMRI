"""Chaîne complète : T1 3D DICOM -> FastSurfer -> volumes, normes -> SEG + SR."""

from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from pydicom.dataset import Dataset

from segbrainmri.dicom.series import DicomSeries, _text, is_our_output
from segbrainmri.dicom.writers import ResultInfo
from segbrainmri.volumetry.analysis import analyze_volumes, patient_age, patient_sex
from segbrainmri.volumetry.fastsurfer import FastSurfer
from segbrainmri.volumetry.normative import NormativeLibrary
from segbrainmri.volumetry.outputs import (
    build_volumetry_report,
    build_volumetry_seg,
    native_labelmap,
)

logger = logging.getLogger(__name__)

DEFAULT_T1_PATTERNS = (
    "bravo",
    "mprage",
    "mp-rage",
    "spgr",
    "t1[ _-]?3d",
    "3d[ _-]?t1",
    "t1w[ _-]?3d",
    "3d[ _-]?tfe",
    "t1[ _-]?tfe",
)
#: T1 injectées (gadolinium) : FastSurfer est entraîné sur des T1 sans injection
DEFAULT_T1_EXCLUDE = ("gado", r"\bgd\b", r"\bpost\b", r"\+ ?c\b", "contrast", r"\binj")


def is_t1_3d(
    ds: Dataset,
    patterns: Iterable[str] = DEFAULT_T1_PATTERNS,
    exclude: Iterable[str] = DEFAULT_T1_EXCLUDE,
) -> bool:
    """T1 3D originale, non injectée, d'après la description / le protocole."""
    if _text(ds, "Modality") != "MR" or is_our_output(ds):
        return False
    if "DERIVED" in _text(ds, "ImageType").upper():
        return False
    acquisition = _text(ds, "MRAcquisitionType").upper()
    if acquisition and acquisition != "3D":
        return False
    text = " ".join(_text(ds, kw) for kw in ("SeriesDescription", "ProtocolName", "SequenceName"))
    if any(re.search(p, text, re.IGNORECASE) for p in exclude):
        return False
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


class VolumetryEngine:
    def __init__(
        self,
        fastsurfer: FastSurfer,
        stripper,
        norms: NormativeLibrary | None,
        *,
        fastsurfer_version: str = "",
        tmp_root: str | Path | None = None,
    ) -> None:
        self.fastsurfer, self.stripper, self.norms = fastsurfer, stripper, norms
        self.fastsurfer_version = fastsurfer_version
        # Fichiers temporaires sur disque (/tmp est en mémoire vive sous Debian 13)
        self.tmp_root = Path(tmp_root) if tmp_root else None

    def intracranial_volume(self, image: sitk.Image, workdir: Path) -> float | None:
        """Volume intracrânien (mm³) : masque SynthStrip (cerveau + LCR)."""
        if self.stripper is None:
            return None
        _, mask = self.stripper.strip(image, workdir)
        voxel = float(np.prod(mask.GetSpacing()))
        return float((sitk.GetArrayViewFromImage(mask) > 0).sum()) * voxel

    def run(self, series: DicomSeries) -> list[Dataset]:
        header = series.first
        age, sex = patient_age(header), patient_sex(header)
        if self.tmp_root is not None:
            self.tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="segbrainmri_vol_", dir=self.tmp_root) as tmp:
            tmp = Path(tmp)
            logger.info("Volume intracrânien (SynthStrip)...")
            icv = self.intracranial_volume(series.image, tmp)
            fs = self.fastsurfer.run(series.image, tmp / "fastsurfer")
        result = analyze_volumes(fs.volumes, age=age, sex=sex, icv_mm3=icv, norms=self.norms)
        hippo = next((s for s in result.deep if s.label == "Hippocampe"), None)
        if hippo is not None:
            logger.info(
                "Hippocampes : G %.2f mL (P%s), D %.2f mL (P%s)",
                hippo.left.volume_ml,
                f"{hippo.left.percentile:.0f}" if hippo.left.percentile is not None else "-",
                hippo.right.volume_ml,
                f"{hippo.right.percentile:.0f}" if hippo.right.percentile is not None else "-",
            )
        info = ResultInfo(
            "FastSurfer", self.fastsurfer_version or "seg_only", len(result.deep), result.cortex_ml
        )
        labelmap = native_labelmap(fs.segmentation, series.image)
        return [
            build_volumetry_seg(labelmap, series.datasets, info),
            build_volumetry_report(
                series.datasets, result, fastsurfer_version=self.fastsurfer_version
            ),
        ]
