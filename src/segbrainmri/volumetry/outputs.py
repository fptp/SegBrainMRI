"""Résultats DICOM de la volumétrie : SEG des structures et compte rendu SR."""

from __future__ import annotations

import highdicom as hd
import numpy as np
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.sr.codedict import codes

from segbrainmri import __version__
from segbrainmri.dicom.report import CONTAINS, ML, _container, _num, _text
from segbrainmri.dicom.series import OUR_MANUFACTURER
from segbrainmri.dicom.writers import ResultInfo, _series_number, build_labelmap_seg
from segbrainmri.volumetry.analysis import StructureResult, VolumetryResult, flag
from segbrainmri.volumetry.regions import SEG_SEGMENTS, segment_labels

PERCENT = hd.sr.CodedConcept("%", "UCUM", "percent")
UNITLESS = hd.sr.CodedConcept("1", "UCUM", "no units")
YEARS = hd.sr.CodedConcept("a", "UCUM", "year")


def native_labelmap(segmentation: sitk.Image, reference: sitk.Image) -> np.ndarray:
    """Segments (1..n de SEG_SEGMENTS) sur la grille de la T1 native (coupes, lignes, colonnes)."""
    resampled = sitk.Resample(
        segmentation, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkInt16
    )
    labels = sitk.GetArrayFromImage(resampled)
    out = np.zeros(labels.shape, np.uint8)
    for k, (_, key, _) in enumerate(SEG_SEGMENTS, start=1):
        out[np.isin(labels, segment_labels(key))] = k
    return out


def build_volumetry_seg(labelmap: np.ndarray, sources: list[Dataset], info: ResultInfo) -> Dataset:
    return build_labelmap_seg(
        labelmap,
        sources,
        info,
        [(label, color) for label, _, color in SEG_SEGMENTS],
        series_description="VOLUMETRIE structures FastSurfer",
        series_offset=9010,
    )


def _sex_text(sex: int | None) -> str:
    return {1: "homme", 0: "femme"}.get(sex, "inconnu")


def _side_items(name: str, side, *, high_is_abnormal: bool) -> list:
    items = [_num(f"{name} volume", side.volume_ml, ML, "VOL")]
    if side.pct_icv is not None:
        items.append(_num(f"{name} % volume intracranien", side.pct_icv, PERCENT, "PICV"))
    if side.percentile is not None:
        items += [
            _num(f"{name} percentile (age, sexe)", side.percentile, PERCENT, "PCTL"),
            _num(f"{name} Z-score", side.z, UNITLESS, "Z"),
            _num(f"{name} volume median attendu", side.expected_ml, ML, "EXP"),
        ]
        note = flag(side.percentile, high_is_abnormal=high_is_abnormal)
        if note:
            items.append(_text(f"{name} interpretation", note, "FLAG"))
    return items


def _structure(item: StructureResult, code: str) -> hd.sr.ContainerContentItem:
    high = "Ventricule" in item.label or "Corne" in item.label
    children = (
        _side_items("Gauche", item.left, high_is_abnormal=high)
        + _side_items("Droite", item.right, high_is_abnormal=high)
        + [_num("Asymetrie (G-D)/moyenne", item.asymmetry, PERCENT, "ASYM")]
    )
    return _container(item.label, code, children)


def _line(item: StructureResult) -> str:
    def side(s):
        text = f"{s.volume_ml:.2f} mL"
        if s.percentile is not None:
            text += f" (P{s.percentile:.0f})"
        return text

    return (
        f"{item.label} : G {side(item.left)}, D {side(item.right)}, "
        f"asymetrie {item.asymmetry:+.0f} %"
    )


def summary_text(result: VolumetryResult) -> str:
    who = (
        f"{_sex_text(result.sex)}, {result.age:.0f} ans"
        if result.age is not None
        else (_sex_text(result.sex))
    )
    parts = [f"Volumetrie cerebrale automatique (FastSurfer) - patient {who}."]
    for item in result.deep:
        if item.label in ("Hippocampe", "Corne temporale"):
            parts.append(_line(item) + ".")
    lows = [
        f"{item.label} {name}"
        for item in result.deep
        for name, s in (("G", item.left), ("D", item.right))
        if flag(s.percentile, high_is_abnormal="Ventricule" in item.label or "Corne" in item.label)
    ]
    if result.norms_used:
        parts.append(
            "Valeurs hors normes (P<5 ou P>95 pour les ventricules) : "
            + (", ".join(lows) if lows else "aucune")
            + "."
        )
        parts.append(
            "Percentiles indicatifs : normes BrainCharts (FreeSurfer), non calibrees "
            "sur ce scanner."
        )
    else:
        parts.append(f"Pas de comparaison aux normes ({result.norms_note}).")
    return " ".join(parts)


def build_volumetry_report(
    sources: list[Dataset], result: VolumetryResult, *, fastsurfer_version: str = ""
) -> Dataset:
    items = [
        _text("Resume", summary_text(result), "SUM"),
        _text(
            "Methode",
            f"Segmentation FastSurfer {fastsurfer_version} (aseg + atlas DKT, volumes corriges "
            "du volume partiel). Volume intracranien : masque SynthStrip. Normes : modeles "
            "BrainCharts lifespan_57K_82sites (Rutherford et al., eLife 2022), sans "
            f"calibration locale. segbrainmri {__version__}.",
            "METH",
        ),
    ]
    patient = []
    if result.age is not None:
        patient.append(_num("Age", result.age, YEARS, "AGE"))
    patient.append(_text("Sexe", _sex_text(result.sex), "SEX"))
    if result.icv_ml:
        patient.append(_num("Volume intracranien", result.icv_ml, ML, "ICV"))
    patient.append(_num("Cortex total (somme des lobes)", result.cortex_ml, ML, "CTX"))
    if result.subcortical_gray is not None:
        patient += _side_items(
            "Substance grise sous-corticale", result.subcortical_gray, high_is_abnormal=False
        )
    items.append(_container("Donnees generales", "GEN", patient))
    items.append(
        _container(
            "Structures profondes (comparees aux normes)",
            "DEEP",
            [_structure(s, f"D{n}") for n, s in enumerate(result.deep, start=1)],
        )
    )
    items.append(
        _container(
            "Lobes - volume cortical (sans normes)",
            "LOBES",
            [_structure(s, f"L{n}") for n, s in enumerate(result.lobes, start=1)],
        )
    )
    items.append(
        _text(
            "Avertissement",
            "Resultat automatique d'un outil de recherche, non certifie comme dispositif medical. "
            "Percentiles indicatifs (normes FreeSurfer, scanner non calibre). A interpreter par un "
            "radiologue en fonction de la clinique et de la qualite de l'acquisition.",
            "WARN",
        )
    )
    root = hd.sr.ContainerContentItem(codes.LN.DiagnosticImagingReport)
    root.ContentSequence = items
    return hd.sr.ComprehensiveSR(
        evidence=sources,
        content=root,
        series_instance_uid=hd.UID(),
        series_number=_series_number(sources[0], 9011),
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer=OUR_MANUFACTURER,
        series_description="VOLUMETRIE compte rendu FastSurfer",
    )


__all__ = ["CONTAINS", "build_volumetry_report", "build_volumetry_seg", "native_labelmap"]
