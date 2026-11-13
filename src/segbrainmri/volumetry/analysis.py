"""Calcul des résultats de volumétrie : volumes, % du volume intracrânien, asymétrie, normes."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

from pydicom.dataset import Dataset

from segbrainmri.volumetry.normative import NormativeLibrary
from segbrainmri.volumetry.regions import ALIASES, LOBES, NORMED, dkt_label

logger = logging.getLogger(__name__)

#: Âges couverts de façon fiable par les modèles BrainCharts
MIN_AGE, MAX_AGE = 3.0, 100.0


def patient_age(ds: Dataset) -> float | None:
    """Âge en années : PatientAge (ex. ``067Y``) ou date de naissance et date d'examen."""
    text = str(ds.get("PatientAge", "") or "").strip().upper()
    match = re.fullmatch(r"(\d{1,3})([DWMY])", text)
    if match:
        value, unit = int(match[1]), match[2]
        return value / {"D": 365.25, "W": 52.18, "M": 12.0, "Y": 1.0}[unit]
    birth, study = str(ds.get("PatientBirthDate", "")), str(ds.get("StudyDate", ""))
    if len(birth) == 8 and len(study) == 8:
        try:
            b = date(int(birth[:4]), int(birth[4:6]), int(birth[6:]))
            s = date(int(study[:4]), int(study[4:6]), int(study[6:]))
        except ValueError:
            return None
        return (s - b).days / 365.25
    return None


def patient_sex(ds: Dataset) -> int | None:
    """1 = homme, 0 = femme (codage BrainCharts) ; None si inconnu."""
    return {"M": 1, "F": 0}.get(str(ds.get("PatientSex", "")).strip().upper())


def asymmetry(left: float, right: float) -> float:
    """Indice d'asymétrie en % : 100 x (G - D) / moyenne(G, D). Positif = gauche plus grand."""
    mean = (left + right) / 2
    return 100.0 * (left - right) / mean if mean > 0 else 0.0


@dataclass
class SideResult:
    volume_ml: float
    pct_icv: float | None = None
    z: float | None = None
    percentile: float | None = None
    expected_ml: float | None = None  # médiane attendue pour l'âge et le sexe


@dataclass
class StructureResult:
    label: str
    left: SideResult
    right: SideResult
    normed: bool = False

    @property
    def asymmetry(self) -> float:
        return asymmetry(self.left.volume_ml, self.right.volume_ml)


@dataclass
class VolumetryResult:
    age: float | None
    sex: int | None
    icv_ml: float | None
    lobes: list[StructureResult] = field(default_factory=list)
    deep: list[StructureResult] = field(default_factory=list)
    cortex_ml: float = 0.0
    subcortical_gray: SideResult | None = None
    norms_used: bool = False
    norms_note: str = ""


def _volume(volumes: dict[str, float], name: str) -> float | None:
    for candidate in (name, *ALIASES.get(name, ())):
        if candidate in volumes:
            return volumes[candidate]
    return None


def analyze_volumes(
    volumes: dict[str, float],
    *,
    age: float | None,
    sex: int | None,
    icv_mm3: float | None,
    norms: NormativeLibrary | None,
) -> VolumetryResult:
    """Construit le résultat à partir des volumes FastSurfer (mm³)."""
    use_norms = norms is not None and age is not None and sex is not None
    note = ""
    if norms is None:
        note = "modeles normatifs non installes"
    elif age is None or sex is None:
        note = "age ou sexe absent du DICOM : pas de comparaison aux normes"
    elif not MIN_AGE <= age <= MAX_AGE:
        use_norms, note = False, f"age ({age:.0f} ans) hors de la plage des normes"

    def side(mm3: float, model: str | None) -> SideResult:
        result = SideResult(volume_ml=mm3 / 1000.0)
        if icv_mm3:
            result.pct_icv = 100.0 * mm3 / icv_mm3
        if use_norms and model and norms.has(model):
            nm = norms.get(model)
            result.z = nm.zscore(mm3, age, sex)
            result.percentile = nm.percentile(mm3, age, sex)
            result.expected_ml = nm.expected(age, sex) / 1000.0
        return result

    result = VolumetryResult(
        age=age,
        sex=sex,
        icv_ml=icv_mm3 / 1000.0 if icv_mm3 else None,
        norms_used=use_norms,
        norms_note=note,
    )
    for lobe, regions in LOBES.items():
        sides = []
        for hemi in ("G", "D"):
            missing = [r for r in regions if dkt_label(hemi, r) not in volumes]
            if missing:
                logger.warning("Régions absentes du lobe %s (%s) : %s", lobe, hemi, missing)
            sides.append(sum(volumes.get(dkt_label(hemi, r), 0.0) for r in regions))
        result.lobes.append(StructureResult(lobe, side(sides[0], None), side(sides[1], None)))
        result.cortex_ml += (sides[0] + sides[1]) / 1000.0
    for label, left_name, right_name in NORMED:
        left, right = _volume(volumes, left_name), _volume(volumes, right_name)
        if left is None or right is None:
            logger.warning("Structure absente des volumes FastSurfer : %s", label)
            continue
        result.deep.append(
            StructureResult(label, side(left, left_name), side(right, right_name), normed=True)
        )
    if "SubCortGrayVol" in volumes:
        result.subcortical_gray = side(volumes["SubCortGrayVol"], "SubCortGrayVol")
    return result


def flag(percentile: float | None, *, high_is_abnormal: bool = False) -> str:
    """Interprétation indicative d'un percentile."""
    if percentile is None:
        return ""
    if high_is_abnormal:
        return "tres eleve" if percentile > 99 else "eleve" if percentile > 95 else ""
    return "tres bas" if percentile < 1 else "bas" if percentile < 5 else ""
