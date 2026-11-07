"""Compte rendu DICOM SR (Comprehensive SR) de l'analyse.

Contenu : résultat global (nombre de lésions, volume), comparaison avec l'antériorité
(date, nombre de lésions nouvelles / élargies / stables) et, pour chaque lésion nouvelle
ou élargie, son volume et sa position (numéro d'image et coordonnées patient).
Le texte est en ASCII (sans accents) pour un affichage correct dans toutes les
visionneuses.
"""

from __future__ import annotations

from dataclasses import dataclass

import highdicom as hd
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.sr.codedict import codes

from segbrainmri import __version__
from segbrainmri.dicom.series import OUR_MANUFACTURER
from segbrainmri.dicom.writers import ResultInfo, _series_number
from segbrainmri.longitudinal import ENLARGING, NEW, SMALL, LesionComparison

_SCHEME = "99SEGBRAINMRI"  # codes locaux (préfixe 99 réservé aux codes privés)


def _code(value: str, meaning: str) -> hd.sr.CodedConcept:
    return hd.sr.CodedConcept(value=value, scheme_designator=_SCHEME, meaning=meaning)


MM3 = hd.sr.CodedConcept("mm3", "UCUM", "cubic millimeter")
ML = hd.sr.CodedConcept("mL", "UCUM", "milliliter")
COUNT = hd.sr.CodedConcept("{lesions}", "UCUM", "lesions")
CONTAINS = hd.sr.RelationshipTypeValues.CONTAINS


@dataclass
class PriorInfo:
    study_date: str
    description: str


def _text(meaning: str, value: str, code: str) -> hd.sr.TextContentItem:
    return hd.sr.TextContentItem(_code(code, meaning), value[:1024], relationship_type=CONTAINS)


def _num(meaning: str, value: float, unit, code: str) -> hd.sr.NumContentItem:
    return hd.sr.NumContentItem(
        _code(code, meaning), round(float(value), 2), unit, relationship_type=CONTAINS
    )


def _container(meaning: str, code: str, items: list) -> hd.sr.ContainerContentItem:
    container = hd.sr.ContainerContentItem(_code(code, meaning), relationship_type=CONTAINS)
    container.ContentSequence = items
    return container


def _date(value: str) -> str:
    return f"{value[6:8]}/{value[4:6]}/{value[:4]}" if len(value) == 8 else value or "?"


def _location(record, image: sitk.Image, sources: list[Dataset]) -> str:
    z, y, x = record.centroid_zyx
    k = min(max(round(z), 0), len(sources) - 1)
    point = image.TransformContinuousIndexToPhysicalPoint((x, y, z))
    number = sources[k].get("InstanceNumber", k + 1)
    # Coordonnées DICOM (LPS) : x vers la gauche, y vers l'arrière, z vers le haut
    return (
        f"image n. {number} (coupe {k + 1}/{len(sources)}), "
        f"x={point[0]:.0f} y={point[1]:.0f} z={point[2]:.0f} mm"
    )


def summary_text(
    info: ResultInfo,
    comparison: LesionComparison | None,
    prior: PriorInfo | None,
    *,
    searched: bool,
) -> str:
    text = (
        f"Segmentation automatique des lesions de SEP ({info.algorithm}) : "
        f"{info.lesion_count} lesion(s), volume lesionnel total {info.lesion_volume_ml:.2f} mL."
    )
    if comparison is not None and prior is not None:
        text += (
            f" Comparaison avec l'examen du {_date(prior.study_date)} : "
            f"{comparison.new} nouvelle(s) lesion(s), {comparison.enlarging} lesion(s) "
            f"elargie(s), {comparison.stable} lesion(s) stable(s)."
        )
    elif searched:
        text += " Aucun examen anterieur trouve dans Orthanc : pas de comparaison."
    return text


def build_report(
    sources: list[Dataset],
    image: sitk.Image,
    info: ResultInfo,
    *,
    comparison: LesionComparison | None = None,
    prior: PriorInfo | None = None,
    longitudinal_searched: bool = False,
    min_new_mm3: float | None = None,
) -> Dataset:
    """Construit le SR. ``image`` : volume SimpleITK de la série (pour les coordonnées)."""
    items = [
        _text(
            "Resume", summary_text(info, comparison, prior, searched=longitudinal_searched), "SUM"
        ),
        _text(
            "Algorithme",
            f"{info.algorithm} ({info.algorithm_version}), segbrainmri {__version__}",
            "ALG",
        ),
        _container(
            "Resultat global",
            "GLOBAL",
            [
                _num("Nombre de lesions", info.lesion_count, COUNT, "NLES"),
                _num("Volume lesionnel total", info.lesion_volume_ml, ML, "VLES"),
            ],
        ),
    ]
    if comparison is not None and prior is not None:
        rules = "lesion sans recouvrement avec une lesion anterieure"
        if min_new_mm3 is not None:
            rules += f" et d'au moins {min_new_mm3:g} mm3"
        items.append(
            _container(
                "Comparaison avec l'anteriorite",
                "COMP",
                [
                    _text(
                        "Examen anterieur",
                        f"{_date(prior.study_date)} - {prior.description}",
                        "PRIOR",
                    ),
                    _num("Nouvelles lesions", comparison.new, COUNT, "NNEW"),
                    _num("Lesions elargies", comparison.enlarging, COUNT, "NENL"),
                    _num("Lesions stables", comparison.stable, COUNT, "NSTA"),
                    _num(
                        "Petites lesions sans antecedent (sous le seuil)",
                        comparison.counts.get(SMALL, 0),
                        COUNT,
                        "NSMA",
                    ),
                    _num(
                        "Volume des nouvelles lesions",
                        comparison.volumes_mm3.get(NEW, 0.0),
                        MM3,
                        "VNEW",
                    ),
                    _text("Definition d'une nouvelle lesion", rules, "RULE"),
                ],
            )
        )
        for category, label, code in (
            (NEW, "Nouvelle lesion", "NEW"),
            (ENLARGING, "Lesion elargie", "ENL"),
        ):
            records = sorted(
                (r for r in comparison.lesions if r.category == category),
                key=lambda r: -r.volume_mm3,
            )
            for n, record in enumerate(records, start=1):
                details = [
                    _num("Volume", record.volume_mm3, MM3, "VOL"),
                    _text("Localisation", _location(record, image, sources), "LOC"),
                ]
                if category == ENLARGING:
                    details.append(_num("Volume anterieur", record.prior_volume_mm3, MM3, "VPRI"))
                items.append(_container(f"{label} {n}", f"{code}{n}", details))
    items.append(
        _text(
            "Avertissement",
            "Resultat automatique d'un outil de recherche, non certifie comme dispositif medical. "
            "Des differences de machine, de protocole ou de segmentation entre examens peuvent "
            "produire de fausses nouvelles lesions : relecture par un radiologue indispensable.",
            "WARN",
        )
    )
    root = hd.sr.ContainerContentItem(codes.LN.DiagnosticImagingReport)
    root.ContentSequence = items
    return hd.sr.ComprehensiveSR(
        evidence=sources,
        content=root,
        series_instance_uid=hd.UID(),
        series_number=_series_number(sources[0], 9004),
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer=OUR_MANUFACTURER,
        series_description=f"SEP compte rendu {info.algorithm}"[:64],
    )
