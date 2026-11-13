"""Structures mesurées : lobes (regroupement de l'atlas DKT) et structures profondes."""

from __future__ import annotations

#: Régions corticales de l'atlas DKT (31 par hémisphère) regroupées par lobe
#: (regroupement usuel de FreeSurfer, « lobesStrict »).
LOBES: dict[str, tuple[str, ...]] = {
    "Frontal": (
        "caudalmiddlefrontal",
        "lateralorbitofrontal",
        "medialorbitofrontal",
        "paracentral",
        "parsopercularis",
        "parsorbitalis",
        "parstriangularis",
        "precentral",
        "rostralmiddlefrontal",
        "superiorfrontal",
    ),
    "Parietal": (
        "inferiorparietal",
        "postcentral",
        "precuneus",
        "superiorparietal",
        "supramarginal",
    ),
    "Temporal": (
        "entorhinal",
        "fusiform",
        "inferiortemporal",
        "middletemporal",
        "parahippocampal",
        "superiortemporal",
        "transversetemporal",
    ),
    "Occipital": ("cuneus", "lateraloccipital", "lingual", "pericalcarine"),
    "Cingulaire": (
        "caudalanteriorcingulate",
        "isthmuscingulate",
        "posteriorcingulate",
        "rostralanteriorcingulate",
    ),
    "Insula": ("insula",),
}


def dkt_label(hemi: str, region: str) -> str:
    """Nom FreeSurfer d'une région corticale, ex. ``ctx-lh-superiorfrontal``."""
    return f"ctx-{'lh' if hemi == 'G' else 'rh'}-{region}"


#: Structures comparées aux normes : (libellé, nom aseg gauche, nom aseg droit).
#: Les noms sont ceux des modèles BrainCharts (FreeSurfer aseg).
NORMED: tuple[tuple[str, str, str], ...] = (
    ("Hippocampe", "Left-Hippocampus", "Right-Hippocampus"),
    ("Amygdale", "Left-Amygdala", "Right-Amygdala"),
    ("Corne temporale", "Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent"),
    ("Ventricule lateral", "Left-Lateral-Ventricle", "Right-Lateral-Ventricle"),
    ("Thalamus", "Left-Thalamus-Proper", "Right-Thalamus-Proper"),
)

#: Noms équivalents (FreeSurfer 7 / FastSurfer ont renommé certaines structures)
ALIASES: dict[str, tuple[str, ...]] = {
    "Left-Thalamus-Proper": ("Left-Thalamus",),
    "Right-Thalamus-Proper": ("Right-Thalamus",),
}

#: Segments du DICOM SEG : (libellé, structures aseg/DKT incluses, couleur CIELab)
SEG_SEGMENTS: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("Lobe frontal", "lobe:Frontal", (60.0, -5.0, -55.0)),  # bleu
    ("Lobe parietal", "lobe:Parietal", (70.0, -55.0, 50.0)),  # vert
    ("Lobe temporal", "lobe:Temporal", (75.0, 25.0, 75.0)),  # orange
    ("Lobe occipital", "lobe:Occipital", (45.0, 60.0, -55.0)),  # violet
    ("Cortex cingulaire", "lobe:Cingulaire", (80.0, -40.0, -10.0)),  # turquoise
    ("Insula", "lobe:Insula", (90.0, -5.0, 80.0)),  # jaune
    ("Hippocampe", "aseg:Hippocampus", (53.0, 80.0, 67.0)),  # rouge
    ("Amygdale", "aseg:Amygdala", (65.0, 65.0, -10.0)),  # rose
    ("Ventricules lateraux", "aseg:Lateral-Ventricle", (85.0, -20.0, -25.0)),  # cyan clair
    ("Cornes temporales", "aseg:Inf-Lat-Vent", (40.0, 20.0, -60.0)),  # bleu foncé
)

#: Numéros FreeSurfer des régions DKT (``ctx-lh-<nom>`` = 1000 + n, ``ctx-rh-<nom>`` = 2000 + n)
DKT_INDEX: dict[str, int] = {
    "caudalanteriorcingulate": 2,
    "caudalmiddlefrontal": 3,
    "cuneus": 5,
    "entorhinal": 6,
    "fusiform": 7,
    "inferiorparietal": 8,
    "inferiortemporal": 9,
    "isthmuscingulate": 10,
    "lateraloccipital": 11,
    "lateralorbitofrontal": 12,
    "lingual": 13,
    "medialorbitofrontal": 14,
    "middletemporal": 15,
    "parahippocampal": 16,
    "paracentral": 17,
    "parsopercularis": 18,
    "parsorbitalis": 19,
    "parstriangularis": 20,
    "pericalcarine": 21,
    "postcentral": 22,
    "posteriorcingulate": 23,
    "precentral": 24,
    "precuneus": 25,
    "rostralanteriorcingulate": 26,
    "rostralmiddlefrontal": 27,
    "superiorfrontal": 28,
    "superiorparietal": 29,
    "superiortemporal": 30,
    "supramarginal": 31,
    "transversetemporal": 34,
    "insula": 35,
}

#: Numéros FreeSurfer (gauche, droite) des structures aseg affichées
ASEG_LABELS: dict[str, tuple[int, int]] = {
    "Hippocampus": (17, 53),
    "Amygdala": (18, 54),
    "Lateral-Ventricle": (4, 43),
    "Inf-Lat-Vent": (5, 44),
    "Thalamus": (10, 49),
}


def segment_labels(key: str) -> list[int]:
    """Numéros FreeSurfer d'un segment ``lobe:<Lobe>`` ou ``aseg:<Structure>``."""
    kind, name = key.split(":", 1)
    if kind == "lobe":
        return [base + DKT_INDEX[r] for r in LOBES[name] for base in (1000, 2000)]
    return list(ASEG_LABELS[name])
