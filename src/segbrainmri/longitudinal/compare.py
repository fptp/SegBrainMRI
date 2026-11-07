"""Recalage de l'antériorité et classement des lésions (nouvelle / élargie / stable).

Méthode :

1. Recalage **rigide** (6 paramètres) de l'examen antérieur sur l'examen actuel, sur
   les cerveaux extraits par SynthStrip (information mutuelle, multi-résolution).
2. Le masque des lésions antérieures est projeté dans l'espace actuel.
3. Chaque lésion actuelle (composante 26-connexe) est classée :

   * **nouvelle** : aucun recouvrement avec une lésion antérieure (dilatée de
     ``tolerance_mm`` pour absorber l'imprécision du recalage) et volume
     >= ``min_new_mm3`` ;
   * **élargie** : volume >= ``enlarge_ratio`` x volume antérieur correspondant et
     augmentation >= ``enlarge_min_mm3`` ;
   * **stable** : les autres ;
   * les lésions sans antécédent mais plus petites que ``min_new_mm3`` sont comptées à
     part (``SMALL``) et affichées avec les stables (« autres lésions »).

Une relecture par le radiologue reste indispensable : des différences de protocole ou
de segmentation entre les deux examens peuvent créer de fausses nouvelles lésions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

logger = logging.getLogger(__name__)

NEW, ENLARGING, STABLE = 1, 2, 3
#: Compteur des lésions sans antécédent mais sous le seuil de taille (affichées avec les stables)
SMALL = 4


def _downsample(image: sitk.Image, voxel_mm: float) -> sitk.Image:
    """Rééchantillonne à ``voxel_mm`` isotrope (lissage anti-repliement), si plus fin."""
    spacing = image.GetSpacing()
    if min(spacing) >= voxel_mm:
        return image
    sigmas = [max(0.0, (voxel_mm - s) / 2) for s in spacing]
    smoothed = sitk.SmoothingRecursiveGaussian(image, sigmas)
    size = [max(1, round(n * s / voxel_mm)) for n, s in zip(image.GetSize(), spacing, strict=True)]
    return sitk.Resample(
        smoothed,
        size,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        (voxel_mm,) * 3,
        image.GetDirection(),
        0.0,
        sitk.sitkFloat32,
    )


def register_rigid(
    fixed: sitk.Image, moving: sitk.Image, *, seed: int = 42, working_mm: float = 1.5
) -> sitk.Transform:
    """Transformation rigide qui amène ``moving`` (antériorité) sur ``fixed`` (actuel).

    Convention SimpleITK : la transformation va de l'espace ``fixed`` vers ``moving`` ;
    ``sitk.Resample(moving, fixed, transform)`` produit ``moving`` dans l'espace ``fixed``.
    """
    # Un recalage rigide est aussi précis à ~1,5 mm et bien plus rapide (images 0,5 mm).
    # La transformation, exprimée en coordonnées physiques, vaut à pleine résolution.
    fixed = _downsample(sitk.Cast(fixed, sitk.sitkFloat32), working_mm)
    moving = _downsample(sitk.Cast(moving, sitk.sitkFloat32), working_mm)
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.05, seed)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=300,
        relaxationFactor=0.5,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])  # ~6 mm, 3 mm, 1,5 mm
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)
    transform = reg.Execute(fixed, moving)
    logger.info(
        "Recalage rigide : métrique finale %.4f (%d itérations)",
        reg.GetMetricValue(),
        reg.GetOptimizerIteration(),
    )
    return transform


@dataclass
class LesionComparison:
    """Résultat de la comparaison, dans l'espace de l'examen actuel."""

    labelmap: np.ndarray  # (coupes, lignes, colonnes) : 0, NEW, ENLARGING, STABLE (+ petites)
    counts: dict[int, int] = field(default_factory=dict)
    volumes_mm3: dict[int, float] = field(default_factory=dict)

    @property
    def new(self) -> int:
        return self.counts.get(NEW, 0)

    @property
    def enlarging(self) -> int:
        return self.counts.get(ENLARGING, 0)

    @property
    def stable(self) -> int:
        return self.counts.get(STABLE, 0)


def _ball(radius_mm: float, spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    radii = [max(0, round(radius_mm / s)) for s in spacing_zyx]
    grids = np.ogrid[tuple(slice(-r, r + 1) for r in radii)]
    dist = sum((g * s / max(radius_mm, 1e-6)) ** 2 for g, s in zip(grids, spacing_zyx, strict=True))
    return dist <= 1.0


def classify_lesions(
    current: np.ndarray,
    prior: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    min_new_mm3: float = 10.0,
    tolerance_mm: float = 1.0,
    enlarge_ratio: float = 1.5,
    enlarge_min_mm3: float = 20.0,
) -> LesionComparison:
    """Classe chaque lésion de ``current`` par rapport à ``prior`` (même grille, ``(z, y, x)``)."""
    if current.shape != prior.shape:
        raise ValueError("Les masques actuel et antérieur doivent avoir la même forme.")
    voxel_mm3 = float(np.prod(spacing_zyx))
    structure = np.ones((3, 3, 3), bool)  # 26-connexité
    current_labels, n_current = ndimage.label(current > 0, structure)
    prior_bin = prior > 0
    prior_labels, _ = ndimage.label(prior_bin, structure)
    prior_volumes = np.bincount(prior_labels.ravel()) * voxel_mm3
    margin = _ball(tolerance_mm, spacing_zyx) if tolerance_mm > 0 else None
    pad = [s // 2 for s in margin.shape] if margin is not None else [0, 0, 0]

    labelmap = np.zeros(current.shape, np.uint8)
    counts = {NEW: 0, ENLARGING: 0, STABLE: 0, SMALL: 0}
    volumes = {NEW: 0.0, ENLARGING: 0.0, STABLE: 0.0, SMALL: 0.0}
    for index, box in enumerate(ndimage.find_objects(current_labels), start=1):
        if box is None:
            continue
        # Boîte élargie de la tolérance, pour travailler localement (rapide)
        box = tuple(
            slice(max(b.start - p, 0), min(b.stop + p, n))
            for b, p, n in zip(box, pad, current.shape, strict=True)
        )
        lesion = current_labels[box] == index
        volume = float(lesion.sum()) * voxel_mm3
        near = ndimage.binary_dilation(lesion, margin) if margin is not None else lesion
        touched = np.unique(prior_labels[box][near & prior_bin[box]])
        touched = touched[touched > 0]
        if touched.size == 0:
            category = NEW if volume >= min_new_mm3 else SMALL
        else:
            before = float(prior_volumes[touched].sum())
            grown = volume >= enlarge_ratio * before and volume - before >= enlarge_min_mm3
            category = ENLARGING if grown else STABLE
        # Les petites lésions sans antécédent sont affichées avec les « autres » (STABLE)
        labelmap[box][lesion] = STABLE if category == SMALL else category
        counts[category] += 1
        volumes[category] += volume
    logger.info(
        "Comparaison : %d nouvelle(s), %d élargie(s), %d stable(s), %d petite(s) sous le seuil"
        " sur %d lésion(s)",
        counts[NEW],
        counts[ENLARGING],
        counts[STABLE],
        counts[SMALL],
        n_current,
    )
    return LesionComparison(labelmap=labelmap, counts=counts, volumes_mm3=volumes)
