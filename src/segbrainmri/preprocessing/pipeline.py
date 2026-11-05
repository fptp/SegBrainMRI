"""Pipeline de prétraitement multi-modal (FLAIR seul ou FLAIR + T1, etc.).

Étapes, pour la modalité de référence (la première, typiquement FLAIR) :

1. chargement NIfTI ;
2. réorientation (``orientation``) ;
3. rééchantillonnage isotrope (``target_spacing``) ;
4. correction N4 optionnelle ;
5. masque d'avant-plan + rognage/normalisation des intensités.

Les autres modalités sont rééchantillonnées sur la grille de la référence.
Hypothèse : les modalités sont déjà co-registrées (même espace physique).
TODO : ajouter un recalage rigide optionnel (SimpleITK / ANTsPy) pour les cas non alignés.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from segbrainmri.preprocessing import intensity, spatial
from segbrainmri.preprocessing.config import PreprocessingConfig
from segbrainmri.preprocessing.io import load_nifti, save_nifti

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedCase:
    """Résultat du prétraitement d'un patient.

    Attributes:
        data: tableau ``(C, Z, Y, X)`` float32, une couche par modalité (ordre de ``modalities``).
        modalities: noms des modalités, dans l'ordre des canaux.
        reference: image SimpleITK (géométrie de la grille prétraitée).
        original: image native de la modalité de référence (pour revenir à l'espace patient).
        mask: masque d'avant-plan ``(Z, Y, X)`` ou ``None``.
    """

    data: np.ndarray
    modalities: list[str]
    reference: sitk.Image
    original: sitk.Image
    mask: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def to_image(self, array: np.ndarray, *, is_label: bool = False) -> sitk.Image:
        """Enveloppe un tableau ``(Z, Y, X)`` avec la géométrie de la grille prétraitée."""
        image = sitk.GetImageFromArray(array.astype(np.uint8 if is_label else np.float32))
        image.CopyInformation(self.reference)
        return image

    def restore_to_original(self, array: np.ndarray, *, is_label: bool = True) -> sitk.Image:
        """Ramène une prédiction ``(Z, Y, X)`` dans l'espace natif du patient."""
        return spatial.resample_like(
            self.to_image(array, is_label=is_label), self.original, is_label=is_label
        )


class PreprocessingPipeline:
    """Applique :class:`PreprocessingConfig` à une ou plusieurs modalités."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    # -- étapes -----------------------------------------------------------------
    def _spatial(self, image: sitk.Image) -> sitk.Image:
        image = spatial.reorient(image, self.config.orientation)
        return spatial.resample_to_spacing(image, self.config.target_spacing)

    def _normalize(self, image: sitk.Image, mask: np.ndarray | None) -> np.ndarray:
        cfg = self.config
        return intensity.normalize_intensity(
            sitk.GetArrayFromImage(image),
            mask,
            method=cfg.normalization,
            clip_percentiles=cfg.clip_percentiles,
        )

    # -- API publique -------------------------------------------------------------
    def run(self, inputs: Mapping[str, str | Path | sitk.Image]) -> PreprocessedCase:
        """Prétraite un patient.

        Args:
            inputs: ``{"flair": chemin, "t1": chemin, ...}``. La première entrée est la
                référence géométrique. Accepte aussi des ``sitk.Image`` déjà chargées.
        """
        if not inputs:
            raise ValueError("Au moins une modalité est requise.")
        names = list(inputs)
        images = {
            name: src if isinstance(src, sitk.Image) else load_nifti(src)
            for name, src in inputs.items()
        }

        ref_name = names[0]
        original = images[ref_name]
        reference = self._spatial(original)
        logger.info(
            "Référence %s : %s @ %s mm -> %s @ %s mm",
            ref_name,
            original.GetSize(),
            tuple(round(s, 2) for s in original.GetSpacing()),
            reference.GetSize(),
            tuple(round(s, 2) for s in reference.GetSpacing()),
        )

        aligned: dict[str, sitk.Image] = {ref_name: reference}
        for name in names[1:]:
            aligned[name] = spatial.resample_like(images[name], reference)

        mask_img = intensity.foreground_mask(reference) if self.config.foreground_mask else None
        mask = sitk.GetArrayFromImage(mask_img).astype(bool) if mask_img is not None else None

        channels = []
        for name in names:
            image = aligned[name]
            if self.config.n4_bias_correction:
                logger.info("Correction N4 du biais de champ (%s)...", name)
                image = intensity.n4_bias_correction(image, mask_img)
            channels.append(self._normalize(image, mask))

        return PreprocessedCase(
            data=np.stack(channels, axis=0),
            modalities=names,
            reference=reference,
            original=original,
            mask=mask,
            metadata={"config": self.config},
        )

    def save(self, case: PreprocessedCase, output_dir: str | Path) -> list[Path]:
        """Écrit chaque modalité prétraitée dans ``output_dir/<modalite>_preproc.nii.gz``."""
        output_dir = Path(output_dir)
        return [
            save_nifti(case.to_image(case.data[i]), output_dir / f"{name}_preproc.nii.gz")
            for i, name in enumerate(case.modalities)
        ]
