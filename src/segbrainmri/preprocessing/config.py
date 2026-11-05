"""Configuration du prétraitement."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Paramètres du pipeline. Les valeurs par défaut sont usuelles pour la SEP (FLAIR 3D).

    Attention : ces paramètres doivent correspondre à ceux utilisés lors de
    l'entraînement du modèle chargé (espacement, orientation, normalisation).
    """

    #: Espacement cible en mm (x, y, z). ``None`` conserve la résolution native.
    target_spacing: tuple[float, float, float] | None = (1.0, 1.0, 1.0)
    #: Orientation cible : axes croissants vers ces directions ("RAS" = RAS+ de nibabel/MONAI,
    #: vérifié avec SimpleITK 2.5). ``None`` conserve l'orientation native.
    orientation: str | None = "RAS"
    #: Correction du biais de champ N4 (lente sur CPU : ~1 min par volume).
    n4_bias_correction: bool = False
    #: Percentiles de rognage des intensités, calculés dans le masque cérébral.
    clip_percentiles: tuple[float, float] | None = (0.5, 99.5)
    #: Normalisation : "zscore", "minmax" ou "none".
    normalization: str = "zscore"
    #: Masque d'avant-plan par seuillage d'Otsu (approximation grossière du cerveau).
    #: Pour une vraie extraction du cerveau, utiliser HD-BET / SynthStrip en amont.
    foreground_mask: bool = True

    def __post_init__(self) -> None:
        if self.normalization not in {"zscore", "minmax", "none"}:
            raise ValueError(f"normalization invalide : {self.normalization!r}")
        if self.target_spacing is not None:
            if len(self.target_spacing) != 3 or min(self.target_spacing) <= 0:
                raise ValueError("target_spacing doit contenir 3 valeurs strictement positives.")
            self.target_spacing = tuple(float(s) for s in self.target_spacing)
        if self.clip_percentiles is not None:
            low, high = self.clip_percentiles
            if not 0 <= low < high <= 100:
                raise ValueError("clip_percentiles doit vérifier 0 <= bas < haut <= 100.")
