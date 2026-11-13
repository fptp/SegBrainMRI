"""Configuration du service (fichier TOML, par défaut /etc/segbrainmri/config.toml)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("/etc/segbrainmri/config.toml")


@dataclass
class ListenerConfig:
    ae_title: str = "SEGBRAINMRI"
    host: str = "0.0.0.0"
    port: int = 11112
    #: AET autorisés à nous envoyer des images (vide = tous).
    allowed_calling_aets: list[str] = field(default_factory=lambda: ["ORTHANC"])


@dataclass
class OrthancConfig:
    host: str = "192.168.10.172"
    port: int = 4242
    ae_title: str = "ORTHANC"


@dataclass
class ProcessingConfig:
    spool_dir: Path = Path("/var/lib/segbrainmri")
    #: Une série est traitée quand plus aucune image n'arrive depuis N secondes.
    settle_seconds: float = 20.0
    poll_seconds: float = 5.0
    device: str = "auto"
    threads: int | None = None
    flair_patterns: list[str] = field(
        default_factory=lambda: ["flair", "dark[ _-]?fluid", "t2[ _-]?fl", "spcir"]
    )
    #: Conserver les images reçues après un traitement réussi (sinon supprimées).
    keep_inputs: bool = False


@dataclass
class ModelConfig:
    backend: str = "flames"
    models_dir: Path = Path("/opt/segbrainmri/models")
    #: Augmentation au test : "auto" (GPU seulement), true ou false.
    tta: str | bool = "auto"
    skull_strip: bool = True


@dataclass
class OutputConfig:
    #: DICOM SEG (OHIF, Stone Web Viewer, 3D Slicer, Weasis)
    seg: bool = True
    #: Séries d'images dérivées : FLAIR + contours en niveaux de gris, et série couleur
    #: « SEP EVOLUTION » (nouvelles en rouge, élargies en orange). Inutiles avec une
    #: visionneuse qui affiche le DICOM SEG ; à activer pour Horos.
    contours: bool = False
    contour_thickness_mm: float = 0.8
    #: Compte rendu DICOM SR (résumé, nouvelles lésions avec volume et position)
    sr: bool = True
    #: Overlay couleur RGB de toutes les lésions (non fenêtrable)
    overlay: bool = False
    overlay_opacity: float = 0.5


@dataclass
class LongitudinalConfig:
    #: Chercher dans Orthanc la FLAIR antérieure du patient et signaler les nouvelles lésions
    enabled: bool = True
    #: Volume minimal (mm³) pour signaler une nouvelle lésion (~ sphère de 2,7 mm)
    min_new_mm3: float = 10.0
    #: Tolérance de recalage : une lésion à moins de N mm d'une ancienne n'est pas nouvelle
    tolerance_mm: float = 1.0
    #: Lésion élargie si volume >= ratio x ancien volume et augmentation >= N mm³
    enlarge_ratio: float = 1.5
    enlarge_min_mm3: float = 20.0


@dataclass
class VolumetryConfig:
    #: Volumétrie des T1 3D (FastSurfer) avec comparaison aux normes par âge et sexe
    enabled: bool = True
    t1_patterns: list[str] = field(
        default_factory=lambda: [
            "bravo",
            "mprage",
            "mp-rage",
            "spgr",
            "t1[ _-]?3d",
            "3d[ _-]?t1",
            "t1w[ _-]?3d",
            "3d[ _-]?tfe",
            "t1[ _-]?tfe",
        ]
    )
    #: T1 injectées exclues (FastSurfer est entraîné sur des T1 sans gadolinium)
    t1_exclude: list[str] = field(
        default_factory=lambda: ["gado", r"\bgd\b", r"\bpost\b", r"\+ ?c\b", "contrast", r"\binj"]
    )
    fastsurfer_dir: Path = Path("/opt/segbrainmri/fastsurfer")
    fastsurfer_python: Path = Path("/opt/segbrainmri/fastsurfer-venv/bin/python")
    norms_dir: Path = Path("/opt/segbrainmri/models/norms")


@dataclass
class ServiceConfig:
    listener: ListenerConfig = field(default_factory=ListenerConfig)
    orthanc: OrthancConfig = field(default_factory=OrthancConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    longitudinal: LongitudinalConfig = field(default_factory=LongitudinalConfig)
    volumetry: VolumetryConfig = field(default_factory=VolumetryConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> ServiceConfig:
        """Charge le TOML ; toute clé absente garde sa valeur par défaut."""
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        raw = tomllib.loads(path.read_text()) if path.is_file() else {}
        config = cls()
        for section in fields(cls):
            values = raw.get(section.name, {})
            target = getattr(config, section.name)
            known = {f.name: f for f in fields(target)}
            unknown = set(values) - set(known)
            if unknown:
                raise ValueError(f"Clé(s) inconnue(s) dans [{section.name}] : {sorted(unknown)}")
            for key, value in values.items():
                if key in (
                    "spool_dir",
                    "models_dir",
                    "fastsurfer_dir",
                    "fastsurfer_python",
                    "norms_dir",
                ):
                    value = Path(value)
                setattr(target, key, value)
        unknown_sections = set(raw) - {f.name for f in fields(cls)}
        if unknown_sections:
            raise ValueError(f"Section(s) inconnue(s) : {sorted(unknown_sections)}")
        return config
