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
    seg: bool = True
    overlay: bool = True
    overlay_opacity: float = 0.5


@dataclass
class ServiceConfig:
    listener: ListenerConfig = field(default_factory=ListenerConfig)
    orthanc: OrthancConfig = field(default_factory=OrthancConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

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
                if key in ("spool_dir", "models_dir"):
                    value = Path(value)
                setattr(target, key, value)
        unknown_sections = set(raw) - {f.name for f in fields(cls)}
        if unknown_sections:
            raise ValueError(f"Section(s) inconnue(s) : {sorted(unknown_sections)}")
        return config
