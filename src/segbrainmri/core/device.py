"""Gestion dynamique de l'appareil de calcul (CPU par défaut, CUDA si disponible).

Le choix se fait dans cet ordre de priorité :

1. argument explicite ``preferred`` (ex. ``--device cuda:0`` dans la CLI) ;
2. variable d'environnement ``SEGBRAINMRI_DEVICE`` ;
3. ``"auto"`` : CUDA si ``torch.cuda.is_available()``, sinon CPU.

Si CUDA est demandé mais indisponible, on bascule sur le CPU avec un
avertissement (sauf en mode ``strict``). Aucun changement de code n'est donc
nécessaire le jour où le GPU est installé : il suffit d'installer les pilotes
NVIDIA et une roue PyTorch CUDA.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

ENV_VAR = "SEGBRAINMRI_DEVICE"
_VALID_PREFIXES = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class DeviceInfo:
    """Description de l'appareil retenu (utile pour les logs et la CLI ``info``)."""

    device: torch.device
    name: str
    total_memory_gb: float | None = None
    cuda_version: str | None = None
    cpu_threads: int | None = None

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def __str__(self) -> str:
        if self.is_cuda:
            return (
                f"{self.device} ({self.name}, {self.total_memory_gb:.1f} Go VRAM, "
                f"CUDA {self.cuda_version})"
            )
        return f"cpu ({self.name}, {self.cpu_threads} threads PyTorch)"


def _cuda_device(spec: str) -> torch.device:
    device = torch.device(spec)
    index = 0 if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise ValueError(
            f"Appareil {spec!r} demandé mais seulement {torch.cuda.device_count()} GPU détecté(s)."
        )
    return torch.device("cuda", index)


def resolve_device(preferred: str | None = None, *, strict: bool = False) -> torch.device:
    """Détermine le ``torch.device`` à utiliser, sans effet de bord (pas de log).

    Args:
        preferred: ``"auto"``, ``"cpu"``, ``"cuda"`` ou ``"cuda:N"``. ``None`` lit
            ``SEGBRAINMRI_DEVICE`` puis retombe sur ``"auto"``.
        strict: si vrai, lève une erreur au lieu de basculer sur le CPU quand
            CUDA est demandé mais indisponible.
    """
    spec = (preferred or os.environ.get(ENV_VAR) or "auto").strip().lower()
    if not spec.startswith(_VALID_PREFIXES):
        raise ValueError(f"Appareil inconnu {spec!r} (attendu : auto, cpu, cuda, cuda:N).")

    if spec == "auto":
        return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if spec == "cpu":
        return torch.device("cpu")

    # spec commence par "cuda"
    if torch.cuda.is_available():
        return _cuda_device(spec)
    if strict:
        raise RuntimeError(f"{spec!r} demandé mais CUDA n'est pas disponible sur cette machine.")
    logger.warning("%r demandé mais CUDA est indisponible : bascule sur le CPU.", spec)
    return torch.device("cpu")


def describe_device(device: torch.device) -> DeviceInfo:
    """Collecte les informations matérielles de ``device``."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return DeviceInfo(
            device=device,
            name=props.name,
            total_memory_gb=props.total_memory / 1024**3,
            cuda_version=torch.version.cuda,
        )
    import platform

    return DeviceInfo(
        device=device,
        name=platform.processor() or platform.machine() or "CPU",
        cpu_threads=torch.get_num_threads(),
    )


def get_device(
    preferred: str | None = None,
    *,
    strict: bool = False,
    num_threads: int | None = None,
) -> torch.device:
    """Résout l'appareil, le configure et journalise clairement le choix.

    Args:
        preferred: voir :func:`resolve_device`.
        strict: voir :func:`resolve_device`.
        num_threads: nombre de threads intra-op PyTorch en mode CPU
            (par défaut : valeur choisie par PyTorch).
    """
    device = resolve_device(preferred, strict=strict)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        # cuDNN choisit l'algorithme de convolution le plus rapide pour des tailles fixes
        torch.backends.cudnn.benchmark = True
        logger.info("Calcul sur GPU : %s", describe_device(device))
    else:
        if num_threads:
            torch.set_num_threads(num_threads)
        logger.info("Calcul sur CPU : %s", describe_device(device))
        logger.warning(
            "Aucun GPU utilisé : l'inférence 3D sur IRM cérébrale peut prendre plusieurs "
            "minutes par patient et consommer beaucoup de RAM. Réduisez la taille des patchs "
            "(--patch-size) si la mémoire est insuffisante."
        )
    return device


def autocast(device: torch.device):
    """Contexte de précision mixte : float16 sur GPU, désactivé sur CPU.

    Sur CPU, l'AMP bfloat16 n'apporte pas de gain fiable pour les convolutions 3D ;
    on reste donc en float32 pour la reproductibilité.
    """
    enabled = device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)
