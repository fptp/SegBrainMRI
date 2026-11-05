"""Registre des architectures disponibles (nom -> classe)."""

from __future__ import annotations

from collections.abc import Callable

from segbrainmri.models.base import SegmentationModel

_REGISTRY: dict[str, type[SegmentationModel]] = {}


def register_model(name: str) -> Callable[[type[SegmentationModel]], type[SegmentationModel]]:
    def decorator(cls: type[SegmentationModel]) -> type[SegmentationModel]:
        if name in _REGISTRY:
            raise KeyError(f"Modèle déjà enregistré : {name!r}")
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def build_model(name: str, **kwargs) -> SegmentationModel:
    try:
        return _REGISTRY[name](**kwargs)
    except KeyError:
        raise KeyError(f"Modèle inconnu {name!r}. Disponibles : {available_models()}") from None
