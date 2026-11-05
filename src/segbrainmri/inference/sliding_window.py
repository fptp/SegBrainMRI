"""Inférence par fenêtre glissante, pour tenir en mémoire sur CPU comme sur GPU."""

from __future__ import annotations

import itertools
from collections.abc import Callable

import torch
import torch.nn.functional as F


def _starts(size: int, patch: int, step: int) -> list[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch, step))
    starts.append(size - patch)  # dernier patch collé au bord
    return starts


def _gaussian_weight(patch: tuple[int, int, int], sigma_scale: float = 0.125) -> torch.Tensor:
    """Pondération gaussienne : le centre des patchs compte plus que les bords."""
    axes = []
    for p in patch:
        coords = torch.arange(p, dtype=torch.float32) - (p - 1) / 2
        axes.append(torch.exp(-(coords**2) / (2 * (p * sigma_scale) ** 2)))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return (weight / weight.max()).clamp_min(1e-3)


@torch.no_grad()
def sliding_window_inference(
    volume: torch.Tensor,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    patch_size: tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.5,
    batch_size: int = 1,
) -> torch.Tensor:
    """Applique ``predictor`` sur des patchs recouvrants et fusionne les sorties.

    Args:
        volume: tenseur ``(1, C, Z, Y, X)`` sur l'appareil de calcul.
        predictor: ``(B, C, z, y, x) -> (B, K, z, y, x)``.
        patch_size: taille des patchs (réduire en cas de manque de mémoire).
        overlap: recouvrement relatif entre patchs, dans ``[0, 1)``.

    Returns:
        Logits ``(1, K, Z, Y, X)`` en float32, sur le même appareil que ``volume``.
    """
    if volume.ndim != 5 or volume.shape[0] != 1:
        raise ValueError(f"Tenseur (1, C, Z, Y, X) attendu, reçu {tuple(volume.shape)}")
    if not 0 <= overlap < 1:
        raise ValueError("overlap doit être dans [0, 1).")

    spatial = volume.shape[2:]
    # Remplissage si le volume est plus petit qu'un patch
    pad = [max(0, p - s) for p, s in zip(patch_size, spatial, strict=True)]
    if any(pad):
        volume = F.pad(volume, [0, pad[2], 0, pad[1], 0, pad[0]])
    padded = volume.shape[2:]

    steps = [max(1, int(p * (1 - overlap))) for p in patch_size]
    grid = list(
        itertools.product(
            *(_starts(s, p, st) for s, p, st in zip(padded, patch_size, steps, strict=True))
        )
    )
    weight = _gaussian_weight(patch_size).to(volume.device)

    output: torch.Tensor | None = None
    norm = torch.zeros((1, 1, *padded), device=volume.device)
    for i in range(0, len(grid), batch_size):
        batch_starts = grid[i : i + batch_size]
        slices = [
            tuple(slice(s, s + p) for s, p in zip(start, patch_size, strict=True))
            for start in batch_starts
        ]
        patches = torch.cat([volume[(..., *sl)] for sl in slices], dim=0)
        logits = predictor(patches).float()
        if output is None:
            output = torch.zeros((1, logits.shape[1], *padded), device=volume.device)
        for j, sl in enumerate(slices):
            output[(..., *sl)] += logits[j : j + 1] * weight
            norm[(..., *sl)] += weight

    assert output is not None
    output /= norm
    return output[..., : spatial[0], : spatial[1], : spatial[2]]
