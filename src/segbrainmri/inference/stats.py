"""Statistiques lésionnelles (sans dépendance à PyTorch)."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def lesion_statistics(mask: sitk.Image, min_size_voxels: int = 3) -> tuple[int, float]:
    """Nombre de lésions (composantes 26-connexes >= ``min_size_voxels``) et volume en mL."""
    cc = sitk.ConnectedComponentImageFilter()
    cc.SetFullyConnected(True)
    components = sitk.RelabelComponent(cc.Execute(mask), minimumObjectSize=min_size_voxels)
    count = int(sitk.GetArrayViewFromImage(components).max())
    voxel_ml = float(np.prod(mask.GetSpacing())) / 1000.0
    volume = float((sitk.GetArrayViewFromImage(mask) > 0).sum()) * voxel_ml
    return count, volume
