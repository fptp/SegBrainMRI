"""Fixtures : volumes NIfTI synthétiques (aucune donnée patient dans le dépôt)."""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk


def make_phantom(size=(48, 40, 16), spacing=(1.0, 1.2, 3.0), seed=0) -> sitk.Image:
    """Ellipsoïde "cerveau" bruité avec deux "lésions" hyperintenses, sur fond nul."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = size
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, nz), np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing="ij"
    )
    brain = (x**2 + y**2 + z**2) < 0.7
    arr = np.where(brain, 400.0, 0.0) + rng.normal(0, 10, brain.shape) * brain
    arr[((x - 0.3) ** 2 + y**2 + z**2) < 0.02] = 900.0
    arr[((x + 0.3) ** 2 + (y - 0.2) ** 2 + z**2) < 0.02] = 900.0
    image = sitk.GetImageFromArray(arr.astype(np.float32))
    image.SetSpacing(spacing)
    image.SetOrigin((-10.0, 5.0, 2.0))
    return image


@pytest.fixture
def phantom() -> sitk.Image:
    return make_phantom()


@pytest.fixture
def flair_path(tmp_path, phantom):
    path = tmp_path / "sub-01_FLAIR.nii.gz"
    sitk.WriteImage(phantom, str(path))
    return path


@pytest.fixture
def t1_path(tmp_path):
    path = tmp_path / "sub-01_T1w.nii.gz"
    sitk.WriteImage(make_phantom(seed=1), str(path))
    return path
