import zipfile

import pytest
import torch

from segbrainmri.models.flames import (
    MODEL_SUBDIR,
    FlamesBackend,
    SynthStrip,
    _safe_extract,
    find_model_folder,
)


def _fake_model_tree(root):
    folder = root / "flames" / "Dataset004_WML" / MODEL_SUBDIR
    (folder / "fold_0").mkdir(parents=True)
    (folder / "plans.json").write_text("{}")
    (folder / "dataset.json").write_text("{}")
    return folder


def test_find_model_folder(tmp_path):
    assert find_model_folder(tmp_path / "absent") is None
    folder = _fake_model_tree(tmp_path)
    assert find_model_folder(tmp_path / "flames") == folder


def test_safe_extract_rejects_zip_slip(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../evil.txt", "x")
    with pytest.raises(RuntimeError):
        _safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_backend_requires_downloaded_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="download-models"):
        FlamesBackend(tmp_path, device="cpu")


def test_backend_tta_defaults(tmp_path):
    _fake_model_tree(tmp_path)
    backend = FlamesBackend(tmp_path, device="cpu", skull_strip=False)
    assert backend.tta is False  # pas d'augmentation au test sur CPU
    assert FlamesBackend(tmp_path, device="cpu", skull_strip=False, tta=True).tta is True


def test_synthstrip_command(tmp_path):
    """Le wrapper appelle le script avec les bons arguments (script factice local)."""
    script = tmp_path / "mri_synthstrip"
    script.write_text(
        "import sys, pathlib\n"
        "a = sys.argv\n"
        "pathlib.Path(a[a.index('-o') + 1]).write_text(' '.join(a[1:]))\n"
        "pathlib.Path(a[a.index('-m') + 1]).write_text('mask')\n"
    )
    weights = tmp_path / "synthstrip.1.pt"
    weights.write_bytes(b"")
    stripper = SynthStrip(script, weights, torch.device("cpu"), threads=2)
    stripper.run(tmp_path / "in.nii.gz", tmp_path / "out.nii.gz", tmp_path / "mask.nii.gz")
    args = (tmp_path / "out.nii.gz").read_text()
    assert f"--model {weights}" in args and "--threads 2" in args and "--gpu" not in args


def test_synthstrip_failure_is_reported(tmp_path):
    script = tmp_path / "mri_synthstrip"
    script.write_text("import sys\nprint('boom', file=sys.stderr)\nsys.exit(3)\n")
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"")
    with pytest.raises(RuntimeError, match="boom"):
        SynthStrip(script, weights, torch.device("cpu")).run(
            tmp_path / "a", tmp_path / "b", tmp_path / "c"
        )


FAKE_SYNTHSTRIP = """
import sys
import numpy as np
import surfa as sf

a = sys.argv
src, out, mask = a[a.index("-i") + 1], a[a.index("-o") + 1], a[a.index("-m") + 1]
image = sf.load_volume(src)
# Mêmes appels surfa que mri_synthstrip (c'est ici que le bug numpy 2 se produisait)
conformed = image.conform(voxsize=1.0, dtype="float32", method="nearest", orientation="LIA")
conformed = conformed.crop_to_bbox()  # 2e bug (scipy récent) : corrigé par _surfa_compat
back = conformed.resample_like(image, fill=0)
brain = image.new((back.data > 0.5).astype(np.uint8))
stripped = image.copy()
stripped[brain == 0] = 0
stripped.save(out)
brain.save(mask)
"""


def _oblique_sagittal(shape=(30, 48, 40)):
    import numpy as np
    import SimpleITK as sitk

    z, y, x = np.meshgrid(*(np.linspace(-1, 1, s) for s in shape), indexing="ij")
    arr = np.where(x**2 + y**2 + z**2 < 0.6, 1.0, 0.0).astype(np.float32)
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((0.5, 0.5, 0.9))
    sag = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], float)  # GE CUBE sagittal
    t = np.deg2rad(7)  # légère obliquité, comme une vraie acquisition
    rot = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    image.SetDirection(tuple((rot @ sag).ravel()))
    image.SetOrigin((-30.0, -100.0, 80.0))
    return image


def test_synthstrip_handles_sagittal_with_numpy2(tmp_path):
    """FLAIR sagittale oblique : surfa plantait (numpy 2) sans la pré-orientation LIA."""
    import numpy as np
    import SimpleITK as sitk

    script = tmp_path / "mri_synthstrip"
    script.write_text(FAKE_SYNTHSTRIP)
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"")
    image = _oblique_sagittal()
    stripper = SynthStrip(script, weights, torch.device("cpu"))

    stripped, brain = stripper.strip(image, tmp_path)
    for result in (stripped, brain):
        assert result.GetSize() == image.GetSize()
        assert result.GetDirection() == image.GetDirection()
        assert result.GetOrigin() == image.GetOrigin()
    inside = sitk.GetArrayFromImage(image) > 0
    overlap = (sitk.GetArrayFromImage(brain) > 0) & inside
    assert overlap.sum() > 0.9 * inside.sum()  # masque aligné sur l'anatomie, pas décalé
    np.testing.assert_array_equal(sitk.GetArrayFromImage(stripped)[~inside], 0)

    # Sans le correctif (image sagittale passée telle quelle), surfa échoue
    raw = tmp_path / "raw.nii.gz"
    sitk.WriteImage(image, str(raw))
    with pytest.raises(RuntimeError, match="sequence"):
        stripper.run(raw, tmp_path / "o.nii.gz", tmp_path / "m.nii.gz")


def test_surfa_compat_fixes_crop_to_bbox(tmp_path):
    """Sans le correctif, crop_to_bbox plante avec scipy récent ; avec, il fonctionne."""
    import subprocess
    import sys

    code = (
        "import numpy as np, surfa as sf\n"
        "{patch}"
        "v = sf.Volume(np.pad(np.ones((4, 5, 6), np.float32), 3))\n"
        "print(v.crop_to_bbox().shape)\n"
    )
    patch = "from segbrainmri.models import _surfa_compat; _surfa_compat.apply()\n"
    fixed = subprocess.run(
        [sys.executable, "-c", code.format(patch=patch)], capture_output=True, text=True
    )
    assert fixed.returncode == 0, fixed.stderr
    assert "(4, 5, 6)" in fixed.stdout
    raw = subprocess.run(
        [sys.executable, "-c", code.format(patch="")], capture_output=True, text=True
    )
    if raw.returncode == 0:
        pytest.skip("surfa/scipy installés n'ont pas le bug (correctif inutile mais sans effet)")
    assert "numpy.bool" in raw.stderr
