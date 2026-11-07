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
