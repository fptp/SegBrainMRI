import numpy as np
import pytest
import SimpleITK as sitk

from segbrainmri.preprocessing import PreprocessingConfig, PreprocessingPipeline, load_nifti
from segbrainmri.preprocessing.intensity import normalize_intensity
from segbrainmri.preprocessing.spatial import reorient, resample_to_spacing


def test_load_nifti(flair_path):
    image = load_nifti(flair_path)
    assert image.GetDimension() == 3
    assert image.GetSize() == (48, 40, 16)


def test_load_rejects_other_formats(tmp_path):
    bad = tmp_path / "scan.png"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError):
        load_nifti(bad)
    with pytest.raises(FileNotFoundError):
        load_nifti(tmp_path / "absent.nii.gz")


def test_resample_preserves_extent(phantom):
    out = resample_to_spacing(phantom, (1.0, 1.0, 1.0))
    assert out.GetSpacing() == (1.0, 1.0, 1.0)
    assert out.GetSize() == (48, 48, 48)
    assert out.GetOrigin() == phantom.GetOrigin()


def test_reorient_las_roundtrip(phantom):
    ras = reorient(phantom, "RAS")
    back = reorient(ras, "LPS")
    np.testing.assert_allclose(sitk.GetArrayFromImage(back), sitk.GetArrayFromImage(phantom))
    assert reorient(phantom, None) is phantom


def test_zscore_inside_mask():
    rng = np.random.default_rng(0)
    arr = rng.normal(100, 20, (10, 10, 10)).astype(np.float32)
    mask = np.zeros_like(arr, dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    out = normalize_intensity(arr, mask, clip_percentiles=None)
    assert abs(out[mask].mean()) < 1e-4
    assert abs(out[mask].std() - 1) < 1e-3
    assert (out[~mask] == 0).all()


def test_minmax():
    arr = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    out = normalize_intensity(arr, method="minmax", clip_percentiles=None)
    assert out.min() == 0 and out.max() == 1


def test_invalid_config():
    with pytest.raises(ValueError):
        PreprocessingConfig(normalization="foo")
    with pytest.raises(ValueError):
        PreprocessingConfig(target_spacing=(1, 0, 1))


def test_pipeline_multimodal_and_restore(flair_path, t1_path, tmp_path):
    pipeline = PreprocessingPipeline(PreprocessingConfig(target_spacing=(2.0, 2.0, 2.0)))
    case = pipeline.run({"flair": flair_path, "t1": t1_path})

    assert case.modalities == ["flair", "t1"]
    assert case.data.dtype == np.float32
    assert case.data.shape[0] == 2
    assert case.data.shape[1:] == case.mask.shape
    assert case.reference.GetSpacing() == (2.0, 2.0, 2.0)
    assert abs(case.data[0][case.mask].mean()) < 1e-3

    restored = case.restore_to_original(case.mask.astype(np.uint8))
    assert restored.GetSize() == case.original.GetSize()
    assert restored.GetSpacing() == case.original.GetSpacing()

    written = pipeline.save(case, tmp_path / "out")
    assert [p.name for p in written] == ["flair_preproc.nii.gz", "t1_preproc.nii.gz"]
    assert all(p.is_file() for p in written)
