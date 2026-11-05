import numpy as np
import SimpleITK as sitk
import torch

from segbrainmri.cli import main
from segbrainmri.inference import LesionSegmenter, sliding_window_inference
from segbrainmri.models import available_models, build_model
from segbrainmri.preprocessing import PreprocessingConfig, PreprocessingPipeline


def test_registry():
    assert "unet3d" in available_models()


def test_unet_shape():
    model = build_model("unet3d", in_channels=2, base_features=4, depth=3)
    out = model(torch.zeros(1, 2, 16, 16, 16))
    assert out.shape == (1, 1, 16, 16, 16)


def test_sliding_window_identity_reconstructs_volume():
    volume = torch.rand(1, 1, 21, 17, 9)  # dimensions non multiples, plus petit qu'un patch en Z
    out = sliding_window_inference(volume, lambda x: x * 2, patch_size=(8, 8, 16), overlap=0.5)
    assert out.shape == volume.shape
    torch.testing.assert_close(out, volume * 2)


def test_weights_roundtrip(tmp_path):
    model = build_model("unet3d", base_features=4, depth=2)
    path = tmp_path / "w.pt"
    torch.save({"state_dict": model.state_dict()}, path)
    other = build_model("unet3d", base_features=4, depth=2)
    other.load_weights(path, torch.device("cpu"))
    for a, b in zip(model.state_dict().values(), other.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b)


def test_end_to_end_cpu(flair_path, tmp_path):
    model = build_model("unet3d", base_features=4, depth=3)
    segmenter = LesionSegmenter(
        model,
        device="cpu",
        pipeline=PreprocessingPipeline(PreprocessingConfig(target_spacing=(2.0, 2.0, 2.0))),
        patch_size=(16, 16, 16),
    )
    result = segmenter.segment({"flair": flair_path})
    original = sitk.ReadImage(str(flair_path))
    assert result.mask.GetSize() == original.GetSize()
    assert set(np.unique(sitk.GetArrayFromImage(result.mask))) <= {0, 1}
    paths = result.save(tmp_path / "out")
    assert paths["mask"].is_file() and paths["probabilities"].is_file()


def test_cli_info(capsys):
    assert main(["--device", "cpu", "info"]) == 0
    assert "Appareil" in capsys.readouterr().out


def test_cli_segment_requires_weights(flair_path, tmp_path):
    assert main(["segment", "--flair", str(flair_path), "-o", str(tmp_path)]) == 2


def test_cli_segment_untrained(flair_path, tmp_path):
    code = main(
        [
            "--device",
            "cpu",
            "segment",
            "--flair",
            str(flair_path),
            "-o",
            str(tmp_path / "o"),
            "--untrained",
            "--spacing",
            "2",
            "--patch-size",
            "16",
        ]
    )
    assert code == 0
    assert (tmp_path / "o" / "lesions_mask.nii.gz").is_file()


def test_cli_preprocess(flair_path, tmp_path):
    assert main(["preprocess", "--flair", str(flair_path), "-o", str(tmp_path / "p")]) == 0
    assert (tmp_path / "p" / "flair_preproc.nii.gz").is_file()


def test_lesion_statistics():
    import SimpleITK as sitk

    from segbrainmri.inference.predictor import lesion_statistics

    arr = np.zeros((10, 10, 10), np.uint8)
    arr[1:3, 1:3, 1:3] = 1  # 8 voxels
    arr[6:8, 6:8, 6:8] = 1  # 8 voxels
    arr[9, 0, 0] = 1  # 1 voxel : sous le seuil de taille, pas compté comme lésion
    mask = sitk.GetImageFromArray(arr)
    mask.SetSpacing((2.0, 2.0, 2.0))  # 8 mm³ par voxel
    count, volume_ml = lesion_statistics(mask)
    assert count == 2
    assert abs(volume_ml - 17 * 8 / 1000) < 1e-9
