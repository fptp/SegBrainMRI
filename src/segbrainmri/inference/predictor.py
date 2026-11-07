"""Orchestration de bout en bout : NIfTI -> prétraitement -> modèle -> masque NIfTI."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from segbrainmri.core.device import autocast, get_device
from segbrainmri.inference.sliding_window import sliding_window_inference
from segbrainmri.inference.stats import lesion_statistics
from segbrainmri.models.base import SegmentationModel
from segbrainmri.preprocessing import PreprocessingPipeline, save_nifti

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Masque binaire dans l'espace natif du patient + statistiques lésionnelles."""

    mask: sitk.Image
    probabilities: sitk.Image
    lesion_count: int
    lesion_volume_ml: float
    elapsed_s: float

    def save(self, output_dir: str | Path, prefix: str = "lesions") -> dict[str, Path]:
        output_dir = Path(output_dir)
        return {
            "mask": save_nifti(self.mask, output_dir / f"{prefix}_mask.nii.gz"),
            "probabilities": save_nifti(self.probabilities, output_dir / f"{prefix}_prob.nii.gz"),
        }


class LesionSegmenter:
    """Segmente les lésions de SEP d'un patient.

    Exemple::

        model = build_model("unet3d", in_channels=1)
        model.load_weights("weights/unet3d_flair.pt", device)
        seg = LesionSegmenter(model, device="auto")
        result = seg.segment({"flair": "sub-01_FLAIR.nii.gz"})
        result.save("data/output/sub-01")
    """

    def __init__(
        self,
        model: SegmentationModel,
        *,
        device: str | torch.device | None = None,
        pipeline: PreprocessingPipeline | None = None,
        patch_size: tuple[int, int, int] = (128, 128, 128),
        overlap: float = 0.5,
        threshold: float = 0.5,
    ) -> None:
        self.device = device if isinstance(device, torch.device) else get_device(device)
        self.model = model.to(self.device).eval()
        self.pipeline = pipeline or PreprocessingPipeline()
        divisor = getattr(model, "divisor", 1)
        if any(p % divisor for p in patch_size):
            raise ValueError(f"patch_size {patch_size} doit être multiple de {divisor}.")
        self.patch_size = tuple(patch_size)
        self.overlap = overlap
        self.threshold = threshold

    def _predict(self, patches: torch.Tensor) -> torch.Tensor:
        with autocast(self.device):
            return self.model(patches)

    @torch.inference_mode()
    def segment(self, inputs: Mapping[str, str | Path | sitk.Image]) -> SegmentationResult:
        start = time.perf_counter()
        case = self.pipeline.run(inputs)

        volume = torch.from_numpy(case.data).unsqueeze(0).to(self.device)
        logits = sliding_window_inference(
            volume, self._predict, patch_size=self.patch_size, overlap=self.overlap
        )
        if logits.shape[1] == 1:
            prob = torch.sigmoid(logits)[0, 0]
        else:  # multi-classe : probabilité de "non-fond"
            prob = 1.0 - torch.softmax(logits, dim=1)[0, 0]
        prob_np = prob.cpu().numpy().astype(np.float32)
        if case.mask is not None:
            prob_np[~case.mask] = 0.0

        prob_native = case.restore_to_original(prob_np, is_label=False)
        mask_native = sitk.Cast(prob_native > self.threshold, sitk.sitkUInt8)
        count, volume_ml = lesion_statistics(mask_native)
        elapsed = time.perf_counter() - start
        logger.info(
            "Segmentation terminée en %.1f s sur %s : %d lésion(s), %.2f mL",
            elapsed,
            self.device,
            count,
            volume_ml,
        )
        return SegmentationResult(mask_native, prob_native, count, volume_ml, elapsed)
