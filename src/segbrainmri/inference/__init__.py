"""Inférence : IRM prétraitée -> masque de lésions."""

from segbrainmri.inference.predictor import LesionSegmenter, SegmentationResult
from segbrainmri.inference.sliding_window import sliding_window_inference

__all__ = ["LesionSegmenter", "SegmentationResult", "sliding_window_inference"]
