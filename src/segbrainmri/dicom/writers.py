"""Création des objets DICOM résultats.

* **DICOM SEG** (standard) : masque binaire des lésions référencé sur la FLAIR source,
  affichable en surimpression dans OHIF, Stone Web Viewer, 3D Slicer, Weasis...
* **Overlay couleur** (Secondary Capture RGB) : FLAIR avec lésions en rouge, lisible par
  n'importe quelle visionneuse, y compris l'explorateur de base d'Orthanc.
"""

from __future__ import annotations

from dataclasses import dataclass

import highdicom as hd
import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sr.codedict import codes
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from segbrainmri import __version__
from segbrainmri.dicom.series import OUR_MANUFACTURER

#: Attributs patient / examen recopiés pour que le résultat rejoigne l'examen d'origine.
_STUDY_TAGS = (
    "PatientName",
    "PatientID",
    "IssuerOfPatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "StudyInstanceUID",
    "StudyDate",
    "StudyTime",
    "StudyID",
    "AccessionNumber",
    "ReferringPhysicianName",
    "StudyDescription",
    "FrameOfReferenceUID",
    "PositionReferenceIndicator",
    "SpecificCharacterSet",
)
_GEOMETRY_TAGS = (
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "SliceThickness",
    "SliceLocation",
)


@dataclass
class ResultInfo:
    """Informations décrivant le résultat, reprises dans les descriptions de séries."""

    algorithm: str
    algorithm_version: str
    lesion_count: int
    lesion_volume_ml: float

    @property
    def summary(self) -> str:
        return f"{self.lesion_count} lesion(s), {self.lesion_volume_ml:.2f} mL"


def _series_number(source: Dataset, offset: int) -> int:
    return int(source.get("SeriesNumber", 0) or 0) + offset


def build_seg(mask: np.ndarray, sources: list[Dataset], info: ResultInfo) -> Dataset:
    """DICOM SEG binaire. ``mask`` : ``(coupes, lignes, colonnes)`` aligné sur ``sources``."""
    if mask.shape != (len(sources), sources[0].Rows, sources[0].Columns):
        raise ValueError(f"Masque {mask.shape} incompatible avec la série source.")
    description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label="Lesions SEP",
        segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
        segmented_property_type=codes.SCT.Lesion,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=hd.AlgorithmIdentificationSequence(
            name=info.algorithm,
            version=info.algorithm_version[:64],
            family=codes.cid7162.ArtificialIntelligence,
        ),
        anatomic_regions=[codes.SCT.Brain],
        display_color=hd.color.CIELabColor(53.0, 80.0, 67.0),  # rouge
    )
    binary = mask.astype(bool)
    seg = hd.seg.Segmentation(
        source_images=sources,
        pixel_array=binary,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[description],
        series_instance_uid=hd.UID(),
        series_number=_series_number(sources[0], 9000),
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer=OUR_MANUFACTURER,
        manufacturer_model_name=info.algorithm,
        software_versions=f"segbrainmri {__version__}",
        device_serial_number="0",
        series_description=f"SEP lesions {info.algorithm} - {info.summary}"[:64],
        content_description=f"MS lesions: {info.summary}"[:64],
        content_label="LESIONS",
        omit_empty_frames=bool(binary.any()),
    )
    return seg


def _window(volume: np.ndarray, source: Dataset) -> tuple[float, float]:
    center, width = source.get("WindowCenter"), source.get("WindowWidth")
    if center is not None and width is not None:
        center = float(center[0] if hasattr(center, "__len__") else center)
        width = float(width[0] if hasattr(width, "__len__") else width)
        if width > 0:
            return center - width / 2, center + width / 2
    foreground = volume[volume > 0]
    values = foreground if foreground.size else volume
    low, high = np.percentile(values, (1.0, 99.5))
    return float(low), float(max(high, low + 1e-3))


def build_overlay(
    volume: np.ndarray,
    mask: np.ndarray,
    sources: list[Dataset],
    info: ResultInfo,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    opacity: float = 0.5,
) -> list[Dataset]:
    """Série Secondary Capture RGB : FLAIR en niveaux de gris + lésions colorées.

    ``volume`` et ``mask`` : ``(coupes, lignes, colonnes)`` alignés sur ``sources``.
    """
    if volume.shape != mask.shape or volume.shape[0] != len(sources):
        raise ValueError("volume, masque et série source doivent avoir la même forme.")
    low, high = _window(volume, sources[0])
    gray = np.clip((volume - low) / (high - low), 0, 1) * 255.0
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    lesion = mask.astype(bool)
    rgb[lesion] = (1 - opacity) * rgb[lesion] + opacity * np.asarray(color, dtype=float)
    rgb = rgb.round().astype(np.uint8)

    series_uid = generate_uid()
    series_number = _series_number(sources[0], 9001)
    description = f"SEP overlay {info.algorithm} - {info.summary}"[:64]
    out = []
    for k, src in enumerate(sources):
        ds = Dataset()
        for tag in _STUDY_TAGS + _GEOMETRY_TAGS:
            if tag in src:
                ds[tag] = src[tag]
        ds.SOPClassUID = SecondaryCaptureImageStorage
        ds.SOPInstanceUID = generate_uid()
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = series_number
        ds.InstanceNumber = k + 1
        ds.SeriesDescription = description
        ds.Modality = src.get("Modality", "MR")
        ds.ConversionType = "WSD"
        ds.ImageType = ["DERIVED", "SECONDARY"]
        ds.Manufacturer = OUR_MANUFACTURER
        ds.ManufacturerModelName = info.algorithm[:64]
        ds.SoftwareVersions = f"segbrainmri {__version__}"
        ds.BurnedInAnnotation = "NO"
        ds.ImageComments = f"Automatic MS lesion segmentation ({info.algorithm}): {info.summary}"
        ref = Dataset()
        ref.ReferencedSOPClassUID = src.SOPClassUID
        ref.ReferencedSOPInstanceUID = src.SOPInstanceUID
        ds.SourceImageSequence = [ref]

        ds.Rows, ds.Columns = rgb.shape[1], rgb.shape[2]
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
        ds.BitsAllocated = ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = rgb[k].tobytes()

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = ds.SOPClassUID
        meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta = meta
        out.append(ds)
    return out
