"""Création des objets DICOM résultats.

* **DICOM SEG** (standard) : masque binaire des lésions référencé sur la FLAIR source,
  affichable en surimpression dans OHIF, Stone Web Viewer, 3D Slicer, Weasis...
* **Série contours** (MR dérivée, niveaux de gris) : la FLAIR d'origine, pixels inchangés,
  avec le contour des lésions tracé à la valeur maximale. Elle se fenêtre comme la série
  native dans toutes les visionneuses, et les contours restent blancs.
* **Overlay couleur** (Secondary Capture RGB, optionnel) : lésions en rouge, mais une image
  couleur ne peut pas être fenêtrée.

Toutes les chaînes écrites sont en ASCII (pas de jeu de caractères à déclarer).
"""

from __future__ import annotations

import copy
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


#: Couleurs CIELab des segments (rouge, orange, jaune).
RED, ORANGE, YELLOW = (53.0, 80.0, 67.0), (67.0, 43.0, 74.0), (97.0, -22.0, 94.0)


def _check_shape(array: np.ndarray, sources: list[Dataset]) -> None:
    if array.shape != (len(sources), sources[0].Rows, sources[0].Columns):
        raise ValueError(f"Masque {array.shape} incompatible avec la série source.")


def build_labelmap_seg(
    labelmap: np.ndarray,
    sources: list[Dataset],
    info: ResultInfo,
    segments: list[tuple[str, tuple[float, float, float]]],
    *,
    series_description: str,
    series_offset: int = 9000,
) -> Dataset:
    """DICOM SEG à plusieurs segments. ``labelmap`` : 0 = fond, ``k`` = ``segments[k-1]``."""
    _check_shape(labelmap, sources)
    algorithm = hd.AlgorithmIdentificationSequence(
        name=info.algorithm,
        version=info.algorithm_version[:64],
        family=codes.cid7162.ArtificialIntelligence,
    )
    descriptions = [
        hd.seg.SegmentDescription(
            segment_number=k,
            segment_label=label[:64],
            segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
            segmented_property_type=codes.SCT.Lesion,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm,
            anatomic_regions=[codes.SCT.Brain],
            display_color=hd.color.CIELabColor(*color),
        )
        for k, (label, color) in enumerate(segments, start=1)
    ]
    # Une couche binaire par segment (accepte les segments vides)
    stacked = np.stack([labelmap == k for k in range(1, len(segments) + 1)], axis=-1)
    return hd.seg.Segmentation(
        source_images=sources,
        pixel_array=stacked,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=descriptions,
        series_instance_uid=hd.UID(),
        series_number=_series_number(sources[0], series_offset),
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer=OUR_MANUFACTURER,
        manufacturer_model_name=info.algorithm,
        software_versions=f"segbrainmri {__version__}",
        device_serial_number="0",
        series_description=series_description[:64],
        content_description=f"MS lesions: {info.summary}"[:64],
        content_label="LESIONS",
        omit_empty_frames=bool(labelmap.any()),
    )


def build_seg(mask: np.ndarray, sources: list[Dataset], info: ResultInfo) -> Dataset:
    """DICOM SEG binaire. ``mask`` : ``(coupes, lignes, colonnes)`` aligné sur ``sources``."""
    return build_labelmap_seg(
        (mask > 0).astype(np.uint8),
        sources,
        info,
        [("Lesions SEP", RED)],
        series_description=f"SEP lesions {info.algorithm} - {info.summary}",
    )


def _contour(mask: np.ndarray, thickness: int) -> np.ndarray:
    """Contour extérieur 2D (coupe par coupe) : ne masque aucun pixel de la lésion."""
    from scipy.ndimage import binary_dilation

    structure = np.zeros((3, 3, 3), bool)
    structure[1] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]  # dilatation dans le plan uniquement
    lesion = mask.astype(bool)
    return binary_dilation(lesion, structure, iterations=thickness) & ~lesion


def build_contour_series(
    mask: np.ndarray,
    sources: list[Dataset],
    info: ResultInfo,
    *,
    series_description: str,
    series_offset: int = 9001,
    thickness_mm: float = 0.8,
) -> list[Dataset]:
    """Série MR dérivée : FLAIR d'origine (pixels inchangés) + contours des lésions.

    Les contours prennent la valeur maximale représentable : ils restent blancs quel que
    soit le fenêtrage, tandis que le reste de l'image se fenêtre comme la série native.
    """
    _check_shape(mask, sources)
    spacing = min(float(v) for v in sources[0].get("PixelSpacing", [1.0, 1.0]))
    contour = _contour(mask, max(1, round(thickness_mm / spacing)))
    series_uid = generate_uid()
    series_number = _series_number(sources[0], series_offset)
    out = []
    for k, src in enumerate(sources):
        ds = copy.deepcopy(src)
        pixels = src.pixel_array.copy()
        bits = int(src.get("BitsStored", src.BitsAllocated))
        signed = int(src.get("PixelRepresentation", 0)) == 1
        top = 2 ** (bits - 1) - 1 if signed else 2**bits - 1
        pixels[contour[k]] = top
        ds.PixelData = pixels.astype(pixels.dtype).tobytes()
        for tag in (
            "LargestImagePixelValue",
            "SmallestImagePixelValue",
            "LargestPixelValueInSeries",
            "SmallestPixelValueInSeries",
        ):
            if tag in ds:
                del ds[tag]
        ds.SOPInstanceUID = generate_uid()
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = series_number
        ds.SeriesDescription = series_description[:64]
        ds.ImageType = ["DERIVED", "SECONDARY"]
        ds.Manufacturer = OUR_MANUFACTURER
        ds.ManufacturerModelName = info.algorithm[:64]
        ds.SoftwareVersions = f"segbrainmri {__version__}"
        ds.DerivationDescription = f"Automatic MS lesion contours ({info.algorithm})"[:1024]
        ds.ImageComments = f"Automatic MS lesion segmentation ({info.algorithm}): {info.summary}"
        ref = Dataset()
        ref.ReferencedSOPClassUID = src.SOPClassUID
        ref.ReferencedSOPInstanceUID = src.SOPInstanceUID
        ds.SourceImageSequence = [ref]
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = ds.SOPClassUID
        meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        meta.TransferSyntaxUID = ExplicitVRLittleEndian  # pixels décompressés
        ds.file_meta = meta
        ds.set_original_encoding(False, True)  # explicite, petit-boutiste (comme file_meta)
        out.append(ds)
    return out


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
    rgb = _gray_rgb(volume, sources[0])
    lesion = mask.astype(bool)
    rgb[lesion] = (1 - opacity) * rgb[lesion] + opacity * np.asarray(color, dtype=float)
    rgb = rgb.round().astype(np.uint8)

    return _rgb_series(
        rgb,
        sources,
        info,
        series_description=f"SEP overlay {info.algorithm} - {info.summary}",
        series_offset=9002,
    )


def _gray_rgb(volume: np.ndarray, source: Dataset) -> np.ndarray:
    low, high = _window(volume, source)
    gray = np.clip((volume - low) / (high - low), 0, 1) * 255.0
    return np.repeat(gray[..., None], 3, axis=-1)


def build_color_contour_series(
    volume: np.ndarray,
    labelmap: np.ndarray,
    colors: dict[int, tuple[int, int, int]],
    sources: list[Dataset],
    info: ResultInfo,
    *,
    series_description: str,
    series_offset: int = 9003,
    thickness_mm: float = 0.8,
) -> list[Dataset]:
    """Série couleur : FLAIR en niveaux de gris + contour coloré de chaque catégorie.

    ``colors`` : ``{valeur de labelmap: (R, G, B)}`` ; les catégories absentes de
    ``colors`` ne sont pas dessinées. L'image est figée (non fenêtrable) : le fenêtrage
    par défaut de la série source est appliqué.
    """
    if volume.shape != labelmap.shape or volume.shape[0] != len(sources):
        raise ValueError("volume, masque et série source doivent avoir la même forme.")
    spacing = min(float(v) for v in sources[0].get("PixelSpacing", [1.0, 1.0]))
    thickness = max(1, round(thickness_mm / spacing))
    rgb = _gray_rgb(volume, sources[0])
    # Ordre inverse : la première catégorie de ``colors`` est dessinée par-dessus
    for value, color in reversed(list(colors.items())):
        rgb[_contour(labelmap == value, thickness)] = color
    return _rgb_series(
        rgb.round().astype(np.uint8),
        sources,
        info,
        series_description=series_description,
        series_offset=series_offset,
    )


def _rgb_series(
    rgb: np.ndarray,
    sources: list[Dataset],
    info: ResultInfo,
    *,
    series_description: str,
    series_offset: int,
) -> list[Dataset]:
    """Série Secondary Capture RGB (une image par coupe, géométrie de la source)."""
    series_uid = generate_uid()
    series_number = _series_number(sources[0], series_offset)
    description = series_description[:64]
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
