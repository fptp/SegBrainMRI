"""Génération de séries DICOM MR synthétiques (aucune donnée patient)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid


def write_mr_series(
    directory: Path,
    volume: np.ndarray,
    *,
    description: str = "AX 3D FLAIR",
    spacing: tuple[float, float, float] = (1.0, 1.0, 2.0),
    study_uid: str | None = None,
    shuffle: bool = True,
) -> list[Path]:
    """Écrit ``volume`` ``(coupes, lignes, colonnes)`` en série MR mono-trame (uint16)."""
    directory.mkdir(parents=True, exist_ok=True)
    study_uid = study_uid or generate_uid()
    series_uid, frame_uid = generate_uid(), generate_uid()
    data = np.clip(volume, 0, 4095).astype(np.uint16)
    order = np.random.default_rng(0).permutation(len(data)) if shuffle else range(len(data))
    paths = []
    for k in order:
        ds = Dataset()
        ds.PatientName, ds.PatientID = "TEST^PHANTOM", "PHANTOM001"
        ds.PatientBirthDate, ds.PatientSex = "19800101", "O"
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, generate_uid()
        ds.FrameOfReferenceUID = frame_uid
        ds.StudyDate, ds.StudyTime, ds.StudyID, ds.AccessionNumber = "20260101", "120000", "1", "A1"
        ds.ReferringPhysicianName = ""
        ds.Modality, ds.Manufacturer = "MR", "ACME"
        ds.SeriesDescription, ds.SeriesNumber, ds.InstanceNumber = description, 3, int(k) + 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND"]
        ds.ScanningSequence, ds.SequenceVariant, ds.MRAcquisitionType = "SE\\IR", "SK", "3D"
        ds.ImagePositionPatient = [-50.0, -60.0, float(k) * spacing[2]]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [spacing[1], spacing[0]]
        ds.SliceThickness = spacing[2]
        ds.Rows, ds.Columns = data.shape[1], data.shape[2]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
        ds.PixelData = data[k].tobytes()
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = (
            ds.SOPClassUID,
            ds.SOPInstanceUID,
        )
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta = meta
        path = directory / f"{ds.SOPInstanceUID}.dcm"
        ds.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths


def phantom_volume(shape=(20, 48, 40)) -> tuple[np.ndarray, np.ndarray]:
    """Cerveau ellipsoïde + 2 lésions hyperintenses ; renvoie (volume, masque lésions)."""
    z, y, x = np.meshgrid(*(np.linspace(-1, 1, s) for s in shape), indexing="ij")
    brain = (x**2 + y**2 + z**2) < 0.7
    lesions = (((x - 0.3) ** 2 + y**2 + z**2) < 0.03) | (((x + 0.3) ** 2 + y**2 + z**2) < 0.03)
    volume = np.where(brain, 800.0, 0.0)
    volume[lesions] = 2000.0
    return volume, lesions.astype(np.uint8)
