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
    patient_id: str = "PHANTOM001",
    study_date: str = "20260101",
    origin: tuple[float, float, float] = (-50.0, -60.0, 0.0),
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
        ds.PatientName, ds.PatientID = "TEST^PHANTOM", patient_id
        ds.PatientBirthDate, ds.PatientSex = "19800101", "O"
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, generate_uid()
        ds.FrameOfReferenceUID = frame_uid
        ds.StudyDate, ds.StudyTime, ds.StudyID, ds.AccessionNumber = study_date, "120000", "1", "A1"
        ds.ReferringPhysicianName = ""
        ds.Modality, ds.Manufacturer = "MR", "ACME"
        ds.SeriesDescription, ds.SeriesNumber, ds.InstanceNumber = description, 3, int(k) + 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND"]
        ds.ScanningSequence, ds.SequenceVariant, ds.MRAcquisitionType = "SE\\IR", "SK", "3D"
        ds.ImagePositionPatient = [origin[0], origin[1], origin[2] + float(k) * spacing[2]]
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


def brain_phantom(
    lesions: list[tuple[tuple[float, float, float], float]],
    *,
    shape: tuple[int, int, int] = (36, 72, 64),
    spacing: tuple[float, float, float] = (2.0, 2.0, 3.0),
    origin: tuple[float, float, float] = (-64.0, -72.0, -54.0),
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    seed: int = 0,
) -> np.ndarray:
    """ "Cerveau" asymétrique (recalage bien posé) + lésions hyperintenses.

    La tête du patient est tournée/translatée de ``rotation_deg`` / ``translation_mm``
    (repositionnement entre deux examens). Lésions : ``((x, y, z) mm, rayon mm)`` dans
    le repère anatomique. Renvoie ``(coupes, lignes, colonnes)`` ; géométrie identique à
    :func:`write_mr_series` avec les mêmes ``spacing`` et ``origin``.
    """
    from scipy.spatial.transform import Rotation

    k, j, i = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    points = np.stack(
        [origin[0] + i * spacing[0], origin[1] + j * spacing[1], origin[2] + k * spacing[2]],
        axis=-1,
    )
    rot = Rotation.from_euler("xyz", rotation_deg, degrees=True)
    q = rot.inv().apply((points - np.asarray(translation_mm)).reshape(-1, 3)).reshape(points.shape)
    x, y, z = q[..., 0], q[..., 1], q[..., 2]

    def ellipsoid(c, r):
        return sum(((v - cv) / rv) ** 2 for v, cv, rv in zip((x, y, z), c, r, strict=True)) < 1

    volume = np.zeros(shape, np.float32)
    volume[ellipsoid((0, 0, 0), (55, 62, 45))] = 800.0
    volume[ellipsoid((-12, 8, 4), (7, 20, 9))] = 250.0  # "ventricule" gauche
    volume[ellipsoid((14, 2, 4), (5, 14, 7))] = 250.0  # "ventricule" droit, plus petit
    volume[ellipsoid((5, -40, -15), (20, 10, 12))] = 1100.0  # structure postérieure
    for center, radius in lesions:
        volume[ellipsoid(center, (radius, radius, radius))] = 2000.0
    rng = np.random.default_rng(seed)
    return volume + rng.normal(0, 15, shape).astype(np.float32) * (volume > 0)
