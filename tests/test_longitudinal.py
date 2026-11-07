import json
import shutil
import socket
import subprocess
import time
import urllib.request

import highdicom as hd
import numpy as np
import pytest
import SimpleITK as sitk
from scipy import ndimage

from segbrainmri.dicom.series import load_series
from segbrainmri.dicom.writers import ResultInfo, build_color_contour_series, build_contour_series
from segbrainmri.longitudinal import ENLARGING, NEW, SMALL, STABLE, classify_lesions, register_rigid
from segbrainmri.service.config import ServiceConfig
from segbrainmri.service.worker import process_series
from tests.dicom_utils import brain_phantom, phantom_volume, write_mr_series

SPACING, ORIGIN = (2.0, 2.0, 3.0), (-64.0, -72.0, -54.0)
OLD_LESIONS = [((-20, 15, 10), 6.0), ((25, -10, -5), 6.0)]
NEW_LESION = ((0, 30, 20), 6.0)
MOVE = {"rotation_deg": (4.0, -3.0, 6.0), "translation_mm": (5.0, -4.0, 3.0)}


class ThresholdBackend:
    name, version = "Threshold", "test"

    def segment(self, flair: sitk.Image) -> sitk.Image:
        return sitk.Cast(flair > 1500, sitk.sitkUInt8)


def _image(volume: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing(SPACING)
    image.SetOrigin(ORIGIN)
    return image


# --------------------------------------------------------------------------- unités
def test_classify_lesions():
    shape, spacing = (20, 40, 40), (1.0, 1.0, 1.0)
    prior = np.zeros(shape, np.uint8)
    current = np.zeros(shape, np.uint8)
    prior[5:8, 5:8, 5:8] = current[5:8, 5:8, 5:8] = 1  # stable (27 mm³)
    prior[5:8, 20:23, 20:23] = 1  # 27 mm³ ...
    current[4:10, 19:25, 19:25] = 1  # ... devenue 216 mm³ : élargie
    current[14:18, 30:34, 5:9] = 1  # nouvelle (64 mm³)
    current[15, 10, 30] = 1  # nouvelle mais 1 mm³ : sous le seuil
    result = classify_lesions(current, prior, spacing)
    assert (result.new, result.enlarging, result.stable, result.counts[SMALL]) == (1, 1, 1, 1)
    assert result.labelmap[15, 31, 6] == NEW
    assert result.labelmap[6, 21, 21] == ENLARGING
    assert result.labelmap[6, 6, 6] == STABLE
    assert result.labelmap[15, 10, 30] == STABLE  # petite : affichée avec les "autres"


def test_classify_tolerates_small_misregistration():
    shape, spacing = (20, 40, 40), (1.0, 1.0, 1.0)
    prior = np.zeros(shape, np.uint8)
    current = np.zeros(shape, np.uint8)
    prior[5:9, 5:9, 5:9] = 1
    current[5:9, 5:9, 9:13] = 1  # accolée : aucun voxel commun, mais à 1 mm
    assert classify_lesions(current, prior, spacing, tolerance_mm=1.0).new == 0
    assert classify_lesions(current, prior, spacing, tolerance_mm=0).new == 1
    far = np.zeros(shape, np.uint8)
    far[5:9, 5:9, 3:7] = 1  # 2 mm d'écart : au-delà d'une tolérance de 1 mm
    assert classify_lesions(current, far, spacing, tolerance_mm=1.0).new == 1


def test_register_rigid_recovers_head_motion():
    prior = _image(brain_phantom(OLD_LESIONS))
    current = _image(brain_phantom(OLD_LESIONS, **MOVE, seed=1))
    transform = register_rigid(current, prior)
    moved = sitk.Resample(prior, current, transform, sitk.sitkNearestNeighbor)
    a = sitk.GetArrayFromImage(moved) > 1500
    b = sitk.GetArrayFromImage(current) > 1500
    dice = 2 * (a & b).sum() / (a.sum() + b.sum())
    assert dice > 0.7, dice  # lésions superposées après recalage
    unregistered = sitk.GetArrayFromImage(prior) > 1500
    assert 2 * (unregistered & b).sum() / (unregistered.sum() + b.sum()) < dice


def test_contour_series_is_windowable_grayscale(tmp_path):
    volume, lesions = phantom_volume()
    write_mr_series(tmp_path / "s", volume)
    series = load_series(tmp_path / "s")
    out = build_contour_series(
        lesions, series.datasets, ResultInfo("T", "t", 2, 1.0), series_description="SEP test"
    )
    src = series.datasets
    assert len(out) == len(src)
    k = int(np.argmax(lesions.sum(axis=(1, 2))))
    before, after = src[k].pixel_array, out[k].pixel_array
    assert out[k].PhotometricInterpretation == "MONOCHROME2"
    assert out[k].SOPClassUID == src[k].SOPClassUID  # reste une image MR (fenêtrable)
    assert out[k].SeriesInstanceUID != src[k].SeriesInstanceUID
    assert list(out[k].ImageType)[:2] == ["DERIVED", "SECONDARY"]
    changed = before != after
    assert changed.any()
    assert (after[changed] == 2**12 - 1).all()  # contours à la valeur maximale (12 bits)
    assert not (changed & lesions[k].astype(bool)).any()  # la lésion elle-même est intacte


def _flatten_sr(sequence, out=None) -> dict:
    """{nom du concept: valeur} pour tout l'arbre SR (texte ou nombre)."""
    out = {} if out is None else out
    for item in sequence:
        name = item.ConceptNameCodeSequence[0].CodeMeaning
        if item.ValueType == "TEXT":
            out.setdefault(name, item.TextValue)
        elif item.ValueType == "NUM":
            out.setdefault(name, float(item.MeasuredValueSequence[0].NumericValue))
        if "ContentSequence" in item:
            _flatten_sr(item.ContentSequence, out)
    return out


def test_evolution_series_colors(tmp_path):
    volume, _ = phantom_volume()
    write_mr_series(tmp_path / "s", volume)
    series = load_series(tmp_path / "s")
    labelmap = np.zeros(volume.shape, np.uint8)
    labelmap[8:12, 10:16, 8:14] = NEW
    labelmap[8:12, 30:36, 25:31] = ENLARGING
    labelmap[8:12, 20:24, 30:34] = STABLE
    out = build_color_contour_series(
        sitk.GetArrayFromImage(series.image).astype(np.float32),
        labelmap,
        {NEW: (255, 0, 0), ENLARGING: (255, 140, 0)},
        series.datasets,
        ResultInfo("T", "t", 3, 1.0),
        series_description="SEP EVOLUTION test",
    )
    rgb = out[10].pixel_array
    assert out[10].PhotometricInterpretation == "RGB"
    assert (rgb[9, 8:14] == (255, 0, 0)).all()  # contour au-dessus de la nouvelle : rouge
    assert (rgb[29, 25:31] == (255, 140, 0)).all()  # contour de l'élargie : orange
    stable_ring = rgb[19, 30:34]
    assert (stable_ring[:, 0] == stable_ring[:, 1]).all()  # stable non dessinée (gris)


# --------------------------------------------------------------------------- Orthanc
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def orthanc(tmp_path):
    """Un vrai serveur Orthanc local (test ignoré s'il n'est pas installé)."""
    binary = shutil.which("Orthanc")
    if binary is None:
        pytest.skip("Orthanc n'est pas installé")
    http, dicom, ours = _free_port(), _free_port(), _free_port()
    config = {
        "Name": "Test",
        "StorageDirectory": str(tmp_path / "db"),
        "IndexDirectory": str(tmp_path / "db"),
        "DicomAet": "ORTHANC",
        "DicomPort": dicom,
        "HttpPort": http,
        "RemoteAccessAllowed": False,
        "AuthenticationEnabled": False,
        "DicomModalities": {
            "segbrainmri": {"AET": "SEGBRAINMRI", "Host": "127.0.0.1", "Port": ours}
        },
        "Plugins": [],
    }
    (tmp_path / "orthanc.json").write_text(json.dumps(config))
    proc = subprocess.Popen(
        [binary, str(tmp_path / "orthanc.json")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{http}"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "/system", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    yield base, dicom
    proc.terminate()
    proc.wait(timeout=10)


def _upload(base: str, directory) -> None:
    for path in directory.glob("*.dcm"):
        request = urllib.request.Request(base + "/instances", data=path.read_bytes(), method="POST")
        urllib.request.urlopen(request).read()


def _get(base: str, path: str):
    return json.load(urllib.request.urlopen(base + path))


def test_end_to_end_new_lesion_against_prior_in_orthanc(orthanc, tmp_path):
    base, dicom_port = orthanc
    # Antériorité (2 lésions) dans Orthanc ; examen actuel : tête bougée + 1 nouvelle lésion
    write_mr_series(
        tmp_path / "prior",
        brain_phantom(OLD_LESIONS),
        spacing=SPACING,
        origin=ORIGIN,
        study_date="20250301",
        description="AX FLAIR 3D",
    )
    _upload(base, tmp_path / "prior")
    current = brain_phantom(OLD_LESIONS + [NEW_LESION], **MOVE, seed=1)
    write_mr_series(
        tmp_path / "current",
        current,
        spacing=SPACING,
        origin=ORIGIN,
        study_date="20260101",
        description="AX FLAIR 3D",
    )

    config = ServiceConfig()
    config.orthanc.host, config.orthanc.port = "127.0.0.1", dicom_port
    results = process_series(tmp_path / "current", ThresholdBackend(), config)

    descriptions = sorted(ds.SeriesDescription for ds in results if ds.InstanceNumber == 1)
    assert any("vs 20250301 - 1 nouv, 0 elarg" in d for d in descriptions), descriptions
    assert "SEP EVOLUTION vs 20250301 - 1 nouv (rouge), 0 elarg (orange)" in descriptions

    # Série évolution : contour rouge autour de la nouvelle lésion, rien autour des anciennes
    evolution = sorted(
        (ds for ds in results if ds.SeriesDescription.startswith("SEP EVOLUTION")),
        key=lambda ds: int(ds.InstanceNumber),
    )
    rgb = np.stack([ds.pixel_array for ds in evolution])
    red = (rgb[..., 0] == 255) & (rgb[..., 1] == 0) & (rgb[..., 2] == 0)
    new_zone = ndimage.binary_dilation(
        brain_phantom([NEW_LESION], **MOVE, seed=1) > 1500, iterations=3
    )
    assert red.sum() > 0 and new_zone[red].all()  # tout le rouge est autour de la nouvelle lésion

    # Compte rendu SR : résumé, nombre de nouvelles lésions et localisation de chacune
    sr = next(ds for ds in results if ds.Modality == "SR")
    flat = _flatten_sr(sr.ContentSequence)
    assert "1 nouvelle(s) lesion(s)" in flat["Resume"]
    assert "01/03/2025" in flat["Resume"]
    assert flat["Nouvelles lesions"] == 1 and flat["Lesions elargies"] == 0
    assert flat["Localisation"].startswith("image n. ")

    seg = next(ds for ds in results if ds.Modality == "SEG")
    decoded = hd.seg.Segmentation.from_dataset(seg)
    series = load_series(tmp_path / "current")
    uids = [ds.SOPInstanceUID for ds in series.datasets]
    new = decoded.get_pixels_by_source_instance(uids, segment_numbers=[1])[..., 0]
    labels, n = ndimage.label(new)
    assert n == 1  # exactement une nouvelle lésion...
    expected = brain_phantom([NEW_LESION], **MOVE, seed=1) > 1500
    assert (labels > 0)[expected].mean() > 0.8  # ... à l'endroit de la vraie nouvelle lésion

    # Les résultats sont bien arrivés dans Orthanc, dans l'examen actuel
    study = series.first.StudyInstanceUID
    found = _get(base, "/studies?expand")
    current_study = next(s for s in found if s["MainDicomTags"]["StudyInstanceUID"] == study)
    names = sorted(
        _get(base, f"/series/{sid}")["MainDicomTags"].get("SeriesDescription", "")
        for sid in current_study["Series"]
    )
    assert any(n.startswith("SEP EVOLUTION") for n in names)
    assert any(n.startswith("SEP lesions") for n in names)
    assert any(n.startswith("SEP compte rendu") for n in names)  # le SR est dans Orthanc


def test_no_prior_in_orthanc(orthanc, tmp_path):
    base, dicom_port = orthanc
    write_mr_series(
        tmp_path / "current",
        brain_phantom(OLD_LESIONS),
        spacing=SPACING,
        origin=ORIGIN,
        patient_id="OTHER",
    )
    config = ServiceConfig()
    config.orthanc.host, config.orthanc.port = "127.0.0.1", dicom_port
    results = process_series(tmp_path / "current", ThresholdBackend(), config)
    assert any("pas d'anteriorite" in ds.SeriesDescription for ds in results if ds.Modality == "MR")
    assert sum(ds.Modality == "SEG" for ds in results) == 1
