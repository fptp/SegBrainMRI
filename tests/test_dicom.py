import highdicom as hd
import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from segbrainmri.dicom.network import StorageReceiver, echo, send_datasets
from segbrainmri.dicom.series import is_flair, load_series, read_header
from segbrainmri.dicom.writers import ResultInfo, build_overlay, build_seg
from segbrainmri.service.config import ServiceConfig
from segbrainmri.service.worker import SpoolWorker, process_series
from tests.dicom_utils import phantom_volume, write_mr_series


class ThresholdBackend:
    """Moteur factice : seuillage (remplace FLAMeS, dont les poids ne sont pas versionnés)."""

    name, version = "Threshold", "test"

    def segment(self, flair: sitk.Image) -> sitk.Image:
        return sitk.Cast(flair > 1500, sitk.sitkUInt8)


@pytest.fixture
def series_dir(tmp_path):
    volume, _ = phantom_volume()
    write_mr_series(tmp_path / "flair", volume)
    return tmp_path / "flair"


def test_is_flair():
    ds = pydicom.Dataset()
    ds.Modality, ds.SeriesDescription, ds.ImageType = "MR", "Sag 3D FLAIR fs", ["ORIGINAL"]
    assert is_flair(ds)
    ds.SeriesDescription = "T1 MPRAGE"
    assert not is_flair(ds)
    ds.SeriesDescription, ds.ImageType = "FLAIR", ["DERIVED", "SECONDARY"]
    assert not is_flair(ds)
    ds.ImageType, ds.Manufacturer = ["ORIGINAL"], "SegBrainMRI"
    assert not is_flair(ds)


def test_load_series_orders_slices(series_dir):
    volume, _ = phantom_volume()
    series = load_series(series_dir)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(series.image), volume.astype(np.uint16))
    positions = [float(ds.ImagePositionPatient[2]) for ds in series.datasets]
    assert positions == sorted(positions)
    assert series.image.GetSpacing() == (1.0, 1.0, 2.0)


def test_seg_and_overlay(series_dir):
    _, lesions = phantom_volume()
    series = load_series(series_dir)
    info = ResultInfo("Threshold", "test", 2, 1.5)

    seg = build_seg(lesions, series.datasets, info)
    assert seg.Modality == "SEG"
    assert seg.StudyInstanceUID == series.first.StudyInstanceUID
    decoded = hd.seg.Segmentation.from_dataset(seg)
    sop_uids = [ds.SOPInstanceUID for ds in series.datasets]
    labelmap = decoded.get_pixels_by_source_instance(sop_uids, segment_numbers=[1])[..., 0]
    np.testing.assert_array_equal(labelmap.astype(np.uint8), lesions)

    volume = sitk.GetArrayFromImage(series.image).astype(np.float32)
    overlay = build_overlay(volume, lesions, series.datasets, info)
    assert len(overlay) == len(series.datasets)
    k = int(np.argmax(lesions.sum(axis=(1, 2))))
    pixels = overlay[k].pixel_array
    assert pixels.shape == (*lesions.shape[1:], 3)
    lesion_px = pixels[lesions[k].astype(bool)]
    assert (lesion_px[:, 0] > lesion_px[:, 1]).all()  # rouge dominant sur les lésions


def test_seg_empty_mask(series_dir):
    series = load_series(series_dir)
    empty = np.zeros((len(series.datasets), series.first.Rows, series.first.Columns), np.uint8)
    seg = build_seg(empty, series.datasets, ResultInfo("T", "t", 0, 0.0))
    assert seg.NumberOfFrames >= 1


def test_config_load(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[orthanc]\nhost = "10.0.0.1"\n[processing]\nspool_dir = "/tmp/x"\n')
    config = ServiceConfig.load(path)
    assert config.orthanc.host == "10.0.0.1" and config.orthanc.port == 4242
    assert str(config.processing.spool_dir) == "/tmp/x"
    path.write_text("[orthanc]\nhots = 1\n")
    with pytest.raises(ValueError):
        ServiceConfig.load(path)


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_receive_process_and_send(tmp_path, series_dir):
    """Chaîne complète en local : envoi -> réception -> traitement -> envoi au "PACS"."""
    in_port, pacs_port = _free_port(), _free_port()
    config = ServiceConfig()
    config.listener.port, config.listener.allowed_calling_aets = in_port, ["ORTHANC"]
    config.orthanc.host, config.orthanc.port, config.orthanc.ae_title = (
        "127.0.0.1",
        pacs_port,
        "PACS",
    )
    config.processing.spool_dir, config.processing.settle_seconds = tmp_path / "spool", 0

    worker = SpoolWorker(config, ThresholdBackend())
    receiver = StorageReceiver(
        worker.incoming, port=in_port, host="127.0.0.1", allowed_calling_aets=["ORTHANC"]
    )
    pacs = StorageReceiver(tmp_path / "pacs", ae_title="PACS", port=pacs_port, host="127.0.0.1")
    receiver.start()
    pacs.start()
    try:
        assert echo("127.0.0.1", in_port, "SEGBRAINMRI", "ORTHANC")
        sources = [pydicom.dcmread(p) for p in sorted(series_dir.glob("*.dcm"))]
        # un AET non autorisé est refusé
        with pytest.raises(RuntimeError):
            send_datasets(
                sources[:1],
                host="127.0.0.1",
                port=in_port,
                called_aet="SEGBRAINMRI",
                calling_aet="INTRUS",
                retries=1,
            )
        send_datasets(
            sources, host="127.0.0.1", port=in_port, called_aet="SEGBRAINMRI", calling_aet="ORTHANC"
        )
        assert worker.run_once() == 1
    finally:
        receiver.stop()
        pacs.stop()

    received = [read_header(p) for p in (tmp_path / "pacs").rglob("*.dcm")]
    modalities = sorted({ds.Modality for ds in received})
    assert modalities == ["MR", "SEG"]
    assert len(received) == 1 + len(sources)
    assert all(ds.StudyInstanceUID == sources[0].StudyInstanceUID for ds in received)
    assert not any(worker.incoming.iterdir()) and not any(worker.failed.iterdir())


def test_worker_skips_non_flair_and_failures(tmp_path, monkeypatch):
    volume, _ = phantom_volume()
    config = ServiceConfig()
    config.processing.spool_dir, config.processing.settle_seconds = tmp_path / "spool", 0
    worker = SpoolWorker(config, ThresholdBackend())
    write_mr_series(worker.incoming / "t1", volume, description="T1 MPRAGE")
    write_mr_series(worker.incoming / "flair", volume)
    # personne n'écoute sur ce port : l'envoi échoue
    config.orthanc.host, config.orthanc.port = "127.0.0.1", _free_port()
    import segbrainmri.dicom.network as net

    monkeypatch.setattr(net.time, "sleep", lambda s: None)  # pas d'attente entre tentatives
    assert worker.run_once() == 2
    assert not (worker.incoming / "t1").exists()
    assert (worker.failed / "flair" / "error.txt").is_file()


def test_process_series_no_send(series_dir):
    results = process_series(series_dir, ThresholdBackend(), ServiceConfig(), send=False)
    assert results[0].Modality == "SEG"
    assert len(results) == 1 + len(list(series_dir.glob("*.dcm")))


def test_missing_slice_is_rejected(series_dir):
    middle = next(
        p for p in series_dir.glob("*.dcm") if float(read_header(p).ImagePositionPatient[2]) == 20.0
    )
    middle.unlink()
    with pytest.raises(ValueError, match="manquante"):
        load_series(series_dir)


def test_duplicate_instances_received_concurrently(tmp_path, series_dir):
    """Même image reçue deux fois en même temps (plusieurs associations) : pas d'erreur."""
    from concurrent.futures import ThreadPoolExecutor

    port = _free_port()
    receiver = StorageReceiver(tmp_path / "in", port=port, host="127.0.0.1")
    receiver.start()
    try:
        sources = [pydicom.dcmread(p) for p in sorted(series_dir.glob("*.dcm"))]

        def push(_):
            send_datasets(
                sources,
                host="127.0.0.1",
                port=port,
                called_aet="SEGBRAINMRI",
                calling_aet="HOROS",
                retries=1,
            )

        with ThreadPoolExecutor(4) as pool:
            list(pool.map(push, range(4)))  # lève une exception si un C-STORE échoue
    finally:
        receiver.stop()
    stored = list((tmp_path / "in").rglob("*"))
    assert len([p for p in stored if p.suffix == ".dcm"]) == len(sources)
    assert not [p for p in stored if p.suffix == ".part"]
    assert load_series(next((tmp_path / "in").iterdir())).image.GetSize()[2] == len(sources)
