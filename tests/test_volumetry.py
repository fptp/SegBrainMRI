import os
import stat
from pathlib import Path

import highdicom as hd
import nibabel as nib
import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from segbrainmri.dicom.series import load_series
from segbrainmri.volumetry.analysis import analyze_volumes, asymmetry, patient_age, patient_sex
from segbrainmri.volumetry.engine import VolumetryEngine, is_t1_3d
from segbrainmri.volumetry.fastsurfer import FastSurfer, parse_stats
from segbrainmri.volumetry.normative import NormativeLibrary, design_row, unwarp, warp
from segbrainmri.volumetry.regions import LOBES, SEG_SEGMENTS
from tests.dicom_utils import brain_phantom, write_mr_series
from tests.test_longitudinal import _flatten_sr

DATA = Path(__file__).parent / "data"
STATS = DATA / "fastsurfer_aseg+DKT.stats"
NORMS = os.environ.get("SEGBRAINMRI_NORMS_DIR")  # modèles BrainCharts réels (facultatif)


# --------------------------------------------------------------------------- unités
def test_parse_real_fastsurfer_stats():
    v = parse_stats(STATS)  # fichier produit par segstats.py de FastSurfer v2.6.1
    assert v["Left-Hippocampus"] == pytest.approx(3821.091)
    assert v["SubCortGrayVol"] == pytest.approx(14543.05099)
    assert sum(k.startswith("ctx-lh-") for k in v) == 31
    assert sum(k.startswith("ctx-rh-") for k in v) == 31
    assert "Left-Thalamus" in v  # nom FreeSurfer 7 (alias de Left-Thalamus-Proper)


def test_lobes_cover_all_dkt_regions():
    regions = [r for rs in LOBES.values() for r in rs]
    assert len(regions) == len(set(regions)) == 31
    v = parse_stats(STATS)
    assert {f"ctx-lh-{r}" for r in regions} == {k for k in v if k.startswith("ctx-lh-")}


def test_spline_basis_and_warp():
    for age in (-5, 0, 37.3, 70, 109.9):
        row = design_row(age, 1)
        assert row.shape == (92,)
        assert row[:3].tolist() == [1.0, age, 1.0]
        assert not row[3:85].any()  # colonnes « site » à zéro (non calibré)
        assert row[85:].sum() == pytest.approx(1.0)  # partition de l'unité (B-spline)
    gamma = np.array([0.3, -0.2])
    y = np.array([500.0, 3500.0, 12000.0])
    np.testing.assert_allclose(unwarp(warp(y, gamma), gamma), y, rtol=1e-10)


def test_patient_age_and_sex():
    ds = pydicom.Dataset()
    ds.PatientAge = "067Y"
    assert patient_age(ds) == 67
    ds = pydicom.Dataset()
    ds.PatientBirthDate, ds.StudyDate = "19560315", "20261113"
    assert patient_age(ds) == pytest.approx(70.66, abs=0.01)
    ds.PatientSex = "F"
    assert patient_sex(ds) == 0
    assert patient_age(pydicom.Dataset()) is None


@pytest.mark.parametrize(
    ("description", "acquisition", "expected"),
    [
        ("Ax 3D T1 BRAVO", "3D", True),
        ("SAG T1 MPRAGE", "3D", True),
        ("Ax T1 BRAVO +C", "3D", False),  # injectée
        ("3D T1 BRAVO post gado", "3D", False),
        ("Sag CUBE FLAIR", "3D", False),
        ("Ax T1 SE", "2D", False),
        ("Ax T1 BRAVO", "2D", False),
    ],
)
def test_t1_detection(description, acquisition, expected):
    ds = pydicom.Dataset()
    ds.Modality, ds.SeriesDescription, ds.ImageType = "MR", description, ["ORIGINAL"]
    ds.MRAcquisitionType = acquisition
    assert is_t1_3d(ds) is expected


class FakeModel:
    """Normes factices : médiane 4000 mm³, écart-type 400 mm³ (gaussien)."""

    def zscore(self, value, age, sex):
        return (value - 4000.0) / 400.0

    def percentile(self, value, age, sex):
        from scipy.stats import norm

        return 100 * norm.cdf(self.zscore(value, age, sex))

    def expected(self, age, sex, centile=50.0):
        return 4000.0


class FakeNorms:
    def has(self, name):
        return name.endswith("Hippocampus")

    def get(self, name):
        return FakeModel()


def test_analyze_volumes():
    v = parse_stats(STATS)
    result = analyze_volumes(v, age=72, sex=1, icv_mm3=1_500_000, norms=FakeNorms())
    hippo = next(s for s in result.deep if s.label == "Hippocampe")
    assert hippo.left.volume_ml == pytest.approx(3.821091)
    assert hippo.left.pct_icv == pytest.approx(100 * 3821.091 / 1_500_000)
    assert hippo.left.z == pytest.approx((3821.091 - 4000) / 400)
    assert hippo.left.expected_ml == pytest.approx(4.0)
    assert hippo.asymmetry == pytest.approx(asymmetry(3821.091, 3936.525))
    thalamus = next(s for s in result.deep if s.label == "Thalamus")
    assert thalamus.left.percentile is None  # pas de norme (factice) pour le thalamus
    frontal = next(s for s in result.lobes if s.label == "Frontal")
    expected = sum(v[f"ctx-lh-{r}"] for r in LOBES["Frontal"]) / 1000
    assert frontal.left.volume_ml == pytest.approx(expected)
    assert result.norms_used
    no_age = analyze_volumes(v, age=None, sex=1, icv_mm3=None, norms=FakeNorms())
    assert not no_age.norms_used and "age" in no_age.norms_note
    assert all(s.left.percentile is None for s in no_age.deep)


@pytest.mark.skipif(not NORMS, reason="SEGBRAINMRI_NORMS_DIR non défini (modèles BrainCharts)")
def test_real_braincharts_models():
    lib = NormativeLibrary(NORMS)
    hippo = lib.get("Left-Hippocampus")
    # Valeurs de référence calculées avec le code d'origine pcntoolkit 0.30
    assert hippo.expected(70, 0) == pytest.approx(3427, abs=1)
    assert hippo.expected(70, 1) == pytest.approx(3678, abs=1)
    assert hippo.percentile(hippo.expected(70, 1), 70, 1) == pytest.approx(50, abs=1e-6)
    assert hippo.expected(85, 1) < hippo.expected(60, 1)  # atrophie liée à l'âge


# --------------------------------------------------------------------------- chaîne complète
FAKE_RUN = """#!/bin/bash
# Faux run_fastsurfer.sh : recopie des sorties réelles préparées par le test
while [[ $# -gt 0 ]]; do
  case "$1" in --sd) SD="$2"; shift;; --sid) SID="$2"; shift;; esac; shift
done
mkdir -p "$SD/$SID/mri" "$SD/$SID/stats"
cp "$FAKE_DIR/aparc.DKTatlas+aseg.deep.mgz" "$SD/$SID/mri/"
cp "$FAKE_DIR/aseg+DKT.stats" "$SD/$SID/stats/"
"""


class FakeStripper:
    def strip(self, image, workdir):
        mask = sitk.Cast(image > 100, sitk.sitkUInt8)
        return image, mask


def _fake_fastsurfer(tmp_path, t1: sitk.Image) -> FastSurfer:
    home = tmp_path / "fastsurfer"
    (home / "checkpoints").mkdir(parents=True)
    for plane in ("axial", "coronal", "sagittal"):
        (home / "checkpoints" / f"aparc_vinn_{plane}_v2.0.0.pkl").write_bytes(b"")
    script = home / "run_fastsurfer.sh"
    script.write_text(FAKE_RUN)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    # Segmentation alignée sur la T1 : hippocampe gauche (17) et lobe frontal gauche (1028)
    arr = np.zeros(sitk.GetArrayFromImage(t1).shape, np.int16)
    arr[10:14, 20:30, 20:30] = 17
    arr[20:24, 30:40, 30:40] = 1028
    seg = sitk.GetImageFromArray(arr)
    seg.CopyInformation(t1)
    sitk.WriteImage(seg, str(tmp_path / "seg.nii.gz"))
    nii = nib.load(str(tmp_path / "seg.nii.gz"))
    nib.save(
        nib.MGHImage(np.asarray(nii.dataobj).astype(np.int32), nii.affine),
        str(tmp_path / "aparc.DKTatlas+aseg.deep.mgz"),
    )
    (tmp_path / "aseg+DKT.stats").write_text(STATS.read_text())
    os.environ["FAKE_DIR"] = str(tmp_path)
    return FastSurfer(home, "python3")


def test_engine_end_to_end(tmp_path):
    volume = brain_phantom([((-20, 15, 10), 6.0)])
    write_mr_series(
        tmp_path / "t1",
        volume,
        spacing=(2.0, 2.0, 3.0),
        origin=(-64.0, -72.0, -54.0),
        description="Ax 3D T1 BRAVO",
        sex="M",
    )
    series = load_series(tmp_path / "t1")
    engine = VolumetryEngine(
        _fake_fastsurfer(tmp_path, series.image),
        FakeStripper(),
        FakeNorms(),
        fastsurfer_version="v2.6.1",
    )
    seg, sr = engine.run(series)

    # SEG : segments nommés, hippocampe et lobe frontal à leur place sur la T1 native
    labels = [s.SegmentLabel for s in seg.SegmentSequence]
    assert labels == [label for label, _, _ in SEG_SEGMENTS]
    decoded = hd.seg.Segmentation.from_dataset(seg)
    uids = [ds.SOPInstanceUID for ds in series.datasets]
    hippo = decoded.get_pixels_by_source_instance(uids, segment_numbers=[7])[..., 0]
    frontal = decoded.get_pixels_by_source_instance(uids, segment_numbers=[1])[..., 0]
    assert hippo[10:14, 20:30, 20:30].all() and hippo.sum() == 4 * 10 * 10
    assert frontal[20:24, 30:40, 30:40].all() and frontal.sum() == 4 * 10 * 10

    # SR : résumé, hippocampes avec percentiles, lobes avec asymétrie
    flat = _flatten_sr(sr.ContentSequence)
    assert "Hippocampe : G 3.82 mL (P" in flat["Resume"]
    assert "non calibrees" in flat["Resume"]
    assert flat["Gauche volume"] == pytest.approx(3.82, abs=0.01)
    assert sr.SeriesDescription == "VOLUMETRIE compte rendu FastSurfer"
    assert "Volume intracranien" in flat
    assert flat["Sexe"] == "homme" and flat["Age"] == pytest.approx(46, abs=0.1)


def test_fastsurfer_command_and_missing_install(tmp_path):
    with pytest.raises(FileNotFoundError, match="install.sh"):
        FastSurfer(tmp_path, "python3")
    t1 = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    fs = _fake_fastsurfer(tmp_path, t1)
    cmd = fs.command(Path("/x/t1.nii.gz"), Path("/x/sd"))
    for flag in ("--seg_only", "--no_cereb", "--no_hypothal", "--no_cc"):
        assert flag in cmd
    assert cmd[cmd.index("--device") + 1] == "cpu"


def test_worker_routes_t1_to_volumetry(tmp_path):
    from segbrainmri.service.config import ServiceConfig
    from segbrainmri.service.worker import SpoolWorker

    class FakeEngine:
        series = []

        def run(self, series):
            self.series.append(series.first.SeriesDescription)
            return []

    class NoBackend:
        name, version = "x", "x"

        def segment(self, flair):
            raise AssertionError("une T1 ne doit pas passer par FLAMeS")

    config = ServiceConfig()
    config.processing.spool_dir, config.processing.settle_seconds = tmp_path / "spool", 0
    engine = FakeEngine()
    worker = SpoolWorker(config, NoBackend(), engine)
    volume = brain_phantom([])
    write_mr_series(worker.incoming / "t1", volume, description="Ax 3D T1 BRAVO")
    write_mr_series(worker.incoming / "t1c", volume, description="Ax 3D T1 BRAVO +C")
    assert worker.run_once() == 2
    assert engine.series == ["Ax 3D T1 BRAVO"]  # la T1 injectée est ignorée
    assert not any(worker.failed.iterdir())
