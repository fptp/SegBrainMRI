"""Intégration de FLAMeS (FLAIR Lesion Analysis in Multiple Sclerosis).

FLAMeS est un modèle nnU-Net v2 entraîné sur 668 FLAIR (1,5 T et 3 T, Siemens, GE,
Philips), publié en 2025 (Dereskewicz et al., medRxiv 2025.05.19.25327707). Sur les
jeux de test publiés il dépasse LST-AI, SAMSEG et LST-LPA.

Chaîne reproduite à l'identique de l'inférence de référence :

1. extraction du cerveau par SynthStrip (poids ``synthstrip.1.pt``) ;
2. ``nnU-Net v2`` — Dataset004_WML, configuration ``3d_fullres``,
   entraîneur ``nnUNetTrainer_8000epochs``, plans ``nnUNetPlans``, tous les folds.

nnU-Net applique lui-même rééchantillonnage et normalisation : le prétraitement
générique de :mod:`segbrainmri.preprocessing` ne doit PAS être appliqué ici.
Le masque renvoyé est dans la géométrie exacte du FLAIR d'entrée.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import SimpleITK as sitk
import torch

from segbrainmri.core.device import get_device

logger = logging.getLogger(__name__)

ZENODO_RECORD = "17955359"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
MODEL_ZIP = "Dataset004_WML.zip"
DATASET_NAME = "Dataset004_WML"
TRAINER = "nnUNetTrainer_8000epochs"
PLANS = "nnUNetPlans"
CONFIGURATION = "3d_fullres"
MODEL_SUBDIR = f"{TRAINER}__{PLANS}__{CONFIGURATION}"

SYNTHSTRIP_SCRIPT_URL = (
    "https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthstrip/mri_synthstrip"
)
SYNTHSTRIP_WEIGHTS_URL = (
    "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt"
)


# --------------------------------------------------------------------------- téléchargement
def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Téléchargement %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "segbrainmri"})
    with urllib.request.urlopen(request, timeout=60) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    tmp.replace(dest)
    return dest


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(zip_path: Path, dest: Path) -> None:
    """Décompresse en refusant tout chemin sortant de ``dest`` (zip slip)."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not target.is_relative_to(dest):
                raise RuntimeError(f"Chemin suspect dans l'archive : {member}")
        zf.extractall(dest)


def download_models(models_dir: str | Path) -> Path:
    """Télécharge FLAMeS (Zenodo, somme MD5 vérifiée) et SynthStrip dans ``models_dir``."""
    models_dir = Path(models_dir)
    flames_dir = models_dir / "flames"
    if find_model_folder(flames_dir) is None:
        with urllib.request.urlopen(ZENODO_API, timeout=60) as resp:
            record = json.load(resp)
        entry = next((f for f in record["files"] if f["key"] == MODEL_ZIP), None)
        if entry is None:
            raise RuntimeError(f"{MODEL_ZIP} absent de l'enregistrement Zenodo {ZENODO_RECORD}")
        zip_path = _download(entry["links"]["self"], models_dir / MODEL_ZIP)
        expected = entry.get("checksum", "").removeprefix("md5:")
        if expected and _md5(zip_path) != expected:
            zip_path.unlink()
            raise RuntimeError("Somme de contrôle MD5 invalide pour le modèle FLAMeS.")
        _safe_extract(zip_path, flames_dir)
        zip_path.unlink()
        logger.info("FLAMeS installé dans %s", flames_dir)
    else:
        logger.info("FLAMeS déjà présent dans %s", flames_dir)

    synth_dir = models_dir / "synthstrip"
    if not (synth_dir / "mri_synthstrip").is_file():
        _download(SYNTHSTRIP_SCRIPT_URL, synth_dir / "mri_synthstrip")
    if not (synth_dir / "synthstrip.1.pt").is_file():
        _download(SYNTHSTRIP_WEIGHTS_URL, synth_dir / "synthstrip.1.pt")
    return models_dir


def find_model_folder(root: str | Path) -> Path | None:
    """Trouve le dossier ``<trainer>__<plans>__<config>`` contenant plans.json et dataset.json."""
    root = Path(root)
    if not root.is_dir():
        return None
    candidates = [
        p.parent for p in root.rglob("plans.json") if (p.parent / "dataset.json").is_file()
    ]
    preferred = [c for c in candidates if c.name == MODEL_SUBDIR]
    found = preferred or candidates
    return sorted(found)[0] if found else None


# --------------------------------------------------------------------------- SynthStrip
class SynthStrip:
    """Extraction du cerveau via le script officiel ``mri_synthstrip`` (FreeSurfer)."""

    def __init__(
        self, script: Path, weights: Path, device: torch.device, threads: int | None = None
    ) -> None:
        for path in (script, weights):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} introuvable : lancez `segbrainmri download-models` d'abord."
                )
        self.script, self.weights, self.device, self.threads = script, weights, device, threads

    def run(self, image: Path, out_image: Path, out_mask: Path) -> None:
        cmd = [
            sys.executable,
            str(self.script),
            "-i",
            str(image),
            "-o",
            str(out_image),
            "-m",
            str(out_mask),
            "--model",
            str(self.weights),
        ]
        if self.device.type == "cuda":
            cmd.append("--gpu")
        if self.threads:
            cmd += ["--threads", str(self.threads)]
        env = {**os.environ, "MPLBACKEND": "Agg"}
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"SynthStrip a échoué :\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )

    def strip(self, image: sitk.Image, workdir: Path) -> tuple[sitk.Image, sitk.Image]:
        """Renvoie (image sans crâne, masque cérébral) sur la grille exacte de ``image``.

        L'image est d'abord réorientée en LIA, l'orientation de travail de SynthStrip :
        ``surfa`` (<= 0.6.3) plante avec numpy 2 dès qu'il doit lui-même permuter les axes
        (ex. FLAIR sagittale : "setting an array element with a sequence"). La
        réorientation (permutation / retournement d'axes) est exacte et inversée ensuite.
        """
        original = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
            image.GetDirection()
        )
        src, out, mask = (
            workdir / "ss_in.nii.gz",
            workdir / "ss_out.nii.gz",
            workdir / "ss_mask.nii.gz",
        )
        sitk.WriteImage(sitk.DICOMOrient(sitk.Cast(image, sitk.sitkFloat32), "LIA"), str(src))
        self.run(src, out, mask)
        results = []
        for path, pixel in ((out, sitk.sitkFloat32), (mask, sitk.sitkUInt8)):
            restored = sitk.DICOMOrient(sitk.ReadImage(str(path), pixel), original)
            if restored.GetSize() != image.GetSize():
                raise RuntimeError(
                    f"SynthStrip : taille inattendue {restored.GetSize()} != {image.GetSize()}"
                )
            restored.CopyInformation(image)
            results.append(restored)
        return results[0], results[1]


# --------------------------------------------------------------------------- FLAMeS
class FlamesBackend:
    """Segmentation des lésions de SEP par FLAMeS (implémente ``LesionBackend``)."""

    name = "FLAMeS"
    version = f"nnU-Net v2 {DATASET_NAME} {CONFIGURATION}"

    def __init__(
        self,
        models_dir: str | Path,
        *,
        device: str | torch.device | None = None,
        tta: bool | None = None,
        threads: int | None = None,
        skull_strip: bool = True,
    ) -> None:
        """
        Args:
            models_dir: dossier rempli par :func:`download_models`.
            device: ``auto``/``cpu``/``cuda``... (voir :mod:`segbrainmri.core.device`).
            tta: augmentation au test (miroirs, ~8x plus lent). ``None`` = activée sur GPU
                seulement.
            threads: threads CPU PyTorch.
            skull_strip: ``False`` si les FLAIR fournis sont déjà sans crâne.
        """
        self.models_dir = Path(models_dir)
        self.device = (
            device if isinstance(device, torch.device) else get_device(device, num_threads=threads)
        )
        self.tta = (self.device.type == "cuda") if tta is None else tta
        self.model_folder = find_model_folder(self.models_dir / "flames")
        if self.model_folder is None:
            raise FileNotFoundError(
                f"Modèle FLAMeS introuvable sous {self.models_dir / 'flames'} : "
                "lancez `segbrainmri download-models`."
            )
        synth = self.models_dir / "synthstrip"
        self.stripper = (
            SynthStrip(synth / "mri_synthstrip", synth / "synthstrip.1.pt", self.device, threads)
            if skull_strip
            else None
        )
        self._predictor = None

    def _load_predictor(self):
        if self._predictor is not None:
            return self._predictor
        # nnU-Net avertit si ces variables manquent ; elles ne servent pas à l'inférence.
        for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            os.environ.setdefault(key, str(self.models_dir / "flames"))
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=self.tta,
            perform_everything_on_device=self.device.type == "cuda",
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(str(self.model_folder), use_folds=None)
        logger.info(
            "FLAMeS chargé (%s, folds=%s, TTA=%s) sur %s",
            self.model_folder.name,
            len(predictor.list_of_parameters),
            self.tta,
            self.device,
        )
        self._predictor = predictor
        return predictor

    def segment(self, flair: sitk.Image) -> sitk.Image:
        start = time.perf_counter()
        predictor = self._load_predictor()
        with tempfile.TemporaryDirectory(prefix="segbrainmri_") as tmp:
            tmp = Path(tmp)
            image, brain_mask = sitk.Cast(flair, sitk.sitkFloat32), None
            if self.stripper is not None:
                logger.info("Extraction du cerveau (SynthStrip)...")
                image, brain_mask = self.stripper.strip(image, tmp)
            # Entrée nnU-Net dans l'orientation d'origine, comme l'inférence de référence
            src = tmp / "flair_0000.nii.gz"
            sitk.WriteImage(image, str(src))

            logger.info("Segmentation des lésions (nnU-Net)...")
            # Lecteur déclaré dans plans.json : garantit l'ordre des axes de l'entraînement
            reader = predictor.plans_manager.image_reader_writer_class()
            data, props = reader.read_images([str(src)])
            seg = predictor.predict_single_npy_array(data, props, None, None, False)
            out = tmp / "seg.nii.gz"
            reader.write_seg(seg, str(out), props)
            mask = sitk.ReadImage(str(out))

        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        mask.CopyInformation(flair)
        if brain_mask is not None:
            mask = sitk.And(mask, brain_mask > 0)
            mask = sitk.Cast(mask, sitk.sitkUInt8)
        logger.info("FLAMeS terminé en %.0f s", time.perf_counter() - start)
        return mask
