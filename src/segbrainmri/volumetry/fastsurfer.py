"""Appel de FastSurfer (module de segmentation seul, sans FreeSurfer ni licence).

FastSurfer (Henschel et al., NeuroImage 2020 ; licence Apache 2.0) segmente une T1 3D en
structures de l'atlas aseg + régions corticales DKT, puis calcule les volumes avec
correction de volume partiel (comme ``mri_segstats`` de FreeSurfer), ce qui les rend
comparables aux normes BrainCharts. Il est installé dans son propre environnement Python
(dépendances figées par FastSurfer) et appelé par son script officiel.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def parse_stats(path: str | Path) -> dict[str, float]:
    """Volumes (mm³) par structure d'un fichier .stats FreeSurfer/FastSurfer.

    Lit aussi les mesures globales d'en-tête (``# Measure ...``) sous leur nom court.
    """
    volumes: dict[str, float] = {}
    columns: list[str] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("# Measure"):
            parts = [p.strip() for p in line[len("# Measure") :].split(",")]
            if len(parts) >= 4:
                try:
                    volumes[parts[1]] = float(parts[3])
                except ValueError:
                    pass
        elif line.startswith("# ColHeaders"):
            columns = line.split()[2:]
        elif line.strip() and not line.startswith("#") and columns:
            row = dict(zip(columns, line.split(), strict=False))
            if "StructName" in row and "Volume_mm3" in row:
                volumes[row["StructName"]] = float(row["Volume_mm3"])
    if not volumes:
        raise ValueError(f"Aucun volume lu dans {path}")
    return volumes


def read_mgz(path: str | Path) -> sitk.Image:
    """Lit un .mgz (format FreeSurfer) en image SimpleITK, géométrie comprise."""
    import nibabel as nib

    img = nib.load(str(path))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "seg.nii.gz"
        data = np.asarray(img.dataobj).astype(np.int16)
        nib.save(nib.Nifti1Image(data, img.affine), str(out))
        return sitk.ReadImage(str(out))


@dataclass
class FastSurferResult:
    volumes: dict[str, float]  # mm³ par structure (StructName) + mesures globales
    segmentation: sitk.Image  # aparc.DKTatlas+aseg (espace conforme 1 mm de FastSurfer)
    seconds: float


class FastSurfer:
    def __init__(
        self,
        home: str | Path,
        python: str | Path,
        *,
        device: str = "cpu",
        threads: int | None = None,
        timeout_s: int = 3 * 3600,
    ) -> None:
        self.home, self.python = Path(home), Path(python)
        script = self.home / "run_fastsurfer.sh"
        if not script.is_file():
            raise FileNotFoundError(
                f"FastSurfer introuvable ({script}) : relancer deploy/install.sh."
            )
        checkpoints = list((self.home / "checkpoints").glob("aparc_vinn_*.pkl"))
        if len(checkpoints) < 3:
            raise FileNotFoundError(
                f"Poids FastSurfer manquants dans {self.home / 'checkpoints'} : "
                "relancer deploy/install.sh."
            )
        self.device, self.threads, self.timeout_s = device, threads, timeout_s

    def command(self, t1: Path, subjects_dir: Path, subject: str = "case") -> list[str]:
        cmd = [
            "bash",
            str(self.home / "run_fastsurfer.sh"),
            "--t1",
            str(t1),
            "--sid",
            subject,
            "--sd",
            str(subjects_dir),
            "--seg_only",
            "--no_cereb",
            "--no_hypothal",
            "--no_cc",
            # Résolution standard de 1 mm : sinon FastSurfer travaille à la plus petite taille
            # de voxel de l'image (ex. 0,47 mm sur une BRAVO -> cube 512³, mode expérimental,
            # 8 fois plus de mémoire). Les normes BrainCharts sont aussi établies à 1 mm.
            "--vox_size",
            "1",
            "--device",
            "cuda" if self.device.startswith("cuda") else "cpu",
            "--py",
            f"{self.python} -s",
        ]
        if self.threads:
            cmd += ["--threads", str(self.threads)]
        if os.geteuid() == 0:
            cmd.append("--allow_root")
        return cmd

    def run(self, t1: sitk.Image, workdir: str | Path) -> FastSurferResult:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        t1_path = workdir / "t1.nii.gz"
        sitk.WriteImage(sitk.Cast(t1, sitk.sitkFloat32), str(t1_path))
        subjects = workdir / "subjects"
        start = time.perf_counter()
        logger.info("FastSurfer (segmentation) en cours...")
        env = {
            **os.environ,
            "MPLBACKEND": "Agg",
            "FASTSURFER_HOME": str(self.home),
            # FastSurfer lit sa version avec git ; le dépôt appartient à root, pas au service
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": str(self.home),
        }
        proc = subprocess.run(
            self.command(t1_path, subjects),
            capture_output=True,
            text=True,
            env=env,
            timeout=self.timeout_s,
            cwd=str(self.home),
        )
        seconds = time.perf_counter() - start
        case = subjects / "case"
        stats = case / "stats" / "aseg+DKT.stats"
        seg = case / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
        if proc.returncode != 0 or not stats.is_file() or not seg.is_file():
            raise RuntimeError(
                f"FastSurfer a échoué (code {proc.returncode}) :\n"
                f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
            )
        logger.info("FastSurfer terminé en %.0f s", seconds)
        return FastSurferResult(parse_stats(stats), read_mgz(seg), seconds)
