"""Traitement d'une série FLAIR reçue : segmentation puis envoi des résultats à Orthanc."""

from __future__ import annotations

import logging
import shutil
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from segbrainmri.dicom.network import send_datasets
from segbrainmri.dicom.series import is_flair, is_our_output, load_series, read_header
from segbrainmri.dicom.writers import ResultInfo, build_overlay, build_seg
from segbrainmri.inference.stats import lesion_statistics
from segbrainmri.models.backends import LesionBackend
from segbrainmri.service.config import ServiceConfig

logger = logging.getLogger(__name__)


def build_backend(config: ServiceConfig) -> LesionBackend:
    model = config.model
    if model.backend != "flames":
        raise ValueError(f"Moteur inconnu : {model.backend!r} (disponible : flames)")
    from segbrainmri.models.flames import FlamesBackend

    tta = None if model.tta == "auto" else bool(model.tta)
    return FlamesBackend(
        model.models_dir,
        device=config.processing.device,
        tta=tta,
        threads=config.processing.threads,
        skull_strip=model.skull_strip,
    )


def process_series(
    series_dir: Path, backend: LesionBackend, config: ServiceConfig, *, send: bool = True
) -> list:
    """Segmente une série et renvoie (et envoie) les datasets résultats."""
    series = load_series(series_dir)
    mask_img = backend.segment(series.image)
    mask = sitk.GetArrayFromImage(mask_img).astype(np.uint8)  # (coupes, lignes, colonnes)
    count, volume_ml = lesion_statistics(mask_img)
    info = ResultInfo(backend.name, backend.version, count, volume_ml)
    logger.info("Résultat : %s", info.summary)

    results = []
    if config.output.seg:
        results.append(build_seg(mask, series.datasets, info))
    if config.output.overlay:
        volume = sitk.GetArrayFromImage(series.image).astype(np.float32)
        results += build_overlay(
            volume, mask, series.datasets, info, opacity=config.output.overlay_opacity
        )
    if send:
        o = config.orthanc
        send_datasets(
            results,
            host=o.host,
            port=o.port,
            called_aet=o.ae_title,
            calling_aet=config.listener.ae_title,
        )
    return results


class SpoolWorker:
    """Surveille le dossier de réception et traite les séries une par une.

    Arborescence sous ``spool_dir`` : ``incoming/`` (réception), ``processing/``,
    ``failed/`` (séries en erreur + ``error.txt``) et ``done/`` si ``keep_inputs``.
    """

    def __init__(self, config: ServiceConfig, backend: LesionBackend) -> None:
        self.config, self.backend = config, backend
        root = config.processing.spool_dir
        self.incoming, self.processing = root / "incoming", root / "processing"
        self.failed, self.done = root / "failed", root / "done"
        for d in (self.incoming, self.processing, self.failed, self.done):
            d.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        # Séries interrompues par un arrêt du service : on les remet en file
        for leftover in self.processing.iterdir():
            leftover.replace(self.incoming / leftover.name)

    def _ready(self) -> list[Path]:
        now, settle = time.time(), self.config.processing.settle_seconds
        ready = []
        for d in sorted(self.incoming.iterdir()):
            files = list(d.glob("*.dcm"))
            if any(d.glob("*.part")) or not files:
                continue
            if now - max(f.stat().st_mtime for f in files) >= settle:
                ready.append(d)
        return ready

    def handle(self, series_dir: Path) -> None:
        work = self.processing / series_dir.name
        series_dir.replace(work)
        try:
            header = read_header(next(work.glob("*.dcm")))
            if is_our_output(header) or not is_flair(header, self.config.processing.flair_patterns):
                logger.info(
                    "Série ignorée (pas une FLAIR originale) : %s",
                    header.get("SeriesDescription", "?"),
                )
                shutil.rmtree(work)
                return
            logger.info(
                "Traitement de la série %s (%s)", header.get("SeriesDescription", "?"), work.name
            )
            process_series(work, self.backend, self.config)
        except Exception:
            logger.exception("Échec du traitement de %s", work.name)
            target = self.failed / work.name
            shutil.rmtree(target, ignore_errors=True)
            work.replace(target)
            (target / "error.txt").write_text(traceback.format_exc())
            return
        if self.config.processing.keep_inputs:
            shutil.rmtree(self.done / work.name, ignore_errors=True)
            work.replace(self.done / work.name)
        else:
            shutil.rmtree(work)

    def run_once(self) -> int:
        ready = self._ready()
        for series_dir in ready:
            if self._stop.is_set():
                break
            self.handle(series_dir)
        return len(ready)

    def run_forever(self) -> None:
        logger.info("Worker démarré, surveillance de %s", self.incoming)
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.config.processing.poll_seconds)

    def stop(self) -> None:
        self._stop.set()
