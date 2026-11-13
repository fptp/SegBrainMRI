"""Traitement d'une série FLAIR reçue : segmentation puis envoi des résultats à Orthanc."""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from segbrainmri.core.device import get_device
from segbrainmri.dicom.network import send_datasets
from segbrainmri.dicom.query import Peer, PriorSeries, find_prior_flair, retrieve_series
from segbrainmri.dicom.report import PriorInfo, build_report
from segbrainmri.dicom.series import (
    DicomSeries,
    is_flair,
    is_our_output,
    load_series,
    read_header,
)
from segbrainmri.dicom.writers import (
    ORANGE,
    RED,
    ResultInfo,
    build_color_contour_series,
    build_contour_series,
    build_labelmap_seg,
    build_overlay,
    build_seg,
)
from segbrainmri.inference.stats import lesion_statistics
from segbrainmri.longitudinal import (
    ENLARGING,
    NEW,
    STABLE,
    LesionComparison,
    classify_lesions,
    register_rigid,
)
from segbrainmri.models.backends import Analysis, LesionBackend, analyze
from segbrainmri.service.config import ServiceConfig
from segbrainmri.volumetry.engine import VolumetryEngine, is_t1_3d

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


def _array(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image).astype(np.uint8)  # (coupes, lignes, colonnes)


def compare_with_prior(
    series: DicomSeries, analysis: Analysis, backend: LesionBackend, config: ServiceConfig
) -> tuple[LesionComparison, PriorSeries] | None:
    """Récupère la FLAIR antérieure dans Orthanc, la segmente, la recale et compare."""
    o = config.orthanc
    peer = Peer(o.host, o.port, o.ae_title, config.listener.ae_title)
    prior = find_prior_flair(series.first, peer, config.processing.flair_patterns)
    if prior is None:
        logger.info("Aucune FLAIR antérieure trouvée dans Orthanc pour ce patient.")
        return None
    logger.info(
        "Antériorité : %s du %s (%d images)", prior.description, prior.study_date, prior.instances
    )
    with tempfile.TemporaryDirectory(prefix="segbrainmri_prior_") as tmp:
        retrieve_series(prior, peer, tmp)
        prior_series = load_series(tmp)
    logger.info("Segmentation de l'antériorité...")
    prior_analysis = analyze(backend, prior_series.image)
    logger.info("Recalage de l'antériorité sur l'examen actuel...")
    fixed = analysis.brain if analysis.brain is not None else series.image
    moving = prior_analysis.brain if prior_analysis.brain is not None else prior_series.image
    transform = register_rigid(fixed, moving)
    prior_mask = sitk.Resample(
        prior_analysis.mask,
        series.image,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    lg = config.longitudinal
    comparison = classify_lesions(
        _array(analysis.mask),
        _array(prior_mask),
        tuple(reversed(series.image.GetSpacing())),
        min_new_mm3=lg.min_new_mm3,
        tolerance_mm=lg.tolerance_mm,
        enlarge_ratio=lg.enlarge_ratio,
        enlarge_min_mm3=lg.enlarge_min_mm3,
    )
    return comparison, prior


def build_results(
    series: DicomSeries,
    mask: np.ndarray,
    info: ResultInfo,
    config: ServiceConfig,
    comparison: tuple[LesionComparison, PriorSeries] | None,
) -> list:
    out = config.output
    sources = series.datasets
    results = []
    if comparison is None:
        suffix = " - pas d'anteriorite" if config.longitudinal.enabled else ""
        if out.seg:
            results.append(build_seg(mask, sources, info))
        if out.contours:
            results += build_contour_series(
                mask,
                sources,
                info,
                series_description=f"SEP lesions {info.algorithm} - {info.summary}{suffix}",
                thickness_mm=out.contour_thickness_mm,
            )
    else:
        result, prior = comparison
        versus = f"vs {prior.study_date} - {result.new} nouv, {result.enlarging} elarg"
        if out.seg:
            # SEG 1 : toutes les lésions en rouge ; SEG 2 : évolution (nouvelles en rouge,
            # élargies en orange). La visionneuse les superpose en transparence sur la FLAIR,
            # qui reste fenêtrable.
            results.append(build_seg(mask, sources, info))
            evolution = np.where(result.labelmap == STABLE, 0, result.labelmap).astype(np.uint8)
            results.append(
                build_labelmap_seg(
                    evolution,
                    sources,
                    info,
                    [("Nouvelles lesions", RED), ("Lesions elargies", ORANGE)],
                    series_description=f"SEP EVOLUTION SEG {versus}",
                    series_offset=9005,
                )
            )
        if out.contours:
            results += build_contour_series(
                mask,
                sources,
                info,
                series_description=f"SEP lesions {info.algorithm} - {info.summary}",
                thickness_mm=out.contour_thickness_mm,
            )
            # Série couleur de l'évolution : nouvelles en rouge, élargies en orange
            results += build_color_contour_series(
                sitk.GetArrayFromImage(series.image).astype(np.float32),
                result.labelmap,
                {NEW: (255, 0, 0), ENLARGING: (255, 140, 0)},
                sources,
                info,
                series_description=f"SEP EVOLUTION vs {prior.study_date} - {result.new} nouv "
                f"(rouge), {result.enlarging} elarg (orange)",
                thickness_mm=out.contour_thickness_mm,
            )
    if out.sr:
        results.append(
            build_report(
                sources,
                series.image,
                info,
                comparison=comparison[0] if comparison else None,
                prior=PriorInfo(comparison[1].study_date, comparison[1].description)
                if comparison
                else None,
                longitudinal_searched=config.longitudinal.enabled,
                min_new_mm3=config.longitudinal.min_new_mm3,
            )
        )
    if out.overlay:
        volume = sitk.GetArrayFromImage(series.image).astype(np.float32)
        results += build_overlay(volume, mask, sources, info, opacity=out.overlay_opacity)
    return results


def process_series(
    series_dir: Path, backend: LesionBackend, config: ServiceConfig, *, send: bool = True
) -> list:
    """Segmente une série (et la compare à l'antériorité), renvoie et envoie les résultats."""
    series = load_series(series_dir)
    analysis = analyze(backend, series.image)
    mask = _array(analysis.mask)
    count, volume_ml = lesion_statistics(analysis.mask)
    info = ResultInfo(backend.name, backend.version, count, volume_ml)
    logger.info("Résultat : %s", info.summary)

    comparison = None
    if config.longitudinal.enabled:
        try:
            comparison = compare_with_prior(series, analysis, backend, config)
        except Exception:
            logger.exception("Comparaison avec l'antériorité impossible : envoi sans comparaison.")

    results = build_results(series, mask, info, config, comparison)
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


def build_volumetry_engine(config: ServiceConfig) -> VolumetryEngine:
    from segbrainmri.models.flames import SynthStrip
    from segbrainmri.volumetry.fastsurfer import FastSurfer
    from segbrainmri.volumetry.normative import NormativeLibrary

    vol, device = config.volumetry, get_device(config.processing.device)
    synth = config.model.models_dir / "synthstrip"
    stripper = SynthStrip(
        synth / "mri_synthstrip", synth / "synthstrip.1.pt", device, config.processing.threads
    )
    fastsurfer = FastSurfer(
        vol.fastsurfer_dir,
        vol.fastsurfer_python,
        device=device.type,
        threads=config.processing.threads,
    )
    try:
        norms = NormativeLibrary(vol.norms_dir)
    except FileNotFoundError:
        logger.warning("Modèles normatifs absents (%s) : volumes sans percentiles.", vol.norms_dir)
        norms = None
    version_file = vol.fastsurfer_dir / "VERSION_SEGBRAINMRI"
    version = version_file.read_text().strip() if version_file.is_file() else ""
    return VolumetryEngine(
        fastsurfer,
        stripper,
        norms,
        fastsurfer_version=version,
        tmp_root=config.processing.spool_dir / "tmp",
    )


def process_t1(
    series_dir: Path, engine: VolumetryEngine, config: ServiceConfig, *, send: bool = True
) -> list:
    """Volumétrie d'une T1 3D : SEG des structures + compte rendu SR, envoyés à Orthanc."""
    results = engine.run(load_series(series_dir))
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

    def __init__(
        self,
        config: ServiceConfig,
        backend: LesionBackend,
        volumetry: VolumetryEngine | None = None,
    ) -> None:
        self.config, self.backend = config, backend
        self._volumetry = volumetry
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
            description = header.get("SeriesDescription", "?")
            vol = self.config.volumetry
            if not is_our_output(header) and is_flair(
                header, self.config.processing.flair_patterns
            ):
                logger.info("Traitement de la FLAIR %s (%s)", description, work.name)
                process_series(work, self.backend, self.config)
            elif vol.enabled and is_t1_3d(header, vol.t1_patterns, vol.t1_exclude):
                logger.info("Volumétrie de la T1 3D %s (%s)", description, work.name)
                process_t1(work, self.volumetry_engine(), self.config)
            else:
                logger.info("Série ignorée (ni FLAIR ni T1 3D originale) : %s", description)
                shutil.rmtree(work)
                return
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

    def volumetry_engine(self) -> VolumetryEngine:
        """Créé au premier besoin : le service démarre même si FastSurfer manque."""
        if self._volumetry is None:
            self._volumetry = build_volumetry_engine(self.config)
        return self._volumetry

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
