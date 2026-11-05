"""Point d'entrée du service ``segbrainmri serve`` (géré par systemd)."""

from __future__ import annotations

import logging
import signal

from segbrainmri.dicom.network import StorageReceiver, echo
from segbrainmri.service.config import ServiceConfig
from segbrainmri.service.worker import SpoolWorker, build_backend

logger = logging.getLogger(__name__)


def serve(config: ServiceConfig) -> None:
    backend = build_backend(config)
    worker = SpoolWorker(config, backend)
    receiver = StorageReceiver(
        worker.incoming,
        ae_title=config.listener.ae_title,
        host=config.listener.host,
        port=config.listener.port,
        allowed_calling_aets=config.listener.allowed_calling_aets,
    )
    o = config.orthanc
    if echo(o.host, o.port, o.ae_title, config.listener.ae_title):
        logger.info("Orthanc joignable : %s@%s:%d", o.ae_title, o.host, o.port)
    else:
        logger.warning(
            "Orthanc %s@%s:%d injoignable pour l'instant (C-ECHO échoué).",
            o.ae_title,
            o.host,
            o.port,
        )

    def _shutdown(signum, _frame):
        logger.info("Signal %s reçu : arrêt après la série en cours.", signal.Signals(signum).name)
        worker.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    receiver.start()
    try:
        worker.run_forever()
    finally:
        receiver.stop()
        logger.info("Service arrêté.")
