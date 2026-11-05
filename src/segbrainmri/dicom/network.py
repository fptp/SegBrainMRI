"""Réseau DICOM : réception (C-STORE SCP) et envoi vers Orthanc (C-STORE SCU)."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

from pydicom.dataset import Dataset
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian
from pynetdicom import AE, StoragePresentationContexts, evt
from pynetdicom.sop_class import Verification

logger = logging.getLogger(__name__)

_SUCCESS = 0x0000
_OUT_OF_RESOURCES = 0xA700
_NOT_AUTHORIZED = 0x0124


class StorageReceiver:
    """Serveur C-STORE : range chaque instance dans ``spool/<SeriesInstanceUID>/``.

    Écriture atomique (fichier ``.part`` puis renommage) pour que le worker ne lise
    jamais un fichier incomplet.
    """

    def __init__(
        self,
        spool_dir: str | Path,
        *,
        ae_title: str = "SEGBRAINMRI",
        host: str = "0.0.0.0",
        port: int = 11112,
        allowed_calling_aets: Sequence[str] = (),
    ) -> None:
        self.spool_dir = Path(spool_dir)
        self.ae_title, self.host, self.port = ae_title, host, port
        self.allowed = {a.strip() for a in allowed_calling_aets if a.strip()}
        self._server = None

    def _on_store(self, event) -> int:
        calling = event.assoc.requestor.ae_title.strip()
        if self.allowed and calling not in self.allowed:
            logger.warning("C-STORE refusé depuis l'AET non autorisé %r", calling)
            return _NOT_AUTHORIZED
        try:
            ds = event.dataset
            series_dir = self.spool_dir / str(ds.SeriesInstanceUID)
            series_dir.mkdir(parents=True, exist_ok=True)
            target = series_dir / f"{ds.SOPInstanceUID}.dcm"
            tmp = target.with_suffix(".part")
            tmp.write_bytes(event.encoded_dataset())
            tmp.replace(target)
        except Exception:
            logger.exception("Échec d'enregistrement d'une instance reçue")
            return _OUT_OF_RESOURCES
        return _SUCCESS

    def start(self) -> None:
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        ae = AE(ae_title=self.ae_title)
        ae.supported_contexts = StoragePresentationContexts
        ae.add_supported_context(Verification)  # C-ECHO, pour tester depuis Orthanc
        self._server = ae.start_server(
            (self.host, self.port),
            block=False,
            evt_handlers=[(evt.EVT_C_STORE, self._on_store)],
        )
        logger.info("Récepteur DICOM %s à l'écoute sur %s:%d", self.ae_title, self.host, self.port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None


def send_datasets(
    datasets: Sequence[Dataset],
    *,
    host: str,
    port: int,
    called_aet: str = "ORTHANC",
    calling_aet: str = "SEGBRAINMRI",
    retries: int = 3,
    timeout: float = 60.0,
    connect_timeout: float = 10.0,
) -> None:
    """Envoie ``datasets`` par C-STORE ; nouvelle tentative (attente croissante) si échec."""
    if not datasets:
        return
    sop_classes = sorted({str(ds.SOPClassUID) for ds in datasets})
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        ae = AE(ae_title=calling_aet)
        ae.acse_timeout = ae.dimse_timeout = ae.network_timeout = timeout
        ae.connection_timeout = connect_timeout
        for sop in sop_classes:
            ae.add_requested_context(sop, [ExplicitVRLittleEndian, ImplicitVRLittleEndian])
        assoc = ae.associate(host, port, ae_title=called_aet)
        try:
            if not assoc.is_established:
                raise ConnectionError(
                    f"Association refusée ou impossible avec {called_aet}@{host}:{port}"
                )
            for ds in datasets:
                status = assoc.send_c_store(ds)
                code = getattr(status, "Status", None)
                if code != _SUCCESS:
                    raise RuntimeError(f"C-STORE refusé (statut {code!r}) pour {ds.SOPInstanceUID}")
            logger.info(
                "%d instance(s) envoyée(s) à %s@%s:%d", len(datasets), called_aet, host, port
            )
            return
        except (ConnectionError, RuntimeError) as exc:
            last_error = exc
            logger.warning("Envoi DICOM : tentative %d/%d échouée : %s", attempt, retries, exc)
        finally:
            if assoc.is_established:
                assoc.release()
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Échec de l'envoi vers {called_aet}@{host}:{port}") from last_error


def echo(host: str, port: int, called_aet: str, calling_aet: str = "SEGBRAINMRI") -> bool:
    """C-ECHO : vérifie la connexion DICOM avec Orthanc."""
    ae = AE(ae_title=calling_aet)
    ae.connection_timeout = ae.acse_timeout = 10
    ae.add_requested_context(Verification)
    assoc = ae.associate(host, port, ae_title=called_aet)
    if not assoc.is_established:
        return False
    try:
        status = assoc.send_c_echo()
        return getattr(status, "Status", None) == _SUCCESS
    finally:
        assoc.release()
