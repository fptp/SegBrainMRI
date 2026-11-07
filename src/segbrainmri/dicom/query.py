"""Recherche et récupération de l'examen antérieur dans Orthanc (C-FIND + C-GET).

Orthanc doit connaître notre AET (entrée ``segbrainmri`` de ``DicomModalities``) : il
autorise alors par défaut C-FIND et C-GET depuis la VM. C-GET récupère les images sur la
même association : elles n'arrivent pas dans le dossier de réception du service.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydicom.dataset import Dataset
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian
from pynetdicom import AE, build_role, evt
from pynetdicom.sop_class import (
    EnhancedMRImageStorage,
    MRImageStorage,
    StudyRootQueryRetrieveInformationModelFind,
    StudyRootQueryRetrieveInformationModelGet,
)

from segbrainmri.dicom.series import OUR_MANUFACTURER

logger = logging.getLogger(__name__)

_PENDING = (0xFF00, 0xFF01)


@dataclass(frozen=True)
class PriorSeries:
    study_uid: str
    series_uid: str
    study_date: str
    description: str
    instances: int


@dataclass(frozen=True)
class Peer:
    host: str
    port: int
    called_aet: str
    calling_aet: str
    timeout: float = 30.0


def _associate(peer: Peer, contexts: Iterable, **kwargs):
    ae = AE(ae_title=peer.calling_aet)
    ae.connection_timeout = ae.acse_timeout = peer.timeout
    ae.dimse_timeout = ae.network_timeout = max(peer.timeout, 120)
    for context in contexts:
        if isinstance(context, tuple):
            ae.add_requested_context(*context)
        else:
            ae.add_requested_context(context)
    assoc = ae.associate(peer.host, peer.port, ae_title=peer.called_aet, **kwargs)
    if not assoc.is_established:
        raise ConnectionError(
            f"Association refusée par {peer.called_aet}@{peer.host}:{peer.port} "
            "(l'AET de la VM est-il déclaré dans DicomModalities d'Orthanc ?)"
        )
    return assoc


def _find(peer: Peer, query: Dataset) -> list[Dataset]:
    assoc = _associate(peer, [StudyRootQueryRetrieveInformationModelFind])
    try:
        answers = []
        for status, identifier in assoc.send_c_find(
            query, StudyRootQueryRetrieveInformationModelFind
        ):
            code = getattr(status, "Status", None)
            if code in _PENDING and identifier is not None:
                answers.append(identifier)
            elif code not in (0x0000, *_PENDING):
                raise RuntimeError(f"C-FIND refusé (statut {code!r})")
        return answers
    finally:
        assoc.release()


def _text(ds: Dataset, keyword: str) -> str:
    return str(ds.get(keyword, "") or "")


def find_prior_flair(
    current: Dataset, peer: Peer, flair_patterns: Iterable[str]
) -> PriorSeries | None:
    """Série FLAIR de l'examen le plus récent du patient, antérieur à ``current``."""
    patient_id = _text(current, "PatientID")
    if not patient_id:
        logger.warning("Pas de PatientID : recherche d'antériorité impossible.")
        return None
    current_key = (_text(current, "StudyDate"), _text(current, "StudyTime"))

    query = Dataset()
    query.QueryRetrieveLevel = "STUDY"
    query.PatientID = patient_id
    query.StudyInstanceUID = query.StudyDate = query.StudyTime = ""
    studies = [
        s
        for s in _find(peer, query)
        if _text(s, "StudyInstanceUID") != _text(current, "StudyInstanceUID")
        and (_text(s, "StudyDate"), _text(s, "StudyTime")) < current_key
    ]
    studies.sort(key=lambda s: (_text(s, "StudyDate"), _text(s, "StudyTime")), reverse=True)
    patterns = [re.compile(p, re.IGNORECASE) for p in flair_patterns]
    wanted = _text(current, "SeriesDescription").lower()

    for study in studies:
        query = Dataset()
        query.QueryRetrieveLevel = "SERIES"
        query.StudyInstanceUID = _text(study, "StudyInstanceUID")
        query.SeriesInstanceUID = query.SeriesDescription = query.ProtocolName = ""
        query.Modality = query.Manufacturer = query.NumberOfSeriesRelatedInstances = ""
        candidates = []
        for series in _find(peer, query):
            text = f"{_text(series, 'SeriesDescription')} {_text(series, 'ProtocolName')}"
            count = int(_text(series, "NumberOfSeriesRelatedInstances") or 0)
            if (
                _text(series, "Modality") == "MR"
                and _text(series, "Manufacturer") != OUR_MANUFACTURER
                and count >= 3
                and any(p.search(text) for p in patterns)
            ):
                candidates.append((series, count))
        if candidates:
            # Même description que la série actuelle en priorité, puis la plus fournie
            series, count = max(
                candidates,
                key=lambda c: (_text(c[0], "SeriesDescription").lower() == wanted, c[1]),
            )
            return PriorSeries(
                study_uid=_text(study, "StudyInstanceUID"),
                series_uid=_text(series, "SeriesInstanceUID"),
                study_date=_text(study, "StudyDate"),
                description=_text(series, "SeriesDescription"),
                instances=count,
            )
    return None


def retrieve_series(prior: PriorSeries, peer: Peer, dest: str | Path) -> int:
    """Récupère la série par C-GET dans ``dest`` ; renvoie le nombre d'images reçues."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    received = 0

    def on_store(event) -> int:
        nonlocal received
        ds = event.dataset
        (dest / f"{ds.SOPInstanceUID}.dcm").write_bytes(event.encoded_dataset())
        received += 1
        return 0x0000

    storage = [MRImageStorage, EnhancedMRImageStorage]
    contexts = [StudyRootQueryRetrieveInformationModelGet] + [
        (sop, [ExplicitVRLittleEndian, ImplicitVRLittleEndian]) for sop in storage
    ]
    assoc = _associate(
        peer,
        contexts,
        ext_neg=[build_role(sop, scp_role=True) for sop in storage],
        evt_handlers=[(evt.EVT_C_STORE, on_store)],
    )
    try:
        query = Dataset()
        query.QueryRetrieveLevel = "SERIES"
        query.StudyInstanceUID = prior.study_uid
        query.SeriesInstanceUID = prior.series_uid
        final = None
        for status, _ in assoc.send_c_get(query, StudyRootQueryRetrieveInformationModelGet):
            final = getattr(status, "Status", None)
        if final not in (0x0000, None) and final not in _PENDING:
            raise RuntimeError(f"C-GET terminé avec le statut {final!r}")
    finally:
        assoc.release()
    if received < prior.instances:
        raise RuntimeError(f"C-GET incomplet : {received}/{prior.instances} images reçues")
    logger.info("Antériorité récupérée : %d images (%s)", received, prior.study_date)
    return received
