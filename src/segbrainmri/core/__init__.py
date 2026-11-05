"""Briques transverses : gestion du matériel (CPU/GPU) et logging."""

from segbrainmri.core.device import DeviceInfo, get_device, resolve_device
from segbrainmri.core.logging import setup_logging

__all__ = ["DeviceInfo", "get_device", "resolve_device", "setup_logging"]
