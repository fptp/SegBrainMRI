import logging

import pytest
import torch

from segbrainmri.core import device as device_mod
from segbrainmri.core.device import get_device, resolve_device


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv(device_mod.ENV_VAR, raising=False)


def test_auto_falls_back_to_cpu(no_cuda):
    assert resolve_device("auto") == torch.device("cpu")
    assert resolve_device(None) == torch.device("cpu")


def test_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_cuda_requested_without_gpu_falls_back(no_cuda, caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_device("cuda") == torch.device("cpu")
    assert "bascule sur le CPU" in caplog.text


def test_cuda_strict_raises(no_cuda):
    with pytest.raises(RuntimeError):
        resolve_device("cuda:0", strict=True)


def test_env_variable(no_cuda, monkeypatch):
    monkeypatch.setenv(device_mod.ENV_VAR, "cpu")
    assert resolve_device() == torch.device("cpu")


def test_auto_selects_cuda_when_available(monkeypatch):
    monkeypatch.delenv(device_mod.ENV_VAR, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert resolve_device("auto") == torch.device("cuda", 0)
    with pytest.raises(ValueError):
        resolve_device("cuda:3")


def test_invalid_device():
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_get_device_logs_cpu_warning(no_cuda, caplog):
    with caplog.at_level(logging.INFO, logger="segbrainmri"):
        assert get_device("auto") == torch.device("cpu")
    assert "Calcul sur CPU" in caplog.text
    assert any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="aucun GPU")
def test_real_gpu():
    assert get_device("cuda").type == "cuda"
