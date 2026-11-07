#!/usr/bin/env bash
# Installation de SegBrainMRI en service sur Debian 12/13 (à lancer en root depuis le dépôt).
#   sudo ./deploy/install.sh          # PyTorch CPU (par défaut)
#   sudo ./deploy/install.sh --gpu    # PyTorch CUDA (pilote NVIDIA déjà installé)
set -euo pipefail

PREFIX=/opt/segbrainmri
TORCH_INDEX=https://download.pytorch.org/whl/cpu
[[ "${1:-}" == "--gpu" ]] && TORCH_INDEX=https://download.pytorch.org/whl/cu126
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

[[ $EUID -eq 0 ]] || { echo "À lancer avec sudo." >&2; exit 1; }

echo "==> Paquets système"
apt-get update -q
apt-get install -y -q python3 python3-venv python3-pip git

echo "==> Utilisateur de service"
id segbrainmri &>/dev/null || useradd --system --home-dir /var/lib/segbrainmri --shell /usr/sbin/nologin segbrainmri
install -d -o segbrainmri -g segbrainmri -m 0750 /var/lib/segbrainmri
install -d -m 0755 "$PREFIX" /etc/segbrainmri

echo "==> Environnement Python ($PREFIX/venv)"
[[ -d "$PREFIX/venv" ]] || python3 -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/pip" install -q --upgrade pip
# torch et torchvision doivent venir du même index (sinon : "torchvision::nms does not exist").
# On retire d'abord un éventuel torchvision installé depuis PyPI par une version précédente.
"$PREFIX/venv/bin/pip" uninstall -q -y torchvision 2>/dev/null || true
"$PREFIX/venv/bin/pip" install -q torch torchvision --index-url "$TORCH_INDEX"
"$PREFIX/venv/bin/pip" install -q "$REPO_DIR"
# Réinstalle toujours le code du dépôt (mise à jour), sans toucher aux dépendances
"$PREFIX/venv/bin/pip" install -q --force-reinstall --no-deps "$REPO_DIR"

echo "==> Modèles (FLAMeS + SynthStrip, ~1 Go)"
"$PREFIX/venv/bin/segbrainmri" download-models --dest "$PREFIX/models"

echo "==> Configuration"
[[ -f /etc/segbrainmri/config.toml ]] || install -m 0644 "$REPO_DIR/deploy/config.toml" /etc/segbrainmri/config.toml

echo "==> Service systemd"
install -m 0644 "$REPO_DIR/deploy/segbrainmri.service" /etc/systemd/system/segbrainmri.service
systemctl daemon-reload
systemctl enable segbrainmri
systemctl restart segbrainmri

echo "==> Test de connexion avec Orthanc"
"$PREFIX/venv/bin/segbrainmri" echo --config /etc/segbrainmri/config.toml || \
  echo "ATTENTION : Orthanc injoignable, vérifiez /etc/segbrainmri/config.toml et le pare-feu."

echo "Terminé. Journal du service : journalctl -u segbrainmri -f"
