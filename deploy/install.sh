#!/usr/bin/env bash
# Installation de SegBrainMRI en service sur Debian 12/13 (à lancer en root depuis le dépôt).
#   sudo ./deploy/install.sh          # PyTorch CPU (par défaut)
#   sudo ./deploy/install.sh --gpu    # PyTorch CUDA (pilote NVIDIA déjà installé)
set -euo pipefail

PREFIX=/opt/segbrainmri
FASTSURFER_TAG=v2.6.1
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

echo "==> Modèles (FLAMeS + SynthStrip + normes de volumétrie, ~1 Go)"
"$PREFIX/venv/bin/segbrainmri" download-models --dest "$PREFIX/models"

echo "==> FastSurfer $FASTSURFER_TAG (volumétrie des T1 3D, environnement séparé)"
FS_HOME="$PREFIX/fastsurfer"
FS_VENV="$PREFIX/fastsurfer-venv"
if [[ ! -d "$FS_HOME/.git" ]]; then
  git clone -q --depth 1 --branch "$FASTSURFER_TAG" https://github.com/Deep-MI/FastSurfer "$FS_HOME"
elif [[ "$(cat "$FS_HOME/VERSION_SEGBRAINMRI" 2>/dev/null)" != "$FASTSURFER_TAG" ]]; then
  git -C "$FS_HOME" fetch -q --depth 1 origin tag "$FASTSURFER_TAG"
  git -C "$FS_HOME" checkout -q "$FASTSURFER_TAG"
fi
echo "$FASTSURFER_TAG" > "$FS_HOME/VERSION_SEGBRAINMRI"
[[ -d "$FS_VENV" ]] || python3 -m venv "$FS_VENV"
"$FS_VENV/bin/pip" install -q --upgrade pip
# PyTorch (CPU ou CUDA) aux versions figées par FastSurfer, puis le reste de ses dépendances
FS_TORCH=$(grep -E '^torch==' "$FS_HOME/requirements.txt")
FS_TORCHVISION=$(grep -E '^torchvision==' "$FS_HOME/requirements.txt")
"$FS_VENV/bin/pip" uninstall -q -y torchvision 2>/dev/null || true
"$FS_VENV/bin/pip" install -q "$FS_TORCH" "$FS_TORCHVISION" --index-url "$TORCH_INDEX"
"$FS_VENV/bin/pip" install -q -r "$FS_HOME/requirements.txt"
# Poids du réseau FastSurferVINN (le service n'a pas le droit d'écrire dans /opt)
(cd "$FS_HOME" && FASTSURFER_HOME="$FS_HOME" PYTHONPATH="$FS_HOME" \
  "$FS_VENV/bin/python" FastSurferCNN/download_checkpoints.py --vinn)
ls "$FS_HOME"/checkpoints/aparc_vinn_*.pkl >/dev/null

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
