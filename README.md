# SegBrainMRI

Module Python open-source pour la **segmentation des lésions de Sclérose en Plaques (SEP)**
sur IRM cérébrale (T2-FLAIR, T1 optionnelle) au format NIfTI.

- **CPU par défaut**, bascule **automatique vers CUDA** dès qu'un GPU est disponible
  (aucune modification de code nécessaire).
- Pipeline de prétraitement : réorientation, rééchantillonnage isotrope, correction N4
  optionnelle, normalisation d'intensité.
- Inférence par **fenêtre glissante** (mémoire maîtrisée sur CPU), masque renvoyé dans
  l'espace natif du patient, avec le nombre et le volume des lésions.

> ⚠️ **Outil de recherche, non certifié dispositif médical.** Ne pas utiliser pour un
> diagnostic clinique.

## Structure

```
SegBrainMRI/
├── pyproject.toml
├── src/segbrainmri/
│   ├── core/            # device.py (CPU/GPU), logging.py
│   ├── preprocessing/   # io.py, spatial.py, intensity.py, pipeline.py, config.py
│   ├── models/          # base.py (interface), registry.py, unet3d.py (référence)
│   ├── inference/       # sliding_window.py, predictor.py
│   └── cli.py           # commande `segbrainmri`
├── tests/               # pytest, volumes synthétiques (aucune donnée réelle)
├── data/{raw,processed,output}/   # ignorés par git
└── weights/                       # ignoré par git
```

## Installation sur Debian (12 / 13)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

```bash
git clone https://github.com/fptp/SegBrainMRI.git
cd SegBrainMRI
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Mode CPU (actuel)

Installer d'abord la roue PyTorch **CPU** (≈ 200 Mo au lieu de plusieurs Go pour la version CUDA) :

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -e ".[dev]"
```

### Passage au GPU (plus tard)

1. Rendre le GPU visible dans la VM (PCI passthrough côté hyperviseur).
2. Installer le pilote NVIDIA (dépôts `contrib non-free non-free-firmware` activés dans
   `/etc/apt/sources.list`) :
   ```bash
   sudo apt install -y nvidia-driver firmware-misc-nonfree
   ```
   puis redémarrer et vérifier avec `nvidia-smi`.
3. Remplacer la roue CPU par une roue CUDA (le runtime CUDA est embarqué dans la roue,
   le *CUDA Toolkit* n'est pas nécessaire). Choisir l'index adapté au pilote sur
   <https://pytorch.org/get-started/locally/>, par exemple :
   ```bash
   pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
   ```
4. Vérifier : `segbrainmri info` doit afficher `CUDA dispo. : True`.

## Utilisation

```bash
segbrainmri info
```

```bash
segbrainmri preprocess --flair data/raw/sub-01_FLAIR.nii.gz -o data/processed/sub-01
```

```bash
segbrainmri segment --flair data/raw/sub-01_FLAIR.nii.gz --t1 data/raw/sub-01_T1w.nii.gz \
    --weights weights/unet3d_flair_t1.pt -o data/output/sub-01
```

Sorties : `lesions_mask.nii.gz` (masque binaire, espace natif) et `lesions_prob.nii.gz`.

Options utiles :

| Option | Rôle |
|---|---|
| `--device auto\|cpu\|cuda\|cuda:N` | choix de l'appareil (ou variable `SEGBRAINMRI_DEVICE`) |
| `--patch-size 96` | réduire si la RAM/VRAM est insuffisante (multiple de 8) |
| `--threads 8` | nombre de threads CPU PyTorch |
| `--spacing 1,1,1` / `--native-spacing` | résolution de travail |
| `--n4` | correction du biais de champ (lent sur CPU) |
| `--untrained` | poids aléatoires : **test du pipeline uniquement** |

Si `--device cuda` est demandé sans GPU, le programme bascule sur le CPU avec un
avertissement. En Python, `resolve_device("cuda", strict=True)` lève une erreur à la place.

### API Python

```python
from segbrainmri.core import get_device
from segbrainmri.inference import LesionSegmenter
from segbrainmri.models import build_model

device = get_device()                       # CUDA si disponible, sinon CPU (+ avertissement)
model = build_model("unet3d", in_channels=1)
model.load_weights("weights/unet3d_flair.pt", device)
result = LesionSegmenter(model, device=device).segment({"flair": "sub-01_FLAIR.nii.gz"})
result.save("data/output/sub-01")
print(result.lesion_count, result.lesion_volume_ml)
```

## Hypothèses et limites actuelles

- Les modalités (FLAIR, T1) doivent être **déjà co-registrées** ; la première fournie
  (FLAIR) sert de grille de référence.
- Le masque d'avant-plan (Otsu) n'est **pas** une extraction du cerveau : pour de meilleurs
  résultats, appliquer HD-BET ou SynthStrip en amont.
- Le `UNet3D` fourni est une architecture de référence **non entraînée**. Aucun poids
  n'est distribué ici ; la suite consiste à brancher un modèle pré-entraîné (nnU-Net,
  bundle MONAI, LST-AI…) via l'interface `SegmentationModel`, en reproduisant exactement
  son prétraitement d'entraînement.

## Tests

```bash
pytest
```

## Données et confidentialité

Le `.gitignore` exclut tous les formats d'imagerie (`.nii`, `.nii.gz`, DICOM, `.mgz`…),
les tableaux (`.csv`, `.tsv`), et les poids (`.pt`, `.pth`, `.ckpt`, `.onnx`…).
Vérifiez avec `git status` avant chaque commit qu'aucune donnée patient n'est indexée.
