# SegBrainMRI

Service open-source de **segmentation automatique des lésions de Sclérose en Plaques (SEP)**
sur IRM cérébrale, intégré à **Orthanc** :

```
IRM ──► Orthanc ──(FLAIR, DICOM)──► VM SegBrainMRI ──(résultats, DICOM)──► Orthanc
                                     SynthStrip + FLAMeS
```

1. Orthanc reçoit un examen. Un script Lua envoie automatiquement chaque série **FLAIR**
   à la VM SegBrainMRI.
2. La VM segmente les lésions en arrière-plan avec **FLAMeS**, sans intervention.
3. Les résultats rejoignent **le même examen** dans Orthanc (192.168.10.172, port 4242) :
   - **« SEP lesions … »** : la FLAIR avec le **contour des lésions** en blanc. Les pixels
     sont ceux de la série d'origine : elle **se fenêtre normalement** dans toutes les
     visionneuses (Orthanc, Horos…) ;
   - une série **DICOM SEG** (masque des lésions, affichable en couleur en surimpression
     dans OHIF, Stone Web Viewer, 3D Slicer, Weasis…).

   La description des séries indique le nombre de lésions et le volume lésionnel total.

### Afficher les lésions en couleur transparente sur la FLAIR fenêtrable

Une image DICOM est soit en niveaux de gris (fenêtrable), soit en couleur (figée) : un
masque coloré **transparent** au-dessus d'une FLAIR **qui reste fenêtrable** est affiché
par la visionneuse, à partir du **DICOM SEG** :

- **SEG « SEP lesions … »** : toutes les lésions, en rouge ;
- **SEG « SEP EVOLUTION SEG vs … »** (s'il y a une antériorité) : nouvelles lésions en
  rouge, élargies en orange.

Le viewer CAI ([fptp/Viewer](https://github.com/fptp/Viewer), module `seg-overlay.js`)
affiche ces SEG en surimpression. Les séries d'images dérivées (contours, couleur) ne sont
donc plus envoyées par défaut : `[output] contours = true` les réactive.

Autres visionneuses compatibles : **Weasis** (gratuit, Windows / macOS / Linux, version ≥ 4.3 :
opacité réglable par lésion), **OHIF** (plugin Orthanc), **3D Slicer**. Horos n'affiche pas
le DICOM SEG : il ne montre que les séries d'images (contours, série couleur).

### Comparaison avec l'examen antérieur (nouvelles lésions)

Si Orthanc contient une **FLAIR antérieure du même patient** (même identifiant patient,
date antérieure), la VM la récupère automatiquement, la segmente, la **recale** sur
l'examen actuel (recalage rigide, sur le cerveau extrait) et classe chaque lésion :

| Catégorie | Règle (réglable dans `[longitudinal]`) |
|---|---|
| **Nouvelle** | aucun recouvrement avec une lésion antérieure (à 1 mm près) et ≥ 10 mm³ |
| **Élargie** | volume ≥ 1,5 × l'ancien et au moins +20 mm³ |
| **Autre** | stable, ou trop petite pour être signalée |

Séries supplémentaires renvoyées :
- **« SEP EVOLUTION vs AAAAMMJJ - N nouv (rouge), M elarg (orange) »** : la FLAIR avec le
  contour **rouge** des nouvelles lésions et **orange** des lésions élargies (les autres ne
  sont pas dessinées). Série couleur : elle s'affiche avec le fenêtrage par défaut de la
  série d'origine et ne peut pas être refenêtrée ;
- le DICOM SEG contient trois segments : nouvelles (rouge), élargies (orange), autres (jaune).

### Compte rendu DICOM SR

Chaque analyse renvoie aussi une série **« SEP compte rendu … »** (DICOM SR) contenant :
un résumé en une phrase, le nombre de lésions et le volume lésionnel total, puis en cas
de comparaison la date de l'antériorité, les nombres de lésions nouvelles / élargies /
stables et, **pour chaque lésion nouvelle ou élargie, son volume et sa position** (numéro
d'image et coordonnées en mm), pour la retrouver rapidement.

Sans antériorité dans Orthanc, la description indique « pas d'anteriorite ». Le traitement
dure environ deux fois plus longtemps (deux examens segmentés).

> ⚠️ Des différences de machine, de protocole ou de segmentation entre les deux examens
> peuvent produire de **fausses nouvelles lésions** : la relecture par le radiologue reste
> indispensable.

Fonctionne **sur CPU** (plusieurs minutes par examen) et bascule automatiquement sur le
**GPU** dès qu'il est installé.

> ⚠️ **Outil de recherche, non certifié dispositif médical.** Les résultats ne doivent pas
> servir seuls à un diagnostic.

## Volumétrie cérébrale (troubles cognitifs) : T1 3D

Une **T1 3D non injectée** (BRAVO, MPRAGE, SPGR…) envoyée à la VM est analysée par
**FastSurfer** v2.6.1 (Apache 2.0) : segmentation en structures (atlas aseg + 31 régions
corticales DKT par hémisphère), volumes corrigés du volume partiel.

Résultats renvoyés dans l'examen, dans Orthanc :
- **SEG « VOLUMETRIE structures FastSurfer »** : lobes frontal, pariétal, temporal,
  occipital, cingulaire, insula, hippocampes, amygdales, ventricules latéraux et cornes
  temporales, chacun dans sa couleur ;
- **SR « VOLUMETRIE compte rendu FastSurfer »** : pour chaque structure, volume gauche et
  droit (mL), % du volume intracrânien, indice d'asymétrie et, pour les **hippocampes,
  amygdales, cornes temporales, ventricules latéraux, thalamus et substance grise
  sous-corticale**, le **percentile et le Z-score pour l'âge et le sexe**, avec le volume
  médian attendu. Le résumé signale les valeurs hors normes (< P5, ou > P95 pour les
  ventricules).

Normes : modèles **BrainCharts** `lifespan_57K_82sites` (Rutherford et al., eLife 2022 ;
~58 000 sujets sains, volumes FreeSurfer), licence GPL-3. Leur calcul est réimplémenté
(pcntoolkit 0.30 ne s'installe plus en Python 3.13) et validé contre le code d'origine
(écart < 10⁻¹⁴).

> ⚠️ **Limites** : percentiles **indicatifs**, les normes n'étant **pas calibrées sur le
> scanner local** (une calibration demande quelques dizaines de T1 3D normales de ce
> scanner). Les volumes des **lobes** n'ont pas de normes publiées : volumes, % du volume
> intracrânien et asymétrie seulement. Le volume intracrânien est estimé par SynthStrip.
> Les T1 injectées sont ignorées. Sur CPU, compter de l'ordre de 5 à 10 minutes par examen
> et ~8 Go de RAM (FastSurfer).

## Le modèle : FLAMeS

[FLAMeS](https://doi.org/10.1101/2025.05.19.25327707) (*FLAIR Lesion Analysis in Multiple
Sclerosis*, 2025) est un modèle nnU-Net v2 entraîné sur 668 FLAIR (1,5 T et 3 T ; Siemens,
GE, Philips). Sur les jeux de test publiés, il obtient un Dice moyen de 0,74 et dépasse
LST-AI, SAMSEG et LST-LPA. Il n'a besoin que de la **FLAIR**.

Chaîne reproduite à l'identique de l'inférence de référence :
SynthStrip (extraction du cerveau) → nnU-Net v2 (`Dataset004_WML`, `3d_fullres`,
`nnUNetTrainer_8000epochs`, tous les folds). Les poids sont téléchargés depuis
[Zenodo](https://zenodo.org/records/17955359) (somme MD5 vérifiée) et ne sont jamais versionnés.

Si vous l'utilisez dans une publication, citez FLAMeS (Dereskewicz et al., 2025) et
SynthStrip (Hoopes et al., NeuroImage 2022).

## Installation sur la VM Debian (12 / 13)

```bash
sudo apt install -y git
```

```bash
git clone https://github.com/fptp/SegBrainMRI.git
```

```bash
cd SegBrainMRI && sudo ./deploy/install.sh
```

Le script :
- crée l'utilisateur système `segbrainmri` ;
- installe PyTorch **CPU** et le module dans `/opt/segbrainmri/venv` ;
- télécharge FLAMeS et SynthStrip (≈ 1 Go) dans `/opt/segbrainmri/models` ;
- installe la configuration `/etc/segbrainmri/config.toml` et le service systemd
  `segbrainmri`, puis le démarre ;
- teste la connexion DICOM avec Orthanc.

Le pare-feu de la VM doit accepter les connexions entrantes sur le **port 11112** depuis Orthanc.

### Configuration de la VM

Tout se règle dans `/etc/segbrainmri/config.toml` (modèle commenté : `deploy/config.toml`).
Les valeurs par défaut correspondent à votre installation : résultats envoyés à
`ORTHANC@192.168.10.172:4242`, réception sur le port 11112 (AET `SEGBRAINMRI`).

Après une modification :

```bash
sudo systemctl restart segbrainmri
```

## Configuration d'Orthanc (VM 192.168.10.172)

1. Copier `deploy/orthanc/segbrainmri.lua` dans `/etc/orthanc/segbrainmri.lua`.
2. Ajouter dans la configuration d'Orthanc (ex. `/etc/orthanc/orthanc.json`) le contenu de
   `deploy/orthanc/orthanc-segbrainmri.json`, (adresse de la VM SegBrainMRI : 192.168.10.145, à adapter si elle change).
   Si `DicomModalities` ou `LuaScripts` existent déjà, ajouter les entrées au lieu de dupliquer les clés.
3. Redémarrer Orthanc :
   ```bash
   sudo systemctl restart orthanc
   ```

L'AET d'Orthanc doit être `ORTHANC` (valeur par défaut). Sinon, l'ajuster dans
`allowed_calling_aets` et `[orthanc] ae_title` de `/etc/segbrainmri/config.toml`.

Envoi manuel possible : dans l'interface d'Orthanc, ouvrir une série FLAIR, puis
« Send to DICOM modality » → `segbrainmri`.

## Suivi et vérifications

| Action | Commande |
|---|---|
| Journal en direct | `journalctl -u segbrainmri -f` |
| État du service | `systemctl status segbrainmri` |
| Tester la connexion avec Orthanc | `/opt/segbrainmri/venv/bin/segbrainmri echo` |
| Matériel détecté (CPU/GPU) | `/opt/segbrainmri/venv/bin/segbrainmri info` |
| Séries en erreur | `/var/lib/segbrainmri/failed/<série>/error.txt` |

Les images reçues sont **supprimées après un traitement réussi** (`keep_inputs = false`).
Les séries en erreur sont conservées dans `/var/lib/segbrainmri/failed/` pour diagnostic :
pensez à les purger.

**Sélection des séries.** Une série est traitée si elle est de modalité MR, originale (pas
`DERIVED`), et si sa description, son protocole ou sa séquence contient « FLAIR »
(motifs réglables via `flair_patterns`). Les autres séries sont ignorées.

## Passage au GPU

1. Rendre le GPU visible dans la VM (PCI passthrough côté hyperviseur).
2. Installer le pilote NVIDIA (dépôts `contrib non-free non-free-firmware` activés), redémarrer :
   ```bash
   sudo apt install -y nvidia-driver firmware-misc-nonfree
   ```
3. Vérifier que `nvidia-smi` affiche le GPU, puis réinstaller avec PyTorch CUDA :
   ```bash
   sudo ./deploy/install.sh --gpu
   ```

Avec `device = "auto"`, le service utilise le GPU au redémarrage. Il active alors aussi
l'augmentation au test (`tta = "auto"`), comme l'inférence de référence de FLAMeS.

## Utilisation en ligne de commande (sans Orthanc)

```bash
segbrainmri segment --flair sub-01_FLAIR.nii.gz -o resultats/
```

```bash
segbrainmri process-dicom -i dossier_serie_flair/ -o resultats_dicom/
```

Ajouter `--send` à `process-dicom` pour envoyer aussi les résultats à Orthanc.

## Structure

```
src/segbrainmri/
├── core/           # device.py (CPU/GPU), logging.py
├── models/         # flames.py (FLAMeS + SynthStrip), unet3d.py (référence), backends.py
├── dicom/          # series.py, writers.py (SEG, contours), network.py, query.py (antériorité)
├── longitudinal/   # compare.py (recalage, nouvelles lésions)
├── volumetry/      # FastSurfer, lobes, normes BrainCharts, SEG + SR
├── service/        # config.py, worker.py, app.py (service `segbrainmri serve`)
├── preprocessing/  # pipeline générique NIfTI (pour le UNet3D de référence)
├── inference/      # fenêtre glissante, statistiques lésionnelles
└── cli.py
deploy/             # install.sh, segbrainmri.service, config.toml, orthanc/
tests/              # pytest, données synthétiques uniquement
```

## Limites connues

- Les séries **Enhanced MR multi-trames** ne sont pas encore prises en charge : exporter la
  FLAIR en DICOM « classique » (une image par fichier).
- FLAMeS est entraîné sur des FLAIR cérébrales d'adultes. Les petites lésions (< 10 mm³)
  sont les plus souvent manquées.

## Développement

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -e ".[dev]" && pytest
```

## Données et confidentialité

Le `.gitignore` exclut tous les formats d'imagerie (`.nii`, `.nii.gz`, DICOM, `.mgz`…), les
tableaux (`.csv`, `.tsv`) et les poids de modèles (`.pt`, `.pth`, `.zip`…). Les tests
n'utilisent que des images synthétiques.
