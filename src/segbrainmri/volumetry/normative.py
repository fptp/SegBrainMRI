"""Normes par âge et sexe : modèles BrainCharts / PCNtoolkit (Rutherford et al., eLife 2022).

Modèle ``lifespan_57K_82sites`` : régression bayésienne (BLR) entraînée sur ~58 000 IRM de
sujets sains (82 sites) sur les volumes ``aseg`` de FreeSurfer, avec covariables âge et
sexe, base B-spline sur l'âge, effet fixe par site et vraisemblance « sinh-arcsinh »
(distribution non gaussienne).

``pcntoolkit`` 0.30, qui a produit ces fichiers, ne s'installe plus avec Python 3.13 : la
prédiction est réimplémentée ici à l'identique (validée numériquement contre le code
d'origine). Les fichiers modèles (pickle) sont lus par un désérialiseur restreint qui
n'instancie que des tableaux numpy et des conteneurs de données (aucun code exécuté).

Sans calibration sur le scanner local, les colonnes « site » sont mises à zéro, comme le
font les auteurs pour leurs courbes de référence : les percentiles sont alors indicatifs.

Modèles et code d'origine : https://github.com/predictive-clinical-neuroscience/braincharts
(licence GPL-3).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import norm

#: Paramètres du modèle lifespan_57K_82sites (script apply_normative_models de BrainCharts)
XMIN, XMAX, SPLINE_ORDER, N_KNOTS, N_SITES = -5.0, 110.0, 3, 5, 82
MODEL_NAME = "lifespan_57K_82sites"


class _Record:
    """Conteneur neutre pour les objets pcntoolkit désérialisés (attributs seulement)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _SafeUnpickler(pickle.Unpickler):
    _NUMPY = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module, name):
        if (module, name) in self._NUMPY:
            return super().find_class(module, name)
        if module.split(".")[0] in {"pcntoolkit", "norm_blr", "bayesreg", "utils"}:
            return type(name, (_Record,), {})
        raise pickle.UnpicklingError(f"Classe non autorisée dans le modèle : {module}.{name}")


def _augknt(knots: np.ndarray, order: int) -> np.ndarray:
    """Vecteur de nœuds augmenté (équivalent de bspline.splinelab.augknt)."""
    return np.concatenate([[knots[0]] * order, knots, [knots[-1]] * order])


@lru_cache(maxsize=1)
def _spline_basis():
    knots = _augknt(np.linspace(XMIN, XMAX, N_KNOTS), SPLINE_ORDER)
    n_basis = len(knots) - SPLINE_ORDER - 1
    return knots, n_basis


def design_row(age: float, sex: int) -> np.ndarray:
    """Ligne de la matrice de plan : [1, âge, sexe, sites (zéros), B-spline(âge)].

    ``sex`` : 1 = homme, 0 = femme (codage BrainCharts).
    """
    knots, n_basis = _spline_basis()
    basis = np.array(
        [
            BSpline(knots, np.eye(n_basis)[i], SPLINE_ORDER, extrapolate=False)(age)
            for i in range(n_basis)
        ]
    )
    basis = np.nan_to_num(basis)
    # Convention de bspline.Bspline : la dernière fonction vaut 1 à l'extrémité droite
    if age >= XMAX:
        basis[:] = 0.0
        basis[-1] = 1.0
    return np.concatenate([[1.0, age, float(sex)], np.zeros(N_SITES), basis])


def _warp_params(gamma: np.ndarray) -> tuple[float, float]:
    epsilon, b = float(gamma[0]), float(np.exp(gamma[1]))
    return -epsilon * b, b  # (a, b)


def warp(y, gamma):
    """Transformation sinh-arcsinh (espace des volumes -> espace gaussien)."""
    a, b = _warp_params(gamma)
    return np.sinh(b * np.arcsinh(y) - a)


def unwarp(z, gamma):
    a, b = _warp_params(gamma)
    return np.sinh((np.arcsinh(z) + a) / b)


@dataclass
class NormativeModel:
    name: str
    m: np.ndarray  # coefficients a posteriori
    A: np.ndarray  # précision a posteriori
    hyp: np.ndarray  # [log bêta, epsilon, log b, log alpha]
    warp_reparam: bool = True

    @classmethod
    def load(cls, path: str | Path, name: str = "") -> NormativeModel:
        with open(path, "rb") as fh:
            nm = _SafeUnpickler(fh).load()
        blr = nm.blr
        if type(getattr(blr, "warp", None)).__name__ != "WarpSinArcsinh":
            raise ValueError(f"Modèle {name or path} : transformation inattendue.")
        return cls(
            name=name,
            m=np.asarray(blr.m),
            A=np.asarray(blr.A),
            hyp=np.asarray(blr.hyp),
            warp_reparam=bool(blr.warp_reparam),
        )

    @property
    def gamma(self) -> np.ndarray:
        return self.hyp[1:3]

    def _gaussian(self, age: float, sex: int) -> tuple[float, float]:
        x = design_row(age, sex)
        if x.shape[0] != self.m.shape[0]:
            raise ValueError(f"Modèle {self.name} : {self.m.shape[0]} coefficients attendus.")
        beta = float(np.exp(self.hyp[0]))
        if self.warp_reparam:
            beta /= float(np.exp(self.gamma[1])) ** 2
        mean = float(x @ self.m)
        var = 1.0 / beta + float(x @ np.linalg.solve(self.A, x))
        return mean, var

    def zscore(self, value: float, age: float, sex: int) -> float:
        mean, var = self._gaussian(age, sex)
        return float((warp(value, self.gamma) - mean) / np.sqrt(var))

    def percentile(self, value: float, age: float, sex: int) -> float:
        return float(100.0 * norm.cdf(self.zscore(value, age, sex)))

    def expected(self, age: float, sex: int, centile: float = 50.0) -> float:
        """Volume attendu au centile demandé pour cet âge et ce sexe."""
        mean, var = self._gaussian(age, sex)
        return float(unwarp(mean + norm.ppf(centile / 100.0) * np.sqrt(var), self.gamma))


class NormativeLibrary:
    """Accès aux modèles d'un dossier ``lifespan_57K_82sites/<structure>/Models/``."""

    def __init__(self, root: str | Path) -> None:
        root = Path(root)
        self.root = root / MODEL_NAME if (root / MODEL_NAME).is_dir() else root
        if not self.root.is_dir():
            raise FileNotFoundError(f"Modèles normatifs introuvables : {self.root}")
        self._cache: dict[str, NormativeModel] = {}

    def has(self, structure: str) -> bool:
        return (self.root / structure / "Models" / "NM_0_0_estimate.pkl").is_file()

    def get(self, structure: str) -> NormativeModel:
        if structure not in self._cache:
            path = self.root / structure / "Models" / "NM_0_0_estimate.pkl"
            self._cache[structure] = NormativeModel.load(path, structure)
        return self._cache[structure]


#: Modèles BrainCharts figés sur un commit précis, vérifiés par SHA-256
NORMS_URL = (
    "https://raw.githubusercontent.com/predictive-clinical-neuroscience/braincharts/"
    "5a1a7822e6f54405b0a826cf137d584240af307c/models/lifespan_57K_82sites.zip"
)
NORMS_SHA256 = "60457279debb787ea2ce4175191bd2ac16214e84bd260c7dab3d8bf2f1095cc4"


def download_norms(dest: str | Path) -> Path:
    """Télécharge et décompresse les modèles normatifs dans ``dest`` (idempotent)."""
    import hashlib
    import logging
    import urllib.request

    from segbrainmri.models.flames import _safe_extract

    dest = Path(dest)
    target = dest / MODEL_NAME
    if (target / "Left-Hippocampus" / "Models" / "NM_0_0_estimate.pkl").is_file():
        return target
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{MODEL_NAME}.zip"
    logging.getLogger(__name__).info("Téléchargement des normes BrainCharts")
    request = urllib.request.Request(NORMS_URL, headers={"User-Agent": "segbrainmri"})
    with urllib.request.urlopen(request, timeout=120) as resp:
        archive.write_bytes(resp.read())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != NORMS_SHA256:
        archive.unlink()
        raise RuntimeError(f"Somme SHA-256 inattendue pour les normes : {digest}")
    _safe_extract(archive, dest)
    archive.unlink()
    return target
