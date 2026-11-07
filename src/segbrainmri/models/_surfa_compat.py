"""Correctifs de compatibilité de ``surfa`` (<= 0.6.3) avec numpy 2 / scipy récents.

``mri_synthstrip`` (FreeSurfer) dépend de ``surfa``, dont la dernière version publiée
n'est pas compatible avec numpy 2, lui-même imposé par Python 3.13 (Debian 13).
Plutôt que de modifier le script officiel, on corrige ``surfa`` en mémoire puis on
exécute le script tel quel :

    python -m segbrainmri.models._surfa_compat /chemin/mri_synthstrip -i ... -o ...

Correctifs :
* ``FramedImage.bbox`` : ``scipy.ndimage.find_objects`` refuse un tableau booléen
  ("'numpy.bool' object cannot be interpreted as an integer").

(Le bug de ``reorient`` avec numpy 2 est évité en amont : l'image est fournie déjà
orientée en LIA, voir :meth:`segbrainmri.models.flames.SynthStrip.strip`.)
"""

from __future__ import annotations

import runpy
import sys

import numpy as np


def _bbox(self, margin=None):
    from surfa.core.array import check_array

    mask = self.max(frames=True).data > 0
    if not np.any(mask):
        return tuple(slice(0, s) for s in mask.shape)
    from scipy.ndimage import find_objects

    cropping = find_objects(mask.astype(np.uint8))[0]  # correctif : entier, pas booléen
    if margin is not None:
        margin = np.repeat(margin, self.basedim) if np.isscalar(margin) else np.asarray(margin)
        check_array(margin, ndim=1, shape=self.basedim, name="bbox margin")
        if not np.issubdtype(margin.dtype, np.integer):
            raise ValueError("only integers can be used for valid bbox margins")
        start = [max(0, c.start - margin[i]) for i, c in enumerate(cropping)]
        stop = [min(self.baseshape[i], c.stop + margin[i]) for i, c in enumerate(cropping)]
        step = [c.step for c in cropping]
        cropping = tuple(slice(*s) for s in zip(start, stop, step, strict=True))
    return cropping


def apply() -> None:
    """Applique les correctifs (idempotent)."""
    from surfa.image.framed import FramedImage

    FramedImage.bbox = _bbox


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit("usage : python -m segbrainmri.models._surfa_compat SCRIPT [ARGS...]")
    apply()
    script = argv[0]
    sys.argv = [script, *argv[1:]]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
