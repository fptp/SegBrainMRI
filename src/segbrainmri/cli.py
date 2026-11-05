"""Interface en ligne de commande ``segbrainmri``.

Sous-commandes :
    info        affiche l'appareil détecté (CPU/GPU) et les versions ;
    preprocess  prétraite un scan et écrit les volumes normalisés ;
    segment     segmente les lésions de SEP et écrit le masque NIfTI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from segbrainmri import __version__
from segbrainmri.core.logging import setup_logging

logger = logging.getLogger("segbrainmri.cli")


def _triple(text: str) -> tuple:
    values = [v for v in text.replace("x", ",").split(",") if v]
    if len(values) == 1:
        values *= 3
    if len(values) != 3:
        raise argparse.ArgumentTypeError("3 valeurs attendues, ex. 1,1,1 ou 128x128x128")
    return tuple(values)


def _float3(text: str) -> tuple[float, float, float]:
    return tuple(float(v) for v in _triple(text))


def _int3(text: str) -> tuple[int, int, int]:
    return tuple(int(v) for v in _triple(text))


def _inputs(args: argparse.Namespace) -> dict[str, Path]:
    inputs = {"flair": args.flair}
    if args.t1 is not None:
        inputs["t1"] = args.t1
    return inputs


def _preprocessing_config(args: argparse.Namespace):
    from segbrainmri.preprocessing import PreprocessingConfig

    return PreprocessingConfig(
        target_spacing=None if args.native_spacing else args.spacing,
        orientation=None if args.orientation.lower() == "none" else args.orientation,
        n4_bias_correction=args.n4,
    )


# ---------------------------------------------------------------------------- commandes
def cmd_info(args: argparse.Namespace) -> int:
    import SimpleITK as sitk
    import torch

    from segbrainmri.core.device import describe_device, resolve_device
    from segbrainmri.models import available_models

    device = resolve_device(args.device)
    print(f"segbrainmri  : {__version__}")
    print(f"PyTorch      : {torch.__version__} (CUDA compilé : {torch.version.cuda or 'non'})")
    print(f"SimpleITK    : {sitk.Version_VersionString()}")
    print(f"CUDA dispo.  : {torch.cuda.is_available()} ({torch.cuda.device_count()} GPU)")
    print(f"Appareil     : {describe_device(device)}")
    print(f"Modèles      : {', '.join(available_models())}")
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    from segbrainmri.preprocessing import PreprocessingPipeline

    pipeline = PreprocessingPipeline(_preprocessing_config(args))
    case = pipeline.run(_inputs(args))
    for path in pipeline.save(case, args.output):
        print(path)
    return 0


def cmd_segment(args: argparse.Namespace) -> int:
    from segbrainmri.core.device import get_device
    from segbrainmri.inference import LesionSegmenter
    from segbrainmri.models import build_model
    from segbrainmri.preprocessing import PreprocessingPipeline

    inputs = _inputs(args)
    device = get_device(args.device, num_threads=args.threads)
    model = build_model(args.model, in_channels=len(inputs))
    if args.weights is not None:
        model.load_weights(args.weights, device)
    elif args.untrained:
        logger.warning("--untrained : poids ALÉATOIRES, résultat sans valeur clinique (test).")
    else:
        logger.error("Aucun poids fourni : utilisez --weights CHEMIN (ou --untrained pour tester).")
        return 2

    segmenter = LesionSegmenter(
        model,
        device=device,
        pipeline=PreprocessingPipeline(_preprocessing_config(args)),
        patch_size=args.patch_size,
        overlap=args.overlap,
        threshold=args.threshold,
    )
    result = segmenter.segment(inputs)
    paths = result.save(args.output, prefix=args.prefix)
    print(f"Masque       : {paths['mask']}")
    print(f"Probabilités : {paths['probabilities']}")
    print(f"Lésions      : {result.lesion_count} ({result.lesion_volume_ml:.2f} mL)")
    return 0


# ---------------------------------------------------------------------------- parseur
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segbrainmri",
        description="Segmentation des lésions de SEP sur IRM cérébrale (NIfTI).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="plus de logs (-vv)")
    parser.add_argument("-q", "--quiet", action="store_true", help="avertissements seulement")
    parser.add_argument(
        "--device",
        default=None,
        help="auto (défaut), cpu, cuda, cuda:N. Variable d'env. : SEGBRAINMRI_DEVICE",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="matériel détecté et versions").set_defaults(func=cmd_info)

    def add_io(p: argparse.ArgumentParser) -> None:
        p.add_argument("--flair", type=Path, required=True, help="T2-FLAIR (.nii/.nii.gz)")
        p.add_argument("--t1", type=Path, default=None, help="T1 co-registrée (optionnel)")
        p.add_argument("-o", "--output", type=Path, required=True, help="dossier de sortie")
        p.add_argument(
            "--spacing",
            type=_float3,
            default=(1.0, 1.0, 1.0),
            help="espacement cible en mm (défaut 1,1,1)",
        )
        p.add_argument("--native-spacing", action="store_true", help="pas de rééchantillonnage")
        p.add_argument("--orientation", default="RAS", help="RAS (défaut), LPS... ou none")
        p.add_argument("--n4", action="store_true", help="correction de biais N4 (lent)")

    p_pre = sub.add_parser("preprocess", help="prétraiter un scan")
    add_io(p_pre)
    p_pre.set_defaults(func=cmd_preprocess)

    p_seg = sub.add_parser("segment", help="segmenter les lésions de SEP")
    add_io(p_seg)
    p_seg.add_argument("--model", default="unet3d", help="architecture (défaut unet3d)")
    p_seg.add_argument("--weights", type=Path, default=None, help="fichier de poids .pt/.pth")
    p_seg.add_argument(
        "--untrained",
        action="store_true",
        help="poids aléatoires, pour tester le pipeline sans modèle entraîné",
    )
    p_seg.add_argument(
        "--patch-size",
        type=_int3,
        default=(128, 128, 128),
        help="taille des patchs (défaut 128; réduire à 96/64 si RAM limitée)",
    )
    p_seg.add_argument("--overlap", type=float, default=0.5, help="recouvrement (défaut 0.5)")
    p_seg.add_argument("--threshold", type=float, default=0.5, help="seuil de probabilité")
    p_seg.add_argument("--threads", type=int, default=None, help="threads CPU PyTorch")
    p_seg.add_argument("--prefix", default="lesions", help="préfixe des fichiers de sortie")
    p_seg.set_defaults(func=cmd_segment)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(-1 if args.quiet else args.verbose)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
