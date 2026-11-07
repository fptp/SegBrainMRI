"""Interface en ligne de commande ``segbrainmri``.

Sous-commandes :
    info             appareil détecté (CPU/GPU) et versions ;
    download-models  télécharge FLAMeS et SynthStrip ;
    serve            service DICOM : reçoit les FLAIR, segmente, renvoie à Orthanc ;
    process-dicom    traite une fois un dossier DICOM (et envoie à Orthanc si --send) ;
    echo             teste la connexion DICOM avec Orthanc (C-ECHO) ;
    preprocess       prétraite un scan NIfTI ;
    segment          segmente un scan NIfTI et écrit le masque.
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

    inputs = _inputs(args)
    device = get_device(args.device, num_threads=args.threads)
    if args.model == "flames":
        return _segment_flames(args, device)

    from segbrainmri.inference import LesionSegmenter
    from segbrainmri.models import build_model
    from segbrainmri.preprocessing import PreprocessingPipeline

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


def _segment_flames(args: argparse.Namespace, device) -> int:
    from segbrainmri.inference.stats import lesion_statistics
    from segbrainmri.models.flames import FlamesBackend
    from segbrainmri.preprocessing import load_nifti, save_nifti

    if args.t1 is not None:
        logger.warning("FLAMeS n'utilise que la FLAIR : --t1 est ignorée.")
    backend = FlamesBackend(
        args.models_dir, device=device, threads=args.threads, skull_strip=not args.no_skull_strip
    )
    mask = backend.segment(load_nifti(args.flair))
    path = save_nifti(mask, args.output / f"{args.prefix}_mask.nii.gz")
    count, volume = lesion_statistics(mask)
    print(f"Masque       : {path}")
    print(f"Lésions      : {count} ({volume:.2f} mL)")
    return 0


def _service_config(args: argparse.Namespace):
    from segbrainmri.service.config import ServiceConfig

    config = ServiceConfig.load(args.config)
    if args.device:
        config.processing.device = args.device
    return config


def cmd_serve(args: argparse.Namespace) -> int:
    from segbrainmri.service.app import serve

    serve(_service_config(args))
    return 0


def cmd_process_dicom(args: argparse.Namespace) -> int:
    from segbrainmri.service.worker import build_backend, process_series

    config = _service_config(args)
    results = process_series(args.input, build_backend(config), config, send=args.send)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for ds in results:
            ds.save_as(args.output / f"{ds.SOPInstanceUID}.dcm", enforce_file_format=True)
        print(f"{len(results)} fichier(s) DICOM écrit(s) dans {args.output}")
    return 0


def cmd_echo(args: argparse.Namespace) -> int:
    from segbrainmri.dicom.network import echo

    config = _service_config(args)
    o = config.orthanc
    ok = echo(o.host, o.port, o.ae_title, config.listener.ae_title)
    print(f"C-ECHO {o.ae_title}@{o.host}:{o.port} : {'OK' if ok else 'ÉCHEC'}")
    return 0 if ok else 1


def cmd_download_models(args: argparse.Namespace) -> int:
    from segbrainmri.models.flames import download_models

    print(download_models(args.dest))
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
    p_seg.add_argument(
        "--model",
        default="flames",
        help="flames (défaut, pré-entraîné) ou unet3d (nécessite --weights)",
    )
    p_seg.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/opt/segbrainmri/models"),
        help="dossier des modèles téléchargés (flames)",
    )
    p_seg.add_argument(
        "--no-skull-strip", action="store_true", help="FLAIR déjà sans crâne (flames)"
    )
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

    def add_config(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "-c",
            "--config",
            type=Path,
            default=None,
            help="fichier TOML (défaut /etc/segbrainmri/config.toml)",
        )

    p_serve = sub.add_parser("serve", help="service DICOM (systemd)")
    add_config(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    p_dcm = sub.add_parser("process-dicom", help="traiter un dossier DICOM (une série)")
    add_config(p_dcm)
    p_dcm.add_argument("-i", "--input", type=Path, required=True, help="dossier de la série")
    p_dcm.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="écrire aussi les DICOM résultats dans ce dossier",
    )
    p_dcm.add_argument("--send", action="store_true", help="envoyer les résultats à Orthanc")
    p_dcm.set_defaults(func=cmd_process_dicom)

    p_echo = sub.add_parser("echo", help="tester la connexion DICOM avec Orthanc")
    add_config(p_echo)
    p_echo.set_defaults(func=cmd_echo)

    p_dl = sub.add_parser("download-models", help="télécharger FLAMeS et SynthStrip")
    p_dl.add_argument("--dest", type=Path, default=Path("/opt/segbrainmri/models"))
    p_dl.set_defaults(func=cmd_download_models)
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
