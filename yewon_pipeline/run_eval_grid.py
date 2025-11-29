import os
import argparse
import subprocess
from datetime import datetime


DETECTORS = ["yunet", "haar", "mtcnn", "retinaface"]


def find_experiments(output_root: str):
    exps = []
    if not os.path.isdir(output_root):
        return exps

    for name in sorted(os.listdir(output_root)):
        exp_dir = os.path.join(output_root, name)
        if not os.path.isdir(exp_dir):
            continue
        ckpt_path = os.path.join(exp_dir, "best.pth")
        if os.path.exists(ckpt_path):
            exps.append((name, ckpt_path))
    return exps


def main():
    parser = argparse.ArgumentParser(
        description="Run eval_detector_classifier.py over many (detector, checkpoint) combos."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root (contains images/ and annotations/).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root dir that contains classifier experiment subdirs (with best.pth).",
    )
    parser.add_argument(
        "--python_exe",
        type=str,
        default="python",
        help="Python executable.",
    )
    parser.add_argument(
        "--eval_script",
        type=str,
        default="eval_detector_classifier.py",
        help="Evaluation script filename.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for classifier model.",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        help="IoU threshold for matching GT/pred boxes.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit number of images for quick test (None = all).",
    )
    parser.add_argument(
        "--yunet_onnx",
        type=str,
        default="face_detection_yunet_2023mar.onnx",
        help="Path to YuNet ONNX.",
    )
    parser.add_argument(
        "--haar_xml",
        type=str,
        default="haarcascade_frontalface_default.xml",
        help="Path to Haar cascade XML.",
    )
    parser.add_argument(
        "--retina_thresh",
        type=float,
        default=0.8,
        help="Score threshold for RetinaFace.",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default=None,
        help="CSV path to accumulate all eval results. "
             "If None, a timestamped file in output_root will be used.",
    )
    args = parser.parse_args()

    exps = find_experiments(args.output_root)
    if not exps:
        print(f"[run_eval_grid] No experiments (best.pth) found under {args.output_root}")
        return

    if args.results_csv is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_csv = os.path.join(args.output_root, f"det_cls_grid_{ts}.csv")

    print("========================================")
    print(f"Found {len(exps)} experiments under: {args.output_root}")
    print("Detectors:", DETECTORS)
    print(f"Results CSV: {args.results_csv}")
    print("========================================")

    for exp_name, ckpt in exps:
        for det in DETECTORS:
            print("\n----------------------------------------")
            print(f"Eval: exp={exp_name}, detector={det}")
            print("----------------------------------------")

            cmd = [
                args.python_exe,
                args.eval_script,
                "--data_root", args.data_root,
                "--checkpoint", ckpt,
                "--detector", det,
                "--device", args.device,
                "--iou_thresh", str(args.iou_thresh),
                "--yunet_onnx", args.yunet_onnx,
                "--haar_xml", args.haar_xml,
                "--retina_thresh", str(args.retina_thresh),
                "--results_csv", args.results_csv,
            ]
            if args.max_images is not None:
                cmd += ["--max_images", str(args.max_images)]

            print("Command:")
            print("  " + " ".join(str(c) for c in cmd))

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[WARN] eval script failed with return code {result.returncode}")

    print("\n========================================")
    print("All detector x classifier evaluations finished.")
    print(f"Summary CSV: {args.results_csv}")
    print("========================================")


if __name__ == "__main__":
    main()
