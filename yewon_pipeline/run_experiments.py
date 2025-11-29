import os
import argparse
import subprocess
import csv
from datetime import datetime

import torch


BACKBONES = [
    "resnet18",
    "resnet50",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "vit_b_16",
]

EXPERIMENTS = []

# Common settings (cropped per-face, 2-class, merge incorrect with no-mask)
BASE_CFG = {
    "dataset_mode": "cropped",
    "num_classes": 2,
    "merge_incorrect_with_nomask": True,
    "pretrained": True,
    "freeze_backbone": False,  # finetune
    "batch_size": 256,
    "weight_decay": 1e-4,
    "img_size": 224,
    "amp": True,
    "seed": 42,
}

# (1) 5 backbones × 2 (lr, epochs) settings
for model_name in BACKBONES:
    # Setting A: lr=1e-3, epochs=10
    EXPERIMENTS.append(
        {
            "name": f"cropped_{model_name}_pt_finetune_lr1e-3_ep10",
            "model_name": model_name,
            "lr": 1e-3,
            "epochs": 10,
            **BASE_CFG,
        }
    )
    # Setting B: lr=5e-4, epochs=20
    EXPERIMENTS.append(
        {
            "name": f"cropped_{model_name}_pt_finetune_lr5e-4_ep20",
            "model_name": model_name,
            "lr": 5e-4,
            "epochs": 20,
            **BASE_CFG,
        }
    )

# (2) small_cnn scratch model (train longer)
for lr, epochs in [(1e-3, 20), (5e-4, 30)]:
    EXPERIMENTS.append(
        {
            "name": f"cropped_small_cnn_scratch_lr{lr}_ep{epochs}",
            "model_name": "small_cnn",
            "lr": lr,
            "epochs": epochs,
            **BASE_CFG,
        }
    )




def build_command(args, exp, output_dir):
    if args.launcher == "single":
        base_cmd = [args.python_exe, args.pipeline_script]
    elif args.launcher == "torchrun":
        base_cmd = [
            "torchrun",
            f"--nproc_per_node={args.nproc_per_node}",
            args.pipeline_script,
        ]
    else:
        raise ValueError(f"Unknown launcher: {args.launcher}")

    cmd = base_cmd + [
        "--data_root", args.data_root,
        "--output_dir", output_dir,
        "--dataset_mode", exp["dataset_mode"],
        "--num_classes", str(exp["num_classes"]),
        "--model_name", exp["model_name"],
        "--epochs", str(exp["epochs"]),
        "--batch_size", str(exp["batch_size"]),
        "--lr", str(exp["lr"]),
        "--weight_decay", str(exp.get("weight_decay", 1e-4)),
        "--img_size", str(exp.get("img_size", 224)),
        "--num_workers", str(args.num_workers),
        "--seed", str(exp.get("seed", 42)),
    ]

    # incorrect merge setting
    if not exp.get("merge_incorrect_with_nomask", True):
        cmd.append("--no_merge_incorrect")

    # pretrained / no_pretrained
    if exp.get("pretrained", True):
        cmd.append("--pretrained")
    else:
        cmd.append("--no_pretrained")

    # freeze_backbone / no_freeze_backbone
    if exp.get("freeze_backbone", True):
        cmd.append("--freeze_backbone")
    else:
        cmd.append("--no_freeze_backbone")

    # automatic mixed precision
    if exp.get("amp", True):
        cmd.append("--amp")
    else:
        cmd.append("--no_amp")

    return cmd


def run_single_experiment(args, exp):
    exp_name = exp["name"]
    output_dir = os.path.join(args.output_root, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    cmd = build_command(args, exp, output_dir)

    print("\n========================================")
    print(f"Running experiment: {exp_name}")
    print("Command:")
    print("  " + " ".join(str(c) for c in cmd))
    print("========================================\n")

    # Run pipeline_1.py
    result = subprocess.run(cmd)
    status = "ok" if result.returncode == 0 else f"error_{result.returncode}"

    best_acc = None
    best_epoch = None

    if status == "ok":
        ckpt_path = os.path.join(output_dir, "best.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            best_acc = ckpt.get("best_val_acc", None)
            best_epoch = ckpt.get("epoch", None)
        else:
            status = "no_checkpoint"

    return {
        "name": exp_name,
        "output_dir": output_dir,
        "status": status,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        # Also record config
        "dataset_mode": exp["dataset_mode"],
        "num_classes": exp["num_classes"],
        "merge_incorrect_with_nomask": exp.get("merge_incorrect_with_nomask", True),
        "model_name": exp["model_name"],
        "pretrained": exp.get("pretrained", True),
        "freeze_backbone": exp.get("freeze_backbone", True),
        "epochs": exp["epochs"],
        "batch_size": exp["batch_size"],
        "lr": exp["lr"],
        "weight_decay": exp.get("weight_decay", 1e-4),
        "img_size": exp.get("img_size", 224),
        "amp": exp.get("amp", True),
        "seed": exp.get("seed", 42),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple pipeline_1.py experiments and log results to CSV."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root of dataset (contains 'images/' and 'annotations/').",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="runs/auto",
        help="Root directory for all experiment outputs.",
    )
    parser.add_argument(
        "--pipeline_script",
        type=str,
        default="pipeline_1.py",
        help="Training script file name.",
    )
    parser.add_argument(
        "--python_exe",
        type=str,
        default="python",
        help="Python executable to use when launcher=single.",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        default="torchrun",
        choices=["single", "torchrun"],
        help="How to launch training. 'torchrun' for multi-GPU.",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="Number of processes per node when using torchrun.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader num_workers passed to pipeline_1.py.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    results = []
    for exp in EXPERIMENTS:
        res = run_single_experiment(args, exp)
        results.append(res)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_root, f"summary_{timestamp}.csv")

    fieldnames = sorted({k for r in results for k in r.keys()})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("\n========================================")
    print("All experiments finished.")
    print(f"Summary CSV saved to: {csv_path}")
    print("========================================\n")


if __name__ == "__main__":
    main()
