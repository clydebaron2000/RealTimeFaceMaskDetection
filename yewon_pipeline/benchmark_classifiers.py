import os
import time
import torch

from eval_detector_classifier import load_model_from_checkpoint

# Checkpoints to benchmark
CHECKPOINTS = [
    # (short_name, checkpoint_path)
    ("resnet18_12k_best", "/local-ssd/yl3427/test/runs_12k/resnet18_pt_finetune_lr1e-3_ep10/best.pth"),
    ("resnet50_12k_best", "/local-ssd/yl3427/test/runs_12k/resnet50_pt_finetune_lr1e-3_ep10/best.pth"),
    ("effb0_12k_best",    "/local-ssd/yl3427/test/runs_12k/efficientnet_b0_pt_finetune_lr1e-3_ep10/best.pth"),
    ("mobilenetv3_12k_best", "/local-ssd/yl3427/test/runs_12k/mobilenet_v3_small_pt_finetune_lr1e-3_ep10/best.pth"),
    ("vitb16_12k_best",   "/local-ssd/yl3427/test/runs_12k/vit_b_16_pt_finetune_lr1e-3_ep10/best.pth"),
    ("small_cnn_12k",     "/local-ssd/yl3427/test/runs_12k/small_cnn_scratch_lr0.001_ep20/best.pth"),
    ("custom_cnn_12k",    "/local-ssd/yl3427/test/runs_12k/custom_cnn_scratch_lr0.001_ep20/best.pth")
]

DEVICE_STR = "cuda"  # or "cuda:0"
WARMUP_ITERS = 20
BENCH_ITERS = 200


def human_mb(path):
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 ** 2)


def main():
    device = torch.device(DEVICE_STR if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows = []
    for short_name, ckpt_path in CHECKPOINTS:
        print("\n================================")
        print(f"Benchmarking: {short_name}")
        print(f"Checkpoint : {ckpt_path}")

        # 1) Load model
        model, num_classes, img_size, _ = load_model_from_checkpoint(ckpt_path, device)
        model.eval()

        # Number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        size_mb = human_mb(ckpt_path)

        # 2) Dummy input (batch_size=1)
        dummy = torch.randn(1, 3, img_size, img_size, device=device)

        # 3) Warmup (CUDA kernel initialization + cache warmup)
        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # 4) Actual measurement
        start = time.time()
        with torch.no_grad():
            for _ in range(BENCH_ITERS):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
        end = time.time()
        elapsed = end - start
        fps = BENCH_ITERS / elapsed
        ms_per_image = (elapsed / BENCH_ITERS) * 1000.0

        print(f"img_size       : {img_size}")
        print(f"num_params     : {num_params:,}")
        print(f"ckpt size (MB) : {size_mb:.2f}")
        print(f"latency (ms/img): {ms_per_image:.3f}")
        print(f"throughput (FPS): {fps:.2f}")

        rows.append(
            (short_name, img_size, num_params, size_mb, ms_per_image, fps)
        )

    # Print clean one-line summary
    print("\n=========== SUMMARY ===========")
    header = ["name", "img_size", "num_params", "ckpt_MB", "ms_per_img", "fps"]
    print("\t".join(header))
    for r in rows:
        print("\t".join(str(x) for x in r))


if __name__ == "__main__":
    main()
