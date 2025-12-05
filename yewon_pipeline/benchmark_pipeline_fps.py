import os
import time
import argparse

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from eval_detector_classifier import load_model_from_checkpoint
from detectors_2 import build_detector, BaseFaceDetector


def make_infer_transform(img_size: int):

    if img_size <= 64:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


def list_image_paths(images_dir: str, max_frames: int | None = None):
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG")
    paths = [
        os.path.join(images_dir, f)
        for f in sorted(os.listdir(images_dir))
        if f.endswith(exts)
    ]
    if max_frames is not None:
        paths = paths[:max_frames]
    return paths


def run_pipeline_on_frame(
    image_bgr: np.ndarray,
    detector: BaseFaceDetector,
    model: torch.nn.Module,
    infer_transform,
    device: torch.device,
):

    # 1) detection
    dets = detector.detect(image_bgr)  # list of (x, y, w, h, score)

    if len(dets) == 0:
        return 0  # faces processed

    # 2) crop + transform
    crops = []
    h, w = image_bgr.shape[:2]
    for (x, y, w_box, h_box, score) in dets:
        x1 = max(int(x), 0)
        y1 = max(int(y), 0)
        x2 = min(int(x + w_box), w)
        y2 = min(int(y + h_box), h)
        if x2 <= x1 or y2 <= y1:
            continue
        crop_bgr = image_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        crops.append(infer_transform(crop_pil))

    if not crops:
        return 0

    batch = torch.stack(crops, dim=0).to(device)

    # 3) classifier forward
    with torch.no_grad():
        _ = model(batch)

    return batch.size(0)  # number of faces processed


def measure_fps_for_pipeline(
    data_root: str,
    detector_name: str,
    checkpoint: str,
    device_str: str = "cuda",
    max_frames: int | None = 200,
    warmup_frames: int = 20,
    yunet_onnx: str = "face_detection_yunet_2023mar.onnx",
    haar_xml: str = "haarcascade_frontalface_default.xml",
    retina_thresh: float = 0.8,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # 1) load classifier
    model, num_classes, img_size, _ = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    infer_transform = make_infer_transform(img_size)

    # 2) build detector
    detector = build_detector(
        detector_name,
        device=device_str,
        yunet_onnx=yunet_onnx,
        haar_xml=haar_xml,
        retina_thresh=retina_thresh,
    )

    images_dir = os.path.join(data_root, "images")
    img_paths = list_image_paths(images_dir, max_frames=max_frames)
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in {images_dir}")

    # ---- warmup ----
    with torch.no_grad():
        for p in img_paths[: min(warmup_frames, len(img_paths))]:
            img = cv2.imread(p)
            if img is None:
                continue
            _ = run_pipeline_on_frame(img, detector, model, infer_transform, device)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # ---- timing ----
    total_frames = 0
    total_faces = 0
    start = time.time()
    with torch.no_grad():
        for p in img_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            faces = run_pipeline_on_frame(img, detector, model, infer_transform, device)
            total_frames += 1
            total_faces += faces

        if device.type == "cuda":
            torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    fps = total_frames / elapsed
    faces_per_sec = total_faces / elapsed if elapsed > 0 else 0.0
    avg_faces_per_frame = total_faces / max(total_frames, 1)

    return {
        "frames": total_frames,
        "faces": total_faces,
        "elapsed_sec": elapsed,
        "fps": fps,
        "faces_per_sec": faces_per_sec,
        "avg_faces_per_frame": avg_faces_per_frame,
    }


def build_argparser():
    p = argparse.ArgumentParser(
        description="Benchmark end-to-end FPS for detector+classifier pipelines."
    )
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Detection dataset root (contains 'images/' and 'annotations/').",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string, e.g. 'cuda', 'cuda:0', 'cpu'.",
    )
    p.add_argument(
        "--max_frames",
        type=int,
        default=200,
        help="Max number of frames (images) to use for timing.",
    )
    p.add_argument(
        "--warmup_frames",
        type=int,
        default=20,
        help="Number of warmup frames before timing.",
    )
    p.add_argument(
        "--yunet_onnx",
        type=str,
        default="face_detection_yunet_2023mar.onnx",
    )
    p.add_argument(
        "--haar_xml",
        type=str,
        default="haarcascade_frontalface_default.xml",
    )
    p.add_argument(
        "--retina_thresh",
        type=float,
        default=0.8,
    )
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    # ===== 1) Classifier checkpoints =====
    CHECKPOINTS = [
        # (short_name, checkpoint_path)
        ("resnet18_12k_best",    "/local-ssd/yl3427/test/runs_12k/resnet18_pt_finetune_lr1e-3_ep10/best.pth"),
        ("resnet50_12k_best",    "/local-ssd/yl3427/test/runs_12k/resnet50_pt_finetune_lr1e-3_ep10/best.pth"),
        ("effb0_12k_best",       "/local-ssd/yl3427/test/runs_12k/efficientnet_b0_pt_finetune_lr1e-3_ep10/best.pth"),
        ("mobilenetv3_12k_best", "/local-ssd/yl3427/test/runs_12k/mobilenet_v3_small_pt_finetune_lr1e-3_ep10/best.pth"),
        ("vitb16_12k_best",      "/local-ssd/yl3427/test/runs_12k/vit_b_16_pt_finetune_lr1e-3_ep10/best.pth"),
        ("small_cnn_12k",        "/local-ssd/yl3427/test/runs_12k/small_cnn_scratch_lr0.001_ep20/best.pth"),
        ("custom_cnn_12k",       "/local-ssd/yl3427/test/runs_12k/custom_cnn_scratch_lr0.001_ep20/best.pth"),
    ]

    # ===== 2) Detectors =====
    DETECTORS = ["yunet", "haar", "mtcnn", "retinaface"]

    # ===== 3) Generate PIPELINES for all combinations =====
    # (name, detector_name, checkpoint_path)
    PIPELINES = []
    for det in DETECTORS:
        for short_name, ckpt_path in CHECKPOINTS:
            pipe_name = f"{det}_{short_name}"
            PIPELINES.append((pipe_name, det, ckpt_path))

    print(f"Using frames from: {os.path.join(args.data_root, 'images')}")
    print(f"Max frames: {args.max_frames}, warmup: {args.warmup_frames}")
    print(f"Total pipelines to benchmark: {len(PIPELINES)}")

    rows = []
    for name, det_name, ckpt in PIPELINES:
        print("\n======================================")
        print(f"Benchmarking pipeline: {name}")
        print(f"  detector   : {det_name}")
        print(f"  checkpoint : {ckpt}")
        stats = measure_fps_for_pipeline(
            data_root=args.data_root,
            detector_name=det_name,
            checkpoint=ckpt,
            device_str=args.device,
            max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
            yunet_onnx=args.yunet_onnx,
            haar_xml=args.haar_xml,
            retina_thresh=args.retina_thresh,
        )
        print(f"Frames processed      : {stats['frames']}")
        print(f"Total faces processed : {stats['faces']}")
        print(f"Elapsed sec           : {stats['elapsed_sec']:.3f}")
        print(f"FPS (frames / sec)    : {stats['fps']:.2f}")
        print(f"Faces per second      : {stats['faces_per_sec']:.2f}")
        print(f"Avg faces per frame   : {stats['avg_faces_per_frame']:.2f}")

        rows.append(
            (
                name,
                det_name,
                ckpt,
                stats["fps"],
                stats["faces_per_sec"],
                stats["avg_faces_per_frame"],
            )
        )

    # Print summary
    print("\n=========== SUMMARY (end-to-end FPS) ===========")
    print("name\t detector\t fps_frames\t fps_faces\t faces_per_frame")
    for (name, det_name, ckpt, fps, fps_faces, fpf) in rows:
        print(f"{name}\t{det_name}\t{fps:.2f}\t{fps_faces:.2f}\t{fpf:.2f}")



if __name__ == "__main__":
    main()
