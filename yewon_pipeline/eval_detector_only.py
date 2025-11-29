import os
import argparse
from typing import List, Tuple, Optional

import cv2
import numpy as np

from pipeline_1 import parse_kaggle_annotation
from detectors_2 import build_detector, BaseFaceDetector


def find_image_for_xml(images_dir: str, xml_name: str) -> Optional[str]:
    stem = os.path.splitext(xml_name)[0]
    exts = [".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG"]
    for ext in exts:
        path = os.path.join(images_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None

def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (N, 4), b: (M, 4) in [x1, y1, x2, y2]
    return: (N, M) IoU matrix
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])  # (N,M,2)
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])  # (N,M,2)
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]  # (N,M)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])  # (N,)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # (M,)

    union = area_a[:, None] + area_b[None, :] - inter
    union = np.clip(union, a_min=1e-6, a_max=None)

    iou = inter / union
    return iou.astype(np.float32)


def greedy_match_iou(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    iou_thresh: float = 0.5,
) -> List[Tuple[int, int, float]]:

    matches: List[Tuple[int, int, float]] = []

    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return matches

    ious = box_iou_xyxy(gt_boxes, pred_boxes)  # (G, P)
    gt_used = set()
    pred_used = set()

    G, P = ious.shape
    triplets = []
    for i in range(G):
        for j in range(P):
            triplets.append((i, j, float(ious[i, j])))
    triplets.sort(key=lambda x: x[2], reverse=True)

    for gt_idx, pred_idx, iou in triplets:
        if iou < iou_thresh:
            break
        if gt_idx in gt_used or pred_idx in pred_used:
            continue
        gt_used.add(gt_idx)
        pred_used.add(pred_idx)
        matches.append((gt_idx, pred_idx, iou))

    return matches


# -------------------------
# Main evaluation
# -------------------------

def evaluate_detector_only(
    data_root: str,
    detector_name: str,
    device_str: str = "cuda",
    iou_thresh: float = 0.5,
    max_images: Optional[int] = None,
    yunet_onnx: str = "face_detection_yunet_2023mar.onnx",
    haar_xml: str = "haarcascade_frontalface_default.xml",
    retina_thresh: float = 0.8,
    results_csv: Optional[str] = None,
):

    print(f"[Eval-DetOnly] Using detector: {detector_name}")

    detector: BaseFaceDetector = build_detector(
        detector_name,
        device=device_str,
        yunet_onnx=yunet_onnx,
        haar_xml=haar_xml,
        retina_thresh=retina_thresh,
    )

    images_dir = os.path.join(data_root, "images")
    annotations_dir = os.path.join(data_root, "annotations")

    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]
    xml_files.sort()

    total_images = 0
    total_gt_faces = 0
    total_pred_faces = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    iou_sum = 0.0  # Sum of IoUs for TPs

    for idx, xml_name in enumerate(xml_files):
        if max_images is not None and idx >= max_images:
            break

        xml_path = os.path.join(annotations_dir, xml_name)
        img_path = find_image_for_xml(images_dir, xml_name)
        if img_path is None:
            continue

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        # GT boxes (ignore mask types, treat as "face" only)
        boxes, _ = parse_kaggle_annotation(xml_path)

        if len(boxes) == 0:
            # Skip images without faces (no detection task)
            continue

        gt_boxes_xyxy = np.array(boxes, dtype=np.float32)

        total_images += 1
        total_gt_faces += len(gt_boxes_xyxy)

        # Run detector
        dets = detector.detect(image_bgr)  # [(x, y, w, h, score), ...]
        if len(dets) == 0:
            # All GT boxes become FN
            total_fn += len(gt_boxes_xyxy)
            continue

        pred_boxes_xyxy = []
        for (x, y, w_box, h_box, score) in dets:
            x1 = x
            y1 = y
            x2 = x + w_box
            y2 = y + h_box
            pred_boxes_xyxy.append([x1, y1, x2, y2])
        pred_boxes_xyxy = np.array(pred_boxes_xyxy, dtype=np.float32)

        total_pred_faces += len(pred_boxes_xyxy)

        # IoU-based matching
        matches = greedy_match_iou(gt_boxes_xyxy, pred_boxes_xyxy, iou_thresh=iou_thresh)

        tp = len(matches)
        fn = len(gt_boxes_xyxy) - tp
        fp = len(pred_boxes_xyxy) - tp

        total_tp += tp
        total_fn += fn
        total_fp += fp

        for _, _, iou in matches:
            iou_sum += float(iou)


    recall = total_tp / max(total_gt_faces, 1)
    precision = total_tp / max(total_pred_faces, 1)
    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    mean_iou = iou_sum / max(total_tp, 1)

    print("\n===== Detector-Only Evaluation Result =====")
    print(f"Data root             : {data_root}")
    print(f"Detector              : {detector_name}")
    print(f"IoU threshold         : {iou_thresh}")
    print(f"Images processed      : {total_images}")
    print(f"Total GT faces        : {total_gt_faces}")
    print(f"Total predicted boxes : {total_pred_faces}")
    print(f"TP (matched)          : {total_tp}")
    print(f"FP (unmatched preds)  : {total_fp}")
    print(f"FN (missed GT)        : {total_fn}")
    print(f"Recall                : {recall:.4f}")
    print(f"Precision             : {precision:.4f}")
    print(f"F1-score              : {f1:.4f}")
    print(f"Mean IoU (TP only)    : {mean_iou:.4f}")

    # Save CSV
    if results_csv is not None:
        import csv
        os.makedirs(os.path.dirname(results_csv), exist_ok=True)
        exists = os.path.exists(results_csv)
        with open(results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(
                    [
                        "data_root",
                        "detector",
                        "iou_thresh",
                        "images",
                        "total_gt_faces",
                        "total_pred_faces",
                        "tp",
                        "fp",
                        "fn",
                        "recall",
                        "precision",
                        "f1",
                        "mean_iou",
                    ]
                )
            writer.writerow(
                [
                    data_root,
                    detector_name,
                    iou_thresh,
                    total_images,
                    total_gt_faces,
                    total_pred_faces,
                    total_tp,
                    total_fp,
                    total_fn,
                    recall,
                    precision,
                    f1,
                    mean_iou,
                ]
            )


def build_argparser():
    p = argparse.ArgumentParser(
        description="Detector-only evaluation on face-mask dataset."
    )
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root of dataset (contains 'images/' and 'annotations/').",
    )
    p.add_argument(
        "--detector",
        type=str,
        default="yunet",
        choices=["yunet", "haar", "mtcnn", "retinaface"],
        help="Which detector to use.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string passed to detector (for MTCNN).",
    )
    p.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        help="IoU threshold for TP definition.",
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit number of images for quick test (None = all).",
    )
    p.add_argument(
        "--yunet_onnx",
        type=str,
        default="face_detection_yunet_2023mar.onnx",
        help="Path to YuNet ONNX.",
    )
    p.add_argument(
        "--haar_xml",
        type=str,
        default="haarcascade_frontalface_default.xml",
        help="Path to Haar cascade XML.",
    )
    p.add_argument(
        "--retina_thresh",
        type=float,
        default=0.8,
        help="Score threshold for RetinaFace.",
    )
    p.add_argument(
        "--results_csv",
        type=str,
        default=None,
        help="Path to CSV file to append a one-line summary.",
    )
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    evaluate_detector_only(
        data_root=args.data_root,
        detector_name=args.detector,
        device_str=args.device,
        iou_thresh=args.iou_thresh,
        max_images=args.max_images,
        yunet_onnx=args.yunet_onnx,
        haar_xml=args.haar_xml,
        retina_thresh=args.retina_thresh,
        results_csv=args.results_csv,
    )


if __name__ == "__main__":
    main()
