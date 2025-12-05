# eval_detector_classifier.py
"""
Detector + classifier combination evaluation script.

- classifier: best.pth trained by pipeline_1.py (GT bbox crop based, per-face)
- detector: yunet / haar / mtcnn / retinaface

"""

import os
import argparse
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from pipeline_1 import (
    TrainConfig,
    build_model,
    parse_kaggle_annotation,
    map_face_label,
)
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
) -> List[Tuple[int, int]]:
    """
    Simple greedy matching:
      - Match (gt, pred) pairs starting from highest IoU
      - Ignore pairs with IoU < threshold
    return: list of (gt_idx, pred_idx) pairs
    """
    matches: List[Tuple[int, int]] = []

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
        matches.append((gt_idx, pred_idx))

    return matches


# -------------------------
# Load model from checkpoint
# -------------------------

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = ckpt.get("config", {})

    # Create dummy cfg based on saved config
    dummy_cfg = TrainConfig(data_root=cfg_dict.get("data_root", "."))
    dummy_cfg.model_name = cfg_dict.get("model_name", dummy_cfg.model_name)
    dummy_cfg.num_classes = cfg_dict.get("num_classes", dummy_cfg.num_classes)
    dummy_cfg.img_size = cfg_dict.get("img_size", dummy_cfg.img_size)
    dummy_cfg.pretrained = cfg_dict.get("pretrained", dummy_cfg.pretrained)
    dummy_cfg.freeze_backbone = cfg_dict.get("freeze_backbone", dummy_cfg.freeze_backbone)

    model = build_model(dummy_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    merge_incorrect = cfg_dict.get("merge_incorrect_with_nomask", True)

    return model, dummy_cfg.num_classes, dummy_cfg.img_size, merge_incorrect


# -------------------------
# Main evaluation
# -------------------------

def evaluate_detector_classifier(
    data_root: str,
    checkpoint: str,
    detector_name: str,
    device_str: str = "cuda",
    iou_thresh: float = 0.5,
    max_images: Optional[int] = None,
    yunet_onnx: str = "face_detection_yunet_2023mar.onnx",
    haar_xml: str = "haarcascade_frontalface_default.xml",
    retina_thresh: float = 0.8,
    results_csv: Optional[str] = None,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

  
    # 1) Load model
    model, num_classes, img_size, merge_incorrect = load_model_from_checkpoint(
        checkpoint, device
    )

    # 1-1) Transform branching based on input resolution / normalization policy
    if img_size <= 64:
        # For low-resolution models (40x40) without normalization
        infer_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )
    else:
        # Standard backbones (224x224, ImageNet normalization)
        infer_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


    # 2) Prepare detector
    detector: BaseFaceDetector = build_detector(
        detector_name,
        device=device_str,
        yunet_onnx=yunet_onnx,
        haar_xml=haar_xml,
        retina_thresh=retina_thresh,
    )
    print(f"[Eval] Using detector: {detector_name}")

    # 3) Dataset paths
    images_dir = os.path.join(data_root, "images")
    annotations_dir = os.path.join(data_root, "annotations")

    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]
    xml_files.sort()

    # 4) Statistics variables
    total_images = 0
    total_gt_faces = 0
    total_detected_matched = 0
    total_pred_boxes = 0 

    cls_total = 0
    cls_correct = 0
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    for idx, xml_name in enumerate(xml_files):
        if max_images is not None and idx >= max_images:
            break

        xml_path = os.path.join(annotations_dir, xml_name)
        img_path = find_image_for_xml(images_dir, xml_name)
        if img_path is None:
            # Skip annotations without images
            continue

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        boxes, raw_labels = parse_kaggle_annotation(xml_path)

        # Map GT labels
        gt_boxes_xyxy = []
        gt_labels = []
        for box, raw_label in zip(boxes, raw_labels):
            mapped = map_face_label(
                raw_label,
                num_classes=num_classes,
                merge_incorrect_with_nomask=merge_incorrect,
            )
            if mapped is None:
                continue
            xmin, ymin, xmax, ymax = box  # Already in xyxy format
            gt_boxes_xyxy.append([xmin, ymin, xmax, ymax])
            gt_labels.append(mapped)

        if len(gt_labels) == 0:
            continue

        gt_boxes_xyxy = np.array(gt_boxes_xyxy, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)

        total_images += 1
        total_gt_faces += len(gt_labels)

        # 4-1) Run detector
        dets = detector.detect(image_bgr)  # list of (x, y, w, h, score)
        if len(dets) == 0:
            # All faces in this image are missed
            continue

        pred_boxes_xyxy = []
        for (x, y, w_box, h_box, score) in dets:
            x1 = x
            y1 = y
            x2 = x + w_box
            y2 = y + h_box
            pred_boxes_xyxy.append([x1, y1, x2, y2])
        pred_boxes_xyxy = np.array(pred_boxes_xyxy, dtype=np.float32)
        total_pred_boxes += len(pred_boxes_xyxy)

        # 4-2) IoU-based matching
        matches = greedy_match_iou(gt_boxes_xyxy, pred_boxes_xyxy, iou_thresh=iou_thresh)
        total_detected_matched += len(matches)

        # 4-3) Classification evaluation for matched bbox
        for gt_idx, pred_idx in matches:
            gx1, gy1, gx2, gy2 = pred_boxes_xyxy[pred_idx].astype(int)
            # Defensive check for invalid region
            gx1 = max(gx1, 0)
            gy1 = max(gy1, 0)
            gx2 = min(gx2, image_bgr.shape[1])
            gy2 = min(gy2, image_bgr.shape[0])
            if gx2 <= gx1 or gy2 <= gy1:
                continue

            crop_bgr = image_bgr[gy1:gy2, gx1:gx2]
            if crop_bgr.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            tensor = infer_transform(crop_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                pred_cls = int(torch.argmax(logits, dim=1).item())

            gt_cls = int(gt_labels[gt_idx])

            cls_total += 1
            if pred_cls == gt_cls:
                cls_correct += 1
            conf_mat[gt_cls, pred_cls] += 1

    # ----------------------------
    # Final statistics output
    # ----------------------------
    # det_recall = total_detected_matched / max(total_gt_faces, 1)
    # cls_acc = cls_correct / max(cls_total, 1)

    # Detection metrics
    det_recall = total_detected_matched / max(total_gt_faces, 1)
    det_precision = total_detected_matched / max(total_pred_boxes, 1)
    if det_precision + det_recall > 0:
        det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall)
    else:
        det_f1 = 0.0

    # Classification metrics (micro = accuracy)
    cls_acc = cls_correct / max(cls_total, 1)

    # Classification macro precision / recall / F1 (confusion matrix based)
    per_class_prec = []
    per_class_rec = []
    per_class_f1 = []

    for c in range(num_classes):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp
        fn = conf_mat[c, :].sum() - tp

        prec_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_c  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec_c + rec_c > 0:
            f1_c = 2 * prec_c * rec_c / (prec_c + rec_c)
        else:
            f1_c = 0.0

        per_class_prec.append(prec_c)
        per_class_rec.append(rec_c)
        per_class_f1.append(f1_c)

    macro_prec = sum(per_class_prec) / num_classes
    macro_rec  = sum(per_class_rec) / num_classes
    macro_f1   = sum(per_class_f1) / num_classes

    # Combined pipeline recall metric
    pipeline_recall = det_recall * cls_acc




    print("\n===== Detector + Classifier Evaluation Result =====")
    print(f"Data root         : {data_root}")
    print(f"Checkpoint        : {checkpoint}")
    print(f"Detector          : {detector_name}")
    print(f"IoU threshold     : {iou_thresh}")
    # print(f"Images processed  : {total_images}")
    # print(f"Total GT faces    : {total_gt_faces}")
    # print(f"Matched detections: {total_detected_matched}")
    # print(f"Detection recall  : {det_recall:.4f}")
    # print(f"Cls samples       : {cls_total}")
    # print(f"Cls correct       : {cls_correct}")
    # print(f"Cls accuracy      : {cls_acc:.4f}")
    # print("Confusion matrix (rows=GT, cols=Pred):")
    # print(conf_mat)
    print(f"Total images           : {total_images}")
    print(f"Total GT faces         : {total_gt_faces}")
    print(f"Total predicted boxes  : {total_pred_boxes}")
    print(f"Matched detections     : {total_detected_matched}")
    print(f"Detection recall       : {det_recall:.4f}")
    print(f"Detection precision    : {det_precision:.4f}")
    print(f"Detection F1           : {det_f1:.4f}")
    print(f"Cls samples            : {cls_total}")
    print(f"Cls correct            : {cls_correct}")
    print(f"Cls accuracy (micro F1): {cls_acc:.4f}")
    print(f"Cls macro precision    : {macro_prec:.4f}")
    print(f"Cls macro recall       : {macro_rec:.4f}")
    print(f"Cls macro F1           : {macro_f1:.4f}")
    print(f"Pipeline recall        : {pipeline_recall:.4f}")
    print("Confusion matrix (rows=GT, cols=Pred):")
    print(conf_mat)



    # CSV save option
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
                        "checkpoint",
                        "detector",
                        "iou_thresh",
                        "images",
                        "total_gt_faces",
                        "matched_detections",
                        "total_pred_boxes",
                        "det_recall",
                        "det_precision",
                        "det_f1",
                        "cls_samples",
                        "cls_correct",
                        "cls_acc",              # micro
                        "cls_macro_precision",
                        "cls_macro_recall",
                        "cls_macro_f1",
                        "pipeline_recall",
                    ]
                )

            writer.writerow(
                [
                    data_root,
                    checkpoint,
                    detector_name,
                    iou_thresh,
                    total_images,
                    total_gt_faces,
                    total_detected_matched,
                    total_pred_boxes,
                    det_recall,
                    det_precision,
                    det_f1,
                    cls_total,
                    cls_correct,
                    cls_acc,
                    macro_prec,
                    macro_rec,
                    macro_f1,
                    pipeline_recall,
                ]
            )


def build_argparser():
    p = argparse.ArgumentParser(
        description="Evaluate detector + classifier combo on face-mask dataset."
    )
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root of dataset (contains 'images/' and 'annotations/').",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to classifier checkpoint (best.pth).",
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
        help="Device for classifier model.",
    )
    p.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        help="IoU threshold for matching GT and predicted boxes.",
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit number of images for quick test (None = use all).",
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

    evaluate_detector_classifier(
        data_root=args.data_root,
        checkpoint=args.checkpoint,
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
