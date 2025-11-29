# -*- coding: utf-8 -*-
"""
CS 583 Final Project - Face Mask Detection (Kaggle + Custom CNN + YuNet/Haar + Webcam)

Pipeline:

1) Download / locate Kaggle dataset:
      andrewmvd/face-mask-detection
   using kagglehub.

2) Copy dataset into project folder:
      ./Kaggle downloaded dataset/
   (contains annotations/ and images/).

3) Build FaceMaskDataset from annotations + images.

4) From the original dataset, create ground-truth category folders:

      Category with mask/
          original/
          green with-mask/

      Category without mask/
          original/
          red without mask/

5) Build training dataset from annotated folders:

      green with-mask/      -> label 1 (Mask)
      red without mask/     -> label 0 (No Mask)

   Images are resized to 32x32 grayscale.

6) Balance classes via oversampling, split into train/val,
   and train a small CNN (pure NumPy + scipy.signal.convolve2d).

   Save:
      - mask_detector_model.pkl
      - loss_plot_kaggle.png
      - confusion_matrix_kaggle.png

   Final console output includes:
      - Train accuracy
      - Validation accuracy
      - Per-class precision/recall/F1

7) Live webcam detection with YuNet (preferred) or Haar:

      - If YuNet ONNX file is present:
            face_detection_yunet_2023mar.onnx
        use YuNet for face detection.

      - Otherwise fall back to Haar cascade.

   Classification decision rule (for webcam & logging):

      probs = softmax(logits) = [p_no_mask, p_mask]

      If p_mask >= 0.75 and (p_mask - p_no_mask) >= 0.15:
          Mask (green box, label "Mask (p)")
      Else:
          No Mask (red box, label "No Mask (p)" + beep)

Modes:

      python code.py --mode train   # full pipeline + webcam
      python code.py --mode detect  # just load model + webcam
"""

# ============================================================
# PART 0: IMPORTS & GLOBAL SETTINGS
# ============================================================

import os
import time
import argparse
import pickle
import xml.etree.ElementTree as ET
import shutil
from typing import Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve2d
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from PIL import Image
import winsound  # For beep on Windows

# Global constants / paths
KAGGLE_DATASET_ID = "andrewmvd/face-mask-detection"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mask_detector_model.pkl")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
YUNET_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")

np.random.seed(40)


# ============================================================
# PART 1: DOWNLOAD / COPY DATASET FROM KAGGLE (kagglehub)
# ============================================================

def fetch_dataset_path(dataset_id: str) -> str:
    """
    Download (or locate) the Kaggle dataset using kagglehub.

    Dataset must contain:
        annotations/
        images/

    Install once:
        pip install kagglehub
    """
    try:
        import kagglehub  # type: ignore
    except ImportError:
        raise ImportError(
            "Could not import 'kagglehub'.\n"
            "Install with:\n"
            "   pip install kagglehub\n"
            "Then run:\n"
            "   python code.py --mode train\n"
        )

    print("Using kagglehub to download/locate dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download(dataset_id)
    print(f"  kagglehub dataset path: {dataset_path}")

    annotations = os.path.join(dataset_path, "annotations")
    images = os.path.join(dataset_path, "images")
    if not os.path.isdir(annotations) or not os.path.isdir(images):
        raise FileNotFoundError(
            f"Expected 'annotations/' and 'images/' under {dataset_path}, "
            "but they were not found."
        )

    return dataset_path


def copy_kaggle_dataset_locally(kaggle_path: str) -> str:
    """
    Copy the Kaggle dataset into project folder so we have a local copy:

        BASE_DIR/"Kaggle downloaded dataset"/

    If it already exists, reuse it.
    """
    local_root = os.path.join(BASE_DIR, "Kaggle downloaded dataset")

    if os.path.abspath(kaggle_path) == os.path.abspath(local_root):
        print("  Kaggle dataset already in project folder.")
        return local_root

    if not os.path.exists(local_root):
        print(f"Copying Kaggle dataset into project folder:\n  {local_root}")
        shutil.copytree(kaggle_path, local_root)
        print("  Copy complete.")
    else:
        print(f"Local dataset folder already exists:\n  {local_root}")

    return local_root


# ============================================================
# PART 2: ORIGINAL DATASET WRAPPER (IMAGES + XML)
# ============================================================

class FaceMaskDataset:
    """
    Dataset that loads full images + bounding boxes + class labels.

    __getitem__ returns:
        image (PIL RGB),
        target['boxes']  = list of [xmin, ymin, xmax, ymax]
        target['labels'] = list of integer class indices
    """

    def __init__(self, images_path: str, annotations_path: str, classes: set):
        self.images_path = images_path
        self.annotations_path = annotations_path

        # Map class name -> index
        self.classes_dict = dict(zip(sorted(classes), range(len(classes))))

        self.image_ids = []
        for anno_file in os.listdir(annotations_path):
            if not anno_file.lower().endswith(".xml"):
                continue
            image_id = os.path.splitext(anno_file)[0]
            self.image_ids.append(image_id)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        img_path_png = os.path.join(self.images_path, f"{image_id}.png")
        img_path_jpg = os.path.join(self.images_path, f"{image_id}.jpg")

        if os.path.exists(img_path_png):
            img_path = img_path_png
        elif os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        else:
            raise FileNotFoundError(
                f"No image found for {image_id} (tried .png and .jpg)."
            )

        image = Image.open(img_path).convert("RGB")

        anno_path = os.path.join(self.annotations_path, f"{image_id}.xml")
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")

        tree = ET.parse(anno_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes_dict[name])

        target = {"boxes": boxes, "labels": labels}
        return image, target


# ============================================================
# PART 3: CATEGORY + ANNOTATED FOLDERS CREATION
# ============================================================

def create_category_and_annotated_folders(
    full_dataset: FaceMaskDataset,
) -> Tuple[str, str]:
    """
    From the original dataset (annotations + images), create:

        Category with mask/
            original/
            green with-mask/

        Category without mask/
            original/
            red without mask/

    Rules (GROUND TRUTH from XML):
        - If ANY face is labeled 'without_mask' or 'mask_weared_incorrect'
          -> Category without mask
        - Else if at least one 'with_mask' face present
          -> Category with mask
        - If an image has no faces, skip it.

    Annotated images:
        - 'with_mask' faces   -> GREEN rectangles
        - 'without_mask' or
          'mask_weared_incorrect' -> RED rectangles
    """
    idx_to_class = {v: k for k, v in full_dataset.classes_dict.items()}

    # Folder layout
    base_mask = os.path.join(BASE_DIR, "Category with mask")
    base_nomask = os.path.join(BASE_DIR, "Category without mask")

    orig_mask_dir = os.path.join(base_mask, "original")
    green_mask_dir = os.path.join(base_mask, "green with-mask")

    orig_nomask_dir = os.path.join(base_nomask, "original")
    red_nomask_dir = os.path.join(base_nomask, "red without mask")

    for d in [orig_mask_dir, green_mask_dir, orig_nomask_dir, red_nomask_dir]:
        os.makedirs(d, exist_ok=True)

    print("\n=== Creating category + annotated folders ===")

    for idx in tqdm(range(len(full_dataset)), desc="Categorizing images", unit="img"):
        image_pil, target = full_dataset[idx]
        boxes = target["boxes"]
        labels = target["labels"]
        if len(boxes) == 0:
            continue

        class_names = [idx_to_class[int(l)] for l in labels]

        has_no_mask = any(
            name in ("without_mask", "mask_weared_incorrect") for name in class_names
        )
        has_mask = any(name == "with_mask" for name in class_names)

        if has_no_mask:
            orig_dir = orig_nomask_dir
            ann_dir = red_nomask_dir
        elif has_mask:
            orig_dir = orig_mask_dir
            ann_dir = green_mask_dir
        else:
            continue

        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        image_id = full_dataset.image_ids[idx]
        base_name = f"{image_id}.png"

        # Save original category image
        cv2.imwrite(os.path.join(orig_dir, base_name), image_bgr)

        # Create annotated copy
        annotated = image_bgr.copy()
        h, w, _ = annotated.shape

        for box, lab in zip(boxes, labels):
            name = idx_to_class[int(lab)]
            xmin, ymin, xmax, ymax = map(int, box)

            xmin = max(0, min(xmin, w - 1))
            xmax = max(0, min(xmax, w))
            ymin = max(0, min(ymin, h - 1))
            ymax = max(0, min(ymax, h))
            if xmax <= xmin or ymax <= ymin:
                continue

            if name == "with_mask":
                color = (0, 255, 0)   # green
            else:
                color = (0, 0, 255)   # red

            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)

        cv2.imwrite(os.path.join(ann_dir, base_name), annotated)

    print("Finished creating 'Category with mask' and 'Category without mask'.\n")

    # Return paths to annotated folders used for training
    return green_mask_dir, red_nomask_dir


# ============================================================
# PART 4: BUILD NUMPY DATASET FROM ANNOTATED FOLDERS
# ============================================================

def build_numpy_dataset_from_annotated(
    green_mask_dir: str,
    red_nomask_dir: str,
    img_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create X, y from annotated category folders:

        green_mask_dir (green with-mask)  -> label 1 (Mask)
        red_nomask_dir (red without mask) -> label 0 (No Mask)

    Output:
        X: (N, img_size, img_size, 1), float32 grayscale
        y: (N,), int64
    """
    X_list = []
    y_list = []

    def add_from_folder(folder: str, label: int):
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(folder, fname)
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                continue
            img_resized = cv2.resize(img_bgr, (img_size, img_size)).astype(np.float32) / 255.0
            gray = np.mean(img_resized, axis=2, keepdims=True)
            X_list.append(gray)
            y_list.append(label)

    print("Building NumPy dataset from annotated folders...")
    add_from_folder(green_mask_dir, 1)
    add_from_folder(red_nomask_dir, 0)

    if len(y_list) == 0:
        raise RuntimeError("No images found in annotated folders for training.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    print(f"Annotated dataset: X={X.shape}, y={y.shape}, mask_ratio={y.mean():.3f}")
    return X, y


# ============================================================
# PART 5: CNN ARCHITECTURE (pure NumPy)
# ============================================================

class Conv2D:
    """Simple 2D convolution layer using scipy.signal.convolve2d."""

    def __init__(self, num_filters, kernel_size, input_channels):
        self.W = np.random.randn(num_filters, kernel_size, kernel_size, input_channels) / np.sqrt(
            kernel_size * kernel_size * input_channels / 2
        )
        self.b = np.zeros(num_filters)
        self.kernel_size = kernel_size

    def forward(self, x):
        self.x = x
        N, H, W, C = x.shape
        F, K, _, _ = self.W.shape
        out_H, out_W = H - K + 1, W - K + 1
        out = np.zeros((N, out_H, out_W, F), dtype=np.float32)

        for n in range(N):
            for f in range(F):
                acc = None
                for c in range(C):
                    conv = convolve2d(
                        self.x[n, :, :, c],
                        self.W[f, :, :, c],
                        mode="valid",
                    )
                    acc = conv if acc is None else acc + conv
                out[n, :, :, f] = acc + self.b[f]

        return out

    def backward(self, dout, lr):
        N, H, W, C = self.x.shape
        F, K, _, _ = self.W.shape

        dW = np.zeros_like(self.W)
        db = np.sum(dout, axis=(0, 1, 2))
        dx = np.zeros_like(self.x)

        for n in range(N):
            for f in range(F):
                for c in range(C):
                    dW[f, :, :, c] += convolve2d(
                        self.x[n, :, :, c],
                        dout[n, :, :, f],
                        mode="valid",
                    )
                    dx[n, :, :, c] += convolve2d(
                        dout[n, :, :, f],
                        np.rot90(self.W[f, :, :, c], 2),
                        mode="full",
                    )

        self.W -= lr * dW
        self.b -= lr * db
        return dx


class ReLU:
    """ReLU activation."""

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2D:
    """2x2 Max pooling layer."""

    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, x):
        self.x = x
        N, H, W, C = x.shape
        out_H, out_W = H // self.pool_size, W // self.pool_size
        out = np.zeros((N, out_H, out_W, C), dtype=np.float32)
        self.max_idx = np.zeros((N, out_H, out_W, C), dtype=np.int32)

        for n in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        patch = x[
                            n,
                            i * self.pool_size : i * self.pool_size + self.pool_size,
                            j * self.pool_size : j * self.pool_size + self.pool_size,
                            c,
                        ]
                        out[n, i, j, c] = np.max(patch)
                        self.max_idx[n, i, j, c] = np.argmax(patch)

        return out

    def backward(self, dout):
        N, out_H, out_W, C = dout.shape
        dx = np.zeros_like(self.x)

        for n in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        idx = self.max_idx[n, i, j, c]
                        pi = idx // self.pool_size
                        pj = idx % self.pool_size
                        dx[
                            n,
                            i * self.pool_size + pi,
                            j * self.pool_size + pj,
                            c,
                        ] = dout[n, i, j, c]
        return dx


class Flatten:
    """Flatten (N, H, W, C) -> (N, H*W*C)."""

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class Linear:
    """Fully-connected layer."""

    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) / np.sqrt(in_features / 2)
        self.b = np.zeros(out_features)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout, lr):
        dx = dout @ self.W.T
        dW = self.x.T @ dout
        db = np.sum(dout, axis=0)

        self.W -= lr * dW
        self.b -= lr * db

        return dx


class SoftmaxCrossEntropy:
    """Softmax + cross-entropy loss."""

    def forward(self, x, y):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.y = y
        loss = -np.mean(np.log(self.probs[np.arange(len(y)), y] + 1e-10))
        return loss, self.probs

    def backward(self):
        dout = self.probs.copy()
        dout[np.arange(len(self.y)), self.y] -= 1
        return dout / len(self.y)


class MaskDetectorCNN:
    """
    CNN:
        32x32x1
        -> Conv(16, 3x3) + ReLU + MaxPool(2)
        -> Conv(32, 3x3) + ReLU + MaxPool(2)
        -> Flatten
        -> FC(128) + ReLU
        -> FC(2)  [NoMask, Mask]
    """

    def __init__(self):
        self.layers = [
            Conv2D(16, 3, 1),
            ReLU(),
            MaxPool2D(2),
            Conv2D(32, 3, 16),
            ReLU(),
            MaxPool2D(2),
            Flatten(),
            Linear(6 * 6 * 32, 128),
            ReLU(),
            Linear(128, 2),
        ]
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x, y=None):
        for layer in self.layers:
            x = layer.forward(x)
        if y is None:
            return x
        return self.loss_fn.forward(x, y)

    def backward(self, dout, lr):
        for layer in reversed(self.layers):
            if isinstance(layer, (Conv2D, Linear)):
                dout = layer.backward(dout, lr)
            elif hasattr(layer, "backward"):
                dout = layer.backward(dout)
        return dout


# ============================================================
# PART 6: TRAINING LOOP (BALANCING + METRICS)
# ============================================================

def train_model(epochs: int = 12, batch_size: int = 64, lr: float = 0.001) -> MaskDetectorCNN:
    """
    Full training pipeline:

      1) Download dataset from Kaggle via kagglehub.
      2) Copy into 'Kaggle downloaded dataset/'.
      3) Build FaceMaskDataset.
      4) Create Category-with-mask / Category-without-mask folders,
         plus green/red annotated subfolders.
      5) Build NumPy dataset from:
            - green with-mask/      -> label 1
            - red without mask/     -> label 0
      6) Balance classes via oversampling.
      7) Train CNN with progress bars, track val metrics.
      8) Save model and plots, and print both TRAIN and VAL accuracy.
    """
    print("=== DATASET PREPARATION ===")
    kaggle_path = fetch_dataset_path(KAGGLE_DATASET_ID)
    local_dataset_path = copy_kaggle_dataset_locally(kaggle_path)

    annotations_path = os.path.join(local_dataset_path, "annotations")
    images_path = os.path.join(local_dataset_path, "images")

    # Discover all class names from XML
    classes = set()
    for anno_file in os.listdir(annotations_path):
        if not anno_file.lower().endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(annotations_path, anno_file))
        for obj in tree.findall("object"):
            classes.add(obj.find("name").text)

    print(f"  Classes discovered: {classes}")

    full_dataset = FaceMaskDataset(images_path, annotations_path, classes)
    print(f"  Full dataset size (images): {len(full_dataset)}")

    # Create category + annotated folders
    green_mask_dir, red_nomask_dir = create_category_and_annotated_folders(full_dataset)

    # Build X, y from annotated images
    X, y = build_numpy_dataset_from_annotated(green_mask_dir, red_nomask_dir, img_size=32)

    # ---------- class balancing (oversampling minority) ----------
    print("\n=== CLASS BALANCING (oversampling minority class) ===")
    idx_mask = np.where(y == 1)[0]
    idx_nomask = np.where(y == 0)[0]
    n_mask = len(idx_mask)
    n_nomask = len(idx_nomask)
    print(f"  Count Mask (1):     {n_mask}")
    print(f"  Count No Mask (0):  {n_nomask}")

    if n_mask == 0 or n_nomask == 0:
        print("  WARNING: one class missing, skipping oversampling.")
        X_balanced, y_balanced = X, y
    else:
        max_count = max(n_mask, n_nomask)
        idx_mask_bal = np.random.choice(idx_mask, size=max_count, replace=True)
        idx_nomask_bal = np.random.choice(idx_nomask, size=max_count, replace=True)
        balanced_idx = np.concatenate([idx_mask_bal, idx_nomask_bal])
        np.random.shuffle(balanced_idx)

        X_balanced = X[balanced_idx]
        y_balanced = y[balanced_idx]

        print(f"  Balanced dataset size: {len(y_balanced)}")
        print(f"  New mask_ratio: {y_balanced.mean():.3f}")

    # Train/validation split
    split_idx = int(0.8 * len(X_balanced))
    X_train, X_val = X_balanced[:split_idx], X_balanced[split_idx:]
    y_train, y_val = y_balanced[:split_idx], y_balanced[split_idx:]

    print("=== CNN TRAINING ===")
    print(f"  Train size: {len(X_train)}, Val size: {len(X_val)}")

    model = MaskDetectorCNN()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        idx = np.random.permutation(len(y_train))
        X_train_shuffled = X_train[idx]
        y_train_shuffled = y_train[idx]

        epoch_train_loss = 0.0
        num_batches = 0

        batch_indices = range(0, len(y_train_shuffled), batch_size)
        batch_bar = tqdm(batch_indices, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for start_idx in batch_bar:
            end_idx = start_idx + batch_size
            batch_X = X_train_shuffled[start_idx:end_idx]
            batch_y = y_train_shuffled[start_idx:end_idx]

            loss, _ = model.forward(batch_X, batch_y)
            dout = model.loss_fn.backward()
            model.backward(dout, lr)

            epoch_train_loss += loss
            num_batches += 1
            batch_bar.set_postfix({"loss": f"{loss:.4f}"})

        avg_train_loss = epoch_train_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation
        val_logits = model.forward(X_val)
        val_loss, probs_val = model.loss_fn.forward(val_logits, y_val)
        val_losses.append(val_loss)
        val_preds = np.argmax(probs_val, axis=1)
        val_acc = np.mean(val_preds == y_val)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # Plot loss curves
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(BASE_DIR, "loss_plot_kaggle.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"  Saved loss plot to {loss_plot_path}")

    # Confusion matrix
    val_logits = model.forward(X_val)
    _, probs_val = model.loss_fn.forward(val_logits, y_val)
    val_preds = np.argmax(probs_val, axis=1)
    cm = confusion_matrix(y_val, val_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Mask", "Mask"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Validation)")
    cm_path = os.path.join(BASE_DIR, "confusion_matrix_kaggle.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"  Saved confusion matrix to {cm_path}")

    # Per-class metrics on validation set
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, val_preds, average=None, labels=[0, 1]
    )
    print("\n=== VALIDATION METRICS ===")
    print(
        f"  No Mask - Precision: {precision[0]:.4f}, "
        f"Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}"
    )
    print(
        f"  Mask    - Precision: {precision[1]:.4f}, "
        f"Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}"
    )

    # --- TRAINING ACCURACY (on balanced training set) ---
    train_logits = model.forward(X_train)
    _, probs_train = model.loss_fn.forward(train_logits, y_train)
    train_preds = np.argmax(probs_train, axis=1)
    train_acc = np.mean(train_preds == y_train)
    print(f"\n=== FINAL TRAINING ACCURACY ===\n  Train Acc: {train_acc:.4f}")

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to '{MODEL_PATH}'")

    return model


# ============================================================
# PART 7: DECISION RULE FOR WEBCAM
# ============================================================

def decide_label_from_probs(probs: np.ndarray) -> int:
    """
    Convert probabilities [p_no_mask, p_mask] to hard label.

    Conservative rule:
        If p_mask >= 0.75 and (p_mask - p_no_mask) >= 0.15:
            -> Mask (1)
        else:
            -> No Mask (0)

    This bias makes it much harder to call "Mask" when you
    are actually not wearing a mask.
    """
    p_no = float(probs[0, 0])
    p_mask = float(probs[0, 1])

    if p_mask >= 0.75 and (p_mask - p_no) >= 0.15:
        return 1
    else:
        return 0


# ============================================================
# PART 8: LIVE WEBCAM DETECTION (YuNet + fallback Haar)
# ============================================================

def create_face_detector():
    """
    Try to create a YuNet detector if ONNX file exists.
    Otherwise fall back to Haar cascade.
    """
    if os.path.exists(YUNET_PATH):
        print(f"\n[Detector] Using YuNet: {YUNET_PATH}")
        yunet = cv2.FaceDetectorYN_create(
            YUNET_PATH,
            "",
            (320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )
        return ("yunet", yunet)
    else:
        print(
            "\n[Detector] YuNet ONNX file not found, using Haar cascade instead.\n"
            f"  Expected YuNet file at: {YUNET_PATH}"
        )
        if not os.path.exists(CASCADE_PATH):
            raise FileNotFoundError(
                f"Haar cascade not found at: {CASCADE_PATH}.\n"
                "Place 'haarcascade_frontalface_default.xml' next to code.py."
            )
        haar = cv2.CascadeClassifier(CASCADE_PATH)
        return ("haar", haar)


def run_webcam_detection(model: MaskDetectorCNN):
    """
    Open webcam, detect faces (YuNet preferred, Haar fallback),
    classify each with CNN, draw boxes:

        GREEN = Mask
        RED   = No Mask

    Decision rule:
        probs = softmax(logits) = [p_no, p_mask]
        label = decide_label_from_probs(probs)

    Beeps whenever a No Mask face is detected.
    """
    detector_type, detector = create_face_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera index / permissions.")

    print("\n[Webcam] Starting live mask detection. Press 'q' to quit.\n")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        no_mask_detected = False
        faces = []

        if detector_type == "yunet":
            h, w, _ = frame.shape
            detector.setInputSize((w, h))
            result = detector.detect(frame)
            if result[1] is not None:
                for det in result[1]:
                    x, y, w_box, h_box, score = det[:5]
                    if score < 0.6:
                        continue
                    faces.append((int(x), int(y), int(w_box), int(h_box)))
        else:  # Haar
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_haar = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            faces = list(faces_haar)

        for (x, y, w_box, h_box) in faces:
            face_roi = frame[y : y + h_box, x : x + w_box]
            if face_roi.size == 0:
                continue

            face_resized = cv2.resize(face_roi, (32, 32)).astype(np.float32) / 255.0
            face_gray = np.mean(face_resized, axis=2, keepdims=True)[np.newaxis, :]

            start_infer = time.time()
            logits = model.forward(face_gray)
            infer_time_ms = (time.time() - start_infer) * 1000.0

            exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exps / np.sum(exps, axis=1, keepdims=True)

            p_no = float(probs[0, 0])
            p_mask = float(probs[0, 1])

            pred = decide_label_from_probs(probs)

            if pred == 1:
                label = f"Mask ({p_mask:.2f})"
                color = (0, 255, 0)
            else:
                label = f"No Mask ({p_no:.2f})"
                color = (0, 0, 255)
                no_mask_detected = True

            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            print(
                f"[Webcam] {label} | p_no={p_no:.2f} | p_mask={p_mask:.2f} | "
                f"inference={infer_time_ms:.2f} ms"
            )

        if no_mask_detected:
            winsound.Beep(1000, 200)

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Face Mask Detection (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"\n[Webcam] Average FPS: {fps:.2f}")
    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# PART 9: MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CS 583 Final Project - Face Mask Detection (Kaggle + Custom CNN + YuNet/Haar + Webcam)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "detect"],
        default="train",
        help=(
            "Mode:\n"
            "  train  : full pipeline (download/copy, categorize, train, webcam)\n"
            "  detect : load saved model and run webcam only\n"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Number of training epochs (only used in --mode train)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        print("\n=== MODE: TRAIN (Kaggle + CNN + Categories + Webcam) ===")
        model = train_model(epochs=args.epochs)
        print("\nTraining complete. Launching webcam...")
        run_webcam_detection(model)

    elif args.mode == "detect":
        print("\n=== MODE: DETECT (Webcam only, no training) ===")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file '{MODEL_PATH}' not found.\n"
                "Run once with '--mode train' to create and save the model."
            )
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"Loaded model from '{MODEL_PATH}'. Launching webcam...")
        run_webcam_detection(model)


if __name__ == "__main__":
    main()
