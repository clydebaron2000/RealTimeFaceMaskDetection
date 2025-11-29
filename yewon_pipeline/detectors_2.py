from typing import List, Tuple
import os

import cv2
import numpy as np


# MTCNN (facenet-pytorch)
from facenet_pytorch import MTCNN

# RetinaFace (retina-face package)
from retinaface import RetinaFace


FaceBox = Tuple[int, int, int, int, float]  # x, y, w, h, score


class BaseFaceDetector:
    def detect(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        """
        Input: BGR image (H, W, 3)
        Output: list of (x, y, w, h, score)
        """
        raise NotImplementedError


# ---------------------------
# YuNet
# ---------------------------

class YuNetDetector(BaseFaceDetector):
    def __init__(
        self,
        onnx_path: str = "face_detection_yunet_2023mar.onnx",
        score_thresh: float = 0.6,
        nms_thresh: float = 0.3,
    ):
        self.model = cv2.FaceDetectorYN_create(
            onnx_path,
            "",
            (320, 320),
            score_threshold=score_thresh,
            nms_threshold=nms_thresh,
            top_k=5000,
        )
        self.score_thresh = score_thresh

    def detect(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        h, w, _ = frame_bgr.shape
        self.model.setInputSize((w, h))
        _, result = self.model.detect(frame_bgr)
        boxes: List[FaceBox] = []
        if result is not None:
            for det in result:
                x, y, w_box, h_box, score = det[:5]
                if score < self.score_thresh:
                    continue
                boxes.append((int(x), int(y), int(w_box), int(h_box), float(score)))
        return boxes


# ---------------------------
# Haar Cascade
# ---------------------------

class HaarCascadeDetector(BaseFaceDetector):
    def __init__(self, xml_path: str = "haarcascade_frontalface_default.xml"):
        self.cascade = cv2.CascadeClassifier(xml_path)

    def detect(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        boxes: List[FaceBox] = []
        for (x, y, w_box, h_box) in faces:
            boxes.append((int(x), int(y), int(w_box), int(h_box), 1.0))
        return boxes


# ---------------------------
# MTCNN (facenet-pytorch)
# ---------------------------

class MTCNNDetector(BaseFaceDetector):
    def __init__(self, device: str = "cuda"):
        self.mtcnn = MTCNN(keep_all=True, device=device)

    def detect(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs = self.mtcnn.detect(frame_rgb)
        results: List[FaceBox] = []
        if boxes is not None:
            for box, score in zip(boxes, probs):
                if score is None:
                    continue
                x1, y1, x2, y2 = box
                w_box = x2 - x1
                h_box = y2 - y1
                results.append((int(x1), int(y1), int(w_box), int(h_box), float(score)))
        return results


# ---------------------------
# RetinaFace
# ---------------------------

class RetinaFaceDetector(BaseFaceDetector):

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def detect(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        try:
            resp = RetinaFace.detect_faces(frame_bgr, threshold=self.threshold)
        except Exception:
            # Return empty list on failure
            return []

        boxes: List[FaceBox] = []
        if isinstance(resp, dict):
            for _, info in resp.items():
                area = info.get("facial_area", None)
                score = info.get("score", 1.0)
                if area is None:
                    continue
                x1, y1, x2, y2 = area
                w_box = x2 - x1
                h_box = y2 - y1
                boxes.append(
                    (int(x1), int(y1), int(w_box), int(h_box), float(score))
                )
        return boxes


# ---------------------------
# factory
# ---------------------------

def build_detector(
    name: str,
    device: str = "cuda",
    yunet_onnx: str = "face_detection_yunet_2023mar.onnx",
    haar_xml: str = "haarcascade_frontalface_default.xml",
    retina_thresh: float = 0.8,
) -> BaseFaceDetector:
    name = name.lower()
    if name == "yunet":
        return YuNetDetector(onnx_path=yunet_onnx)
    elif name == "haar":
        return HaarCascadeDetector(xml_path=haar_xml)
    elif name == "mtcnn":
        return MTCNNDetector(device=device)
    elif name == "retinaface":
        return RetinaFaceDetector(threshold=retina_thresh)
    else:
        raise ValueError(f"Unknown detector name: {name}")
