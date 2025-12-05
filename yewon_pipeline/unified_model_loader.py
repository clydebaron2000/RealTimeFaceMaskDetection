"""
Unified Model Loader for Real-time Detection
Provides a flexible interface to load and use any detector/classifier combination from yewon_pipeline
"""

import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms

# Import model architectures from pipeline_1
from pipeline_1 import CustomCNN, SmallCNN, build_model, TrainConfig


@dataclass
class ModelInfo:
    """Information about a loaded model"""
    name: str
    type: str  # 'detector' or 'classifier'
    model_path: str
    num_classes: int
    input_size: int
    device: str


class UnifiedModelLoader:
    """
    Flexible model loader that can handle any model from yewon_pipeline
    """

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models_registry = {}
        self.scan_available_models()

    def scan_available_models(self):
        """Scan for available models in common locations"""
        model_locations = [
            'runs_12k',
            'runs_eval',
            'model_artifacts',
            '.',
        ]

        self.available_models = {
            'detectors': [],
            'classifiers': []
        }

        # Scan for classifier models
        for location in model_locations:
            if not os.path.exists(location):
                continue

            # Look for .pth files
            for root, dirs, files in os.walk(location):
                for file in files:
                    if file.endswith('.pth'):
                        full_path = os.path.join(root, file)
                        model_info = self._analyze_model(full_path)
                        if model_info:
                            self.available_models['classifiers'].append(model_info)

        # Add default detectors
        self.available_models['detectors'] = [
            {'name': 'haar', 'type': 'opencv', 'path': 'haarcascade'},
            {'name': 'yunet', 'type': 'onnx', 'path': 'yewon_pipeline/face_detection_yunet_2023mar.onnx'},
            {'name': 'mtcnn', 'type': 'pytorch', 'path': 'facenet-pytorch'},
            {'name': 'retinaface', 'type': 'pytorch', 'path': 'retina-face'},
        ]

    def _analyze_model(self, model_path: str) -> Optional[Dict]:
        """Analyze a model file to determine its type and configuration"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')

            # Check if it's a wrapped model with config
            if 'config' in checkpoint:
                config = checkpoint['config']
                model_name = config.get('model_name', 'unknown')
                num_classes = config.get('num_classes', 2)
                img_size = config.get('img_size', 224)

                return {
                    'name': model_name,
                    'path': model_path,
                    'num_classes': num_classes,
                    'img_size': img_size,
                    'type': 'wrapped',
                    'config': config
                }

            # Check if it's a raw state dict
            elif isinstance(checkpoint, dict) and any('conv' in k or 'fc' in k for k in checkpoint.keys()):
                # Try to infer from layer shapes
                num_classes = 2  # default
                img_size = 224  # default

                if 'fc.weight' in checkpoint:
                    num_classes = checkpoint['fc.weight'].shape[0]
                elif 'fc.0.weight' in checkpoint:
                    num_classes = checkpoint['fc.0.weight'].shape[0]

                return {
                    'name': os.path.basename(model_path).replace('.pth', ''),
                    'path': model_path,
                    'num_classes': num_classes,
                    'img_size': img_size,
                    'type': 'state_dict'
                }

        except Exception as e:
            print(f"Could not analyze {model_path}: {e}")

        return None

    def load_classifier(self, model_path: str) -> nn.Module:
        """Load a classifier model"""
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'config' in checkpoint:
            # Wrapped model with config
            config_dict = checkpoint['config']
            config = TrainConfig(**{k: v for k, v in config_dict.items()
                                    if k in TrainConfig.__dataclass_fields__})

            # Build model based on config
            model = build_model(config)

            # Load state dict
            if 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])

        else:
            # Raw state dict - try to infer architecture
            state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()

            # Infer num_classes from fc layer
            if 'fc.weight' in state_dict:
                num_classes = state_dict['fc.weight'].shape[0]
            elif 'fc.0.weight' in state_dict:
                num_classes = state_dict['fc.0.weight'].shape[0]
            else:
                num_classes = 2  # default

            # Try CustomCNN first (most common in this codebase)
            model = CustomCNN(in_channels=3, num_classes=num_classes)
            try:
                model.load_state_dict(state_dict, strict=False)
            except:
                # If CustomCNN fails, try SmallCNN
                model = SmallCNN(in_channels=3, num_classes=num_classes)
                model.load_state_dict(state_dict, strict=False)

        model.to(self.device)
        model.eval()
        return model

    def get_transform(self, img_size: int = 224) -> transforms.Compose:
        """Get the appropriate transform for model input"""
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def preprocess_face(self, face_img: np.ndarray, img_size: int = 224) -> torch.Tensor:
        """Preprocess a face image for classification"""
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Convert to PIL
        pil_img = Image.fromarray(face_rgb)

        # Apply transforms
        transform = self.get_transform(img_size)
        tensor = transform(pil_img)

        # Add batch dimension
        return tensor.unsqueeze(0)

    def classify_face(self, model: nn.Module, face_img: np.ndarray,
                     img_size: int = 224) -> Tuple[str, float]:
        """Classify a face image"""
        # Preprocess
        input_tensor = self.preprocess_face(face_img, img_size).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        # Map to label
        class_labels = ['with_mask', 'without_mask']
        if outputs.shape[1] == 3:
            class_labels = ['with_mask', 'without_mask', 'incorrect_mask']

        label = class_labels[predicted.item()]
        conf = confidence.item()

        return label, conf

    def list_available_models(self) -> Dict[str, List[Dict]]:
        """List all available models"""
        return self.available_models


class FlexibleDetector:
    """Flexible face detector that can use different backends"""

    def __init__(self, detector_type: str = 'haar', device: str = 'cpu'):
        self.detector_type = detector_type
        self.device = device
        self.detector = None
        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize the selected detector"""
        try:
            if self.detector_type == 'haar':
                # Import from detectors_2
                from detectors_2 import HaarCascadeDetector
                # Try default OpenCV path first
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if not os.path.exists(cascade_path):
                    cascade_path = 'haarcascade_frontalface_default.xml'
                self.detector = HaarCascadeDetector(cascade_path)

            elif self.detector_type == 'yunet':
                from detectors_2 import YuNetDetector
                model_path = 'face_detection_yunet_2023mar.onnx'
                if not os.path.exists(model_path):
                    model_path = 'yewon_pipeline/face_detection_yunet_2023mar.onnx'
                if os.path.exists(model_path):
                    self.detector = YuNetDetector(model_path)
                else:
                    print(f"YuNet model not found at {model_path}")

            elif self.detector_type == 'mtcnn':
                from detectors_2 import MTCNNDetector
                self.detector = MTCNNDetector(device=self.device)

            elif self.detector_type == 'retinaface':
                from detectors_2 import RetinaFaceDetector
                self.detector = RetinaFaceDetector(thresh=0.8, device=self.device)

        except ImportError as e:
            print(f"Could not import detector {self.detector_type}: {e}")
            print(f"Falling back to Haar cascade")
            # Fallback to basic Haar implementation
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image and return bounding boxes"""
        faces = []

        # If we have a detector from detectors_2
        if hasattr(self.detector, 'detect'):
            detections = self.detector.detect(image)
            # Convert from (x, y, w, h, score) to (x1, y1, x2, y2)
            faces = [(x, y, x+w, y+h) for (x, y, w, h, score) in detections]

        # Fallback for basic cv2.CascadeClassifier
        elif isinstance(self.detector, cv2.CascadeClassifier):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            faces = [(x, y, x+w, y+h) for (x, y, w, h) in detections]

        return faces


class UnifiedPipeline:
    """
    Unified pipeline for face detection and mask classification
    Supports any combination of detectors and classifiers from yewon_pipeline
    """

    def __init__(self, detector_type: str = 'haar',
                 classifier_path: Optional[str] = None,
                 device: str = 'auto'):

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize detector
        self.detector = FlexibleDetector(detector_type, device=self.device)

        # Initialize model loader
        self.model_loader = UnifiedModelLoader(self.device)

        # Load classifier if specified
        self.classifier = None
        self.classifier_info = None
        if classifier_path and os.path.exists(classifier_path):
            self.load_classifier(classifier_path)

    def load_classifier(self, model_path: str):
        """Load a classifier model"""
        self.classifier = self.model_loader.load_classifier(model_path)
        self.classifier_info = self.model_loader._analyze_model(model_path)
        print(f"Loaded classifier: {self.classifier_info}")

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Process a single frame"""
        results = []

        # Detect faces
        faces = self.detector.detect_faces(frame)

        for (x1, y1, x2, y2) in faces:
            result = {
                'box': [x1, y1, x2, y2],
                'type': 'face'
            }

            # Classify if classifier is loaded
            if self.classifier:
                face_img = frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    img_size = self.classifier_info.get('img_size', 224)
                    label, confidence = self.model_loader.classify_face(
                        self.classifier, face_img, img_size
                    )
                    result['label'] = label
                    result['confidence'] = confidence

            results.append(result)

        return results

    def draw_results(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """Draw detection/classification results on frame"""
        output = frame.copy()

        colors = {
            'with_mask': (0, 255, 0),      # Green
            'without_mask': (0, 0, 255),   # Red
            'incorrect_mask': (0, 165, 255), # Orange
            'face': (255, 0, 255)           # Magenta
        }

        for result in results:
            x1, y1, x2, y2 = result['box']

            # Get color based on label
            if 'label' in result:
                color = colors.get(result['label'], (255, 255, 0))
                label_text = f"{result['label']}: {result['confidence']:.2f}"
            else:
                color = colors['face']
                label_text = "Face"

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label
            cv2.putText(output, label_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output