# Face Mask Detection System - Evaluation Pipeline

A comprehensive evaluation framework for face mask detection systems using YuNet face detector and custom CNN classifiers.

## Overview

This evaluation pipeline provides:
- **Face Detection Evaluation**: Test YuNet and Haar cascade detectors
- **Classification Evaluation**: Test custom NumPy CNN and PyTorch ResNet18 classifiers  
- **Full Pipeline Evaluation**: End-to-end detection + classification performance
- **Speed Benchmarking**: Real-time performance analysis (FPS, latency)
- **Memory Benchmarking**: Memory usage and leak detection
- **Comprehensive Reporting**: Automated performance reports and recommendations

## Project Structure

```
RealTimeFaceMaskDetection/
├── evaluation/                    # Main evaluation framework
│   ├── detectors/                # Face detection components
│   │   ├── detector_wrapper.py   # YuNet & Haar detector wrappers
│   │   └── metrics.py            # Detection metrics (mAP, IoU, etc.)
│   ├── classifiers/              # Face mask classification components
│   │   ├── classifier_wrapper.py # NumPy CNN & PyTorch wrappers
│   │   └── metrics.py            # Classification metrics
│   ├── datasets/                 # Dataset loading utilities
│   │   └── dataset_loaders.py    # AndrewMVD, Face12k, Medical Mask loaders
│   ├── benchmarks/               # Performance benchmarking
│   │   ├── speed_benchmark.py    # FPS and latency testing
│   │   └── memory_benchmark.py   # Memory usage and leak detection
│   └── eval_pipeline.py          # Main evaluation orchestrator
├── tests/                        # Unit tests
│   ├── test_detector_wrapper.py  # Detector wrapper tests
│   ├── test_detector_metrics.py  # Detection metrics tests
│   ├── test_classifier_wrapper.py # Classifier wrapper tests
│   └── run_tests.py              # Test runner
├── examples/                     # Usage examples
│   └── basic_evaluation_example.py
├── notebooks/                    # Jupyter notebooks
│   └── CS_583_Final_Project_Face_detection.ipynb
└── reference_code/               # Original implementation
    ├── code.py
    └── maskProject.py
```

## Features

### 🎯 Detection Evaluation
- **Metrics**: Precision, Recall, F1-Score, mAP@0.5
- **Detectors**: YuNet (ONNX), Haar Cascade (XML)
- **IoU Matching**: Configurable IoU thresholds
- **Visualization**: TP/FP/FN analysis

### 🧠 Classification Evaluation  
- **Metrics**: Accuracy, Per-class P/R/F1, Confusion Matrix, ROC Curves
- **Models**: Custom NumPy CNN, PyTorch ResNet18
- **Class Support**: 2-class (mask/no_mask) or 3-class (correct/incorrect/no_mask)
- **Robustness**: Cross-validation and error analysis

### ⚡ Performance Benchmarking
- **Speed**: FPS measurement, latency analysis, scalability testing
- **Memory**: Usage tracking, leak detection, peak memory analysis
- **Real-time**: 30 FPS target assessment
- **Hardware**: CPU/GPU performance comparison

### 📊 Comprehensive Reporting
- **JSON Results**: Machine-readable evaluation data
- **Summary Reports**: Human-readable performance analysis
- **Recommendations**: Automated optimization suggestions
- **System Readiness**: Deployment readiness assessment

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/clydebaron2000/RealTimeFaceMaskDetection.git
cd RealTimeFaceMaskDetection

# Install dependencies
pip install opencv-python numpy torch torchvision scikit-learn matplotlib psutil
```

### 2. Run Unit Tests

```bash
# Run all tests
python tests/run_tests.py

# Run specific test module
python tests/run_tests.py test_detector_wrapper
```

### 3. Basic Evaluation

```python
from evaluation.eval_pipeline import EvaluationPipeline

# Initialize pipeline
pipeline = EvaluationPipeline(output_dir="results")

# Setup components
pipeline.setup_detector('yunet', model_path='models/yunet.onnx')
pipeline.setup_classifier('pytorch', 
                         model_path='models/classifier.pth',
                         class_names=['no_mask', 'with_mask', 'incorrect_mask'])
pipeline.setup_dataset('andrewmvd', 'datasets/andrewmvd')

# Run comprehensive evaluation
results = pipeline.run_comprehensive_evaluation()
```

### 4. Command Line Interface

```bash
# Run comprehensive evaluation
python -m evaluation.eval_pipeline \
    --detector yunet \
    --detector-model models/yunet.onnx \
    --classifier pytorch \
    --classifier-model models/classifier.pth \
    --dataset andrewmvd \
    --dataset-path datasets/andrewmvd \
    --comprehensive

# Run specific evaluations
python -m evaluation.eval_pipeline \
    --dataset-path datasets/andrewmvd \
    --eval-detection \
    --eval-classification \
    --benchmark-speed
```

## Supported Datasets

### 1. AndrewMVD Face Mask Detection
- **Source**: [Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- **Format**: Pascal VOC XML annotations
- **Classes**: with_mask, without_mask, mask_weared_incorrect
- **Size**: ~853 images

### 2. Face Mask 12k Images Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
- **Format**: Directory structure (train/val/test)
- **Classes**: with_mask, without_mask, mask_weared_incorrect
- **Size**: ~12,000 images

### 3. Medical Mask Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/niharika41298/medical-masks-dataset)
- **Format**: YOLO annotations
- **Classes**: with_mask, without_mask
- **Size**: ~6,000 images

## Model Support

### Face Detectors
- **YuNet**: OpenCV's DNN-based face detector (ONNX format)
- **Haar Cascade**: Traditional cascade classifier (XML format)

### Mask Classifiers
- **NumPy CNN**: Custom CNN implementation (32x32 grayscale input)
- **PyTorch ResNet18**: Transfer learning approach (224x224 RGB input)

## Evaluation Metrics

### Detection Metrics
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)  
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **IoU**: Intersection over Union for bounding box matching

### Classification Metrics
- **Accuracy**: Correct predictions / Total predictions
- **Per-class Precision/Recall/F1**: Individual class performance
- **Macro/Micro Averages**: Aggregated performance measures
- **Confusion Matrix**: Detailed error analysis
- **ROC Curves**: Threshold-independent performance

### Performance Metrics
- **FPS**: Frames per second (real-time capability)
- **Latency**: Average processing time per frame/face
- **Memory Usage**: Peak and average memory consumption
- **Scalability**: Performance vs. number of faces/batch size

## Example Results

```
FACE MASK DETECTION SYSTEM - COMPREHENSIVE EVALUATION REPORT
================================================================================
Evaluation Date: 2024-11-28 23:15:00

SYSTEM CONFIGURATION
----------------------------------------
Detector: YuNet (ONNX)
Classifier: PyTorch ResNet18 (PyTorch)
Classes: no_mask, with_mask, incorrect_mask
Dataset: andrewmvd (853 samples)

DETECTION PERFORMANCE
----------------------------------------
Precision: 0.892
Recall: 0.856
F1 Score: 0.874
Average Precision: 0.901

CLASSIFICATION PERFORMANCE
----------------------------------------
Overall Accuracy: 0.934
Macro F1: 0.925
Macro Precision: 0.928
Macro Recall: 0.923

SPEED PERFORMANCE
----------------------------------------
Pipeline FPS: 28.5
Average Processing Time: 35.09 ms
Real-time Target (30 FPS): ✗ NOT ACHIEVED

MEMORY USAGE
----------------------------------------
Peak Memory: 245.3 MB
Average Memory per Image: 1.23 MB
Average Memory per Face: 0.45 MB
Memory Leak Detection: ✓ NO LEAKS DETECTED

OVERALL ASSESSMENT
----------------------------------------
System Readiness: 75% (3/4 checks passed)
Status: ✓ READY FOR DEPLOYMENT

RECOMMENDATIONS
----------------------------------------
• Optimize for real-time performance - consider model quantization or hardware acceleration
```

## Advanced Usage

### Custom Dataset Integration

```python
from evaluation.datasets.dataset_loaders import DatasetLoader

# Create custom dataset loader
class CustomDatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        # Initialize your dataset
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Return (image, target) where target has 'boxes' and 'labels'
        return image, target

# Register with DatasetLoader
DatasetLoader.register_loader('custom', CustomDatasetLoader)
```

### Custom Model Integration

```python
from evaluation.classifiers.classifier_wrapper import ClassifierWrapper

# Create custom classifier
class CustomClassifier:
    def __init__(self, model_path, class_names):
        # Initialize your model
        pass
    
    def predict_single(self, image):
        # Return (predicted_class, probabilities)
        return pred_class, probs
    
    def predict(self, images):
        # Return list of (predicted_class, probabilities)
        return results

# Register with ClassifierWrapper  
ClassifierWrapper.register_classifier('custom', CustomClassifier)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python tests/run_tests.py`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **YuNet**: OpenCV's face detection model
- **AndrewMVD**: Face mask detection dataset
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@misc{facemask_eval_2024,
  title={Face Mask Detection System - Evaluation Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/clydebaron2000/RealTimeFaceMaskDetection}
}
