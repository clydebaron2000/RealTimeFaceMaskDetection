
# Real-time Face Mask Detection Demo

## Quick Start - Run the Interactive Demo

```bash
# Navigate to yewon_pipeline directory
cd yewon_pipeline

# Run the interactive launcher
python run_realtime_demo.py
```

This will show a menu where you can:
1. Test detection-only mode (no classification)
2. Load and test different classifier models
3. Switch between detectors (Haar, YuNet) in real-time
4. Dynamically combine any detector with any classifier

## Direct Demo Usage

```bash
# Basic detection only
python realtime_demo.py

# With specific detector
python realtime_demo.py --detector yunet

# With a specific classifier model
python realtime_demo.py --classifier runs_12k/cylde_custom/best.pth

# With both detector and classifier
python realtime_demo.py --detector haar --classifier runs_12k/cylde_custom/best.pth
```

## Real-time Keyboard Controls

While the demo is running, use these keys:
- `1`: Switch to Haar detector
- `2`: Switch to YuNet detector
- `3/4/5`: Load different classifier models
- `0`: Disable classifier (detection only)
- `I`: Toggle info display
- `F`: Toggle FPS counter
- `S`: Save screenshot
- `Q`: Quit

## Model Compatibility

The system automatically discovers and can load:
- **Detectors**: Haar Cascade, YuNet
- **Classifiers**: Any `.pth` model from:
  - `runs_12k/` (trained on 12k dataset)
  - `runs_eval/` (evaluation models)
  - Custom models following the pipeline format

The unified model loader (`unified_model_loader.py`) handles:
- Automatic model architecture detection
- Dynamic input size adjustment
- Real-time model switching
- Flexible preprocessing pipelines

---

# 1. Fine-tuning classifiers (+ hyperparameter tuning through grid)
```
CUDA_VISIBLE_DEVICES=6,7 python run_experiments.py   --launcher torchrun   --nproc_per_node 2   --dataset_mode mask12k   --data_root "/local-ssd/yl3427/test/input/fm_12k"   --output_root "runs_12k"   --num_workers 4
```

```
CUDA_VISIBLE_DEVICES=4,5 python run_experiments.py   --launcher torchrun   --nproc_per_node 2   --master_port 29601   --dataset_mode cropped   --data_root "/local-ssd/yl3427/test/input/fm_detection"   --output_root "runs_cropped"   --num_workers 4
```

# 2. evaluate detectors (recall)
```
mkdir -p runs_eval

for det in yunet haar mtcnn retinaface; do
  python eval_detector_only.py \
    --data_root "/local-ssd/yl3427/test/input/fm_detection" \
    --detector "$det" \
    --device cuda \
    --iou_thresh 0.5 \
    --yunet_onnx "face_detection_yunet_2023mar.onnx" \
    --haar_xml "haarcascade_frontalface_default.xml" \
    --retina_thresh 0.8 \
    --results_csv "runs_eval/det_only.csv"
done
```


# 3. Evaluate the entire pipeline (detector x classifier)

```
CUDA_VISIBLE_DEVICES=6 \
python run_eval_grid.py \
  --data_root "/local-ssd/yl3427/test/input/fm_detection" \
  --output_root "runs_cropped" \
  --device cuda \
  --results_csv "runs_eval/det_cls_grid_cropped.csv"

```

```
CUDA_VISIBLE_DEVICES=7 \
python run_eval_grid.py \
  --data_root "/local-ssd/yl3427/test/input/fm_detection" \
  --output_root "runs_12k" \
  --device cuda \
  --results_csv "runs_eval/det_cls_grid_12k.csv"

```