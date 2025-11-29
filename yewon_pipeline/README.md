
# 1. Fine-tuning classifiers (+ hyperparameter tuning through grid)

```
DATA_ROOT=/local-ssd/yl3427/test/input/fm_detection

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python run_experiments.py \
  --data_root $DATA_ROOT \
  --output_root /local-ssd/yl3427/test/runs/classifiers \
  --launcher torchrun \
  --num_workers 8 \
  --nproc_per_node 4
```

# 2. evaluate detectors (recall)
```
EVAL_OUT=/local-ssd/yl3427/test/runs/eval

# YuNet
python eval_detector_only.py \
  --data_root $DATA_ROOT \
  --detector yunet \
  --device cuda \
  --iou_thresh 0.5 \
  --results_csv $EVAL_OUT/det_only_summary.csv

# Haar
python eval_detector_only.py \
  --data_root $DATA_ROOT \
  --detector haar \
  --device cuda \
  --iou_thresh 0.5 \
  --results_csv $EVAL_OUT/det_only_summary.csv

# MTCNN
python eval_detector_only.py \
  --data_root $DATA_ROOT \
  --detector mtcnn \
  --device cuda \
  --iou_thresh 0.5 \
  --results_csv $EVAL_OUT/det_only_summary.csv

# RetinaFace
python eval_detector_only.py \
  --data_root $DATA_ROOT \
  --detector retinaface \
  --device cuda \
  --iou_thresh 0.5 \
  --retina_thresh 0.8 \
  --results_csv $EVAL_OUT/det_only_summary.csv
```


# 3. Evaluate the entire pipeline (detector x classifier)
```
CLS_OUT=/local-ssd/yl3427/test/runs/classifiers
EVAL_OUT=/local-ssd/yl3427/test/runs/eval

python run_eval_grid.py \
  --data_root $DATA_ROOT \
  --output_root $CLS_OUT \
  --device cuda \
  --iou_thresh 0.5 \
  --yunet_onnx face_detection_yunet_2023mar.onnx \
  --haar_xml haarcascade_frontalface_default.xml \
  --retina_thresh 0.8 \
  --results_csv $EVAL_OUT/det_cls_grid.csv

```