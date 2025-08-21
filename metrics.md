## Metrics

## mAP (mean Average Precision)

We report mAP as the main evaluation metric across models.

- AP (Average Precision) measures how well a detector ranks correct boxes above incorrect ones. Detections are sorted by confidence score, and precision–recall is computed as boxes are added. AP is the area under this curve.

- IoU thresholds decide if a predicted box matches a ground truth (e.g. IoU ≥ 0.5).

- mAP is AP averaged:

    - mAP@0.5 → average AP across all classes at IoU=0.5 (PASCAL VOC style).

    - mAP@[.5:.95] → average AP across IoUs from 0.5 to 0.95 in steps of 0.05 (COCO style, stricter on localization).

Higher mAP means the model not only detects more objects but also assigns higher confidence to correct detections and localizes them more precisely.
