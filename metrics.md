## Metrics

The following metrics are used to compare the four models (yolov8n, yolov5n, yolov8m, yolov8s)

- **mAP (mean Average Precision)**
Overall accuracy measure for object detection. It averages precision across recall levels and IoU thresholds, showing how well the model balances finding all objects (recall) while avoiding false positives (precision).This is mAP@[0.5:0.95], i.e. the mean of AP values computed at IoU thresholds from 0.50 to 0.95 in steps of 0.05 (the COCO-style metric)

- **latency_p50_ms (50th percentile latency)**
Median single-image inference time. Half of predictions complete faster than this value. Useful for understanding typical performance.

- **latency_p90_ms (90th percentile latency)**
Upper-tail latency. 90% of images finish inference within this time. Highlights performance under heavier loads or on harder inputs.

- **latency_p99_ms (99th percentile latency)**
Extreme latency bound. 99% of requests complete within this time. Critical for latency-sensitive, real-time systems.

- **latency_mean_ms (mean latency)**
Average inference time across all images. Easy to compare models but can be skewed by outliers.

- **mean_precision**
Fraction of detections that are correct across all classes (how many predicted objects were right). High precision means fewer false alarms.

- **mean_recall**
Fraction of ground-truth objects correctly detected (how many real objects were found). High recall means fewer misses.

- **AP50 (Average Precision @ IoU=0.50)**
Accuracy when a predicted box is considered correct if it overlaps the ground truth by at least 50%. A more lenient match criterion.

- **AP75 (Average Precision @ IoU=0.75)**
Stricter version of AP. Requires tighter overlap (75%) for a detection to count as correct. A better test of localization quality.


## Additional Details

### mAP (mean Average Precision)

We report mAP as the main evaluation metric across models.

- AP (Average Precision) measures how well a detector ranks correct boxes above incorrect ones. Detections are sorted by confidence score, and precision–recall is computed as boxes are added. AP is the area under this curve.

- IoU thresholds decide if a predicted box matches a ground truth (e.g. IoU ≥ 0.5).

- mAP is AP averaged:

    - mAP@0.5 → average AP across all classes at IoU=0.5 (PASCAL VOC style).

    - mAP@[.5:.95] → average AP across IoUs from 0.5 to 0.95 in steps of 0.05 (COCO style, stricter on localization).

Higher mAP means the model not only detects more objects but also assigns higher confidence to correct detections and localizes them more precisely.
