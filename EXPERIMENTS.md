# Experiments

Log of every training run. `runs/` is gitignored, so this file is the record that survives.
Newest first. Dataset: HIT-UAV infrared (train 2029 / val 290 / test 579 images, 640x512, 4 classes).

## Summary

| Run   | Model   | imgsz | Epochs (done/planned) | Best ep | P     | R     | mAP50 | mAP50-95 | Train time |
|-------|---------|-------|-----------------------|---------|-------|-------|-------|----------|------------|
| n640  | yolo11n | 640   | 81 / 100 (stopped)    | 73      | 0.841 | 0.836 | 0.882 | 0.595    | ~10.5 min  |
| smoke | yolo11n | 640   | 1 / 1                 | 1       | 0.559 | 0.288 | 0.254 | 0.115    | 12.5 s     |

Metrics are on the **val** split, all classes combined.

---

## 2026-10-01 — n640
- **Goal:** First real baseline: how well does the smallest YOLO11 model detect objects in thermal UAV images at native resolution?
- **Change vs smoke:** 100 epochs planned instead of 1. Everything else the same.
- **Command:** `yolo detect train data=config/hit-uav.yaml model=yolo11n.pt epochs=100 imgsz=640 batch=16 name=n640`
  (defaults: optimizer auto, lr0 0.01, AMP on, seed 0)
- **Result:** Stopped by hand at epoch 81 (~7.7 s/epoch). Best epoch 73: P 0.841, R 0.836, mAP50 0.882, mAP50-95 0.595.
  `best.pt` = epoch 73.
- **Notes:**
  - Val metrics are noisy from one epoch to the next: epoch 74 fell to mAP50 0.849 and epoch 81 came back to 0.882. A
    difference smaller than about 0.03 between two runs may be noise.
  - Train losses were still going down at epoch 81 (box 1.15, cls 0.58). Val losses were flat (box ~1.23–1.25). The
    model is near its limit at this setup. More epochs would probably add only a little.
  - mAP50 (0.882) is much higher than mAP50-95 (0.595): the model finds the objects but its boxes are not tight. Small
    objects make this worse: on a 10×20 px person, a 2 px shift drops IoU to about 0.67. For alerting on people, mAP50 and
    recall are the metrics that matter (see docs/detection-metrics.md).
  - Per-class results are not checked yet. Expect OtherVehicle (0.6% of labels) to score poorly.
- **Decision:** Run `yolo detect val` on `best.pt` to get the per-class table. Then test the hypothesis that small,
  far-away people need more pixels: train `n1280`, and train `s640` to compare a bigger model against more resolution.

## 2026-10-01 — smoke
- **Goal:** Check that the whole pipeline works: dataset config, symlinked folders, labels, GPU training, and saving.
- **Change:** First run.
- **Command:** `yolo detect train data=config/hit-uav.yaml model=yolo11n.pt epochs=1 imgsz=640 batch=16 name=smoke`
- **Result:** Finished in 12.5 s. P 0.559, R 0.288, mAP50 0.254, mAP50-95 0.115.
- **Notes:** The numbers don't matter after only 1 epoch. What matters is that the pipeline runs without errors, so the
  config and labels are read correctly.
- **Decision:** The pipeline works. Start the full run (n640).
