# Experiments

Log of every training run. `runs/` is gitignored, so this file is the record that survives.
Newest first. Dataset: HIT-UAV infrared (train 2029 / val 290 / test 579 images, 640x512, 4 classes).

## Summary

| Run   | Model   | imgsz | Epochs (done/planned) | Best ep | P     | R     | mAP50 | mAP50-95 | Inference* | Train time |
|-------|---------|-------|-----------------------|---------|-------|-------|-------|----------|------------|------------|
| n1280 | yolo11n | 1280  | 100 / 100             | 74      | 0.908 | 0.844 | 0.891 | 0.615    | 5.7 ms     | ~46 min    |
| n640  | yolo11n | 640   | 81 / 100 (stopped)    | 73      | 0.841 | 0.837 | 0.882 | 0.594    | 1.8 ms     | ~10.5 min  |
| smoke | yolo11n | 640   | 1 / 1                 | 1       | 0.559 | 0.288 | 0.254 | 0.115    | –          | 12.5 s     |

Metrics are on the **val** split, all classes combined, from `yolo detect val` on `best.pt`.
\*Inference = model time per image during val (batch 16, RTX 3080 Ti Laptop). This is not the speed for a single live frame.

### Per class (val, best.pt)

| Class (val instances) | Run   | P     | R     | mAP50 | mAP50-95 |
|-----------------------|-------|-------|-------|-------|----------|
| Person (1168)         | n640  | 0.881 | 0.885 | 0.914 | 0.479    |
|                       | n1280 | 0.896 | 0.872 | 0.920 | 0.522    |
| Car (719)             | n640  | 0.933 | 0.972 | 0.982 | 0.738    |
|                       | n1280 | 0.946 | 0.967 | 0.987 | 0.748    |
| Bicycle (554)         | n640  | 0.898 | 0.825 | 0.909 | 0.528    |
|                       | n1280 | 0.903 | 0.872 | 0.933 | 0.582    |
| OtherVehicle (12)     | n640  | 0.651 | 0.667 | 0.722 | 0.630    |
|                       | n1280 | 0.888 | 0.667 | 0.726 | 0.609    |

---

## 2026-10-02 — n1280
- **Goal:** Test the hypothesis that small, far-away people are detected better when the model sees more pixels.
- **Change vs n640:** `imgsz=1280` instead of 640. Everything else the same. Ran all 100 epochs.
- **Command:** `uv run yolo detect train data=config/hit-uav.yaml model=yolo11n.pt epochs=100 imgsz=1280 batch=16 name=n1280`
- **Result:** Best epoch 74: P 0.908, R 0.844, mAP50 0.891, mAP50-95 0.615. ~28 s/epoch, 9.7 GB GPU memory, 81 °C.
- **Notes:**
  - **Person recall did not improve** (0.885 → 0.872, within noise), and Person mAP50 changed by +0.006 (noise). The
    hypothesis "more pixels → find more people" is **not supported**: n640 already found most of them.
  - **Boxes got tighter:** Person mAP50-95 rose from 0.479 to 0.522 (+0.043), and overall mAP50-95 from 0.594 to 0.615. More
    pixels help the model place the edges of small boxes, which matches the 2-pixel IoU table in
    docs/detection-metrics.md.
  - **Bicycle improved the most** (R 0.825 → 0.872, mAP50-95 +0.054). Bicycles are thin, so they may benefit from detail.
  - **OtherVehicle numbers are not reliable:** val has only 12 of them, so one object = 8% recall. R 0.667 = 8/12 in
    both runs. Don't draw conclusions from this class.
  - **Cost:** 3.2× slower inference (1.8 → 5.7 ms) and 4.4× longer training. On a Jetson this difference is large.
  - mAP50 peaked at epoch 74 and ended at 0.863 at epoch 100; `best.pt` keeps epoch 74.
  - Caveat: only one run per setting, and val was also used to pick the best epoch. The test split will give the final
    honest score.
  - *Open question: for a drone that alerts on people, is n1280 worth 3× the inference time?*
- **Decision:** Train `s640` (bigger model, same resolution) to see whether model capacity helps more than resolution.

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
  - Per class (from `val`, see table above): Car is best (mAP50 0.982), Person has the loosest boxes (mAP50-95 0.479),
    and OtherVehicle is weakest (mAP50 0.722).
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
