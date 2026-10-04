# cv-journey

My hands-on computer vision learning path, from the basics of OpenCV to deep learning inference,
with the goal of deploying models on edge hardware (NVIDIA Jetson). Every step ends with a working
script and measured results. I'm learning with Claude Code as a tutor: I write and run the code myself,
and it reviews and explains.

## Roadmap

| Step | Topic | Status |
|---|---|---|
| 0 | Environment: uv, PyTorch + CUDA, OpenCV | ✅ |
| 1 | Images are arrays: live video, color spaces, color-based tracking | ✅ |
| 2 | Camera geometry: calibration, undistortion, measuring on a plane | ✅ |
| 3 | Deep learning inference: YOLO on the webcam, latency per stage | ✅ |
| 4 | Train my own detector on a custom dataset | ✅ |
| 5 | Edge deployment: ONNX, TensorRT, Docker, Jetson | ⏳ |

Notes: [concepts glossary](docs/concepts.md) · [experiment log](EXPERIMENTS.md) · [ML workflow](docs/ml-workflow.md) · [commands](docs/commands.md)

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (the scripts also run on CPU).

```bash
git clone git@github.com:carlos2219/cv-journey.git
cd cv-journey
uv sync
uv run python -m cv_journey.check_env
```

## Scripts

Run each one from the repo root with `uv run python -m cv_journey.<name>`. Press `q` to quit.

| Script | What it does | Keys |
|---|---|---|
| `check_env` | Checks CUDA, PyTorch, OpenCV and the webcam | |
| `live_view` | Live webcam view with a smoothed FPS counter | |
| `color_spaces` | BGR / gray / HSV views, inspects the center pixel | `p` print pixel |
| `tracker` | Tracks a red object: HSV mask → morphology → contours → centroid | `p` print HSV |
| `calib_capture` | Detects a 9×6 checkerboard live and saves calibration images | `c` capture |
| `calibrate` | Computes the camera intrinsics and distortion, saves `config/webcam_calib.json` | |
| `measure` | Undistorts the image and measures distances in mm on the checkerboard plane (homography) | click 2 points |
| `detect` | YOLO object detection with latency per stage (pre-process, inference, post-process) | `d` GPU/CPU, `p` print boxes |

## Results

**Hardware:** Intel Core i7-12800HX (24 threads), NVIDIA RTX 3080 Ti Laptop GPU (16 GB), built-in 640×480 webcam at 30 FPS.
**Software:** Ubuntu 24.04, Python 3.12, PyTorch 2.14 + CUDA 13.0, OpenCV 5.0, Ultralytics 8.4.

### Color tracking (Step 1)
- Hue stays stable under lighting changes, while BGR values change a lot.
- Skin and my red object share the same hue (H ≈ 170). Only saturation separates them
  (object S ≈ 180, skin S ≈ 40–60), so a minimum S threshold is what rejects my face.
- Limit found: under strong light the object washes out (S ≈ 60, V = 255) and overlaps with skin.
  No threshold can separate them, which is a good reason to use learned detectors.

### Camera calibration (Step 2)
Calibrated from 20 checkerboard images (9×6 inner corners, 20 mm squares, shown on an iPad).

| Parameter | Value |
|---|---|
| Reprojection error (RMS) | **0.112 px** |
| Focal length fx, fy | 510.0, 509.7 px |
| Optical center cx, cy | 321.0, 232.9 px |
| Field of view (H × V) | ≈ 64° × 50° |

Measuring with a homography on the board plane reproduced a known 160 mm distance accurately.

### YOLO inference latency (Step 3)
Pretrained COCO models, webcam input 640×480, batch size 1. Values are in ms per frame, after warm-up.

| Model | Device | Pre-process | Inference | Post-process | FPS |
|---|---|---|---|---|---|
| YOLO11n | GPU | 1.2 | 8–10 | 1.0 | 30 (camera-bound) |
| YOLO11n | CPU | 0.5 | 16–18 | 0.5 | 30 (camera-bound) |
| YOLO11x | GPU | 0.5 | 16–18 | 0.6 | 30 (camera-bound) |
| YOLO11x | CPU | 0.6 | ~270 | 0.6 | ~5 |

Conclusions:
- When the camera limits the frame rate, FPS hides the model's speed. The latency per stage is the metric that matters.
- Going from YOLO11n to YOLO11x (about 30× more compute) makes CPU inference about 16× slower, but GPU inference only about 2× slower.
  The nano model is too small to keep the GPU busy.
- Choosing a model means picking the largest one that fits the latency budget on the target hardware.
  On a Jetson, with a much weaker CPU, GPU inference with TensorRT will be essential.

### Thermal UAV detector (Step 4)
Fine-tuned YOLO11 (COCO-pretrained) on the [HIT-UAV](https://github.com/suojiashun/hit-uav-infrared-thermal-dataset)
infrared dataset: 2029 / 290 / 579 train / val / test images taken from a drone at 60–130 m, 4 classes.
Full run history: [EXPERIMENTS.md](EXPERIMENTS.md).

| Run | Model | imgsz | Person recall (val) | mAP50 (val) | mAP50-95 (val) | Inference |
|---|---|---|---|---|---|---|
| **n640** | YOLO11n | 640 | **0.885** | 0.882 | 0.594 | **1.8 ms** |
| n1280 | YOLO11n | 1280 | 0.872 | 0.891 | 0.615 | 5.7 ms |
| s640 | YOLO11s | 640 | 0.858 | 0.898 | 0.612 | 3.1 ms |

**Chosen model: n640.** Final score on the held-out **test** split: mAP50 **0.886**, Person recall **0.890**,
Person mAP50 0.927 (close to val, so the model generalizes).

Conclusions:
- For people alerts, missed people are the critical error, so recall decides. A bigger model or a higher
  resolution gave tighter boxes (mAP50-95) but did **not** find more people, so the smallest, fastest model wins.
- Recall can still be raised without retraining by lowering the confidence threshold, at the cost of more false alarms.
- OtherVehicle (148 training examples, 0.6%) scores worst: too few examples to learn from.
