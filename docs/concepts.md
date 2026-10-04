# Concepts

My reference notes for the terms used in this repo, written in plain words with numbers from my own runs.
Organized by the step where each term first came up.

**Contents:** [Images and color](#images-and-color-step-1) ·
[Camera geometry](#camera-geometry-step-2) · [Inference](#inference-step-3) ·
[Datasets and labels](#datasets-and-labels-step-4) · [Training](#training-step-4) ·
[Detection metrics](#detection-metrics-step-4) · [Edge vs cloud](#edge-vs-cloud)

---

## Images and color (Step 1)

- **Image as an array:** an image is a NumPy array of shape `(h, w, c)`: height (rows), width (columns), and
  channels. My webcam gives `(480, 640, 3)`. Indexing is `img[y, x]`, so the row comes first.
- **BGR:** OpenCV stores the color channels as Blue, Green, Red, not RGB. Convert before passing an image to
  libraries that expect RGB.
- **HSV:** another way to store color. **H**ue = which color (0–179 in OpenCV), **S**aturation = how strong the
  color is, **V**alue = how bright it is. Hue changes less with lighting than BGR values do, which makes HSV
  better for picking out a color.
- **Mask (thresholding):** a black-and-white image where a pixel is white if its value falls inside a range
  (`cv2.inRange`). Red needs two ranges because its hue wraps around: near 0 and near 179.
- **Morphology (open/close):** cleans up a mask. *Open* removes small white specks. *Close* fills small black holes.
- **Contour:** the outline of a white region in the mask. **Centroid:** the region's center point, calculated
  from image moments.
- **FPS and EMA smoothing:** FPS = 1 / time per frame. An exponential moving average
  (`fps = α·new + (1−α)·fps`, α = 0.1) makes the displayed number steady. It works like a first-order low-pass filter.
- **Camera-bound FPS:** if the camera delivers 30 fps, the loop can't run faster than 30 fps, however fast the
  model is. Measure the model's own time to see its real speed.

## Camera geometry (Step 2)

- **Intrinsics:** the camera's own parameters. **fx, fy** = focal length in pixels (mine ≈ 510), and
  **cx, cy** = the principal point, roughly the image center (mine: 321, 233). Together they form the
  **camera matrix K**.
- **Distortion coefficients:** describe how the lens bends straight lines (radial k1 k2 k3, tangential p1 p2).
  **Undistortion** uses them to straighten the image.
- **Calibration:** estimate K and the distortion coefficients from photos of a known pattern (a checkerboard
  with 9×6 inner corners and 20 mm squares).
- **Reprojection error (RMS):** after calibration, project the board corners back into each image and measure how
  far, in pixels, they land from the detected corners. Mine was 0.112 px, which is very good (below 0.5 is good).
- **Homography:** a 3×3 matrix that maps points on one flat plane to another flat plane. I used it to convert
  pixel coordinates on the board plane into millimeters. It only works for points on that plane.

## Inference (Step 3)

- **Inference:** running a trained model on new input to get predictions. No learning happens during inference.
- **Pre-process / inference / post-process:** the three stages of each frame. Pre-process = resize and normalize
  the image. Inference = the neural network itself. Post-process = filter the boxes (confidence threshold, NMS).
  YOLO11n on my GPU: ~1 / 8–10 / 1 ms.
- **NMS (Non-Maximum Suppression):** the model often predicts several overlapping boxes for one object. NMS keeps
  the box with the highest confidence and removes the boxes that overlap it too much (high IoU, see below).
- **Model size (n/s/m/l/x):** bigger models are more accurate but slower. YOLO11x on CPU took ~270 ms per frame,
  against ~17 ms for 11n.

## Datasets and labels (Step 4)

- **Bounding box:** the axis-aligned rectangle around an object.
- **YOLO label format:** one line per object, `class cx cy w h`: the box center, width and height,
  **normalized** to 0–1 by dividing by the image width or height. To get pixels:
  `x_px = cx · img_w`.
- **Ground truth:** the correct labels drawn by humans. Predictions are compared against them.
- **Train / val / test split:** *train* = images the model learns from. *val* = images used during training to
  check progress and pick the best epoch. *test* = images kept untouched for the final, honest score.
  HIT-UAV: 2029 / 290 / 579.
- **Class balance:** how many examples each class has. HIT-UAV: Person 12312, Car 7311, Bicycle 4980,
  OtherVehicle 148 (0.6%). A class with very few examples usually scores poorly.

## Training (Step 4)

Where each step fits in the standard ML workflow: [ml-workflow.md](ml-workflow.md).

- **Fine-tuning:** start from a model already trained on a large dataset (COCO), then train it further on my
  data. This is much faster and needs less data than training from scratch.
- **Epoch:** one full pass over all the training images. **Batch:** the number of images processed before each
  weight update (16 in my runs).
- **Loss:** a number that measures how wrong the model is. Training adjusts the weights to make it smaller. YOLO has
  three losses: **box** (position and size of boxes), **cls** (wrong class), and **dfl** (a finer
  box-edge loss).
- **Train loss vs val loss:** if train loss keeps going down while val loss stays flat or rises, the model is
  memorizing the training images instead of learning to generalize (**overfitting**). n640 at epoch 81: train
  still going down, val flat → close to its limit.
- **best.pt / last.pt:** the weights from the epoch with the best validation fitness, and from the final epoch.
- **imgsz:** the size the images are resized to before going into the model. A larger size gives small objects more
  pixels but costs more time and memory.

## Detection metrics (Step 4)

Deep dive with a worked example: [detection-metrics.md](detection-metrics.md).

- **IoU:** the overlap between a predicted box and the ground-truth box (shared area ÷ total area). 1 = perfect.
- **TP / FP / FN:** correct detection / false alarm / missed object, judged at an IoU threshold.
- **Precision:** of the boxes drawn, the fraction that are real. **Recall:** of the real objects, the fraction found.
  n640: P 0.841, R 0.836.
- **PR curve:** precision plotted against recall as the confidence cutoff is lowered.
- **AP:** the area under the PR curve, computed per class. High AP = real objects are ranked above false alarms.
- **mAP:** the mean of the per-class APs. Every class counts equally.
- **mAP50 / mAP50-95:** mAP with IoU ≥ 0.5 counted as correct / mAP averaged over IoU 0.50…0.95, which also rewards
  tight boxes. n640: 0.882 / 0.595.

## Edge vs cloud

- **Edge:** running the model on the device itself (Jetson, MCU). It's fast, works offline, and keeps the data
  private. Real-time inference and control loops belong here.
- **Cloud:** running it on remote servers. It's slower to respond but has more compute. Use it for verifying with
  bigger models, dashboards, retraining, and updates.
- **Data flywheel:** the edge flags hard cases → the cloud labels them and retrains → the new model is pushed
  back to the edge. This loop is the core of MLOps.
