"""Browse HIT-UAV images with their ground-truth YOLO boxes drawn on top.

Checks that we read the label format correctly before training on it.
Keys: any key = next image, q = quit.
"""
from pathlib import Path

import cv2

# Built from this file's location, so it works from any folder and on any machine
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "hit-uav"
CLASS_NAMES = ["Person", "Car", "Bicycle", "OtherVehicle"]  # index = class ID in the .txt labels
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]  # BGR: green, red, blue, yellow
SPLIT = "val"  # which split to browse: "train", "val" or "test"


def label_path_for(img_path: Path) -> Path:
    """Map an image to its YOLO label: same split and stem, in yolo_labels/, with .txt."""
    split = img_path.parent.name  # "train", "val" or "test"
    return DATA_DIR / "yolo_labels" / split / f"{img_path.stem}.txt"


def read_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read one YOLO label file: one box per line as (class, cx, cy, w, h), normalized 0-1."""
    boxes = []
    with open(label_path) as f:
        for line in f:
            cls, cx, cy, bw, bh = map(float, line.split())
            boxes.append((int(cls), cx, cy, bw, bh))  # class must be int to index lists
    return boxes


def draw_labels(img, boxes: list[tuple[int, float, float, float, float]]) -> None:
    """Draw each box with its class name on img (in place)."""
    h, w = img.shape[:2]
    for cls, cx, cy, bw, bh in boxes:
        # YOLO center + size (normalized) -> OpenCV corners (pixels).
        # round(), not int(): int() truncates 170.999... to 170
        x1 = round((cx - bw / 2) * w)
        y1 = round((cy - bh / 2) * h)
        x2 = round((cx + bw / 2) * w)
        y2 = round((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[cls], 1)
        cv2.putText(img, CLASS_NAMES[cls], (x1, y1 - 3),  # text just above the box
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS[cls], 1)


def main() -> None:
    img_paths = sorted((DATA_DIR / "normal_json" / SPLIT).glob("*.jpg"))
    print(f"{len(img_paths)} images in {SPLIT} (any key = next, q = quit)")

    try:
        for img_path in img_paths:
            boxes = read_labels(label_path_for(img_path))
            img = cv2.imread(str(img_path))  # cv2 wants a str, not a Path
            draw_labels(img, boxes)

            # Filename = daynight_altitude_angle_0_frame, e.g. 0_60_30_0_06440
            parts = img_path.stem.split("_")
            info = f"{img_path.name} alt={parts[1]}m angle={parts[2]}deg boxes={len(boxes)}"
            cv2.putText(img, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            cv2.imshow("labels", img)  # same window name -> reuses the window
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()  # always runs, even on error or Ctrl+C


if __name__ == "__main__":
    main()
