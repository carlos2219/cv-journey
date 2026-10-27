"""Turn HIT-UAV boxes into a classification dataset: one sample per labeled object.

Step A: build the sample index (no PyTorch yet).
"""
from pathlib import Path

from cv_journey.show_labels import DATA_DIR, label_path_for, read_labels

NUM_CLASSES = 3  # Person, Car, Bicycle; OtherVehicle (3) dropped: only 148 boxes

Sample = tuple[Path, int, tuple[float, float, float, float]]  # (img_path, cls, (cx, cy, w, h))


def build_index(split: str) -> list[Sample]:
    """List every box of the kept classes in a split, one entry per box."""
    samples = []  # before the loop, or it resets on every image
    img_paths = sorted((DATA_DIR / "normal_json" / split).glob("*.jpg"))
    for img_path in img_paths:
        boxes = read_labels(label_path_for(img_path))
        for cls, cx, cy, bw, bh in boxes:  # unpack each box in the for line
            if cls >= NUM_CLASSES:
                continue  # skip OtherVehicle
            samples.append((img_path, cls, (cx, cy, bw, bh)))
    return samples  # function level: after ALL images


if __name__ == "__main__":
    index = build_index("train")
    print(f"{len(index)} train samples")
    print("first:", index[0])
