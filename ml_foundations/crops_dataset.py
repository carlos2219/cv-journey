"""Turn HIT-UAV boxes into a classification dataset: one sample per labeled object.

build_index lists every box; CropsDataset turns box i into a (1, 32, 32) float tensor + label.
"""
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

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


class CropsDataset(Dataset):

    def __init__(self, split):
        self.samples = build_index(split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, cls, (cx, cy, bw, bh) = self.samples[index]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        x1 = round((cx - bw / 2) * w)
        y1 = round((cy - bh / 2) * h)
        x2 = round((cx + bw / 2) * w)
        y2 = round((cy + bh / 2) * h)
        crop = img[y1:y2, x1:x2]
        crop = cv2.resize(crop, (32, 32))
        x = torch.from_numpy(crop).float() / 255.0
        x = x.unsqueeze(0)
        return x, cls


if __name__ == "__main__":
    ds = CropsDataset("train")
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    xb, yb = next(iter(loader))
    print(xb.shape, yb.shape)
    print(yb[:10])
    print(len(loader))
