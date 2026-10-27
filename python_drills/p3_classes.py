CLASS_NAMES = ["Person", "Car", "Bicycle"]

# (image, cls, w_px, h_px)

boxes = [
    ("0_60_30_0_001", 0, 15, 30),
    ("0_60_30_0_001", 1, 40, 35),
    ("0_60_30_0_001", 0, 14, 28),
    ("1_130_90_0_002", 0, 10, 17),
    ("1_130_90_0_002", 2, 18, 12),
    ("1_130_90_0_002", 1, 22, 20),
    ("0_100_60_0_003", 0, 12, 22),
    ("0_100_60_0_003", 2, 25, 16),
]


class BoxSet:
    def __init__(self, boxes):
        self.boxes = boxes

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, index):
        return self.boxes[index]

    def count_by_class(self):
        """Return {class_name: number_of_boxes}."""
        counts = {}
        for image, cls, w, h in self.boxes:
            name = CLASS_NAMES[cls]
            counts[name] = counts.get(name, 0) + 1
        return counts


if __name__ == "__main__":
    bs = BoxSet(boxes)
    print(len(bs))
    print(bs[3])
    print(bs.count_by_class())
    print(bs[-1])
    print(bs[1:3])
    for box in bs:
        print(box)
        