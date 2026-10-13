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


def count_by_class(boxes):
    """Return {class_name: number_of_boxes}."""
    counts = {}  # accumulator: created BEFORE the loop
    for image, cls, w, h in boxes:
        name = CLASS_NAMES[cls]
        counts[name] = counts.get(name, 0) + 1  # .get(..., 0): 0 if name is new
    return counts


def avg_height_by_class(boxes):
    """Return {class_name: average box height in px}."""
    avg_heights = {}
    height_sums = {}
    counts = count_by_class(boxes)  # reuse the function above instead of copying it
    for image, cls, w, h in boxes:
        name = CLASS_NAMES[cls]
        height_sums[name] = height_sums.get(name, 0) + h
    # second loop: only after ALL sums are complete can we divide
    for name, total in height_sums.items():
        avg_heights[name] = total / counts[name]
    return avg_heights


def group_by_image(boxes):
    """Return {image_name: [cls, cls, ...]} in the order boxes appear."""
    groups = {}
    for image, cls, w, h in boxes:
        if image not in groups:  # first time we see this image
            groups[image] = []   # start its empty list
        groups[image].append(cls)
    return groups


def box_areas(boxes, cls_filter=None):
    """Return w * h of every box, or only boxes of class cls_filter if given."""
    # None = "no filter": `cls_filter is None` is True, so every box passes.
    # With a class id (e.g. 1), only boxes whose cls matches pass.
    return [w * h for image, cls, w, h in boxes
            if cls_filter is None or cls == cls_filter]


if __name__ == "__main__":
    print(count_by_class(boxes))       # {'Person': 4, 'Car': 2, 'Bicycle': 2}
    print(avg_height_by_class(boxes))  # {'Person': 24.25, 'Car': 27.5, 'Bicycle': 14.0}
    print(group_by_image(boxes))       # {image: [cls, ...], ...}
    print(box_areas(boxes))            # all 8 areas (default: no filter)
    print(box_areas(boxes, 1))         # cars only: [1400, 440]
