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

counts = {}
height_sums = {}
avg_heights = {}
groups = {}

for image, cls, w, h in boxes:
    name = CLASS_NAMES[cls]
    counts[name] = counts.get(name, 0) + 1
    height_sums[name] = height_sums.get(name, 0) + h

    if image not in groups:
        groups[image] = []
    groups[image].append(cls)

for name, total in height_sums.items():
    avg_heights[name] = total / counts[name]

print(avg_heights)
print(counts)
print(height_sums)
print(groups)


