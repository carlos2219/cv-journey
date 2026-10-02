# Detection metrics: from IoU to AP

How an object detector is scored, built up one idea at a time with a small example you can count by hand.
Short definitions are in [concepts.md](concepts.md#detection-metrics-step-4).

## 1. IoU: does one box fit the object?

**IoU (Intersection over Union)** = the area the predicted box and the ground-truth box share ÷ the total area they cover.

- 1.0 = perfect overlap · 0.5 = roughly right · 0 = no overlap

A prediction **counts as correct** only if its IoU with a real object reaches a chosen threshold (for example 0.5) and
the class is right.

## 2. Counting hits and mistakes

| Outcome | Meaning | Thermal-drone example |
|---|---|---|
| **TP**, true positive | the box matches a real object | a box on a real person |
| **FP**, false positive | a box with no real object under it, or one that fits too poorly | a warm rock labeled "Person" |
| **FN**, false negative | a real object that no box found | a person the model missed |

## 3. Precision and recall

- **Precision = TP / (TP + FP):** of the boxes I drew, how many are real? → measures false alarms
- **Recall = TP / (TP + FN):** of the real objects, how many did I find? → measures misses

## 4. The confidence cutoff creates a trade-off

Every box has a confidence score (0–1). We keep only the boxes above a cutoff.

- Strict cutoff → few, very sure boxes → **high precision, low recall**
- Loose cutoff → many boxes → **high recall, lower precision**

So a single (P, R) pair depends on which cutoff you picked. We need a score that covers every cutoff.

## 5. Worked example: building the PR curve

One image contains **4 real people**. The model outputs **6 boxes**, sorted by confidence:

| # | Confidence | Real person? |
|---|---|---|
| 1 | 0.95 | ✅ TP |
| 2 | 0.90 | ✅ TP |
| 3 | 0.80 | ❌ FP (warm rock) |
| 4 | 0.70 | ✅ TP |
| 5 | 0.50 | ❌ FP |
| 6 | 0.30 | ✅ TP |

Lower the cutoff one box at a time and compute P and R each time (recall = TP ÷ 4):

| Boxes kept | TP | FP | Precision | Recall |
|---|---|---|---|---|
| top 1 | 1 | 0 | 1.00 | 0.25 |
| top 2 | 2 | 0 | 1.00 | 0.50 |
| top 3 | 2 | 1 | 0.67 | 0.50 |
| top 4 | 3 | 1 | 0.75 | 0.75 |
| top 5 | 3 | 2 | 0.60 | 0.75 |
| top 6 | 4 | 2 | 0.67 | 1.00 |

Patterns:
- **Recall never goes down.** Keeping more boxes can't lose an object you already found.
- **Precision drops each time a false alarm gets in** (rows 3 and 5).

Plot recall (x) against precision (y). This is the **PR curve**:

```
Precision
1.00 ●───●
     │   │
0.75 │   ●···●
0.67 │   ●   │   ●
0.60 │       ●   │
     │           │
     └───┬───┬───┬───┬── Recall
       0.25 0.5 0.75 1.0
```

## 6. AP: the area under the PR curve

**AP (Average Precision)** = the area under the PR curve, from 0 to 1. It is one number that says
**how well the model ranks real objects above false alarms, across every possible cutoff.**

In practice the curve is first turned into a staircase: the precision at each recall becomes the best precision
reached at that recall *or any higher recall*. For the example:

| Recall range | Precision used | Area |
|---|---|---|
| 0 – 0.50 | 1.00 | 0.500 |
| 0.50 – 0.75 | 0.75 | 0.188 |
| 0.75 – 1.00 | 0.67 | 0.167 |
| | **AP** | **≈ 0.85** |

Reference points:
- **Perfect model:** every real object is ranked above every false alarm → precision stays at 1.00 → AP = 1.0
- **Bad model:** false alarms are mixed in early → the curve sags → AP is small

### What if the rock had low confidence?

Suppose the rock (box 3) gets confidence **0.20** instead of 0.80, so it drops to last place. The new order is
TP, TP, TP, FP, TP, FP:

| Recall range | Precision used | Area |
|---|---|---|
| 0 – 0.75 | 1.00 | 0.750 |
| 0.75 – 1.00 | 0.80 | 0.200 |
| | **AP** | **≈ 0.95** |

AP goes **up** (0.85 → 0.95). The model found the same 4 people with the same 2 false alarms. The only change is
that the false alarm now comes *after* the real objects, so precision stays high while recall builds up. **AP rewards a
model whose confidence scores are trustworthy.**

## 7. Where to see this in my runs

`yolo detect val model=... data=...` saves `BoxPR_curve.png` with one PR curve per class plus the average. (n640 was
stopped early, so its end-of-training plots were never generated.)

## 8. mAP: average over classes

Compute one AP per class, then take the mean:

```
AP(Person), AP(Car), AP(Bicycle), AP(OtherVehicle)  →  mean = mAP
```

Every class counts equally, whatever its number of examples. OtherVehicle (148 labels) has as much weight as Person
(12,312 labels), so one weak, rare class can pull mAP down a lot. **Always check the per-class table, not just the total.**

## 9. mAP50 vs mAP50-95: average over IoU thresholds

The IoU threshold decides what counts as a TP, so changing it changes the whole PR curve.

- **mAP50:** threshold 0.5. A roughly placed box still counts. Measures *"did it find the object?"*
- **mAP50-95:** repeat at 0.50, 0.55, … 0.95 (10 thresholds) and average. At high thresholds a slightly misplaced box
  becomes a FP, *and* its object becomes a FN. Measures *"are the boxes also tight?"*

A large gap (n640: 0.882 vs 0.595) means **the model finds the objects but its boxes are not tight.**

### Why small objects suffer: the same 2-pixel error at two sizes

| Object | Box | Shifted 2 px right | Shared area | Total area | IoU |
|---|---|---|---|---|---|
| Person at ~100 m | 10×20 px | 8×20 shared | 160 | 240 | **0.67** |
| Car | 40×80 px | 38×80 shared | 3040 | 3360 | **0.90** |

The same 2-pixel error barely affects a large box but costs a small box a third of its IoU. In HIT-UAV most people are
tiny, so mAP50-95 is harsh on them. That's partly a property of the metric, not just a weakness of the model.

### Which metric matters depends on the job

| Use of the detection | Box tightness matters? | Watch |
|---|---|---|
| Alert: "person at this location" (search and rescue, intrusion) | No. A few pixels ≈ tens of cm on the ground, much smaller than GPS error | **mAP50, recall** |
| Measuring size or distance from the box, cropping for a second model, separating people standing close together | Yes | **mAP50-95** |

For a drone that reports people, **missing a person (FN) is the costly error**, so recall at IoU 0.5 is the number to watch.
