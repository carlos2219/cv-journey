import json
from pathlib import Path

import cv2
import numpy as np

PATTERN = (9,6) #inner corners
SQUARE_SIZE_MM = 20
IMG_DIR = Path("data/calib")
OUT_FILE = Path("config/webccam_calib.json")

#Stop refining a corner after 30 iterations, or when it moves less than 0.001px
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

#Real corner positions on the board, in mm
cols, rows = PATTERN
objp = np.array(
    [[x * SQUARE_SIZE_MM, y * SQUARE_SIZE_MM, 0] for y in range(rows) for x in range(cols)],
    dtype=np.float32,
)

#Loop over the images
object_points = [] # one objp per good image
image_points = []   # the detected corners per good image

for path in sorted(IMG_DIR.glob("*.png")):
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray,PATTERN, None)
    if not found:
        print(f"Skipped {path.name}: board not found")
        continue

    corners = cv2.cornerSubPix(gray, corners, (11,11),(-1, -1), CRITERIA)
    object_points.append(objp)
    image_points.append(corners)

print(f"Using {len(image_points)} images")
if not image_points:
    print("No usable images found. Check PATTERN and the images in data/calib")
    raise SystemExit(1)

#Calibrate
image_size = gray.shape[::-1] #width, height
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

#Print results
np.set_printoptions(precision=3, suppress=True) # 3 decimals, no scientific notation
print(f"RMS reprojection error: {rms:.3f} px")
print("K =\n", K)
print("dist =",dist.ravel())

#Save to a JSON file
result = {
    "image_size":list(image_size),
    "rms": rms,
    "K": K.tolist(),
    "dist": dist.ravel().tolist(),
}
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(result, f, indent=2)
print(f"Save {OUT_FILE}")