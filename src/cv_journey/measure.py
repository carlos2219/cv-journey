"""Measure real distances (mm) on the checkerboard plane by clicking two points."""
import json
from pathlib import Path

import cv2
import numpy as np

CALIB_FILE = Path("config/webcam_calib.json")
PATTERN = (9, 6)  # inner corners (columns, rows)
SQUARE_SIZE_MM = 20.0

# Stop refining a corner after 30 iterations, or when it moves less than 0.001 px
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# The board corner positions in mm (2D, on the board plane)
cols, rows = PATTERN
board_mm = np.array(
    [[x * SQUARE_SIZE_MM, y * SQUARE_SIZE_MM] for y in range(rows) for x in range(cols)],
    dtype=np.float32,
)

clicks = []  # pixel points clicked by the user (max 2)


def load_calibration(path):
    """Return (K, dist) loaded from a calibration JSON file."""
    with open(path) as f:
        data = json.load(f)
    K = np.array(data["K"])
    dist = np.array(data["dist"])
    return K, dist


def on_mouse(event, x, y, flags, param):
    """Store left-click positions; a 3rd click starts a new measurement."""
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicks) == 2:
            clicks.clear()
        clicks.append((x, y))


def pixel_to_mm(point, H):
    """Map an (x, y) pixel on the board plane to (X, Y) in mm."""
    src = np.array([[point]], dtype=np.float32)  # shape (1, 1, 2)
    return cv2.perspectiveTransform(src, H)[0, 0]  # -> array([X, Y])


def main():
    K, dist = load_calibration(CALIB_FILE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera is not accessible right now")
        raise SystemExit(1)

    # Create the window first, then attach the mouse callback to it
    cv2.namedWindow("Undistorted")
    cv2.setMouseCallback("Undistorted", on_mouse)

    H = None  # last valid homography (image pixels -> board mm)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read from camera")
                break

            # Undistort, then find the board
            undistorted = cv2.undistort(frame, K, dist)
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, PATTERN, None, cv2.CALIB_CB_FAST_CHECK)
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
                H, _ = cv2.findHomography(corners, board_mm)

            # Clicked points (visible from the first click)
            for point in clicks:
                cv2.circle(undistorted, point, 5, (255, 0, 0), -1)  # thickness -1 = filled dot

            # Distance between the two clicked points
            if len(clicks) == 2 and H is not None:
                p1 = pixel_to_mm(clicks[0], H)
                p2 = pixel_to_mm(clicks[1], H)
                distance = np.linalg.norm(p1 - p2)  # Euclidean distance in mm

                cv2.line(undistorted, clicks[0], clicks[1], (0, 255, 0), 2)
                cv2.putText(undistorted, f"{distance:.1f} mm", clicks[1], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Board status, so you know whether H is fresh, old or missing
            if found:
                status, color = "board: detected", (0, 255, 0)
            elif H is not None:
                status, color = "board: not visible (using last H)", (0, 255, 255)
            else:
                status, color = "board: not found yet", (0, 0, 255)
            cv2.putText(undistorted, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Original", frame)
            cv2.imshow("Undistorted", undistorted)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("QUITTING PROGRAM...")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
