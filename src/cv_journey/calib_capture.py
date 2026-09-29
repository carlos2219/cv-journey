"""Capture checkerboard images from the webcam for camera calibration."""
from pathlib import Path

import cv2

PATTERN = (9, 6)  # inner corners of the checkerboard (columns, rows)
OUT_DIR = Path("data/calib")  # where the calibration images are saved


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera is not accessible right now")
        raise SystemExit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0  # number of images saved

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read from camera")
                break

            # Detect the checkerboard corners
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, PATTERN, None, cv2.CALIB_CB_FAST_CHECK)

            # Draw on a copy so that "frame" stays clean for saving
            display = frame.copy()
            cv2.drawChessboardCorners(display, PATTERN, corners, found)
            cv2.putText(display, f"Saved: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Calibration capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("QUITTING PROGRAM...")
                break
            elif key == ord("c") and found:
                path = OUT_DIR / f"img_{count:02d}.png"
                cv2.imwrite(str(path), frame)
                print(f"Saved {path}")
                count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
