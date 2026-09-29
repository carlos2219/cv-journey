"""Track the largest red object from the webcam and show its centroid."""

import time

import cv2
import numpy as np

# HSV thresholds for the red object (OpenCV hue goes 0-179).
# S_min = 100 rejects skin (measured S ~41-63); V_min = 50 rejects very dark pixels.
LOWER_RED = np.array([160, 100, 50])
UPPER_RED = np.array([179, 255, 255])

MIN_AREA = 500  # px^2; smaller blobs are treated as noise
FPS_ALPHA = 0.05  # low-pass filter for the FPS display (smaller = smoother)
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # "brush" for morphology


def make_mask(frame):
    """Return (hsv, mask): the HSV image and a cleaned binary mask of red pixels."""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)  # smooth pixel noise before thresholding
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)  # remove small white specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)  # fill small holes in the object
    return hsv, mask


def find_target(mask):
    """Return (contour, (cx, cy)) of the largest blob, or None if there is no valid target."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_AREA:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return largest, (cx, cy)


def draw_target(frame, target):
    """Draw the target outline and centroid on the frame, or a "no target" label."""
    if target is None:
        cv2.putText(frame, "no target", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return

    contour, (cx, cy) = target
    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # thickness -1 = filled dot
    cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera is not accessible right now")
        raise SystemExit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_y, center_x = height // 2, width // 2

    previous_time = time.perf_counter()
    fps_smooth = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read from camera")
                break

            # Smoothed FPS
            current_time = time.perf_counter()
            fps = 1 / (current_time - previous_time)
            fps_smooth = (1 - FPS_ALPHA) * fps_smooth + FPS_ALPHA * fps
            previous_time = current_time

            # Vision pipeline: frame -> mask -> target
            hsv, mask = make_mask(frame)
            target = find_target(mask)

            # HSV value under the center marker, printed with "p" to tune the thresholds
            pixel_hsv = hsv[center_y, center_x]

            # Overlays
            draw_target(frame, target)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Mask", mask)
            cv2.imshow("Live view", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("QUITTING PROGRAM...")
                break
            elif key == ord("p"):
                print(f"pixel_hsv: {pixel_hsv}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
