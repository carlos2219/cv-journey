"""Detect daily life objects with a pretrained CNN (YOLO) and show the latency per stage."""
import time

import cv2
from ultralytics import YOLO

MODEL_FILE = "yolo11n.pt"  # "n" = nano, the smallest and fastest version
FPS_ALPHA = 0.05  # low-pass filter for the FPS display (smaller = smoother)


def main():
    model = YOLO(MODEL_FILE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera is not accessible right now")
        raise SystemExit(1)

    device = 0  # 0 = first GPU, "cpu" = CPU (toggle with "d")
    previous_time = time.perf_counter()
    fps_smooth = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read from camera")
                break

            # Run the model
            results = model(frame, device=device, verbose=False)
            r = results[0]

            # Draw the detections
            annotated = r.plot()  # a new BGR image with boxes and labels drawn

            # Latency per stage, in ms
            pre = r.speed["preprocess"]
            inf = r.speed["inference"]
            post = r.speed["postprocess"]

            # Smoothed FPS
            current_time = time.perf_counter()
            fps = 1 / (current_time - previous_time)
            fps_smooth = (1 - FPS_ALPHA) * fps_smooth + FPS_ALPHA * fps
            previous_time = current_time

            # Show latency and FPS
            cv2.putText(annotated, f"device: {device}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(annotated, f"preprocess: {pre:.1f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(annotated, f"inference: {inf:.1f} ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(annotated, f"postprocess: {post:.1f} ms", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Detections", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("QUITTING PROGRAM...")
                break
            elif key == ord("d"):  # toggle GPU/CPU
                device = "cpu" if device == 0 else 0
                print(f"Switched to device: {device}")
            elif key == ord("p"):  # print the detections of the current frame
                for box in r.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    print(f"{model.names[cls_id]}: {conf:.2f} at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
