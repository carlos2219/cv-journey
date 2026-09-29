import cv2
import time


cap = cv2.VideoCapture(0)

try:
    is_open = cap.isOpened()
    if not is_open:
        print("Camera is not accessible right now")
        raise SystemExit(1)
    previous_time = time.perf_counter()
    fps_smooth = 0
    while True:
        ok, frame = cap.read()     
        current_time = time.perf_counter()
        diff_time = current_time - previous_time
        fps = 1/diff_time
        alpha = 0.05 #fps low-pass filter alpha value, less=smooth
        fps_smooth = (1-alpha) * fps_smooth + alpha * fps
        previous_time = current_time
        if ok:
            cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Live view",frame)
            if cv2.waitKey(1) & 0xFF==ord("q"):
                break
        else:
            print("Could not read from camera")
            break
finally:
    cap.release()
    cv2.destroyAllWindows()