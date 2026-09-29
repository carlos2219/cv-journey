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
    run_once = True
    alpha = 0.05 #fps low-pass filter alpha value, less=smooth

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_y, center_x = height // 2, width // 2

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read from camera")
            break     
        current_time = time.perf_counter()
        diff_time = current_time - previous_time
        fps = 1/diff_time
        fps_smooth = (1-alpha) * fps_smooth + alpha * fps
        previous_time = current_time


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if run_once:
            print(f"GRAY Shape: {gray.shape}")     
            print(f"HSV Shape: {hsv.shape}")    
            run_once=False
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Extract pixel values
        pixel_bgr = frame[center_y, center_x]
        pixel_hsv = hsv[center_y, center_x]
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), 2) 

        cv2.imshow("Live view",frame)
        cv2.imshow("Gray conversion",gray)
        cv2.imshow("HSV conversion",hsv)

        key = cv2.waitKey(1) & 0xFF

        if key==ord("q"):
            print("QUITTING PROGRAM...")
            break
        elif key==ord("p"):
            #print(f"pixel_bgr: {pixel_bgr}")
            print(f"pixel_hsv: {pixel_hsv}")      
finally:
    cap.release()
    cv2.destroyAllWindows()