import torch 
import cv2

if torch.cuda.is_available():
    print("CUDA is available. GPU will be used.")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. CPU will be used.")

print(f"CUDA version: {torch.version.cuda}")
print(f"pyTorch version: {torch.__version__}")

cap = cv2.VideoCapture(0) #Open the default camera (index 0)

try:
    ok, frame = cap.read()      
    if ok:
        print("Successfully read from camera")
        print(f"Frame shape: {frame.shape}")
    else:
        print("Could not read from camera")
finally:
    cap.release() #Release the camera