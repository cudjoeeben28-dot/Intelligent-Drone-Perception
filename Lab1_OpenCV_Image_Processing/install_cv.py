import subprocess
import sys

def install_opencv():
    try:
        import cv2
        print("OpenCV is already installed.")
    except ImportError:
        print("Installing OpenCV...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
        print("OpenCV installed successfully.")

install_opencv()