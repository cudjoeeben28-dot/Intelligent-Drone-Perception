
import argparse
import os
import cv2
import numpy as np


# ---------- individual operations ----------
def edge_detection(img_gray: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """Canny edge detector on a grayscale image."""
    return cv2.Canny(img_gray, low, high)


def gaussian_blur(img: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Gaussian blur; ksize coerced to the nearest odd integer >= 3."""
    ksize = max(3, ksize // 2 * 2 + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0)


def change_brightness(img: np.ndarray, beta: int = 40) -> np.ndarray:
    """Shift all pixel values by beta (positive = brighten, negative = darken)."""
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)


def rotate(img: np.ndarray, angle: float = 45.0, scale: float = 1.0) -> np.ndarray:
    """Rigid rotation about the image centre (positive angle = CCW)."""
    h, w = img.shape[:2]
    centre = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(centre, angle, scale)
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR)


def translate(img: np.ndarray, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    """
    Pure translation.

    tx > 0 shifts right, ty > 0 shifts down (OpenCV pixel convention).
    """
    h, w = img.shape[:2]
    mat = np.float32([[1, 0, tx],
                      [0, 1, ty]])
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR)






def main():
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.png")  # input image next to this script
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "E_translate")
    os.makedirs(output_dir, exist_ok=True)

    # Required translations: (tx, ty)
    translations = [
        (60, 0),
        (-60, 0),
        (0, 60),
        (40, -40)
    ]

    for tx, ty in translations:
        result = translate(img, tx, ty)
        filename = f"trans_{tx}_{ty}.png"
        output_path = os.path.join(output_dir, filename)

        if not cv2.imwrite(output_path, result):
            raise IOError(f"Could not write {output_path}")

        print(f"Saved: {filename}")

    print("Part E completed successfully.")


if __name__ == "__main__":
    main()



"""
Basic OpenCV operations: Canny edge extraction, Gaussian blur,
brightness shift, rotation, and pure translation.

Usage examples
--------------
1) Pure translation – 40 px right, 15 px up:
       python CV_op.py -i input.jpg -op translate --tx 40 --ty -15 -o shifted.png

2) Rotate 30 degrees counter-clockwise around the image centre:
       python CV_op.py -i input.jpg -op rotate --angle 30 -o rotated.png

3) Rotate + scale – 45 degrees CCW, 1.2x larger:
       python CV_op.py -i input.jpg -op rotate --angle 45 --scale 1.2 -o rot_scale.png

4) Edge detection with custom thresholds:
       python CV_op.py -i input.jpg -op edge --low 50 --high 150 -o edges.png

5) Gaussian blur using an 11 x 11 kernel:
       python CV_op.py -i input.jpg -op blur --ksize 11 -o blur.png

6) Brighten every pixel by +60 levels:
       python CV_op.py -i input.jpg -op brightness --beta 60 -o bright.png
"""
