# Intelligent Drone Perception Systems – Lab Code (COMP11130)

Code and scripts for the six laboratory exercises in COMP11130 *Intelligent Drone Perception Systems*, Assessment 1.

| Lab | Topic | Folder | Main files |
|---|---|---|---|
| 1 | Image processing with OpenCV (Canny, Gaussian blur, brightness, rotation, translation) | [Lab1_OpenCV_Image_Processing](Lab1_OpenCV_Image_Processing) | `CV_op.py`, `install_cv.py` |
| 2 | Image classification with CNNs on MNIST | [Lab2_CNN_MNIST_Classification](Lab2_CNN_MNIST_Classification) | `MNIST_CNN_config_param.py`, `MNIST-CNN.py` |
| 3 | Ground control point (GCP) detection with YOLOv5s | [Lab3_YOLOv5_GCP_Detection](Lab3_YOLOv5_GCP_Detection) | `yolo-pretrained-with-train.py`, `data.yaml`, `args.yaml`, `results.csv` |
| 4 | Semantic segmentation with U-Net (Semantic Drone Dataset) | [Lab4_UNet_Semantic_Segmentation](Lab4_UNet_Semantic_Segmentation) | `unet_lab.py`, `UNET-code.py`, `split.py`, `plot_iou_chart.py`, `UNet_Lab_Colab.ipynb` |
| 5 | LiDAR 3D object detection with pretrained PointPillars (KITTI) | [Lab5_PointPillars_Detection](Lab5_PointPillars_Detection) | `Lab5_PointCloud_Colab.ipynb`, `test_colab.py` |
| 6 | Training PointPillars on a KITTI subset | [Lab6_PointPillars_Training](Lab6_PointPillars_Training) | `Lab6_PointPillars_Colab.ipynb` |

## Running the code

- **Labs 1–3** run locally in Python (PyCharm virtual environment). Install with
  `pip install opencv-python numpy torch torchvision ultralytics torchinfo pyyaml`.
  - Lab 1: put the input `image.png` next to `CV_op.py`, then `python CV_op.py`.
  - Lab 2: `python MNIST_CNN_config_param.py --item 0 --epochs 1` (items 0–6 select each experiment).
  - Lab 3: put the labelled dataset in `yolo-dataset/` (`images/` and `labels/` with train/val/test), then `python yolo-pretrained-with-train.py`.
- **Lab 4** runs locally or in Colab: `python split.py` to create the 80/20 split, then `python unet_lab.py`.
- **Labs 5–6** run in Google Colab with a T4 GPU: open the notebook and run all cells. They clone
  [zhulf0804/PointPillars](https://github.com/zhulf0804/PointPillars) and build its CUDA ops.

## Notes

- Datasets (MNIST, GCP images, Semantic Drone Dataset, KITTI) and trained weights (`.pt`, `.pth`) are not included because of their size and licences.
- Absolute local file paths in the original scripts were replaced with relative paths so the code runs from its own folder.
- Lab 3 `args.yaml` and `results.csv` are the training configuration and per-epoch metrics from the Ultralytics run (128 epochs).
