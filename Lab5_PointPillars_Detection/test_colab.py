import argparse
import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D          # needed for Task 1 legend
import matplotlib.patches as mpatches
from collections import Counter              # Task 3

from pointpillars.utils import setup_seed, read_points, read_calib, read_label, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range, \
    bbox3d2corners_camera, points_camera2image, \
    bbox_camera2lidar
from pointpillars.model import PointPillars


def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts


def visualize_bev(pc, bboxes, labels=None, gt_bboxes=None, gt_labels=None, save_path='detection_result.png'):
    """Visualize bird's eye view of point cloud and bounding boxes"""
    CLASSES = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car'}
    # Task 5 - custom colors (-1 is ground truth)
    COLORS = {0: '#1f77b4', 1: '#2ca02c', 2: '#d62728', -1: '#ffbf00'}

    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot point cloud (bird's eye view)
    ax.scatter(pc[:, 0], pc[:, 1], s=0.5, c='gray', alpha=0.2, label='Point Cloud')

    # Plot predicted bounding boxes
    if bboxes is not None and len(bboxes) > 0:
        for i, bbox in enumerate(bboxes):
            # bbox format: [x, y, z, l, w, h, rotation]
            x, y, l, w, rotation = bbox[0], bbox[1], bbox[3], bbox[4], bbox[6]

            # Get color based on label
            label = int(labels[i]) if labels is not None else 2
            color = COLORS.get(label, 'red')
            class_name = CLASSES.get(label, 'Unknown')

            # Create rotated rectangle
            rect = Rectangle((x - l/2, y - w/2), l, w,
                           angle=np.degrees(rotation),
                           linewidth=2, edgecolor=color, facecolor='none',
                           label=f'Pred: {class_name}' if i == 0 else '')
            ax.add_patch(rect)

            # Add text label
            ax.text(x, y, class_name, fontsize=8, color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Plot ground truth bounding boxes
    # Task 6 - GT boxes are dashed yellow so they stand apart from predictions
    if gt_bboxes is not None and len(gt_bboxes) > 0:
        for i, bbox in enumerate(gt_bboxes):
            x, y, l, w, rotation = bbox[0], bbox[1], bbox[3], bbox[4], bbox[6]

            rect = Rectangle((x - l/2, y - w/2), l, w,
                           angle=np.degrees(rotation),
                           linewidth=2, edgecolor='yellow', facecolor='none',
                           linestyle='--',
                           label='Ground Truth' if i == 0 else '')
            ax.add_patch(rect)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Bird\'s Eye View - 3D Object Detection Results', fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Task 1 - Add legend (fixed: always show all classes)
    legend_elems = [
        Line2D([0],[0], color='gray', lw=0, marker='o', label='Point Cloud'),
        Line2D([0],[0], color=COLORS[0], lw=2, label=CLASSES[0]),
        Line2D([0],[0], color=COLORS[1], lw=2, label=CLASSES[1]),
        Line2D([0],[0], color=COLORS[2], lw=2, label=CLASSES[2]),
    ]
    if gt_bboxes is not None and len(gt_bboxes) > 0:
        legend_elems.append(Line2D([0],[0], color=COLORS[-1], lw=2, linestyle='--', label='Ground Truth'))
    ax.legend(handles=legend_elems, loc='upper right')

    # Task 3 - Show counts in the top-left
    name_map = {0:'Pedestrian', 1:'Cyclist', 2:'Car'}
    cnt = Counter(name_map.get(int(l), 'Unknown') for l in (labels if labels is not None else []))
    summary = f"Detected: {cnt.get('Car',0)} Cars, {cnt.get('Cyclist',0)} Cyclists, {cnt.get('Pedestrian',0)} Pedestrians"
    ax.text(0.01, 0.99, summary, transform=ax.transAxes, ha='left', va='top',
            fontsize=11, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.show()


def main(args):
    CLASSES = {
        'Pedestrian': 0,
        'Cyclist': 1,
        'Car': 2
    }
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    print("Loading model...")
    if not args.no_cuda:
        model = PointPillars(nclasses=len(CLASSES)).cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=len(CLASSES))
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))

    if not os.path.exists(args.pc_path):
        raise FileNotFoundError(f"Point cloud file not found: {args.pc_path}")

    print(f"Reading point cloud from {args.pc_path}...")
    pc = read_points(args.pc_path)
    print(f"Original point cloud shape: {pc.shape}")

    # Task 2 - experiment with the point cloud range [x1, y1, z1, x2, y2, z2]
    # Default below is the one from the task sheet; change it with --point_range
    # e.g. try: --point_range 5 -20 -2 35 20 0.5
    pc = point_range_filter(pc, point_range=args.point_range)
    print(f"Point range: {args.point_range}")
    print(f"Filtered point cloud shape: {pc.shape}")

    pc_torch = torch.from_numpy(pc)

    if args.calib_path and os.path.exists(args.calib_path):
        calib_info = read_calib(args.calib_path)
    else:
        calib_info = None
        print("No calibration file provided or found")

    if args.gt_path and os.path.exists(args.gt_path):
        gt_label = read_label(args.gt_path)
    else:
        gt_label = None
        print("No ground truth file provided or found")

    if args.img_path and os.path.exists(args.img_path):
        img = cv2.imread(args.img_path, 1)
    else:
        img = None

    print("\nRunning inference...")
    model.eval()
    with torch.no_grad():
        if not args.no_cuda:
            pc_torch = pc_torch.cuda()

        result_filter = model(batched_pts=[pc_torch], mode='test')[0]

    if calib_info is not None and img is not None:
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)
        P2 = calib_info['P2'].astype(np.float32)

        image_shape = img.shape[:2]
        result_filter = keep_bbox_from_image_range(result_filter, tr_velo_to_cam, r0_rect, P2, image_shape)

    result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
    lidar_bboxes = result_filter['lidar_bboxes']
    labels, scores = result_filter['labels'], result_filter['scores']

    # Task 4 - confidence thresholding (change with --threshold)
    threshold = args.threshold
    keep = scores >= threshold
    lidar_bboxes, labels, scores = lidar_bboxes[keep], labels[keep], scores[keep]
    print(f"Confidence threshold: {threshold}")

    print(f"\n{'='*50}")
    print(f"Detection Results:")
    print(f"{'='*50}")
    print(f"Number of objects detected: {len(lidar_bboxes)}")

    for i, (bbox, label, score) in enumerate(zip(lidar_bboxes, labels, scores)):
        class_name = LABEL2CLASSES.get(int(label), 'Unknown')
        print(f"  Object {i+1}: {class_name} (confidence: {score:.3f})")
    print(f"{'='*50}\n")

    # Visualize predictions
    gt_lidar_bboxes = None
    if calib_info is not None and gt_label is not None:
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)

        dimensions = gt_label['dimensions']
        location = gt_label['location']
        rotation_y = gt_label['rotation_y']
        gt_labels_array = np.array([CLASSES.get(item, -1) for item in gt_label['name']])
        sel = gt_labels_array != -1
        gt_labels_array = gt_labels_array[sel]
        bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1)
        gt_lidar_bboxes = bbox_camera2lidar(bboxes_camera, tr_velo_to_cam, r0_rect)
        bboxes_camera = bboxes_camera[sel]
        gt_lidar_bboxes = gt_lidar_bboxes[sel]

        print(f"Ground truth objects: {len(gt_lidar_bboxes)}")

    # Task 7 - save each run under a different name (change with --save_path)
    visualize_bev(pc, lidar_bboxes, labels, gt_bboxes=gt_lidar_bboxes,
                  save_path=args.save_path)

    print("\nDetection complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointPillars 3D Object Detection')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--pc_path', required=True,
                       help='Path to point cloud file (.bin)')
    parser.add_argument('--calib_path', default='',
                       help='Path to calibration file (optional)')
    parser.add_argument('--gt_path', default='',
                       help='Path to ground truth label file (optional)')
    parser.add_argument('--img_path', default='',
                       help='Path to image file (optional)')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Use CPU instead of GPU')
    # Added for Tasks 2, 4 and 7 so you can re-run without editing the file
    parser.add_argument('--point_range', type=float, nargs=6,
                       default=[0, -40, -3, 70.4, 40, 1.0],
                       help='x1 y1 z1 x2 y2 z2 (Task 2)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Minimum confidence score to keep (Task 4)')
    parser.add_argument('--save_path', default='bev_default.png',
                       help='Where to save the BEV image (Task 7)')
    args = parser.parse_args()

    main(args)
