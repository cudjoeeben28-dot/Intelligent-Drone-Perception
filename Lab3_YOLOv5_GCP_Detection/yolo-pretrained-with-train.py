from ultralytics import YOLO
from torchinfo import summary
import torch, os, yaml, multiprocessing as mp

DATASET_ROOT = "yolo-dataset"  # dataset folder (images/, labels/) next to this script
DATA_YAML = os.path.join(DATASET_ROOT, "data.yaml")

def ensure_data_yaml():
    # training data dir
    train_p = os.path.join(DATASET_ROOT, "images", "train").replace("\\", "/")
    val_p   = os.path.join(DATASET_ROOT, "images", "val").replace("\\", "/")
    test_p  = os.path.join(DATASET_ROOT, "images", "test").replace("\\", "/")

    # config
    cfg = {"train": train_p, "val": val_p}
    if os.path.isdir(os.path.join(DATASET_ROOT, "images", "test")):
        cfg["test"] = test_p

    cfg["nc"] = 1
    cfg["names"] = ["GCP"]

    # Yaml file
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"data.yaml written → {DATA_YAML}")

def main():
    ensure_data_yaml()

    model = YOLO("yolov5s.pt")

    # Optional model summary
    dummy_input = torch.zeros((1, 3, 640, 640))
    summary(model.model, input_data=dummy_input, depth=3,
            col_names=("input_size", "output_size", "num_params"))

    model.train(
        data=DATA_YAML,
        imgsz=640,
        epochs=128,
        batch=8,
        device=0,     #GPU
        workers=0,
        project="runs",
        name="yolov5s-custom",
        exist_ok=True
    )

    print("\n=== TEST EVALUATION ===")
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        device=0,
        iou=0.6,
        conf=0.01,
        plots=True    # saves plot
    )

    # Ultralytics returns evaluations
    try:
        # mAP@0.5 and mAP@0.5:0.95
        map50_95 = getattr(metrics.box, "map", None)          # mAP@[.5:.95]
        map50     = getattr(metrics.box, "map50", None)       # mAP@.5
        mp_val    = getattr(metrics.box, "mp", None)          # mean precision
        mr_val    = getattr(metrics.box, "mr", None)          # mean recall
        print(f"mAP@0.5:0.95 = {map50_95:.4f} | mAP@0.5 = {map50:.4f} | mPrec = {mp_val:.4f} | mRec = {mr_val:.4f}")
    except Exception:

        try:
            print("Results:", metrics.results_dict)
        except Exception:
            print("Test metrics printed by Ultralytics.")

    try:
        class_maps = getattr(metrics.box, "maps", None)
        names = model.names
        if class_maps is not None:
            print("\nPer-class AP@0.5:0.95:")
            for cls_id, ap in enumerate(class_maps):
                if ap is not None:
                    print(f"  {names.get(cls_id, str(cls_id))}: {ap:.4f}")
    except Exception:
        pass

    # Test with one image
    test_images = os.path.join(DATASET_ROOT, "images", "test")
    if os.path.isdir(test_images):
        pred = model(source=test_images, imgsz=640, conf=0.5, device=0, save=True)
        print("Saved test predictions to:", pred[0].save_dir)



if __name__ == "__main__":
    mp.freeze_support()
    main()

