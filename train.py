

from ultralytics import YOLO

def main():

    # Load YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Train
    model.train(
        data="data.yaml",   # path to dataset yaml
        epochs=15,
        imgsz=640,
        batch=8,
        workers=4,

        # optimization
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,

        # augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,

        # saving
        project="runs",
        name="yolov8n_building",

        # device
        device=0,   # use "cpu" if no GPU
        pretrained=True,

        # segmentation only if using polygons
        # task="segment"
    )

if __name__ == "__main__":
    main()