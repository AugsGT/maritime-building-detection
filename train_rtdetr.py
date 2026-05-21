from ultralytics import RTDETR
import torch

print("\nCUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

model = RTDETR("rtdetr-l.pt")

# ------------------------------------------------
# TRAIN
# ------------------------------------------------

model.train(
    data="data.yaml",

    epochs=50,

    imgsz=640,

    batch=2,

    workers=2,

    device=0,

    pretrained=True,

    cache=False,

    amp=True,

    project="runs",

    name="rtdetr_model"
)