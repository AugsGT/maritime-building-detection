from ultralytics import YOLO
import cv2
import math
#configs

IMAGE_PATH = r"C:/Storage/maritime_2/test_image/test.jpg"

MODEL_PATH = r"C:/Storage/maritime_2/runs/detect/train/weights/best.pt"

CONFIDENCE_THRESHOLD = 0.25

model = YOLO(MODEL_PATH)

results = model(
    IMAGE_PATH,
    conf=CONFIDENCE_THRESHOLD
)

image = cv2.imread(IMAGE_PATH)#read image

img_h, img_w = image.shape[:2]

image_center_x = img_w / 2
image_center_y = img_h / 2

best_score = -1
best_box = None

#select best score & box
for r in results:

    for box in r.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confidence = float(box.conf[0])

        width = x2 - x1
        height = y2 - y1

        area = width * height
        # prominence score based mainly on size
        score = area * confidence

        if score > best_score:

            best_score = score
            best_box = (
                x1, y1, x2, y2,
                confidence
            )


if best_box:

    x1, y1, x2, y2, conf = best_box

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.putText(
        image,
        f"Prominent {conf:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

cv2.imwrite("result.jpg", image)
print("Confidence",conf)
print("Saved result as result.jpg")