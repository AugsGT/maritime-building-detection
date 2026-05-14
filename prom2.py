from ultralytics import YOLO
import cv2
import numpy as np
import math

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

IMAGE_PATH = r"C:/Storage/maritime_2/test_image/test_1.jpg"

MODEL_PATH = r"C:/Storage/maritime_2/runs/detect/train/weights/best.pt"


CONF_THRESHOLD = 0.25

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

model = YOLO(MODEL_PATH)

# -------------------------------------------------------
# LOAD IMAGE
# -------------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise ValueError("Image not found")

img_h, img_w = image.shape[:2]

# -------------------------------------------------------
# RUN DETECTION
# -------------------------------------------------------

results = model(image)

# -------------------------------------------------------
# EXTRACT BUILDINGS
# -------------------------------------------------------

buildings = []

for result in results:

    for box in result.boxes:

        conf = float(box.conf[0])

        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        width = x2 - x1
        height = y2 - y1

        crop = image[y1:y2, x1:x2]

        # ------------------------------------------------
        # CONTOUR / SILHOUETTE COMPLEXITY
        # ------------------------------------------------

        complexity_score = 0

        try:

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            edges = cv2.Canny(gray, 100, 200)

            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) > 0:

                largest = max(contours, key=cv2.contourArea)

                perimeter = cv2.arcLength(largest, True)

                approx = cv2.approxPolyDP(
                    largest,
                    0.02 * perimeter,
                    True
                )

                complexity_score = len(approx)

        except:
            complexity_score = 0

        buildings.append({
            "box": (x1, y1, x2, y2),
            "conf": conf,
            "height": height,
            "width": width,
            "aspect_ratio": height / max(width, 1),
            "complexity": complexity_score
        })

# -------------------------------------------------------
# CHECK DETECTIONS
# -------------------------------------------------------

if len(buildings) == 0:
    print("No buildings detected")
    exit()

# -------------------------------------------------------
# COMPUTE GLOBAL STATS
# -------------------------------------------------------

heights = np.array([b["height"] for b in buildings])
widths = np.array([b["width"] for b in buildings])
aspects = np.array([b["aspect_ratio"] for b in buildings])
complexities = np.array([b["complexity"] for b in buildings])

# Prevent divide-by-zero
EPS = 1e-6

# -------------------------------------------------------
# Z-SCORE FUNCTION
# -------------------------------------------------------

def z_score(value, arr):

    mean = np.mean(arr)
    std = np.std(arr)

    return abs((value - mean) / (std + EPS))

# -------------------------------------------------------
# PROMINENCE SCORE
# -------------------------------------------------------

def prominence_score(target, buildings):

    x1, y1, x2, y2 = target["box"]

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # ---------------------------------------------------
    # 1. HEIGHT UNIQUENESS
    # ---------------------------------------------------

    height_uniqueness = z_score(
        target["height"],
        heights
    )

    # ---------------------------------------------------
    # 2. WIDTH UNIQUENESS
    # ---------------------------------------------------

    width_uniqueness = z_score(
        target["width"],
        widths
    )

    # ---------------------------------------------------
    # 3. ASPECT RATIO UNIQUENESS
    # ---------------------------------------------------

    aspect_uniqueness = z_score(
        target["aspect_ratio"],
        aspects
    )

    # ---------------------------------------------------
    # 4. SILHOUETTE COMPLEXITY UNIQUENESS
    # ---------------------------------------------------

    complexity_uniqueness = z_score(
        target["complexity"],
        complexities
    )

    # ---------------------------------------------------
    # 5. ISOLATION SCORE
    # ---------------------------------------------------

    min_dist = float("inf")

    for b in buildings:

        if b == target:
            continue

        bx1, by1, bx2, by2 = b["box"]

        bcx = (bx1 + bx2) / 2
        bcy = (by1 + by2) / 2

        dist = math.sqrt(
            (cx - bcx) ** 2 +
            (cy - bcy) ** 2
        )

        min_dist = min(min_dist, dist)

    isolation_score = min_dist / max(img_w, img_h)

    # ---------------------------------------------------
    # 6. CENTER EMPHASIS
    # ---------------------------------------------------

    img_cx = img_w / 2
    img_cy = img_h / 2

    center_dist = math.sqrt(
        (cx - img_cx) ** 2 +
        (cy - img_cy) ** 2
    )

    max_dist = math.sqrt(
        img_cx**2 + img_cy**2
    )

    center_score = 1 - (center_dist / max_dist)

    # ---------------------------------------------------
    # FINAL SCORE
    # ---------------------------------------------------

    score = (
        0.25 * height_uniqueness +
        0.10 * width_uniqueness +
        0.25 * aspect_uniqueness +
        0.25 * complexity_uniqueness +
        0.10 * isolation_score +
        0.05 * center_score
    )

    return score

# -------------------------------------------------------
# SCORE BUILDINGS
# -------------------------------------------------------

best_building = None
best_score = -1

for b in buildings:

    score = prominence_score(b, buildings)

    b["score"] = score

    if score > best_score:
        best_score = score
        best_building = b

# -------------------------------------------------------
# DRAW ONLY MOST PROMINENT BUILDING
# -------------------------------------------------------

x1, y1, x2, y2 = best_building["box"]

cv2.rectangle(
    image,
    (x1, y1),
    (x2, y2),
    (0, 0, 255),
    4
)

cv2.putText(
    image,
    f"Prominent {best_score:.2f}",
    (x1, y1 - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    2
)

# -------------------------------------------------------
# SHOW RESULT
# -------------------------------------------------------


cv2.imwrite("resultsss.jpg",image)
print("Confidence",conf)