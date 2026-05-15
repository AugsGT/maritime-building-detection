# YOLO + Saliency + Skyline Prominence Pipeline

from ultralytics import YOLO
import cv2
import numpy as np
import math
from pathlib import Path
from rembg import remove
from PIL import Image

# =====================================================
# CONFIG
# =====================================================

TEST_FOLDER = r"C:/Storage/maritime_2/test_image"

MODEL_PATH = r"C:/Storage/maritime_2/runs/detect/train/weights/best.pt"

OUTPUT_FOLDER = r"C:/Storage/maritime_2/output"

CONF_THRESHOLD = 0.25

DRAW_ALL_DETECTIONS = True

# =====================================================
# CREATE OUTPUT DIRECTORY
# =====================================================

Path(OUTPUT_FOLDER).mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# LOAD MODEL
# =====================================================

model = YOLO(MODEL_PATH)

# =====================================================
# GET ALL IMAGES
# =====================================================

image_paths = list(
    Path(TEST_FOLDER).rglob("*")
)

image_paths = [
    p for p in image_paths
    if p.suffix.lower() in [
        ".jpg",
        ".jpeg",
        ".png"
    ]
]

print(f"\nFound {len(image_paths)} images\n")

# =====================================================
# PROCESS EACH IMAGE
# =====================================================

for image_path in image_paths:

    print("=" * 60)
    print(f"Processing: {image_path.name}")

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not load image")
        continue

    img_h, img_w = image.shape[:2]

    # =================================================
    # GENERATE SALIENCY MAP
    # =================================================

    pil_image = Image.fromarray(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    )

    saliency_map = remove(
        pil_image,
        only_mask=True
    )

    saliency_map = np.array(saliency_map)

    saliency_map = cv2.resize(
        saliency_map,
        (img_w, img_h)
    )

    # =================================================
    # CREATE HEATMAP
    # =================================================

    heatmap = cv2.applyColorMap(
        saliency_map,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image,
        0.85,
        heatmap,
        0.15,
        0
    )

    # =================================================
    # RUN YOLO DETECTION
    # =================================================

    results = model(
        image,
        conf=0.25,
        iou=0.35
    )

    buildings = []

    # =================================================
    # EXTRACT DETECTIONS
    # =================================================

    for result in results:

        for box in result.boxes:

            conf = float(box.conf[0])

            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            crop = image[y1:y2, x1:x2]

            # =============================================
            # SALIENCY SCORE
            # =============================================

            saliency_crop = saliency_map[
                y1:y2,
                x1:x2
            ]

            if saliency_crop.size == 0:

                saliency_score = 0

            else:

                saliency_score = (
                    np.mean(saliency_crop) / 255.0
                )

            # =============================================
            # LIGHT VARIANCE
            # =============================================

            gray_crop = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2GRAY
            )

            light_variance = (
                np.std(gray_crop) / 255.0
            )

            # =============================================
            # CONTOUR COMPLEXITY
            # =============================================

            complexity_score = 0

            try:

                edges = cv2.Canny(
                    gray_crop,
                    100,
                    200
                )

                contours, _ = cv2.findContours(
                    edges,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if len(contours) > 0:

                    largest = max(
                        contours,
                        key=cv2.contourArea
                    )

                    perimeter = cv2.arcLength(
                        largest,
                        True
                    )

                    approx = cv2.approxPolyDP(
                        largest,
                        0.02 * perimeter,
                        True
                    )

                    complexity_score = len(approx)

            except:
                complexity_score = 0

            # =============================================
            # STORE BUILDING FEATURES
            # =============================================

            buildings.append({

                "box": (x1, y1, x2, y2),

                "conf": conf,

                "height": height,

                "width": width,

                "area": width * height,

                "aspect_ratio":
                    height / max(width, 1),

                "complexity":
                    complexity_score,

                "saliency":
                    saliency_score,

                "light_variance":
                    light_variance
            })

    # =================================================
    # NO DETECTIONS
    # =================================================

    if len(buildings) == 0:

        print("No buildings detected")
        continue

    # =================================================
    # GLOBAL ARRAYS
    # =================================================

    heights = np.array([
        b["height"]
        for b in buildings
    ])

    areas = np.array([
        b["area"]
        for b in buildings
    ])

    complexities = np.array([
        b["complexity"]
        for b in buildings
    ])

    EPS = 1e-6

    # =================================================
    # Z SCORE FUNCTION
    # =================================================

    def z_score(value, arr):

        mean = np.mean(arr)
        std = np.std(arr)

        return (
            (value - mean) /
            (std + EPS)
        )

    # =================================================
    # PROMINENCE SCORE
    # =================================================

    def prominence_score(target, buildings):

        x1, y1, x2, y2 = target["box"]

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # =============================================
        # AREA DOMINANCE
        # =============================================

        area_score = (
            target["area"] /
            (np.max(areas) + EPS)
        )

        # =============================================
        # HEIGHT DOMINANCE
        # =============================================

        height_ratio = (
            target["height"] /
            (np.max(heights) + EPS)
        )

        # =============================================
        # VERTICALITY
        # =============================================

        verticality_score = (
            target["height"] /
            (target["width"] + EPS)
        )

        verticality_score = min(
            verticality_score / 5.0,
            1.0
        )

        # =============================================
        # TOP SKYLINE EMPHASIS
        # =============================================

        top_score = (
            1 - (y1 / img_h)
        )

        # =============================================
        # HORIZONTAL CENTER
        # =============================================

        img_cx = img_w / 2

        horizontal_center = (
            1 -
            abs(cx - img_cx) / img_cx
        )

        # =============================================
        # FULL CENTER SCORE
        # =============================================

        img_cy = img_h / 2

        center_dist = math.sqrt(
            (cx - img_cx) ** 2 +
            (cy - img_cy) ** 2
        )

        max_dist = math.sqrt(
            img_cx**2 + img_cy**2
        )

        center_score = (
            1 - (center_dist / max_dist)
        )

        # =============================================
        # SKYLINE CLUSTER SCORE
        # =============================================

        neighbor_count = 0

        for b in buildings:

            if b == target:
                continue

            bx1, by1, bx2, by2 = b["box"]

            bcx = (bx1 + bx2) / 2

            if abs(cx - bcx) < img_w * 0.15:

                neighbor_count += 1

        cluster_score = min(
            neighbor_count / 5,
            1.0
        )

        # =============================================
        # SALIENCY
        # =============================================

        saliency_score = target["saliency"]

        # =============================================
        # LIGHT VARIANCE
        # =============================================

        light_score = target["light_variance"]

        # =============================================
        # COMPLEXITY
        # =============================================

        complexity_uniqueness = max(
            0,
            z_score(
                target["complexity"],
                complexities
            )
        )

        # =============================================
        # ISOLATION SCORE
        # =============================================

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

            min_dist = min(
                min_dist,
                dist
            )

        isolation_score = (
            min_dist /
            max(img_w, img_h)
        )

        isolation_score = min(
            isolation_score,
            0.3
        )

        # =============================================
        # FINAL SCORE
        # =============================================

        score = (

            0.20 * area_score +

            0.18 * height_ratio +

            0.10 * verticality_score +

            0.08 * top_score +

            0.20 * center_score +

            0.10 * horizontal_center +

            0.10 * cluster_score +

            0.03 * saliency_score +

            0.01 * light_score
        )

        return score

    # =================================================
    # SCORE BUILDINGS
    # =================================================

    best_building = None
    best_score = -1

    for i, b in enumerate(buildings):

        score = prominence_score(
            b,
            buildings
        )

        b["score"] = score

        print(
            f"Building {i+1} | "
            f"Conf: {b['conf']:.2f} | "
            f"Height: {b['height']} | "
            f"Area: {b['area']} | "
            f"Score: {score:.2f}"
        )

        if score > best_score:

            best_score = score
            best_building = b

    # =================================================
    # DRAW DETECTIONS
    # =================================================

    for b in buildings:

        x1, y1, x2, y2 = b["box"]

        if b == best_building:

            color = (0, 0, 255)
            thickness = 4

            cv2.putText(
                overlay,
                f"PROMINENT {b['score']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        else:

            if not DRAW_ALL_DETECTIONS:
                continue

            color = (0, 255, 0)
            thickness = 2

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

    # =================================================
    # SAVE RESULT
    # =================================================

    output_path = (
        Path(OUTPUT_FOLDER) /
        image_path.name
    )

    cv2.imwrite(
        str(output_path),
        overlay
    )

    print(f"Saved: {output_path}")

print("\nDone")