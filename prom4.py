# =====================================================
# YOLO + NAVIGATOR PROMINENCE PIPELINE (V2 - NO CENTER BIAS)
# =====================================================

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
OUTPUT_FOLDER = r"C:/Storage/maritime_2/output1"
CONF_THRESHOLD = 0.25
DRAW_ALL_DETECTIONS = True

# =====================================================
# INITIALIZATION
# =====================================================
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
model = YOLO(MODEL_PATH)

image_paths = list(Path(TEST_FOLDER).rglob("*"))
image_paths = [p for p in image_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

print(f"\nFound {len(image_paths)} images. Initializing Navigator Brain...\n")

# =====================================================
# CORE FUNCTIONS
# =====================================================

def calculate_prominence(target, all_buildings, img_w, img_h, avg_brightness):
    """ The 'Perfect Brain' logic for maritime landmark identification. """
    
    # 1. Contextual Normalization
    max_height = max([b['height'] for b in all_buildings]) + 1e-6
    max_entropy = max([b['entropy'] for b in all_buildings]) + 1e-6
    min_y_global = min([b['box'][1] for b in all_buildings]) # Highest point in skyline
    
    # 2. STRUCTURAL DOMINANCE (The Silhouette)
    height_norm = target['height'] / max_height
    # Skyline Break: How high does the top of the building reach relative to others?
    skyline_break = (img_h - target['box'][1]) / (img_h - min_y_global + 1e-6)
    structural_score = (0.5 * height_norm) + (0.5 * skyline_break)

    # 3. VISUAL COMPLEXITY (Architectural Detail)
    entropy_norm = min(target['entropy'] / (max_entropy + 1e-6), 1.0)
    complexity_score = (0.7 * entropy_norm) + (0.3 * target['symmetry'])

    # 4. RADIANCE (Night/Beacon Detection)
    # Beacon Spike: Max pixel vs Box Average
    beacon_score = min(target['beacon_spike'] / 6.0, 1.0) 
    lum_norm = target['max_lum'] / 255.0
    radiance_score = (0.8 * beacon_score) + (0.2 * lum_norm)

    # 5. ISOLATION (Gestalt Standalone Factor)
    # Isolated buildings on a coast are more likely intended as landmarks
    isolation_score = target['isolation_score']

    # =============================================
    # DYNAMIC WEIGHTING (Day vs. Night)
    # =============================================
    if avg_brightness < 75: # Night Mode
        w_struct, w_complex, w_radiance, w_iso = 0.25, 0.15, 0.50, 0.10
    else: # Day Mode
        w_struct, w_complex, w_radiance, w_iso = 0.45, 0.30, 0.10, 0.15

    final_score = (
        (structural_score * w_struct) +
        (complexity_score * w_complex) +
        (radiance_score * w_radiance) +
        (isolation_score * w_iso)
    )

    return final_score

# =====================================================
# MAIN PROCESSING LOOP
# =====================================================

for image_path in image_paths:
    print("-" * 70)
    print(f"Processing: {image_path.name}")
    
    image = cv2.imread(str(image_path))
    if image is None: continue
    img_h, img_w = image.shape[:2]
    avg_img_brightness = np.mean(image)

    # 1. Generate Saliency Mask (rembg)
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask = np.array(remove(pil_img, only_mask=True))
    mask = cv2.resize(mask, (img_w, img_h))

    # 2. YOLO Detection
    results = model(image, conf=CONF_THRESHOLD, iou=0.35)
    raw_buildings = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0: continue
            
            crop = image[y1:y2, x1:x2]
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Feature: Saliency
            sal_crop = mask[y1:y2, x1:x2]
            sal_val = np.mean(sal_crop) / 255.0 if sal_crop.size > 0 else 0
            
            # Feature: Visual Entropy (Laplacian Variance)
            entropy = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
            
            # Feature: Symmetry
            flipped = cv2.flip(gray_crop, 1)
            diff = cv2.absdiff(gray_crop, flipped)
            symm = 1.0 - (np.mean(diff) / 255.0)
            
            # Feature: Beacon Spike & Radiance
            _, max_val, _, _ = cv2.minMaxLoc(gray_crop)
            beacon_spike = max_val / (np.mean(gray_crop) + 1e-6)

            raw_buildings.append({
                "box": (x1, y1, x2, y2),
                "height": h, "width": w, "area": w*h,
                "entropy": entropy, "symmetry": symm,
                "saliency": sal_val, "beacon_spike": beacon_spike,
                "max_lum": max_val, "conf": float(box.conf[0])
            })

    if not raw_buildings:
        print("No buildings detected.")
        continue

    # 3. Calculate Global Isolation Scores
    for b1 in raw_buildings:
        cx1, cy1 = (b1["box"][0] + b1["box"][2])/2, (b1["box"][1] + b1["box"][3])/2
        min_dist = float('inf')
        for b2 in raw_buildings:
            if b1 == b2: continue
            cx2, cy2 = (b2["box"][0] + b2["box"][2])/2, (b2["box"][1] + b2["box"][3])/2
            dist = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            min_dist = min(min_dist, dist)
        b1["isolation_score"] = min(min_dist / (max(img_w, img_h) * 0.2), 1.0) if len(raw_buildings) > 1 else 1.0

    # 4. Final Scoring
    best_b = None
    max_s = -1
    
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

    for b in raw_buildings:
        score = calculate_prominence(b, raw_buildings, img_w, img_h, avg_img_brightness)
        b["final_score"] = score
        
        if score > max_s:
            max_s = score
            best_b = b

    # 5. Visualization
    for b in raw_buildings:
        x1, y1, x2, y2 = b["box"]
        is_prominent = (b == best_b)
        color = (0, 0, 255) if is_prominent else (0, 255, 0)
        thick = 3 if is_prominent else 1
        
        if is_prominent or DRAW_ALL_DETECTIONS:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thick)
            label = f"SCORE: {b['final_score']:.2f}"
            if is_prominent: label = "[PROMINENT] " + label
            cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(str(Path(OUTPUT_FOLDER) / image_path.name), overlay)
    print(f"Landmark Found! Score: {max_s:.2f} -> Saved to output.")

print("\nProcessing Complete.")