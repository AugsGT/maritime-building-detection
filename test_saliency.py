from rembg import remove
from PIL import Image
import numpy as np
import cv2

IMAGE_PATH = r"C:\Storage\maritime_2\test_image\test_1.jpg"

image = Image.open(IMAGE_PATH)

output = remove(image)

output_np = np.array(output)

cv2.imwrite(
    "saliency_result.png",
    cv2.cvtColor(output_np, cv2.COLOR_RGBA2BGRA)
)

print("done")