from rembg import remove
from PIL import Image
import numpy as np
import cv2

IMAGE_PATH = r"C:\Storage\maritime_2\test_image\test_1.jpg"

image = Image.open(IMAGE_PATH)

output = remove(
    image,
    only_mask=True
)

mask = np.array(output)

cv2.imwrite("mask.png", mask)

print(mask.shape)