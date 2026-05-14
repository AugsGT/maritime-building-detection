import os
import cv2
import random
import albumentations as A

IMAGE_DIR = r"C:/Storage/maritime_2/dataset/train/images"
LABEL_DIR = r"C:/Storage/maritime_2/dataset/train/labels"

OUT_IMAGE_DIR = r"C:/Storage/maritime_2/dataset/train_aug/images"
OUT_LABEL_DIR = r"C:/Storage/maritime_2/dataset/train_aug/labels"

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

transform = A.Compose(
    [
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.7
        ),

        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.3
        ),

        A.MotionBlur(
            blur_limit=5,
            p=0.3
        ),

        A.RandomFog(
            fog_coef_range=(0.1, 0.3),
            alpha_coef=0.08,
            p=0.4
        ),

        A.RandomRain(
            p=0.2
        ),

        A.HueSaturationValue(
            p=0.3
        ),

        A.Resize(640, 640)
    ],
    bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    )
)

images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith((".jpg", ".jpeg", ".png"))
]

AUGS_PER_IMAGE = 3

for image_file in images:

    image_path = os.path.join(IMAGE_DIR, image_file)

    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_file)

    image = cv2.imread(image_path)

    if image is None:
        continue

    bboxes = []
    class_labels = []

    if os.path.exists(label_path):

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls, x, y, w, h = parts

            bboxes.append([
                float(x),
                float(y),
                float(w),
                float(h)
            ])

            class_labels.append(int(cls))

    # save original resized image too
    resized = cv2.resize(image, (640, 640))

    cv2.imwrite(
        os.path.join(OUT_IMAGE_DIR, image_file),
        resized
    )

    with open(os.path.join(OUT_LABEL_DIR, label_file), "w") as f:
        for cls, box in zip(class_labels, bboxes):
            f.write(
                f"{cls} {' '.join(map(str, box))}\n"
            )

    for i in range(AUGS_PER_IMAGE):

        transformed = transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )

        aug_image = transformed["image"]
        aug_boxes = transformed["bboxes"]
        aug_labels = transformed["class_labels"]

        new_image_name = (
            f"{os.path.splitext(image_file)[0]}_aug_{i}.jpg"
        )

        new_label_name = (
            f"{os.path.splitext(image_file)[0]}_aug_{i}.txt"
        )

        cv2.imwrite(
            os.path.join(OUT_IMAGE_DIR, new_image_name),
            aug_image
        )

        with open(
            os.path.join(OUT_LABEL_DIR, new_label_name),
            "w"
        ) as f:

            for cls, box in zip(aug_labels, aug_boxes):

                x, y, w, h = box

                f.write(
                    f"{cls} {x} {y} {w} {h}\n"
                )

print("Augmentation complete.")