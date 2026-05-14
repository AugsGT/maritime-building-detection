import os
import random
import shutil

images_dir = r"C:/Storage/maritime_2/images"
labels_dir = r"C:/Storage/maritime_2/labels"

train_img_dir = r"C:/Storage/maritime_2/dataset/train/images"
train_lbl_dir = r"C:/Storage/maritime_2/dataset/train/labels"

val_img_dir = r"C:/Storage/maritime_2/dataset/val/images"
val_lbl_dir = r"C:/Storage/maritime_2/dataset/val/labels"

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

images = [f for f in os.listdir(images_dir)
          if f.endswith((".jpg", ".png", ".jpeg"))]

random.shuffle(images)

split_idx = int(len(images) * 0.8)

train_images = images[:split_idx]
val_images = images[split_idx:]

def move_files(image_list, img_dest, lbl_dest):
    for img in image_list:
        base = os.path.splitext(img)[0]
        label = base + ".txt"

        shutil.copy(
            os.path.join(images_dir, img),
            os.path.join(img_dest, img)
        )

        label_path = os.path.join(labels_dir, label)

        if os.path.exists(label_path):
            shutil.copy(
                label_path,
                os.path.join(lbl_dest, label)
            )

move_files(train_images, train_img_dir, train_lbl_dir)
move_files(val_images, val_img_dir, val_lbl_dir)

print("Dataset split complete.")