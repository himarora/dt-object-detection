import os
import numpy as np
import cv2
import math
import random
from shutil import copyfile


def load_npz_files(npz_dir):
    files = os.listdir(npz_dir)
    files = [f for f in files if f[-4:] == ".npz"]
    n_samples = len(files)
    images = np.zeros((n_samples, 224, 224, 3), dtype=np.uint8)
    bboxes, classes = [], []
    for i, f in enumerate(files):
        sample = np.load(os.path.join(npz_dir, f), allow_pickle=True)
        img, bbox, c = sample["arr_0"], sample["arr_1"], sample["arr_2"]
        images[i] = img
        bboxes.append(np.array(bbox))
        classes.append(np.array(c))
    return images, bboxes, classes, files


def convert_to_yolo_format(img, bbox, cls):
    h, w = img.shape[:2]
    data = []
    for i, (box, c) in enumerate(zip(bbox, cls)):
        top_left = box[:2]
        bottom_right = box[2:]
        x_center = (top_left[0] + bottom_right[0]) * 0.5
        y_center = (top_left[1] + bottom_right[1]) * 0.5
        box_h = np.abs(top_left[1] - bottom_right[1])
        box_w = np.abs(top_left[0] - bottom_right[0])
        target = (c-1, x_center/w, y_center/h, box_h/h, box_w/w)    # -1 from label
        data.append(target)
    assert len(bbox) == len(cls) == len(data)
    return data


def convert_all_files(images, bboxes, classes, files, yolo_data_dir):
    images_dir = os.path.join(yolo_data_dir, "images", "all")
    labels_dir = os.path.join(yolo_data_dir, "labels", "all")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    count = 0
    for i, (img, bbox, cls, f) in enumerate(zip(images, bboxes, classes, files)):
        f = f[:-4]
        cv2.imwrite(os.path.join(images_dir, f"{f}.png"), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        labels = convert_to_yolo_format(img, bbox, cls)
        text = ""
        for l in labels:
            assert l[0] in [0, 1, 2, 3, 4]
            line = ' '.join([str(i) for i in l])
            text += line.strip() + "\n"
        text = text.strip()
        with open(os.path.join(labels_dir, f"{f}.txt"), 'w') as _:
            _.write(text)
        count += 1
    print(f"Converted {count} files from npz to YOLO format")
    return count


def split_yolo_data_train_val(yolo_data_dir, split_ratio_val=0.25):
    images_dir = os.path.join(yolo_data_dir, "images", "all")
    labels_dir = os.path.join(yolo_data_dir, "labels", "all")
    images_train_dir = os.path.join(yolo_data_dir, "images", "train")
    images_val_dir = os.path.join(yolo_data_dir, "images", "val")
    labels_train_dir = os.path.join(yolo_data_dir, "labels", "train")
    labels_val_dir = os.path.join(yolo_data_dir, "labels", "val")
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)
    total_images = len(os.listdir(images_dir))
    total_labels = len(os.listdir(labels_dir))
    assert total_images == total_labels
    n_val_samples = math.floor(split_ratio_val * total_images)
    n_train_samples = math.ceil((1 - split_ratio_val) * total_images)
    assert (n_val_samples + n_train_samples) == total_images == total_labels
    indices = set(range(total_images))
    train_indices = set(random.sample(indices, n_train_samples))
    val_indices = indices - train_indices
    for idx in train_indices:
        image_path = os.path.join(images_dir, f"{idx}.png")
        label_path = os.path.join(labels_dir, f"{idx}.txt")
        new_image_path = os.path.join(images_train_dir, f"{idx}.png")
        new_label_path = os.path.join(labels_train_dir, f"{idx}.txt")
        copyfile(image_path, new_image_path)
        copyfile(label_path, new_label_path)
    print(f"Copied {len(train_indices)} files to train directory")
    for idx in val_indices:
        image_path = os.path.join(images_dir, f"{idx}.png")
        label_path = os.path.join(labels_dir, f"{idx}.txt")
        new_image_path = os.path.join(images_val_dir, f"{idx}.png")
        new_label_path = os.path.join(labels_val_dir, f"{idx}.txt")
        copyfile(image_path, new_image_path)
        copyfile(label_path, new_label_path)
    print(f"Copied {len(val_indices)} files to val directory")
