#!/usr/bin/env python3

import numpy as np
import os
from agent import PurePursuitPolicy
import cv2
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask
from shutil import rmtree

DATASET_DIR="/home/himarora/PycharmProjects/dt-exercises-object-det/object_detection/dataset/sim_data/images"
COLORS = [(0, 0, 0), (100, 117, 226), (226, 111, 101), (116, 114, 117), (216, 171, 15)]    # (255, 0, 255) Background
npz_index = 0


def save_npz(img, boxes, classes):
    global npz_index
    with makedirs(DATASET_DIR):
        np.savez(f"{DATASET_DIR}/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1


def find_color(patch):
    # Too slow
    colors, count = np.unique(patch.reshape(-1, patch.shape[-1]), axis=0, return_counts=True)
    color = tuple(colors[count.argmax()])
    return color
    # return COLORS[1]


def get_bbox_classes(clean_img, min_area=63):
    gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    bboxes, classes = [], []
    for cnt in contours:
        xywh = cv2.boundingRect(cnt)
        area = xywh[2] * xywh[3]
        if area > min_area:
            box = [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]
            cropped_img = clean_img[box[1]:box[3], box[0]:box[2]]
            color = find_color(cropped_img)
            try:
                class_id = COLORS.index(color)
            except ValueError:
                class_id = 0
            if class_id:
                # Cone
                if class_id == 2:
                    box[1] -= 15    # increase height
                classes.append(class_id)
                bboxes.append(box)
    return bboxes, classes


def save_image(img, path):
    global npz_index
    os.makedirs(path, exist_ok=True)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{path}/{npz_index}.png", img)


def remove_lanes(image):
    masks = []
    for c in COLORS:
        mask = cv2.inRange(image, c, c) / 255
        masks.append(mask)
    masks = np.array(masks)
    mask = (masks.sum(axis=0) * 255).astype(np.uint8)
    image = cv2.bitwise_and(image, image, mask=mask)
    return image


def remove_colored_snow(image, k_erode=7, k_dilate=7):
    kernel = np.ones((k_erode, k_erode), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    kernel = np.ones((k_dilate, k_dilate), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    return image


def clean_segmented_image(seg_img):
    seg_img = remove_lanes(seg_img)
    seg_img = remove_colored_snow(seg_img)
    return seg_img


def plot_boxes(true_image, boxes, classes, save_dir=None):
    for i, (box, c) in enumerate(zip(boxes, classes)):
        # color = self.colors[label]
        if save_dir:
            patch_dir = os.path.join(save_dir, str(c))
            patch_path = os.path.join(patch_dir, f"{npz_index}_{i}.png")
            patch = true_image[box[1]:box[3], box[0]:box[2]]
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            cv2.imwrite(patch_path, patch)
        true_image = cv2.rectangle(true_image, (box[0], box[1]), (box[2], box[3]), COLORS[c], 1)
    return true_image


if __name__ == "__main__":
    seed(123)
    environment = launch_env()

    policy = PurePursuitPolicy(environment)

    MAX_STEPS = 250

    rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR)

    n_classes = 4
    # patches_dir = os.path.join(DATASET_DIR, "patches")
    patches_dir = None
    if patches_dir:
        for i in range(1, n_classes + 1):
            os.makedirs(os.path.join(patches_dir, str(i)), exist_ok=True)

    while True:
        obs = environment.reset()
        environment.render(segment=True)
        rewards = []

        nb_of_steps = 0
        count = 0
        while True:
            # if nb_of_steps % 500 == 0:
            #     print(f"Completed {nb_of_steps} steps")
            action = policy.predict(np.array(obs))

            obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
            segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

            rewards.append(rew)
            # Skip the first image and save every 20 steps
            if count > 19 and count % 20 == 0:
                environment.render(segment=int(nb_of_steps / 50) % 2 == 0)
                clean_image = clean_segmented_image(segmented_obs)
                boxes, classes = get_bbox_classes(clean_image)
                save_image(segmented_obs, path=f"{DATASET_DIR}/seg")
                save_image(clean_image, path=f"{DATASET_DIR}/clean")

                # Rescale image and bbox coordinates
                h, w = obs.shape[:2]
                x_scale, y_scale = 224/w, 224/h
                obs = cv2.resize(obs, (224, 224))
                save_image(obs, path=f"{DATASET_DIR}/true")
                for box in boxes:
                    box[0] = int(box[0] * x_scale)
                    box[1] = int(box[1] * y_scale)
                    box[2] = int(box[2] * x_scale)
                    box[3] = int(box[3] * y_scale)
                # All images
                img_true_bx = obs.copy()
                img_true_bx = plot_boxes(img_true_bx, boxes, classes, save_dir=patches_dir)
                # all_images = cv2.hconcat([img_true_bx, segmented_obs, clean_image])
                save_image(img_true_bx, path=f"{DATASET_DIR}/bbox")

                save_npz(obs, boxes, classes)
                nb_of_steps += 1
                if done or count > MAX_STEPS:
                    break
            count += 1
        if npz_index == 1999:
            exit()
