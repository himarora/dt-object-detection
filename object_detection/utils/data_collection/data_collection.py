#!/usr/bin/env python3

import numpy as np
import os
from agent import PurePursuitPolicy
from utils import launch_env, seed
import cv2
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask

DATASET_DIR="/home/himarora/PycharmProjects/dt-exercises-object-det/object_detection/dataset/sim_data/images"

npz_index = 0
def save_npz(img, boxes, classes):
    global npz_index
    with makedirs(DATASET_DIR):
        np.savez(f"{DATASET_DIR}/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1


def get_bbox_classes(seg_img):
    pass


def save_image(img, seg=False):
    global npz_index
    dir_name = "true" if not seg else "seg"
    dir_path = f"{DATASET_DIR}/{dir_name}"
    os.makedirs(dir_path, exist_ok=True)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{dir_path}/{npz_index}.png", img)
    npz_index += 1 if seg else 0


def clean_segmented_image(seg_img):
    # Tip: use either of the two display functions found in util.py to ensure that your cleaning produces clean masks
    # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes
    return seg_img
    # return boxes, classes

seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500

while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        image = clean_segmented_image(segmented_obs)
        save_image(obs, seg=False)
        save_image(image, seg=True)
        # boxes, classes = clean_segmented_image(segmented_obs)
        # save_npz(obs, boxes, classes)

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break