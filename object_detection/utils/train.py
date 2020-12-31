#!/usr/bin/env python3
import npz_to_yolo_format
import os
import subprocess


if __name__ == "__main__":
    npz_dir = "../dataset/sim_data/images"
    yolo_data_dir = "../dataset/sim_data/sim-data-yolo"
    images, bboxes, classes, files = npz_to_yolo_format.load_npz_files(npz_dir)
    _ = npz_to_yolo_format.convert_all_files(images, bboxes, classes, files, yolo_data_dir)
    npz_to_yolo_format.split_yolo_data_train_val(yolo_data_dir)
    # yaml_file = "dataset_real.yaml"
    yaml_file = "dataset_sim.yaml"
    subprocess.run(["python3", "./yolov5/train.py", "--img", "224", "--batch", "16", "--epochs", "200", "--data", yaml_file, "--weights", "yolov5s.pt"])
