import torch
import numpy as np
from .yolov5.models.yolo import Model
from .yolov5.utils.general import non_max_suppression, scale_coords
import cv2


class Wrapper:
    """
    Loads a pre-trained YOLOv5 model for inference
    """

    def __init__(self, model_file, image_size=224, n_classes=2):
        self.weights_path = model_file
        self.device = "cpu" # "cuda" if torch.cuda.is_available() else # YOLOv5-s should work fine on CPU as well
        self.model = Model("/code/exercise_ws/src/object_detection/include/object_detection/yolov5/models/yolov5s.yaml",
                           ch=3, nc=n_classes)
        weights = torch.load(self.weights_path, map_location=self.device)
        print(self.model.load_state_dict(weights))
        self.model.eval()
        self.classes = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.class_id = tuple(range(len(self.classes)))
        if len(self.classes) == 2:
            self.colors = ((255, 255, 0), (0, 165, 255))  # 0: duckie, 1: cone
        else:
            self.colors = ((100, 117, 226), (226, 111, 101), (116, 114, 117), (216, 171, 15))  # 0: duckie, 1: cone, 2: truck, 3: bus

        self.im_sz = image_size
        print(f"Loaded model {self.weights_path} to detect {self.classes} with ids {self.class_id}")

    def predict(self, batch_or_image):
        boxes_batch = []
        labels_batch = []
        scores_batch = []
        if batch_or_image.ndim == 3:
            batch_or_image = np.expand_dims(batch_or_image, axis=0)
        for img in batch_or_image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).to(self.device).float().permute(2, 0, 1).unsqueeze(0)
            img /= 255.0
            with torch.no_grad():
                pred = self.model(img, augment=False)
            pred = non_max_suppression(pred[0], 0.25, 0.45, classes=self.class_id, agnostic=False)
            boxes, labels, scores = [], [], []
            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (self.im_sz, self.im_sz, 3)).round()
                    for d in det:
                        c, s, b = int(d[-1].cpu().numpy()), d[-2].cpu().numpy(), d[:4].cpu().numpy()
                        boxes.append(b)
                        labels.append(c)
                        scores.append(s)
            boxes_batch.append(boxes)
            labels_batch.append(labels)
            scores_batch.append(scores)
        return boxes_batch, labels_batch, scores_batch

    def visualize(self, img: np.ndarray, boxes: np.ndarray, labels: np.ndarray):
        if not len(boxes):
            return img
        labels = np.array(labels, dtype=np.int)
        img = img.copy()
        for i, (box, label) in enumerate(zip(boxes, labels)):
            color = self.colors[label]
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 1)
        return img


class NoGPUAvailable(Exception):
    pass
