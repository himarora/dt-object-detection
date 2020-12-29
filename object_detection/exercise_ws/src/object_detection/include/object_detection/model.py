import torch
from .yolov5.models.yolo import Model


class NoGPUAvailable(Exception):
    pass


class Wrapper:
    """
    Loads a pre-trained YOLOv5 model for inference
    """
    def __init__(self, model_file, image_size=224):
        self.weights_path = model_file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"    # YOLOv5-s should work fine on CPU as well
        self.model = Model("/code/exercise_ws/src/object_detection/include/object_detection/yolov5/models/yolov5s.yaml", ch=3, nc=2)
        weights = torch.load(self.weights_path, map_location=self.device)
        print(self.model.load_state_dict(weights))
        self.model.eval()
        self.classes = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.class_id = tuple(range(len(self.classes)))
        self.im_sz = image_size
        print(f"Loaded model {self.weights_path} to detect {self.classes} with ids {self.class_id}")

    def predict(self, batch_or_image):
        # TODO Make your model predict here!

        # TODO The given batch_or_image parameter will be a numpy array (ie either a 224 x 224 x 3 image, or a
        # TODO batch_size x 224 x 224 x 3 batch of images)
        # TODO These images will be 224 x 224, but feel free to have any model, so long as it can handle these
        # TODO dimensions. You could resize the images before predicting, make your model dimension-agnostic somehow,
        # TODO etc.

        # TODO This method should return a tuple of three lists of numpy arrays. The first is the bounding boxes, the
        # TODO second is the corresponding labels, the third is the scores (the probabilities)

        # See this pseudocode for inspiration
        boxes = []
        labels = []
        scores = []
        for img in batch_or_image:  # or simply pipe the whole batch to the model instead of using a loop!

            box, label, score = self.model.predict(img) # TODO you probably need to send the image to a tensor, etc.
            boxes.append(box)
            labels.append(label)
            scores.append(score)

        return boxes, labels, scores
