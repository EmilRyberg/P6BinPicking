import torch
from vision.yolo.models import Darknet
from vision.yolo.utils.datasets import resize, pad_to_square
from vision.yolo.utils.utils import non_max_suppression, rescale_boxes

from torchvision import transforms
import numpy as np


class Detector:
    def __init__(self, model_def, weights_path, img_size=608):
        print("Initializing YOLO Model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Darknet(model_def, self.device, img_size=img_size).to(self.device)
        self.img_size = img_size
        if weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(weights_path))

        self.model.eval()  # Set in evaluation mode
        print("Finished initializing YOLO Model")

    def predict(self, np_img):
        img = transforms.ToTensor()(np_img)
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)
        input_img = torch.tensor(img, dtype=torch.float32, device=self.device)
        input_img = input_img.reshape((3, self.img_size, self.img_size)).unsqueeze(0)

        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(detections, 0.8, 0.4)

        detections = detections[0]
        detections = rescale_boxes(detections, self.img_size, np_img.shape[:2])
        detections = detections.cpu()
        print("detections: ", np.array(detections))
        return np.array(detections)
