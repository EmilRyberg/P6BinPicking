from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import pkg_resources
import os


class InstanceDetector:
    def __init__(self, weights_path):
        self.cfg = self.__setup_cfg(weights_path)
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, img):
        """
        Args:
            img (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict)

        classes should be: ["PCB", "BottomCover", "BlueCover", "WhiteCover", "BlackCover"]
        """
        return self.predictor(img)

    def __setup_cfg(self, weights_path):
        cfg = get_cfg()
        try:
            cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml"))
        except:  # for windows installs
            cfg_file = pkg_resources.resource_filename(
                "detectron2.model_zoo", os.path.join("configs", "Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml")
            )
            cfg.merge_from_file(cfg_file)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
        #cfg.MODEL.DEVICE = 'cpu'
        cfg.freeze()
        return cfg