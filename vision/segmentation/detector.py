from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


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
        except: #sowwy my windows install is a bit out of date
            cfg.merge_from_file("C:/Users/Benedek/detectron2/detectron2/model_zoo/configs/Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml")
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
        cfg.freeze()
        return cfg