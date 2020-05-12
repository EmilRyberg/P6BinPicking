from vision.segmentation.detector import InstanceDetector
import cv2
import numpy as np

if __name__ == "__main__":
    detector = InstanceDetector('P6BinPicking/vision/model_final_sim.pth')
    img = cv2.imread('color1582022679.966812-0.png')
    results = detector.predict(img)
    #["PCB", "BottomCover", "BlueCover", "WhiteCover", "BlackCover"]
    mask = results["instances"].pred_masks[0].cpu().numpy()
    print(results["instances"].pred_classes[0])
    print(mask * 255)
    cv2.imwrite('box.jpg', mask * 255)