from vision.orientation.orientation_detector import OrientationDetectorNet
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from vision.orientation.orientation_dataset import OrientationDataset

def predict_on_image(image_name, detector):
    pil_image = Image.open(image_name)
    #print(np.array(pil_image).shape)
    resized_image = pil_image.resize((224, 224))
    resized_image_np = np.array(resized_image) / 255
    #resized_image_np = np.expand_dims(resized_image_np, axis=0) / 255
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    image_tensor = torch.from_numpy(resized_image_np).permute(2, 0, 1).float().to(device)
    #image_tensor = normalize(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)
    #print(image_tensor)
    #print(resized_image_np.shape)
    #print(image_tensor.shape)
    detector.eval()
    with torch.no_grad():
        prediction = detector(image_tensor)
    return prediction[0][0]

if __name__ == "__main__":
    device = torch.device("cuda")
    detector = OrientationDetectorNet().to(device)
    detector.load_hdf5_weights('orientation_cnn.hdf5')
    #detector.load_state_dict(torch.load('orientation_weights/model_0_loss_0.0002_acc_100.0.pth'))
    print(detector)
    print(f"left: {predict_on_image('WIN_20191120_12_28_57_Pro.jpg', detector)}")
    print(f"left: {predict_on_image('WIN_20191120_12_22_56_Pro.jpg', detector)}")
    print(f"left: {predict_on_image('WIN_20191120_12_38_10_Pro.jpg', detector)}")
    print(f"left: {predict_on_image('WIN_20191120_12_28_50_Pro.jpg', detector)}")
    print(f"right: {predict_on_image('WIN_20191120_13_10_06_Pro.jpg', detector)}")
    print(f"left: {predict_on_image('WIN_20191205_12_30_33_Pro.jpg', detector)}")
    # predict_on_image("WIN_20191120_12_38_10_Pro.jpg", detector)
    #predict_on_image("WIN_20191120_12_28_50_Pro.jpg", detector)
    #predict_on_image("WIN_20191120_13_10_06_Pro.jpg", detector)
