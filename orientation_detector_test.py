from vision.orientation.orientation_detector import OrientationDetectorNet
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch

if __name__ == "__main__":
    device = torch.device("cuda")
    detector = OrientationDetectorNet().to(device)
    detector.load_hdf5_weights('orientation_cnn.hdf5')
    pil_image = Image.open('WIN_20191120_12_28_57_Pro.jpg')
    #print(np.array(pil_image).shape)
    resized_image = pil_image.resize((224, 224))
    resized_image_np = np.array(resized_image) / 255
    #resized_image_np = np.expand_dims(resized_image_np, axis=0) / 255
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    image_tensor = torch.from_numpy(resized_image_np).permute(2, 0, 1).float().to(device)
    #image_tensor = normalize(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)
    print(image_tensor)
    print(resized_image_np.shape)
    print(image_tensor.shape)
    detector.eval()
    with torch.no_grad():
        prediction = detector.forward(image_tensor)
    print(prediction.detach()[0][0])