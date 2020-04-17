from vision.orientation.orientation_detector import OrientationDetectorNet
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from vision.orientation.orientation_dataset import OrientationDataset
import math

if __name__ == "__main__":
    TRAIN_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    train_dataset = OrientationDataset("orientation_dataset/training_set")
    trainloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)
    test_dataset = OrientationDataset("orientation_dataset/test_set")
    testloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=2)
    val_dataset = OrientationDataset("orientation_dataset/validation_set")
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=2)
    #image_paths, images, labels = dataiter.next()
    #print(images)

    device = torch.device("cuda")
    net = OrientationDetectorNet().to(device)
    net.load_hdf5_weights('orientation_cnn.hdf5')

    criterion = torch.nn.BCELoss()
    optimizer = Adam(net.parameters(), lr=0.00006)

    for epoch in range(2):
        running_loss = 0.0
        running_acc = 0.0
        for index, data in enumerate(trainloader):
            image_paths, images, labels = data
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)

            #print(f"outputs: {outputs}, labels: {labels}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            correct = np.array([1 if (output[0] >= 0.5 and label[0] == 1) or (output[0] < 0.5 and label[0] == 0) else 0 for output, label in zip(outputs, labels)])
            #print(f"outputs: {outputs}\n labels: {labels}\n correct: {correct}")
            running_acc += correct.mean()
            running_loss += loss.item()
            if index % 5 == 4:
                print('[%d, %5d] loss: %.3f, acc: %.3f%%' %
                      (epoch + 1, index + 1, running_loss / 5, (running_acc / 5) * 100))
                running_loss = 0.0
                running_acc = 0.0

        test_loss = 0.0
        test_acc = 0.0
        print("running test set")
        for index, data in enumerate(testloader):
            image_paths, images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)

            loss = criterion(outputs, labels)

            correct = np.array([1 if (output[0] >= 0.5 and label[0] == 1) or (output[0] < 0.5 and label[0] == 0) else 0 for output, label in zip(outputs, labels)])
            test_acc += correct.mean()
            test_loss += loss.item()

        avg_loss = test_loss / (math.ceil(len(test_dataset) / TEST_BATCH_SIZE))
        avg_acc = (test_acc / (math.ceil(len(test_dataset) / TEST_BATCH_SIZE))) * 100
        print('[%d] test loss: %.3f, test acc: %.3f%%' % (epoch + 1, avg_loss, avg_acc))
        print('saving checkpoint')
        torch.save(net.state_dict(), f'orientation_weights/model_{epoch}_loss_{avg_loss:.3}_acc_{avg_acc:.3}.pth')

