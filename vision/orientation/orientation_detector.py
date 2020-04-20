import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.linear import Linear
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import h5py


class OrientationDetectorNet(nn.Module):
    def __init__(self):
        super(OrientationDetectorNet, self).__init__()
        self.model = models.vgg16()
        self.model.avgpool = nn.Identity()
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.model.classifier(x)
        return x

    def load_hdf5_weights(self, hdf5_path):
        file = h5py.File(hdf5_path, 'r')
        weights = file['model_weights']
        feature_index = 0
        classifier_index = 0
        model_features = list(self.model.features.children())
        model_classifier = list(self.model.classifier.children())

        is_loading_features = True

        # Load
        for index, weight in enumerate(weights.values()):
            if "pool" in weight.name:
                continue

            layer = None
            if is_loading_features:
                layer = model_features[feature_index]
                if type(layer) is ReLU or type(layer) is Sigmoid or type(layer) is MaxPool2d:
                    valid_layer = False
                    while not valid_layer:
                        if feature_index >= len(model_features):
                            is_loading_features = False
                            break
                        layer = model_features[feature_index]
                        if type(layer) is ReLU or type(layer) is Sigmoid or type(layer) is MaxPool2d:
                            feature_index += 1
                        else:
                            valid_layer = True
            if not is_loading_features:
                layer = model_classifier[classifier_index]
                if type(layer) is ReLU or type(layer) is Sigmoid or type(layer) is MaxPool2d:
                    valid_layer = False
                    while not valid_layer:
                        if classifier_index >= len(model_classifier):
                            break
                        layer = model_classifier[classifier_index]
                        if type(layer) is ReLU or type(layer) is Sigmoid or type(layer) is MaxPool2d:
                            classifier_index += 1
                        else:
                            valid_layer = True

            #  print(f"weight: {weight}, model: {layer}")
            if type(layer) is Conv2d:
                key = list(file[weight.name].keys())[0]
                weight_block = file[f"{weight.name}/{key}"]
                bias = torch.from_numpy(np.array(weight_block["bias:0"]))
                weight_tensor = torch.from_numpy(np.array(weight_block["kernel:0"])).permute(3, 2, 0, 1)
                layer.bias.detach().copy_(bias)
                layer.weight.detach().copy_(weight_tensor)
            elif type(layer) is Linear:
                key = list(file[weight.name].keys())[0]
                weight_block = file[f"{weight.name}/{key}"]
                bias = torch.from_numpy(np.array(weight_block["bias:0"]))
                weight_tensor = torch.from_numpy(np.array(weight_block["kernel:0"])).t()
                layer.bias.detach().copy_(bias)
                layer.weight.detach().copy_(weight_tensor)

            if is_loading_features:
                feature_index += 1
                if feature_index >= len(model_features):
                    is_loading_features = False
            else:
                classifier_index += 1
                if classifier_index >= len(model_classifier):
                    break