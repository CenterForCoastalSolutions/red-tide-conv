import torch
import torch.nn as nn
from nasnet_mobile import *

class nasnet_wrapper_model(nn.Module): 
    def __init__(self, num_classes, use_nn_feature):
        super(nasnet_wrapper_model, self).__init__()
        self.use_nn_feature = use_nn_feature
        self.nasnet_model = NASNetAMobile(num_classes=num_classes).cuda()

        self.FC1 = nn.Linear(9504, 256)
        self.BN1 = nn.BatchNorm1d(num_features=256)
        self.FC2 = nn.Linear(256, 9)
        self.BN2 = nn.BatchNorm1d(num_features=9)
        if(use_nn_feature == 0):
            self.FC3 = nn.Linear(9, num_classes)
        else:
            self.FC3 = nn.Linear(10, num_classes)
        self.LeakyReLU = nn.LeakyReLU(negative_slope = 0.2)
        self.Softmax = nn.Softmax(dim = 1)

    def forward(self, input_images, input_knn):
        model_features, output = self.nasnet_model(input_images)

        model_features = model_features[:, :, 2:5, 2:5]

        model_features = torch.reshape(model_features, (-1, 9504))

        layer1 = self.LeakyReLU(self.BN1(self.FC1(model_features)))
        layer2 = self.LeakyReLU(self.BN2(self.FC2(layer1)))
        if(self.use_nn_feature == 0):
            output = self.Softmax(self.FC3(layer2))
        else:
            input_knn = torch.unsqueeze(input_knn, 1)
            layer3_input = torch.cat((layer2, input_knn), dim=1)
            output = self.Softmax(self.FC3(layer3_input))
        return output