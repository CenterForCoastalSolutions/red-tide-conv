import torch.nn as nn

class Predictor(nn.Module): 
    def __init__(self, input_dim, num_classes):
        super(Predictor, self).__init__()
        self.FC1 = nn.Linear(input_dim, 15)
        self.FC2 = nn.Linear(15, 10)
        self.FC3 = nn.Linear(10, 10)
        self.FC4 = nn.Linear(10, 10)
        self.FC5 = nn.Linear(10, num_classes)
        self.LeakyReLU = nn.LeakyReLU(negative_slope = 0.2)
        self.Softmax = nn.Softmax(dim = 1)

    def forward(self, inputs):
        layer1 = self.LeakyReLU(self.FC1(inputs))
        layer2 = self.LeakyReLU(self.FC2(layer1))
        layer3 = self.LeakyReLU(self.FC3(layer2))
        layer4 = self.LeakyReLU(self.FC4(layer3))
        output = self.Softmax(self.FC5(layer4))
        return output