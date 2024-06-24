import torch
from torch import nn

class PKLotDetector(nn.Module):
    def __init__(self):
        super(PKLotDetector, self).__init__()
        
        # convolution layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # batch norm
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout(0.5)
        
        # fully connected layers
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 1)

        # relu, sigmoid and maxpool funcs
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: torch.Tensor):
        # first convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        # second convolution block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        # third convolution block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        # print(x.shape)


        
        # transponing tensor to vector
        x = x.view(x.shape[0], -1)

        # first fully-connected block
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # second fully-connected block
        x = self.fc2(x)


        # using sigmoid to get probability
        x = self.sigmoid(x)
        return x
    
    
    def predict_single_image(self, image):
        # this function shoudn't be used in train so we don't need to count gradient's
        with torch.no_grad():
            # add batch dimension
            image = image.unsqueeze(0)
            # getting output and turning it into the probabitily
            output = self(image)
            probability = output.item()
            # return 1 if occupied and 0 if not
            return int(probability > 0.5)
