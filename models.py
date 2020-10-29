## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        #input image is 224 by 224
        #output  = (w-f)+2p / s +1 ==> 224 - 5 +2*0 / 1 +1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5)
        # after maxpooling will be 110
        self.maxpool = nn.MaxPool2d(2,2)
#         output = 110 - 5 / 1 + 1 = 106
        self.conv2 = nn.Conv2d(32,36,5)
#         after maxpooling will be 53
        self.maxpool = nn.MaxPool2d(2,2)
#         output = 53 -5 / 1 + 1 = 49
        self.conv3 = nn.Conv2d(36,48,5)
        self.maxpool = nn.MaxPool2d(2,2)
        # after max pool would be 24 because it would be rounded 
        # output = 24 -3 / 1 + 1 = 22
        self.conv4 = nn.Conv2d(48,64,3)
        self.maxpool = nn.MaxPool2d(2,2)
        # output is 11 after max pooling (256,11,11)
        self.fc1 = nn.Linear(64*11*11,256)
        self.dropout = nn.Dropout(0.35)
        self.fc2 = nn.Linear(256,136)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        x = self.maxpool(F.relu(self.conv4(x)))
        
        x = x.view(x.size(0),-1)
       
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        
                         
        # a modified x, having gone through all the layers of your model, should be returned
        return x
