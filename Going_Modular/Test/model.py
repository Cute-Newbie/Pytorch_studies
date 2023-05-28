import torch
import torch.nn as nn 
import torch.utils
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms


#Creat CNN (Convolutional Neural Network)

class Basemodel(nn.Module):
    """

    MOdel architecture that replicates the TinyVGG
    Model from blahblah

    """

    def __init__(self,
                 input_shape:int,
                 hidden_unit:int,
                 output_shape:int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_unit, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=0), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_unit, 
                      out_channels=hidden_unit,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_unit, hidden_unit, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_unit, hidden_unit, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_unit*13*13,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

    

        
