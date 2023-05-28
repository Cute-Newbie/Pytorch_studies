# Create a file called trainpy to leverage all of our other code scripts to train a pytorch model 
# !python train.py 

import numpy as np
import pandas as pd
import os
import random
import requests
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms


from data_setup import *
from engine import *
from model import *

# Download Data 

data_path = Path("data")
image_path = data_path/'pizza_steak_sushi'

if image_path.is_dir():
    print("Image-path directory alreay exist skip dowload")
else:
    print("Creating ONe")
    image_path.mkdir(parents = True,exist_ok = True)


with open(data_path/"pizza_steak_sushi.zip","wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading Pizza,steak,Sushi")
    f.write(request.content)

with zipfile.ZipFile(data_path/"pizza_steak_sushi.zip",'r') as zip_ref:
    print("unzipping data")
    zip_ref.extractall(image_path)
    print("Done")

## Set_up

train_dir = image_path/"train"
test_dir = image_path/"test"

data_transform = transforms.Compose([transforms.Resize(size = (64,64)),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ToTensor()])

train_dataloader,test_dataloader = data(train_dir = train_dir,
                                        test_dir=test_dir,
                                        data_transform=data_transform,
                                        batch_size = 32)



device = "cuda" if torch.cuda.is_available() else "cpu"


model0 = Basemodel(input_shape = 3,
                   hidden_unit = 10,
                   output_shape = 3).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model0.parameters(),lr = 0.001)
## 

## Train and test Model 

model0 = train(device = device,
      epochs = 50,
      model = model0,
      dataloader = train_dataloader,
      loss_fn = loss_fn,
      optimizer = optimizer)


test(device = device,
     model = model0,
     dataloader = test_dataloader,
     loss_fn = loss_fn,
     optimizer = optimizer)








