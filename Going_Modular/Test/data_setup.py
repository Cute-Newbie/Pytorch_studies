import requests
import pathlib
from pathlib import Path 

import torchvision
from torchvision import datasets,transforms
import torch
import torch.utils
from torch.utils.data import Dataset,DataLoader







def data(train_dir,test_dir,data_transform,batch_size):

    
    ## Custom Dataset for traint,test 
    train_data = datasets.ImageFolder(root = train_dir,
                                      transform = data_transform,
                                      target_transform = None)
    
    test_data = datasets.ImageFolder(root = test_dir,
                                      transform = data_transform,
                                      target_transform = None)
    
    train_dataloader = DataLoader(dataset = train_data,
                                  batch_size = batch_size,
                                  shuffle = True
                                  )
    

    test_dataloader = DataLoader(dataset = test_data,
                                  batch_size = batch_size,
                                  shuffle = False
                                  )
    

    return train_dataloader,test_dataloader 




