import os
import glob

import numpy as np
import skimage.io as sio
import skimage.util as cropper
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torchvision.transforms as transforms
from skimage.transform import  resize
from PIL import Image 

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, folders, labels, frames, transform=None):

        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'image-slice{:03d}.jpg'.format(i))).convert('L')
            
            if use_transform is not None:
                image = use_transform(image)          
            
            X.append(image.squeeze_(0))
        
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform).unsqueeze_(0)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])
        X = torch.squeeze(X)
        return X, y