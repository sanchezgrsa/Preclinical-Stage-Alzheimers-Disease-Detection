  
import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

## ------------------- label conversion tools ------------------ ##
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


## ---------------------- Dataloaders ---------------------- ##
# for 3DCNN
class Dataset_3DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
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
        y = torch.LongTensor([self.labels[index]])                             # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y


## ---------------------- end of Dataloaders ---------------------- ##



## -------------------- (reload) model prediction ---------------------- ##
def Conv3d_final_prediction(model, device, loader):
    model.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = model(X)
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy())
               # to compute accuracy
      
             
    return all_y_pred



## -------------------- end of model prediction ---------------------- ##



## ------------------------ 3D CNN module ---------------------- ##
def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))



    print(outshape)
    return outshape

class CNN3D(nn.Module):
    def __init__(self, t_dim=120, img_x=90, img_y=120, drop_p=0.2, fc_hidden1=128, fc_hidden2=128, num_classes=4):
        super(CNN3D, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        
        self.ch1, self.ch2,self.ch3,self.ch4 ,self.ch5 = 48, 64 , 80 , 64, 48
        self.k1, self.k2, self.k3, self.k4, self.k5= (3, 3, 3),(3,3, 3), (3, 3, 3), (3, 3, 3) , (3, 3, 3)# 3d kernel size
        self.s1, self.s2, self.s3, self.s4, self.s5= (1, 1, 1), (2, 2, 2) ,(2, 2, 2), (2, 2, 2) , (2, 2, 2)# 3d strides
        self.pd1, self.pd2,self.pd3,self.pd4,self.pd5= (0, 0, 0), (0, 0, 0),(0, 0, 0), (0, 0, 0) , (0, 0, 0) # 3d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = conv3D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        self.conv4_outshape = conv3D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)
        self.conv5_outshape = conv3D_output_size(self.conv4_outshape, self.pd5, self.k5, self.s5)

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2) 
        self.bn2 = nn.BatchNorm3d(self.ch2)
        
        self.conv3 = nn.Conv3d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3)   
        self.bn3 = nn.BatchNorm3d(self.ch3)
        
        self.conv4 = nn.Conv3d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4)   
        self.bn4 = nn.BatchNorm3d(self.ch4)
        
        self.conv5 = nn.Conv3d(in_channels=self.ch4, out_channels=self.ch5, kernel_size=self.k5, stride=self.s5,
                               padding=self.pd5)   
        self.bn5 = nn.BatchNorm3d(self.ch5)
      

      
        
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)  
        self.fc1 = nn.Linear(self.ch5 * self.conv5_outshape[0] * self.conv5_outshape[1] * self.conv5_outshape[2],self.fc_hidden1) 
        # fully connected hidden layer       
        # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = multi-classes

    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.drop(x)

        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x

## --------------------- end of 3D CNN module ---------------- ##



