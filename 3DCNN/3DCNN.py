import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions_V2 import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from numpy import save
torch.cuda.empty_cache()

# set path
data_path = "/home/faltay/3DCNN/Dataset_Brain_Binary_Variation"    
dementia_labels_path = "/home/faltay/3DCNN/Labels_Binary.pkl"  # load preprocessed dementia types
save_model_path = "/home/faltay/3DCNN/Results_9"  # save Pytorch models

# 3D CNN parameters
fc_hidden1, fc_hidden2 = 256, 256
dropout = 0.15        # dropout probability

# training parameters
k = 2            # number of target category
epochs = 40
batch_size = 10
learning_rate = 1e-5
log_interval = 10
img_x, img_y = 115 ,115  # resize video 2d frame size

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 70, 190, 1

#train(log_interval, cnn3d, device_2, train_loader_2, optimizer, epoch)
def train(log_interv_3, model_3, device_3, train_loader_3, optimizer_3, epoch_3):
    # set model as training mode
    model_3.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader_3):
        # distribute data to device

        X, y = X.to(device_3), y.to(device_3).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()

        output = model_3(X)  # output size = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_nues = y.cpu().data.squeeze().numpy()
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        y_pre = y_pred.cpu().data.squeeze().numpy()

        if y_nues.size == batch_size: 
            step_score = accuracy_score(y_nues, y_pre)


            scores.append(step_score)         # computed on CPU

            loss.backward()
            optimizer.step()

                # show information
            if (batch_idx + 1) % log_interv_3 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                        epoch_3 + 1, N_count, len(train_loader_3.dataset), 100. * (batch_idx + 1) / len(train_loader_3), loss.item(), 100 * step_score))

    return losses, scores


def validation(model, device, optimizer, test_loader):
    # set model as testing mode

    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():

        for X, y in test_loader:
            # distribute data to device

            X, y = X.to(device), y.to(device).view(-1, )

            output = model(X)

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(model.state_dict(), os.path.join(save_model_path, '3dcnn_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, '3dcnn_optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device_2 = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU


# load dementia types names
with open(dementia_labels_path, 'rb') as f:
    action_names = pickle.load(f)   # load dementia types names

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)


dementia_type = []
fnames = os.listdir(data_path)

all_names = []
for f in fnames:
    if f != ".ipynb_checkpoints": 
       
        loc1 = f.find('_')
        test = (f[0: loc1])
        # Temporarily we do not classify the unknown patients
        if test != "4" and test != "3":
            dementia_type.append(f[0: loc1])
            all_names.append(f)
            
# list all data files
all_X_list = all_names              # all video file names
all_y_list = labels2cat(le, dementia_type)    # all video labels

# train, test, val split so no patients are repeated 



patients = []
labels = []
prev = []
index = 0

for pat in all_X_list: 
    prev.append(pat[2:10])
for pat in all_X_list: 
    if pat[2:10] not in patients: 
        index = prev.index(pat[2:10])        

        patients.append(pat[2:10])
        labels.append(dementia_type[index])
        
## Undersampling process 
i = 0
for lab in labels: 
    if lab == "0":
        undersampling = np.random.choice([2,5], p=[0.9, 0.1])
        if undersampling ==2:
            del labels[i]
            del patients[i]


    i = i + 1
i = 0

for lab in labels: 
    if lab == "0":
        undersampling = np.random.choice([2,5], p=[0.4, 0.6])
        if undersampling ==2:
            del labels[i]
            del patients[i]


    i = i + 1
 


    
train_list_prev, val_list, train_label_prev, val_label = train_test_split(patients, labels, test_size=0.20, random_state=42)
train_list,test_list , train_label, test_label = train_test_split(train_list_prev, train_label_prev, test_size=0.15, random_state=42) 




# Training set (To match patients with scanners)

train_list_def = []
train_label_def = []

for pat in train_list:
    for scan in all_X_list: 
        if pat == scan[2:10]:
            train_list_def.append(scan)
            
for scan in train_list_def: 
    index = all_X_list.index(scan)        
    train_label_def.append(dementia_type[index])

train_list = train_list_def
train_label = np.array(train_label_def).astype(np.int)   

# Validation set 

val_list_def = []
val_label_def = []

for pat in val_list:
    for scan in all_X_list: 
        if pat == scan[2:10]:
            val_list_def.append(scan)
            
for scan in val_list_def: 
    index = all_X_list.index(scan)        
    val_label_def.append(dementia_type[index])

val_list = val_list_def
val_label = np.array(val_label_def).astype(np.int)  

# Test set 

test_list_def = []
test_label_def = []

for pat in test_list:
    for scan in all_X_list: 
        if pat == scan[2:10]:
            test_list_def.append(scan)
            
for scan in test_list_def: 
    index = all_X_list.index(scan)        
    test_label_def.append(dementia_type[index])

test_list = test_list_def
test_label = np.array(test_label_def).astype(np.int) 

# print("Size Dataset: ",len(all_X_list))
# print("Train Dataset: ",len(train_list))
# print("Test Dataset: ",len(test_list))
# print("Val Dataset: ",len(val_list))

save(save_model_path+'/train_list.npy', train_list)
save(save_model_path+'/test_list.npy', test_list)
save(save_model_path+'/train_label.npy', train_label)
save(save_model_path+'/test_label.npy', test_label)
save(save_model_path+'/val_list.npy', val_list)
save(save_model_path+'/val_label.npy', val_label)


# image transformation
transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

train_set  = Dataset_3DCNN(data_path, train_list, train_label, selected_frames, transform=transform)
valid_set = Dataset_3DCNN(data_path, val_list, val_label, selected_frames, transform=transform)



# Dealing with class imbalances

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images: 
        count[item[1]] += 1
        print(count, end="\r")
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))  
    print("")

    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])

        print(weight_per_class[i], end="\r")

    weight = [0] * len(images)  
    print("")

    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]] 

        print(weight[idx], end="\r")

    return weight 

#dataset_train = datasets.ImageFolder(traindir)                                                                         
                                                                                
# For unbalanced dataset we create a weighted sampler 

#weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes)) 

weights = make_weights_for_balanced_classes(train_set, k)                 

weights = torch.DoubleTensor(weights)  
print(len(weights))
samp = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights),replacement=True)    

# load dementia types names
params_train = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 6, 'pin_memory': True,'sampler' : samp} if use_cuda else {}
params_val =  {'batch_size': batch_size, 'shuffle': True, 'num_workers': 6, 'pin_memory': True} if use_cuda else {}


train_loader_2 = data.DataLoader(train_set, **params_train)

valid_loader_2 = data.DataLoader(valid_set, **params_val)




# create model
cnn3d = CNN3D(t_dim=len(selected_frames), img_x=img_x, img_y=img_y, drop_p=dropout, fc_hidden1=fc_hidden1,  fc_hidden2=fc_hidden2, num_classes=k).to(device_2)


# Total number of parameters
pytorch_total_params = sum(p.numel() for p in cnn3d.parameters())

# Total number of trainable parameters

pytorch_total_trainable_params = sum(p.numel() for p in cnn3d.parameters() if p.requires_grad)

print(pytorch_total_params)
print(pytorch_total_trainable_params)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn3d = nn.DataParallel(cnn3d)
    

optimizer = torch.optim.AdamW(cnn3d.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)
# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# start training
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train(log_interval, cnn3d, device_2, train_loader_2, optimizer, epoch)
    with torch.no_grad():
        epoch_test_loss, epoch_test_score = validation(cnn3d, device_2, optimizer, valid_loader_2)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save('/home/faltay/3DCNN/Results_9/3DCNN_epoch_training_losses.npy', A)
    np.save('/home/faltay/3DCNN/Results_9/3DCNN_epoch_training_scores.npy', B)
    np.save('/home/faltay/3DCNN/Results_9/3DCNN_epoch_test_loss.npy', C)
    np.save('/home/faltay/3DCNN/Results_9/3DCNN_epoch_test_score.npy', D)
   

# plot
fig = plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
# plt.plot(histories.losses_val)
plt.title("training scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train1', 'test'], loc="upper left")
title = "/home/faltay/3DCNN/Results_9/fig_DEMENTIA_3DCNN.png"
plt.savefig(title, dpi=600)
# plt.close(fig)
plt.show()