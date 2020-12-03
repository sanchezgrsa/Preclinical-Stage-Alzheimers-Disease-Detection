import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import model_org
import data_loader as brain_data_loader
import original_model
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score,  confusion_matrix
from sklearn.metrics import f1_score

import torch as T

## -------------------- (reload) model prediction ---------------------- ##
def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()

def testing(model, data_loader, sequence_len, device):

    model.eval()

    all_y = []
    all_y_pred = []

    with torch.no_grad():
          for X, y in data_loader:
            # distribute data to device

            X, y = X.to(device), y.to(device).view(-1, )

            h_1, c_1, h_2, c_2, l_0 = model(X, None, None, None, None, None, first=True)
            l_t = l_0

            
            for t in range(sequence_len-1):

                    h_1, c_1, h_2, c_2, l_t, b_t, p = model(X, l_t, h_1, c_1, h_2, c_2)


            # Last time step
            h_1, c_1, h_2, c_2, l_t, b_t, p, log_probas = model(X, l_t, h_1, c_1, h_2, c_2, last=True)

            
            
            y_pred = torch.max(log_probas,1)[1]

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)


    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)

    original_labels = all_y.cpu().data.squeeze().numpy()
    predicted_labels = all_y_pred.cpu().data.squeeze().numpy()    

    test_score = accuracy_score(original_labels,predicted_labels)

    # show information
        
    return  original_labels , predicted_labels, test_score*100



# set path
  
dementia_labels_path ="/home/faltay/Glimpse/Labels_Binary.pkl"  # load preprocessed dementia types
save_model_path ="/home/faltay/Glimpse/Saved_Models_200"
data_path ="/home/faltay/Glimpse/Dataset_Brain_Binary_Variation"


sequence_len = 6
glimse_size= [40,40,40]
std = 0.2
h_g = h_l = 512
hidden_size = 512
num_classes = 2
batch_size = 10
img_x = 224
img_y = 160 
begin_frame, end_frame, skip_frame = 70, 190, 1


# Detect devices
use_cuda = T.cuda.is_available()                   # check if GPU exists
T.cuda.set_device(1)
device = T.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}


# image transformation
with open(dementia_labels_path, 'rb') as f:
    action_names = pickle.load(f)   # load  dementia_type names

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
        if test != "4":
            dementia_type.append(f[0: loc1])
            all_names.append(f)

# list all data files
all_X_list_prev = np.load(save_model_path+'/test_list.npy')
all_X_list = all_X_list_prev.tolist()

all_y_list_prev = np.load(save_model_path+'/test_label.npy')
all_y_list = all_y_list_prev.tolist()


# data loading parameters
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# image transformation

transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()





# reset data loader
all_data_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

all_data_loader = data.DataLoader(brain_data_loader.BrainDataset(data_path, all_X_list, all_y_list, selected_frames, transform=transform), **all_data_params)

model = original_model.RecurrentAttention(glimse_size, h_g, h_l, std, hidden_size, num_classes).to(device) 




model.load_state_dict(torch.load(os.path.join(save_model_path, 'Attention_Model_epoch200.pth')),strict=False)

 # Detect devices
use_cuda = T.cuda.is_available()                   # check if GPU exists
T.cuda.set_device(1)
device = T.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

print('[INFO] model reloaded!')


print('Predicting all {} videos:'.format(len(all_data_loader.dataset)))

original_labels, all_y_pred, acc = testing(model, all_data_loader, sequence_len, device)



df = {}
# write in pandas dataframe
df = pd.DataFrame(data={'filename': all_X_list, 'y': cat2labels(le, all_y_list), 'y_pred': cat2labels(le, all_y_pred)})
df.to_pickle("/home/faltay/3DCNN/Results_9/Dementia_prediction.pkl")  # save pandas dataframe
# pd.read_pickle("./all_videos_prediction.pkl")

print('video prediction finished!')



df_wrong = df[df['y'] != df['y_pred']] 
df_correct = df[df['y'] == df['y_pred']] 
print("Accuracy: ", (len(df_correct)/len(df))*100)

cm_00 = len(df.loc[(df['y'] == "0") & (df['y_pred'] == "0")])
cm_01 = len(df.loc[(df['y'] == "0") & (df['y_pred'] == "1")])
cm_02 = len(df.loc[(df['y'] == "0") & (df['y_pred'] == "2")])

cm_10 = len(df.loc[(df['y'] == "1") & (df['y_pred'] == "0")])
cm_11 = len(df.loc[(df['y'] == "1") & (df['y_pred'] == "1")])
cm_12 = len(df.loc[(df['y'] == "1") & (df['y_pred'] == "2")])

cm_20 = len(df.loc[(df['y'] == "2") & (df['y_pred'] == "0")])
cm_21 = len(df.loc[(df['y'] == "2") & (df['y_pred'] == "1")])
cm_22 = len(df.loc[(df['y'] == "2") & (df['y_pred'] == "2")])

cm_30 = len(df.loc[(df['y'] == "3") & (df['y_pred'] == "0")])
cm_31 = len(df.loc[(df['y'] == "3") & (df['y_pred'] == "1")])
cm_32 = len(df.loc[(df['y'] == "3") & (df['y_pred'] == "2")])


#confussion_matriz = pd.DataFrame(np.array([[cm_00, cm_01, cm_02], [cm_10, cm_11, cm_12], [cm_20, cm_21, cm_22]]),
                #columns=['0', '1', '2'])
confussion_matriz = pd.DataFrame(np.array([[cm_00, cm_01], [cm_10, cm_11]]),
                columns=['0', '1'])
precision_0 = cm_00/(cm_00+cm_01+cm_02)
recall_0 = cm_00 /(cm_00 + cm_10 + cm_20)
f1_0 = 2*(precision_0*recall_0)/(precision_0+recall_0)

print("Class 0: Precision: ", precision_0)
print("Class 0: Recall: ", recall_0)
print("Class 0: F1 Score: ", f1_0)
print("")

precision_1 = cm_11/(cm_11+cm_10+cm_12)
recall_1 = cm_11 /(cm_11 + cm_01 + cm_21)
f1_1 = 2*(precision_1*recall_1)/(precision_1+recall_1)

print("Class 1: Precision: ", precision_1)
print("Class 1: Recall: ", recall_1)
print("Class 1: F1 Score: ", f1_1)
print("")
  
    
micro_fi = f1_score(df['y'].tolist(), df['y_pred'].tolist(),   average='micro')
    
    
print("Micro F1_Score",micro_fi )    
    
    
print(confussion_matriz)