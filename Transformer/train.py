import os
import numpy as np
import torchvision
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.transforms import transforms
import transformer_v3 as transformer
import data_loader as brain_data_loader
import random
from tqdm import tqdm
import torch.utils.data as data
import time
import pickle5 as pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
## ------------------- label conversion tools ------------------ #

def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


def train(model, data_loader, optimizer, epoch, device, loss_update_interval=10):
    """Definition of one epoch procedure.
    """
    model.train()   
    epoch_loss = []
    criterion = nn.CrossEntropyLoss()
    device_2 = next(model.parameters()).device
        
    for i, (X_cpu, y_cpu) in enumerate(data_loader):
        X, y = X_cpu.to(device_2, dtype=torch.float), y_cpu.to(device_2, dtype=torch.float)
        y = torch.squeeze(y)

        optimizer.zero_grad()
        
        output, _ = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Display
        if i%loss_update_interval == 0:
        
            print("[INFO] iteration: %s" %(i), " Training Loss: %.3f" % (loss.item()))

        epoch_loss.append(loss.item())
    
    final_epoch_loss = np.mean(epoch_loss)
    print("[INFO] Epoch (%s) Training Summary: "%(epoch), "Loss: %.3f" % (final_epoch_loss) )

    return final_epoch_loss


    


def validation(model, data_loader, optimizer, epoch, device, holder, test):

    """Definition of one epoch procedure.
    """
 
    model.eval()
    device_2 = next(model.parameters()).device

    test_correct = test_total = 0

    for i, (X_cpu, y_cpu) in enumerate(data_loader):
        X, y = X_cpu.to(device_2, dtype=torch.float), y_cpu.to(device_2, dtype=torch.float)
        
        y = torch.squeeze(y)
        output, _ = model(X)
        
        _, predicted = T.max(output, 1)

        correct = (predicted == y).sum().item()
        
        total = len(y)

        test_correct += correct
        test_total += total

    print("[INFO] predicted correctly :", test_correct)
    print("[INFO] total # of predictions:", test_total)
    acc = test_correct*100/float(test_total)
    bigger = False
    
    if not test:

        if holder < acc:

            bigger = True

        if bigger:
            save_model_path = "/home/faltay/transformer/Saved_Models/"
            torch.save(model.state_dict(), os.path.join(save_model_path, 'transformer_epoch{}_{}.pth'.format(epoch, acc)))  # save spatial_encoder
            # torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'transformer_optimizer_epoch{}.pth'.format(epoch)))      # save optimizer
            print("Epoch {} model saved!".format(epoch))
            print("[INFO] Epoch (%s) Validation Summary: "%(epoch), " Validation Accuracy: ", '%.1f' % (acc))
        else:
            print("[INFO] Epoch (%s) Validation Summary: "%(epoch), " Validation Accuracy: ", '%.1f' % (acc))

    elif test:
        predicted_labels = predicted.cpu().data.squeeze().numpy()
        original_labels = y.cpu().data.squeeze().numpy()
        confusione = confusion_matrix(predicted_labels, original_labels)
        print("[INFO] Test Summary: Test Accuracy: ", '%.1f' % (acc))
        print(confusione)

        micro_fi = f1_score(original_labels, predicted_labels, average='micro')
        print('Micro F1_score: ', micro_fi)

    return test_correct, test_total



def all_main():

    start_time = time.time()

    epochs = 200
    batch_size = 4    
    num_classes = 2
    
    data_path = '/home/faltay/Scripts/Attention_Mechanisms/Dataset_Brain_Binary_Variation'
    dementia_labels_path = '/home/faltay/Scripts/3_CNN/Labels_Binary.pkl'
    save_model_path = '/home/faltay/transformer/Saved_Models'

    print("[INFO] Starting ...")
    
    # Detect devices
    use_cuda = T.cuda.is_available()                   # check if GPU exists
    # T.cuda.set_device(0)
    device = T.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
    
     
    # record training process
    epoch_train_losses = []
    epoch_test_losses = []
    
    print("[INFO] Data is now loading ...")
    # Load data
    T.manual_seed(30)

    # image transformation
    img_x = 224
    img_y = 224 
    begin_frame, end_frame, skip_frame = 70, 190, 1  # or 64 - 172 based on GPU ram 
    seq_len = end_frame - begin_frame

    
    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    ## Data processing until line 290
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
            if undersampling == 2:
                del labels[i]
                del patients[i]        
        i = i + 1

    i = 0
    for lab in labels: 
        if lab == "0":
            undersampling = np.random.choice([2,5], p=[0.4, 0.6])
            if undersampling == 2:
                del labels[i]
                del patients[i]
        i = i + 1

    
    train_list_prev, val_list, train_label_prev, val_label = train_test_split(patients, labels, test_size=0.20, random_state=42)
    train_list, test_list, train_label, test_label = train_test_split(train_list_prev, train_label_prev, test_size=0.15, random_state=42) 


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


    print("Size Dataset: ",len(all_X_list))
    print("Train Dataset: ",len(train_list))
    print("Val Dataset: ",len(val_list))
    print("Test Dataset: ",len(test_list))



    np.save(save_model_path+'/train_list.npy', train_list)
    np.save(save_model_path+'/test_list.npy', test_list)
    np.save(save_model_path+'/train_label.npy', train_label)
    np.save(save_model_path+'/test_label.npy', test_label)
    np.save(save_model_path+'/val_list.npy', val_list)
    np.save(save_model_path+'/val_label.npy', val_label)

    ## if wanna load saved data
    # train_list = np.load(save_model_path+'/binary_with_images_64-172' +'/train_list.npy')
    # test_list = np.load(save_model_path+'/binary_with_images_64-172'+'/test_list.npy')
    # train_label = np.load(save_model_path+'/binary_with_images_64-172'+'/train_label.npy')
    # test_label = np.load(save_model_path+'/binary_with_images_64-172'+'/test_label.npy')
    # val_list = np.load(save_model_path+'/binary_with_images_64-172'+'/val_list.npy')
    # val_label = np.load(save_model_path+'/binary_with_images_64-172'+'/val_label.npy')

    print("Train Dataset: ",len(train_list))
    print("Val Dataset: ",len(val_list))
    print("Test Dataset: ",len(test_list))

    # ------------------------------------------------------------------------------------

    transform = transforms.Compose([transforms.Resize([img_x, img_y]), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])   

    dataset_train = brain_data_loader.BrainDataset(data_path, train_list, train_label, selected_frames, transform = transform)
    dataset_validation = brain_data_loader.BrainDataset(data_path, val_list, val_label, selected_frames, transform = transform)
    dataset_test = brain_data_loader.BrainDataset(data_path, test_list, test_label, selected_frames, transform = transform)

    ## Dealing with class imbalances
    # def make_weights_for_balanced_classes(images, nclasses):                        
    #     count = [0] * nclasses                                                      
    #     for item in images: 
    #         count[item[1]] += 1
    #         print(count, end="\r")
    #     weight_per_class = [0.] * nclasses                                      
    #     N = float(sum(count))  
    #     print("")

    #     for i in range(nclasses):                                                   
    #         weight_per_class[i] = N/float(count[i])
    #         print(weight_per_class[i], end="\r")

    #     weight = [0] * len(images)  
    #     print("")

    #     for idx, val in enumerate(images):                                          
    #         weight[idx] = weight_per_class[val[1]] 
    #         print(weight[idx], end="\r")

    #     return weight 

    # weights = make_weights_for_balanced_classes(dataset_train, num_classes)     
    # weights = torch.DoubleTensor(weights)  
    # samp = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights),replacement=True)    


    # # load dementia types names
    # params_train = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True, 'sampler':samp} if use_cuda else {}
    
    params_train = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': False} if use_cuda else {}
    params_val =  {'batch_size': batch_size, 'shuffle': False, 'num_workers': 2, 'pin_memory': False} if use_cuda else {}


    train_loader = data.DataLoader(dataset_train, **params_train) # when batch_size > 1, use drop_last=True,
    valid_loader = data.DataLoader(dataset_validation, **params_val)
    test_loader  = data.DataLoader(dataset_test, **params_val)


    print("[INFO] Data is loaded. ")
       

    # Model definition
    model = transformer.Semi_Transformer(num_classes, seq_len).to(device)
    # model.to(device)
    
    # Running on multiple nodes
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Set optimizer (default SGD with momentum)
    #optimizer = optim.AdamW(model.parameters(), lr=1e-4, amsgrad=True)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.0009, nesterov=True)
    # print('The lenght of the holder is: ',len(holder))
    
    holder = 88 # for saving models above 88% accuracy

    # start training
    for epoch in range(epochs):
        # train, test model
        
        final_epoch_loss = train(model, train_loader, optimizer, epoch, device)

        with torch.no_grad():
            correct, total = validation(model, valid_loader, optimizer, epoch, device, holder, test=False)

    
    # Test
    with torch.no_grad():
        correct, total = validation(model, test_loader, optimizer, epoch, device, holder, test=True)
    

    # Total amount of time 
    print('[INFO] Total amount of time that training takes: %s minutes' % ((time.time() - start_time)/60))


if __name__== "__main__":
    all_main()