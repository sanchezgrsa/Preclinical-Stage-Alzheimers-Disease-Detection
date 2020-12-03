import os
import glob
import numpy as np
import torch
import torch as T
import torch.nn.functional as F
import original_model
import data_loader as brain_data_loader
from sklearn.metrics import accuracy_score
import torch.utils.data as data
from tensorboard_logger import configure, log_value
import time
from numpy import save
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix
import shutil
import random
from sklearn.model_selection import train_test_split



## ------------------- label conversion tools ------------------ #

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

torch.cuda.empty_cache()
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, data_loader, optimizer, sequence_len, epoch, device, loss_update_interval=10):
    """Definition of one epoch procedure.
    """

    model.train()  
    batch_time = AverageMeter()

    losses = AverageMeter()
    reinloss = AverageMeter()
    accs = AverageMeter() 
    tic = time.time()
    train_iteration = 0


    index = 1

    device_2 = next(model.parameters()).device
    num_train = len(data_loader) * 10
    # print("[INFO] size of l_t at the beginning: %s" %(l_t.shape))
    with tqdm(total=num_train) as pbar:  # visualize process bar

        for i, (X_cpu, y_cpu) in enumerate(data_loader):
            X, y = X_cpu.to(device_2), y_cpu.to(device_2)

            imgs = []
            log_pi = []
            baselines = []
            locs = [] # Locations


            # First time step
            h_1, c_1, h_2, c_2, l_0 = model(X, None, None, None, None, None, first=True)
            l_t = l_0
            locs.append(l_t)

            
            for t in range(sequence_len-1):

                    h_1, c_1, h_2, c_2, l_t, b_t, p = model(X, l_t, h_1, c_1, h_2, c_2)

                    baselines.append(b_t)
                    log_pi.append(p)
                    locs.append(l_t)
                    imgs.append(X)
            # Last time step
            h_1, c_1, h_2, c_2, l_t, b_t, p, log_probas = model(X, l_t, h_1, c_1, h_2, c_2, last=True)
            
            baselines.append(b_t)
            log_pi.append(p)

            # Information to plot the glimpses

            locs.append(l_t)
            imgs.append(X)

            
            imgs = [g.cpu().data.numpy().squeeze() for g in imgs]

            locs = [l.cpu().data.numpy() for l in locs ]

    
            baselines = T.stack(baselines).transpose(1,0)
            log_pi = T.stack(log_pi).transpose(1,0)



            predicted = torch.max(log_probas,1)[1]
              
            
            #_, predicted = T.max(output, 1)

            y = y.squeeze()
            R = (predicted.detach() == y).float()
            R = R.unsqueeze(1).repeat(1, sequence_len)

            # Calculating training accuracy


            # Calculation of losses
            loss_action = F.nll_loss(log_probas, y)

            loss_baseline = F.mse_loss(baselines, R)
            adjusted_reward = R - baselines.detach()

            loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)        
            loss = loss_action + loss_baseline + loss_reinforce*0.01

            # Accuracy
            correct = (predicted == y).float()
            acc = 100*(correct.sum()/len(y))

            # Store
            losses.update(loss.item(), 10)
            accs.update(acc.item(), 10)
            reinloss.update(loss_reinforce.item(), 10)

            index = index + 1  
            # compute gradients and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

         # print in process bar
            toc = time.time()
            batch_time.update(toc - tic)
            pbar.set_description(
                (
                    "{:.1f}s - loss: {:.3f}-rein loss: {:.3f} - acc: {:.3f}".format(
                        (toc - tic), loss.item(), loss_reinforce.item(), acc.item()
                    )
                )
            )
            pbar.update(len(y))
            # record in tensorboard for visualize
            log_value('train_loss', losses.avg, train_iteration)
            log_value('train_acc', accs.avg, train_iteration)
            log_value('train_rein_loss', reinloss.avg, train_iteration)
            train_iteration += 1


    return losses.avg, accs.avg, imgs, locs


def validation(model, data_loader, optimizer, sequence_len, epoch, device, save_model_path,test):
 
   
    model.eval()

    test_loss = 0
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



    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())


    # show information

    
    if not test:
        if  test_score*100 > 8:
            torch.save(model.state_dict(), os.path.join(save_model_path, 'Attention_Model_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
            torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'Attention_Model_optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
            print("Epoch {} model saved!".format(epoch + 1))

    elif test:
        print("[Test] Test Summary: Test Accuracy: ", '%.1f' % (test_score*100))

    return  test_score*100


def main():

    epochs = 200
    sequence_len = 6
    glimse_size= [40 , 40 , 40]

    std = 0.2 
    h_g = h_l = 512
    hidden_size = 512

    num_classes = 2
    batch_size = 10

    learning_rate = 1e-4

    data_path = "/home/faltay/Dataset_Binary_Mirrored_Rot"    

    dementia_labels_path = "/home/faltay/3DCNN/Labels_Binary.pkl"  # load preprocessed dementia types
    save_model_path = "/home/faltay/Glimpse/Saved_Models_200"
    plot_path = "/home/faltay/Glimpse/Saved_Models_200/Plots/"


   # image transformation
    img_x = 224
    img_y = 160 
    
    begin_frame, end_frame, skip_frame = 70, 190, 1


    print("[INFO] Starting ...")
    
    # Detect devices
    use_cuda = T.cuda.is_available()                   # check if GPU exists
    T.cuda.set_device(1)
    device = T.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
   

    # init tensorboard, a Plugin used for visualize the loss and acc
    logs_dir = './logs/'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    tensorboard_dir = logs_dir
    configure(tensorboard_dir)

    #model = neuro_dram_net.NeuroDram_network()
    model = original_model.RecurrentAttention(glimse_size, h_g, h_l, std, hidden_size, num_classes) # Whole line added by FATIH

    # Total number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    # Total number of trainable parameters

    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)



    model.to(device) # Added by FATIH

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=False)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)

    print("[INFO] Data is now loading ...")
    # Load data
        
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


    dataset_train = brain_data_loader.BrainDataset(data_path, train_list, train_label, selected_frames, transform = transform)
    dataset_validation = brain_data_loader.BrainDataset(data_path, val_list, val_label, selected_frames, transform = transform)
    dataset_test = brain_data_loader.BrainDataset(data_path, test_list, test_label, selected_frames, transform = transform)

    # # Dealing with class imbalances

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

    weights = make_weights_for_balanced_classes(dataset_train, num_classes)     
    weights = torch.DoubleTensor(weights)  
    samp = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights),replacement=True)    


    # load dementia types names

    params_train = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True, 'sampler':samp} if use_cuda else {}

    
    #params_train = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    params_val =  {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}


    train_loader = data.DataLoader(dataset_train, **params_train)
    valid_loader = data.DataLoader(dataset_validation, **params_val)
    test_loader = data.DataLoader(dataset_test, **params_val)

    print("[INFO] Data is loaded. ")
    # start training
    for epoch in range(epochs):
        # train, test model
        train_loss, train_acc, imgs, locs = train(model, train_loader, optimizer, sequence_len, epoch, device)
        
        with torch.no_grad():
            valid_acc = validation(model, valid_loader, optimizer, sequence_len, epoch, device,save_model_path, test=False)

        # print messages
        msg1 = "[Training] train loss: {:.3f} - train acc: {:.3f} "
        msg2 = " [Validation] val acc: {:.3f}"
        msg = msg1 + msg2
        print(msg.format(train_loss, train_acc, valid_acc))
        if train_acc > 98: 
                print("Entro  ")

                pickle.dump(imgs, open(plot_path + str(train_acc) + "train_g_{}.p".format(epoch),"wb"))
                pickle.dump(locs, open(plot_path + str(train_acc) +"train_l_{}.p".format(epoch),"wb"))

    
    # Test
    test_acc = validation(model, test_loader, optimizer , sequence_len, epoch, device, save_model_path,test=True)

def save_checkpoint(log_dir, container):
    """Save given information (eg. model, optimizer, epoch number etc.) into log_idr

    Parameters
    ----------
    log_dir : str
        Path to save
    container : dict
        Information to be saved
    """
    path = os.path.join(log_dir, "checkpoints", "ckpt_%s.pt"%datetime.now().strftime("%Y%m%d_%H%M%S"))
    torch.save(container, path)

def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

if __name__=="__main__": 
    main() 