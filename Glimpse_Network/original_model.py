import math
import torch as T
import torch.nn as nn
from torch.distributions import Normal
import model_org as written_model

import train as main_train # import get_cuda


class RecurrentAttention(nn.Module):
    """
    A 3D recurrent visual attention model for interpretable neuroimaging 
    classification, as presented in https://arxiv.org/abs/1910.04721. 
    """
    def __init__(self,
                 g,
                 h_g,
                 h_l,
                 std,
                 hidden_size,
                 num_classes):
        """
        Initialize the recurrent attention model and its
        different components.

        Args
        ----
        - g: size of the square patches in the glimpses extracted
          by the retina.
        - h_g: hidden layer size of the fc layer for 'what' representation
        - h_l: hidden layer size of the fc layer for 'where' representation
        - std: standard deviation of the Gaussian policy.
        - hidden_size: hidden size of the LSTM
        - num_classes: number of classes in the dataset.
        - num_glimpses: number of glimpses to take per image,
          i.e. number of BPTT steps.
        """
        super(RecurrentAttention, self).__init__()
        self.h_g = h_g
        self.h_l = h_l 
        self.hidden_size = hidden_size
        self.g = g
        self.std = std
        self.num_classes = num_classes
        self.glimpse_shape = g
        self.ret = written_model.retina(g) # Commented out by FATIH
        

        self.sensor = written_model.glimpse_3d(self.h_g, self.h_l, self.glimpse_shape) # , k, self.glimpse_shape, c removed by FATIH
        self.rnn = written_model.core_network(self.hidden_size, self.hidden_size)
        self.locator = written_model.location_network(self.hidden_size, 3, self.std)
        self.classifier = written_model.action_network(self.hidden_size, self.num_classes)
        self.baseliner = written_model.baseline_network(self.hidden_size, 1)
        #elf.context = context_network_clin(self.hidden_size) # Commented out by FATIH

    def forward(self, x, l_t, h_1, c_1, h_2, c_2, first = False, last=False):
        """
        Run the recurrent attention model for 1 timestep
        on the minibatch of images `x`.

        Args
        ----
        - x: a 5D Tensor of shape (B, C, H, W, D). The minibatch
          of images.
        - l_t_prev: a 3D tensor of shape (B, 3). The location vector
          containing the glimpse coordinates [x, y,z] for the previous
          timestep t-1.
        - h_1_prev, c_1_prev: a 2D tensor of shape (B, hidden_size). The 
          lower LSTM hidden state vector for the previous timestep t-1.
        - h_2_prev, c_2_prev: a 2D tensor of shape (B, hidden_size). The 
          upper LSTM hidden state vector for the previous timestep t-1.
        - last: a bool indicating whether this is the last timestep.
          If True, the action network returns an output probability
          vector over the classes and the baseline b_t for the
          current timestep t. 
          
        Returns
        -------
        - h_1_t, c_1_t, h_2_t, c_2_t: hidden LSTM states for current step
        - mu: a 3D tensor of shape (B, 3). The mean that parametrizes
          the Gaussian policy.
        - l_t: a 3D tensor of shape (B, 3). The location vector
          containing the glimpse coordinates [x, y,z] for the
          current timestep t.
        - b_t: a vector of length (B,). The baseline for the
          current time step t.
        - log_probas: a 2D tensor of shape (B, num_classes). The
          output log probability vector over the classes.
        - log_pi: a vector of length (B,).
        """
        if first: #if t=0, return get first location to attend to using the context
            # h_0 = self.context(x)
            # mu_0, l_0 = self.locator(h_0,)
            # return h_0, l_0
            x = x.unsqueeze(1)
            B, C, H, W, D = x.shape
            location = T.Tensor(B, 3)
            
            # shape of current images [224, 160 , 256]
            # giving center location
            # point_1 = 112
            # point_2 = 80
            # point_3 = 62
            
            position_1 = 0
            position_2 = 0
            position_3 = 0

            location[:,0].fill_(position_1)
            location[:,1].fill_(position_2)
            location[:,2].fill_(position_3)
            # print(location)
            # print(location.shape)
            l_0 = location.fill_(0)
            # print("[INFO] first location :", l_0)
            h_1, c_1, h_2, c_2 = (T.randn(1, B, self.hidden_size),
                                  T.randn(1, B, self.hidden_size),
                                  T.randn(1, B, self.hidden_size),
                                  T.randn(1, B, self.hidden_size))
            #mu_0, l_0 = self.locator(h_0,)

            return main_train.get_cuda(h_1), main_train.get_cuda(c_1), main_train.get_cuda(h_2), main_train.get_cuda(c_2), main_train.get_cuda(l_0)
        
        # print("[INFO] location :", l_t)
        x = x.unsqueeze(1)
        # l_t = get_cuda(l_t) #Added by FATIH
        g_t = self.sensor(x, l_t) #,display,axes,labels,dem
        h_1, c_1, h_2, c_2 = self.rnn(g_t.unsqueeze(0), h_1, c_1, h_2, c_2)
        mu, l_t = self.locator(h_2)
        b_t = self.baseliner(h_2).squeeze()


        log_pi = Normal(mu, self.std).log_prob(l_t)
        log_pi = T.sum(log_pi, dim=1) #policy probabilities for REINFORCE training
        
        if last:
            log_probas = self.classifier(h_1) # perform classification and get class probabilities
            return h_1, c_1, h_2, c_2, l_t, b_t, log_pi, log_probas

        return h_1, c_1, h_2, c_2, l_t, b_t, log_pi