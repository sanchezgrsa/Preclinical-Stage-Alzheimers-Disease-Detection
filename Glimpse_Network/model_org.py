import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import train as training_File


class retina(object):
    """
    A retina which extracts foveated glimpses centred
    around location l, from an input image x. This is 
    passed to the glimpse network for processing
    Args
    ----
    - x: a 5D Tensor of shape (B, C, H, W, D) - the minibatch
      of images MRI images.
    - l: a 3D Tensor of shape (B, 3). Contains normalized
      coordinates in the range [-1, 1].
    - g: size of glimpse (g x g x g).

    Returns
    -------
    - phi: a 5D tensor of shape (B, C, g, g, g). The
      foveated glimpse of the image.
    """
    def __init__(self, g):
        self.g = g

    def foveate(self, x, l):
        """
        Extract a  cube of size g x g x g, centered
        at location l.
        """
        
        size = self.g

        # extract glimpse
        phi = self.extract_patch(x, l, size)
        
        return phi

    def extract_patch(self, x, l, size):
        """
        Extract a single patch for each image in the
        minibatch x.

        Args
        ----
        - x: a 5D Tensor of shape (B, C, H, W, D). The minibatch
          of images.
        - l: a 3D Tensor of shape (B, 3).

        Returns
        -------
        - patch: a 5D Tensor of shape (B, C, size, size, size)
        """
        B, C, H, W, D = x.shape
     

        # denormalize coords of patch center
        coords = self.denormalize(x.shape[2:],l, size) # size is added by FATIH
    
        patch_x = coords[:, 0] - (size // 2)
        patch_y = coords[:, 1] - (size // 2)
        patch_z = coords[:,2] - (size//2)

        # loop through mini-batch and extract
        patch = []
        for i in range(B):
            # print('[INFO] shape of the x[i]: ', x[i].shape)
            im = x[i].unsqueeze(dim=0)
            # print('[INFO] shape of the im: ', im.shape)
            # get image shape for denormalizing coordinates and padding 
            T = im.shape[2:]
            # print('[INFO] shape of the T: ', T)
            # compute slice indices
            from_x, to_x = patch_x[i], patch_x[i] + size
            from_y, to_y = patch_y[i], patch_y[i] + size
            from_z, to_z = patch_z[i], patch_z[i] + size
            
  

            # cast to ints
            from_x, to_x = from_x.item(), to_x.item()
            from_y, to_y = from_y.item(), to_y.item()
            from_z, to_z = from_z.item(), to_z.item()
            
                # pad tensor in case exceeds
            if self.exceeds(from_x, to_x, from_y, to_y, T, from_z, to_z):
                pad_dims = (
                    size//2+1, size//2+1,
                    size//2+1, size//2+1,
                    size//2+1, size//2+1,
                    0, 0,
                    0, 0,
                )
                #print('[INFO] image before padding: ', im.shape)
                im = F.pad(im, pad_dims, "constant", im[0,:,-2:,-2:,-2:].float().mean().item())
                #print('[INFO] image after padding: ', im.shape)
                
                # add correction factor
                from_x += (size//2+1)
                to_x += (size//2+1)
                from_y += (size//2+1)
                to_y += (size//2+1)
                from_z += (size//2+1)
                to_z += (size//2+1)

            # and finally extract
            #print('[INFO] extracted glimpse: ', im[:, :, from_x:to_x, from_y:to_y, from_z:to_z].shape)
            patch.append(im[:, :, from_x:to_x, from_y:to_y, from_z:to_z])
            
            #print('[INFO] patch : ', patch)


        # concatenate into a single tensor

        patch = torch.cat(patch, dim=0)
        #print(patch)
        return patch

    def denormalize(self,dims,l, size):
        """
        Convert coordinates in the range [-1, 1] to
        Cartesian coordinates 
        """
        coords = torch.zeros(l.shape[0],l.shape[1])
        # print(dims)
        # print(coords.shape)
        # print(coords)
        # print(l.shape)
        # print(l)
        for i in range(l.shape[1]):
            coords[:,i] = (0.5 * ((l[:,i] + 1.0) * dims[i])).long()
        return coords.long()

    def exceeds(self, from_x, to_x, from_y, to_y, T, from_z = None, to_z = None):
        """
        Check whether the extracted glimpse exceeds
        the boundaries of the image of size T.
        """

        if ((from_x < 0) or (from_y < 0) or (from_z < 0) or (to_x > T[0]) or (to_y > T[1])
            or (to_z > T[2])):
            return True
        return False


class glimpse_3d(nn.Module):
    """
    The glimpse network used in the paper.
    Comprises 4 blocks of filters (f = 8,16,32,64) of size (3 x 3 x 3) 
    with max pooling and batch normalisation, as in 
    Bohle et al. (2019).
        
    Args
    ----
    - h_g: dimensionality of 'what' representation g_x_t
    - h_l: dimensionality of 'where' representation g_l_t
    - g: glimpse shape (g x g x g)

    Returns
    -------
    - g_t: glimpse representation summarising what the agent has seen
      and where it has seen it
    """
  
    def __init__(self, h_g, h_l, g):
        super(glimpse_3d,self).__init__()
        
        D_in = 3 # location input size
        temp = 2*2048
        
        # architecture defined
        self.retina = retina(g[0])
        self.Conv_1 = nn.Conv3d(1, 8, 3)
        self.Conv_1_bn = nn.BatchNorm3d(8)
        self.Conv_2 = nn.Conv3d(8, 16, 3)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        self.Conv_3 = nn.Conv3d(16, 32, 3)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.Conv_4 = nn.Conv3d(32, 64, 3)
        self.Conv_4_bn = nn.BatchNorm3d(64)
        self.fc = nn.Linear(temp, h_g)
        self.fc2 = nn.Linear(D_in, h_l)
        self.pool = nn.MaxPool3d(2)


    def forward(self, x, l_t_prev):
        
       
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)
        # print("[INFO] first l_t_prev shape: ", l_t_prev.shape)

        l_out = F.relu(self.fc2(l_t_prev))# g_l_t representation
      
        x = self.retina.foveate(x, l_t_prev) # extract glimpse
        x = self.pool(F.relu(self.Conv_1_bn(self.Conv_1(x.float()))))
        x = self.pool(F.relu(self.Conv_2_bn(self.Conv_2(x))))
        x = F.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = (self.Conv_4_bn(self.Conv_4(x)))
        x = F.relu(self.fc(x.view(x.shape[0], -1)))
        #x = F.relu(self.fc_glimpse( 2*2048))
                       
        what = x
        where = l_out
        
        # feed to fc layer - pointwise multiplication of 'what' and 'where'
        # as in Larochelle and Hinton (2010).
        g_t = F.relu(torch.mul(what,where))
        
        #g_t = get_cuda(g_t) # added by FATIH
        
        return g_t

# Recurrent Network
class core_network(nn.Module):
    """
    A recurrent neural network that maintains an internal state integrating
    information extracted from the history of past observations.
    It is comprised of two stacked LSTM units

    Args
    ----
    - input_size: input size of the rnn.
    - hidden_size: hidden size of the rnn.
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    - h_1_prev, c_1_prev: hidden and cell state of lower LSTM at step t-1
    - h_2_prev, c_2_prev: hidden and cell state of upper LSTM at step t-1
    Returns
    -------
    - h_1_t, c_1_t: hidden and cell state of lower LSTM at step t
    - h_2_t, c_2_t: hidden and cell state of upper LSTM at step t
    """
    def __init__(self, input_size, hidden_size):
        super(core_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_1 = nn.LSTM(input_size,hidden_size,1)
        self.lstm_2 = nn.LSTM(input_size,hidden_size,1)

    def forward(self, g_t, h_1_prev, c_1_prev, h_2_prev, c_2_prev):
        h_1, (_,c_1) = self.lstm_1(g_t, (h_1_prev, c_1_prev))
        h_2, (_,c_2) = self.lstm_2(h_1, (h_2_prev, c_2_prev))

        return h_1, c_1, h_2, c_2

# Classification Network
class action_network(nn.Module):
    """
    Uses the final upper LSTM state of the core network network to
    produce the final output classification.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer - for binary classifcation output_size= 2
    - h_1_T: the hidden state vector of the core network for
      the final time step T.

    Returns
    -------
    - a_t: output probability vector over the categories i.e AD and HC.
    """
    def __init__(self, input_size, output_size):
        super(action_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_1):
        a_t = F.log_softmax(self.fc(h_1.squeeze()), dim=1)
        return a_t


class location_network(nn.Module):
    """
    Uses the internal state h_2_t of the upper LSTM 
    unit to produce the location coordinates l_t of the glimpse
    at the next time step.

    Concretely, the hidden state h_2_t is fed through a fc
    layer followed, by a tanh to clamp the output beween
    [-1, 1]. This produces a 3D vector of means used to
    parametrize a three-component Gaussian with a fixed
    variance from which the location coordinates l_t
    for the next time step are sampled.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - std: standard deviation of the normal distribution.
    - h_2_t: the hidden state vector of the upper LSTM unit for
      the current time step t.

    Returns
    -------
    - mu: a 3D vector of shape (B, 3).
    - l_t: a 3D vector of shape (B, 3).
    """
    def __init__(self, input_size, output_size, std):
        super(location_network, self).__init__()
        self.std = std = 0.2

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
      
        # compute mean
        mu = F.tanh(self.fc(h_t.squeeze(0)))
        # reparametrization trick
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std) #sample from gaussian
        l_t = mu + noise
        # bound between [-1, 1]
        l_t = torch.clamp(l_t,-1,1)
        l_t = l_t.detach() # detach from network. Don't want to propagate 
        # gradients for this sampled value

        return mu, l_t


class baseline_network(nn.Module):
    """
    This network predicts the mean expected reward, and is used 
    to subtract from the true reward during training as a variance 
    reduction technique (as in Mnih et al. (2014)).
    This is trained using MSE regression

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network
      for the current time step `t`.

    Returns
    -------
    - b_t: a 2D vector of shape (B, 1). The baseline
      for the current time step `t`.
    """
    def __init__(self, input_size, output_size):
        super(baseline_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = F.relu(self.fc(h_t.detach())) # detach from computation graph
        return b_t
    
#Non-imaging context network
class context_network_clin(nn.Module):
    """ 
    Single-layer fully connected NN which takes as input available 
    non-imaging data. The output is passed to the location 
    network to provide l_0, and also  becomes the initial hidden 
    layer for the second lstm.
    """
    def __init__(self,hidden_size):
        super(context_network_clin, self).__init__()
        self.fc = nn.Linear(13,hidden_size)

    def forward(self, x):
        out = F.relu(self.fc(x)).unsqueeze(0)
        
        return out