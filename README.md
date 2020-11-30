# Preclinical Stage Alzheimer’s Disease Detection Using MRI Scans
Implementation of IAAI-21 paper 'Preclinical Stage Alzheimer’s Disease Detection Using Magnetic Resonance Image Scans' in Pytorch

## Prerequisites
    -Python 3.7.4
    -Numpy 1.19.0
    -Pytorch 1.5.1
    -Torchvision 0.6.1

## Dataset
In this work, we employ the recently published longitudinalneuroimaging, clinical and cognitive dataset, called OASIS-3  (LaMontagne  et  al.  2019).  It  consists  of  MRI  and  PET imaging from 1098 individuals collected across several studies over the course of 15 years. There are 605 cognitivelynormal adults and 493 individuals at different stages of cognitive decline. Ages of the participants range from 42 to 95years. The dataset contains over 2000 MRI sessions. This dataset contains T1-weighted and T2-weighted MRI scans. The number of T1-weighted and T2-weighted scans are 2117 and 1985, respectively. Since, the majority of the health assessment studies have performed analysis on T1‐weighted MRI data, we evaluated on T1‐weighted MRI scans.

## Models
The model that is used as baseline is based on a 3D CNN model, which was initially used for video classification tasks (HHTseng 2020). This model uses 3D kernels and channels to convolve videoinput, where the videos are viewed as 3D data (2D images over time dimension). For our baseline model, we stack all the images in a brain scan, turn them into 3D input data, and then feed it to the network.

For our first model, we employ a recently proposed 3D recurrent visual attention model, which is tailored for neuroimaging classification (Wood, Cole, andBooth 2019) and focuses on already developed AD detection task. This model uses a recurrent attention mechanism (Sermanet, Frome, and Real 2014) that tries to find relevant locations of brain scan indicative of AD. The model consists of an agent that is trained with reinforcement learning. It is built around a two-layer recurrent neural network (RNN). At each timestamp, the agent receives a small portion of the entire image, which is a glimpse, centered around a position l,and decides which location to select at the next timestamp. After a fixed number of steps, a classification  decision  is made. The aim of using an agent is to maximize the rewards along the timestamps, and then decide to attend the most informative regions of the images. 
![GitHub Logo](/images/glimpse_network.png)

As our second model, we employ a transformer network for the task of preclinical AD detection. Transformer models  have  been  used  for  different  tasks  such  as  human  action recognition from videos (Girdhar et al. 2018) and text translation (Vaswani et al. 2017). Although transformer networks have been used for other tasks and applications, we firmly believe that this is the first work that employs a transformer network on MRI images of brain for preclinical stage Alzheimeir’s disease detection. Slices from a brain scan are fed to the network, and the network is expected to detect whether any sign of dementia is observable or not, even the subject is showing no signs nor symptoms of the disease yet.
![GitHub Logo](/images/transformer.png)

## Numerical Comparison
![GitHub Logo](/images/numerical_comparison.png)
