# Remote sensing image classification
This project focuses on remote sensing image classification using deep learning. 

The current implementations are based on PyTorch and Keras with TensorFlow backend. 

Feel free to contact me: liushengjie0756 (at) gmail.com

## Overview
In the script, we first conduct image segmentation and divide the image to several objects.
Then, we generate training samples and train a network. The network is used to predict the whole image.
Finally, the object-based post-classification refinement strategy is utilized to refine the classification maps.

### Networks
    [2] Wide Contextual Residual Network - WCRN
    [3] Double Branch Multi Attention Mechanism Network - DBMA
    Residual Network with Average Pooling - ResNet99_avg
    [4] Residual Network - ResNet99

### Requirements
    pytorch==1.1.0 # for PyTorch implementation
    skimage==0.15.0
    sciPy==1.0.0
    sklearn==0.19.1
    keras==2.1.6 # for Keras implementation
    tensorflow==1.9.0 # for Keras implementation
    
### Data sets
You can download the hyperspectral data sets in matlab format at: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

Then, you can change the data sets to numpy.array

## How to use
### Run main.py
You will see two predicted maps under the current directory when finished.
One is raw classification, and the other is after object-based post-classification refinement (superpixel-based regularization).

This implementation is based on PyTorch using the Wide Contextual Residual Network [2].

### Run demo_keras.py
This implementation is based on Keras with TensorFlow backend using the Double-Branch Multi-Attention Mechanism Network [3].

For this demo, the dafault network is DBMA. By changing the parameter - patch, which controls the window size of each sample, other networks will be applied.

#### Patch and the corresponding network
- patch==5: WCRN
- patch==7: DBMA
- patch==9: ResNet99_avg

## Networks' performance
Network | WCRN | DBMA | ResNet99 | ResNet99_avg
:-: | :-: | :-: | :-: | :-:
train time (s) | 18 | 222 | 21 | 20| 
test time (s) | 12| 199 | 22 | 21|
OA (%) | 83.00 | 86.86 | 72.34 | 86.68 |

The experiments are based on Keras with TensorFlow backend using 10 samples per class with augmentation, conducted on a machine equipped with Intel i5-8400, GTX1050Ti 4G and 8G RAM. 

The OA is of raw classification averaged from 10 Monte Carlo runs.

## References
  [1] Liu, S., Qi, Z., Li, X. and Yeh, A.G.O., 2019. Integration of Convolutional Neural Networks and Object-Based Post-Classification
Refinement for Land Use and Land Cover Mapping with Optical and SAR Data. Remote Sens., 11(6), p.690. doi.org/10.3390/rs11060690

  [2] Liu, S., Luo, H., Tu, Y., He, Z. and Li, J., 2018, July. Wide Contextual Residual Network with Active Learning for Remote
Sensing Image Classification. In IGARSS 2018, pp. 7145-7148. doi.org/10.1109/IGARSS.2018.8517855

  [3] Ma, W.; Yang, Q.; Wu, Y.; Zhao, W.; Zhang, X. Double-Branch Multi-Attention Mechanism Network for Hyperspectral Image Classification. Remote Sens. 2019, 11, 1307. doi.org/10.3390/rs11111307
  
  [4] Liu, S., and Shi, Q., 2019. Multitask Deep Learning with Spectral Knowledge for Hyperspectral Image Classification. arXiv preprint arXiv:1905.04535. arxiv.org/abs/1905.04535

