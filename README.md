# Remote sensing image classification
This project focuses on remote sensing image classification using deep learning. 

The current implementation is based on PyTorch. 

## Overview
In the script, we first conduct image segmentation and divide the image to several objects.
Then, we generate training samples and train a network. The network is used to predict the whole image.
Finally, the object-based post-classification refinement strategy is utilized to refine the classification maps.


## Requirements
    PyTorch==1.1.0
    skimage==0.15.0
    SciPy==1.0.0
    sklearn==0.19.1
    
## How to use
### Run main.py
You will see two predicted maps under the current directory when finished.
One is raw classification, and the other is after object-based post-classification refinement (superpixel-based regularization).

### References
  [1] Liu, S., Qi, Z., Li, X. and Yeh, A.G.O., 2019. Integration of Convolutional Neural Networks and Object-Based Post-Classification
Refinement for Land Use and Land Cover Mapping with Optical and SAR Data. Remote Sens., 11(6), p.690. 

  [2] Liu, S., Luo, H., Tu, Y., He, Z. and Li, J., 2018, July. Wide Contextual Residual Network with Active Learning for Remote
Sensing Image Classification. In IGARSS 2018, pp. 7145-7148.

