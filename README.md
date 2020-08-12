# 2D super-resolution on cardiac MR images

A Keras implementation of super-resolution using perceptual loss from ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://arxiv.org/abs/1603.08155), as a part of a master thesis project "Super-resolving cardiac MR images using deep learning" at Linköping University.

## Dataset

2D balanced-ssfp slices were used for training/inference. 2D slices were obtained from cine balanced-ssfp volume with spatial resolution of $1 \times  1 \times 8 \mm^3$. Obtained 2D slices are used as a hight-resolution target for training the network. Low-resolution input was created by downsampling with bicubic interpolation and adding Gaussian blurring with \sigma = 1.0.

## Network architecture
![perceptual loss](./images/perceptualNetwork.png)













