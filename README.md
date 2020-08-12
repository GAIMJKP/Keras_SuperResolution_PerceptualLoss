# 2D super-resolution on cardiac MR images

A Keras implementation of super-resolution using perceptual loss from ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://arxiv.org/abs/1603.08155), as a part of a master thesis project "Super-resolving cardiac MR images using deep learning" at Linköping University.

## Dataset

2D balanced-ssfp slices were used for training/inference. 2D slices were obtained from cine balanced-ssfp volume with spatial resolution of 1 x 1 x 8 mm^3. Obtained 2D slices are used as a hight-resolution target for training the network. Low-resolution input was created by downsampling with bicubic interpolation and adding Gaussian blurring with sigma = 1.0. 

## Network architecture and more
![perceptual loss](./images/perceptualNetwork.png)
The network consists of 4 residual blocks and 2 upscaling blocks with perceptual loss as a loss function. The network defines perceptual loss by making use of a pretrained VGG16 network. The proposed idea is that the network optimized with perceptual loss conveys better feature representation in terms of perceptual quality, compared to pixel-wise MSE loss. Also, the network utilize transposed convolution as an upscaling method. However transposed convolution often creates unevenly overlappedpixels (when the kernel size is not divisible by the stride). This behavior ends up adding more of the figurative emphasis in some pixels, which could lead to generating checkerboard artifacts in the super-resolved image. ["Deconvolution and Checkerboard Artifacts"](http://distill.pub/2016/deconv-checkerboard) suggests alternative approach by using, e.g., nearest-neighbor interpolation which does not have overlapping behavior by default.

Hence, in this project following experiments were of interest :  first,  loss function <code> <b>perceptual loss VS MSE loss</b> </code> and second, upscaling method (upscaling factor x4) <code> <b> transposed convolution VS nearest-neighbor interpolation</b> </code>. 

|         |  Loss function  | Upscaling method |
|:-------:|:---------------:|:----------------:|
| Model 1 | Perceptual loss |  Transposed Conv |
| Model 2 | Perceptual loss | NN interpolation |
| Model 3 |     MSE loss    |  Transposed Conv |
| Model 4 |     MSE loss    | NN interpolation |


## Training detail
Training was performed on a workstation with a 3.6GHz, 6-core processor with 64GB RAM, NVIDIA Quadro P6000 GPU.

## Usage





