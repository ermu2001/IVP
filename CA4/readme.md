
#  CNN on the Fudan Pedestrian segmentation dataset 
## (a)
The data is cutted into  into an 80-10-10 train-val-test split.

For detail implementation, see self_split function of PennFudanDataset from data.py. The code is also attached to this report.

## (b)
The training data was augmented with the folloing processing steps:
1. *resize_with_long_edge*: the image is resized having the long edge to match the specific resolution.
2. *color_jitter*: the resized image is then processed with random color jittering with the folloing parameters: [TODO]
3. *random_adjust_sharpness*: the image is then adjust by the sharpness with factor [TODO] at probability [TODO]
4. *center_pad*: the final processed image is then paded with black having the image at center to form a square image at the specific resolution.

It is worth mentioning for validation, the image and mask is only processed with *resize_with_long_edge* and *center_pad* for the pixel mask to match the training image.

## (c)

The model is implemented in model.py. The code is also attached to this document.

This section briefly introduces the architecture of the implemented unet in a top-down narrative.

-  At highest abstraction, the unet is made up of *depth* number of downsapmling block module and *depth* number of upsampling  block module to ensure same input/output spatial resolution. At the lowest spatial resolution at middle of the model, there are two middle module that does not change the spatial resolution.

- Each upsample/downsample block module consist of two spatial shape identity convolution block and a upsample/downsample layer at the end.

- Each convolution block consist of a root mean square normalization layer, a convolution layer and a activation layer.

Generally, this unet architecture could be configured with:
- *depth*: number of upsample/downsample layer.
- *spatial_scale_factor*: the scaling factor for spatial resolution for each upsample/downsample module.
- *channel_scale_factor*: the scaling factor for channel number of each upsample/downsample layer.

The model will scale the channel number at first block of each upsample/downsample block module and scale the spatial resolution at each ending block of it.

The downsampling uses max pooling while upsampling uses bilinear interpolation.

## (d)
For training, adam optimizer is used with no *weight_decay*. A cosine annealing with warm up learning rate shcheduler is used. The details of the learning rate scheduler implementation could be found in class CosineAnnealingWarmupRestarts, utils.py.

Upon training, wandb is used to monitor the training statstics such as training and validation loss.

Aside from this, wandb sweep is used for hyperparameters tuning on some key hyperparameters: *learning_rate*, *batch_size*, *epochs* and *img_size*. The hyperparameter sweep experiment curves could be found at: https://wandb.ai/ermuzzz2001/pedestrian-detection/sweeps/bp8sw90v?nw=nwuserermu2001

From the result, we can determine the best hyperparameter setting as:
- *learning_rate*=1e-4
- *batch_size*=16
- *epochs*=300
- *img_size*=128

The rest of this section shows the training dice loss, validation dice loss and validation dice score.
[TODO]

## (e)

## (f)

## (g)

# code
