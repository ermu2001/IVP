
#  CNN on the Fudan Pedestrian segmentation dataset 
## (a)
The data is cutted into  into an 80-10-10 train-val-test split.

For detail implementation, see self_split function of PennFudanDataset from data.py. The code is also attached to this report.

## (b)
The training data was augmented with the folloing processing steps:
1. *resize_with_long_edge*: the image is resized having the long edge to match the specific resolution.
2. *color_jitter*: the resized image is then processed with random color jittering with the folloing parameters:
   - brightness=0.1
   - contrast=0.1
   - saturation=0.1
   - hue=0.1
3. *random_adjust_sharpness*: the image is then adjust by the sharpness with factor 0.1 at probability 0.5.
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

Aside from this, wandb sweep is used for hyperparameters tuning on some key hyperparameters: *learning_rate*, *batch_size*, *epochs* and *img_size*. The hyperparameter sweep experiment curves could be found at: https://wandb.ai/ermuzzz2001/pedestrian-detection/sweeps/bp8sw90v?nw=nwuserermu2001.

In the sweeping view, we can see that the model converges with validation dice score higher than 0.7 only when using learning rate 1e-4 and image size 128. And the having longer training epochs would result in better final output.

![alt text](./assets/sweep.png)

From the result, we can determine the best hyperparameter setting as:
- *learning_rate*=1e-4
- *batch_size*=16
- *epochs*=300
- *img_size*=128

The rest of this section shows the training dice loss, validation dice loss and validation dice score for the main run with the tuned hyper-parameters.

The training loss curve and validation loss is as listed, the model converges to a low loss after 100 epochs of training, ensuring a valid convergence. The validation loss did not rise since a consine learning was used and should count for less overfitting.

![alt text](./assets/train_loss.png)

![alt text](./assets/val_loss.png)


While during the process, the validation dice score rise to high and started gradually decrease, this could be recognized as overfitting to the training set.

![alt text](./assets/val_score.png)



## (e)
With the final model, the evaluation was run on the test set, where no data autmentation was added in the preprocessing. The final test set evaluation metrics are listed as following.
```
Test set: IoU: 0.7282012584100946, Dice Score: 0.7950285077095032, Soft Dice Loss: 0.2354467511177063
```

## (f)
The model was used to take pedstrain mask for in-distribution images. The image was from held out test set from the original FudanPed dataset. The results are as listed. From the result, the pedstrain was masked out, but the edge of the mask is noisy. And for some cases, there are hallucination where some part of the image where there is not pedstrain was marked as true in the mask.

![alt text](asset/image.png)
![alt text](asset/image-1.png)
![alt text](asset/image-2.png)
![alt text](asset/image-3.png)

## (g)

Finally, the model was used to take pedstrain mask for out-distribution images. Comparing to in distribution samples, the out of distribution performance was rather worse, with more noisy on the edges for the mask and larger false mask in the image.
![alt text](image-4.png)
![alt text](image-5.png)

# code
Attached to this document is the code for this experiments.
## train.py
```python
[TODO]
```

## data.py
```python
[TODO]
```

## utils.py
```python
[TODO]
```

## configs
```yaml
[TODO]
```

# Notebooks
The sweep notebook main.ipynb is mainly run on goolge colab compute platform. The visualization notebook eval.ipynb is run after training for visualization and test set evaluation.
[TODO]