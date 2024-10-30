
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


## (d)

## (e)

## (f)

## (g)

# code
