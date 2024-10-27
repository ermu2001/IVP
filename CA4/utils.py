import torch

def get_device():
    """Get the current device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dice_loss(pred_mask, gt_mask):
    """Calculate the Dice loss."""
    eps = 1e-6  # Smoothing factor to avoid division by zero
    pred_mask_flat = pred_mask.view(-1)
    gt_mask_flat = gt_mask.view(-1)
    intersection = (pred_mask_flat * gt_mask_flat).sum()
    return 1 - (2. * intersection + eps) / (pred_mask_flat.sum() + gt_mask_flat.sum() + eps)

def get_iou(pred_masks, masks):
    """Calculate the Intersection over Union (IoU)."""
    intersection = (pred_masks & masks).sum(dim=(1, 2))
    union = (pred_masks | masks).sum(dim=(1, 2))
    return (intersection + 1e-6) / (union + 1e-6)  # Add small value to avoid division by zero