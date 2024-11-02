import random
import torch
import numpy as np

def get_device(device=None):
    """Get the current device (GPU or CPU)."""
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device)
    
def get_dtype(dtype='float32'):
    """Get the current data type (float32 or float16)."""
    dtype = dtype.lower()
    dtype = getattr(torch, dtype)
    return dtype

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

def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def compute_dice_score(pred_mask, true_mask):
    intersection = torch.sum(pred_mask * true_mask)
    return (2. * intersection) / (torch.sum(pred_mask) + torch.sum(true_mask))


class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0, max_lr=0.1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = total_epochs - warmup_epochs
        self.current_epoch = 0

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, self.get_lr)

    def get_lr(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            return [self.min_lr + (self.max_lr - self.min_lr) * (self.current_epoch / self.warmup_epochs) for _ in self.base_lrs]
        else:
            # Cosine annealing
            cosine_decay = 0.5 * (1 + np.cos((self.current_epoch - self.warmup_epochs) / self.cycle_length * 3.141592653589793))
            return [self.min_lr + (self.max_lr - self.min_lr) * cosine_decay for _ in self.base_lrs]

    def step(self):
        self.current_epoch += 1
        super(CosineAnnealingWarmupRestarts, self).step()

