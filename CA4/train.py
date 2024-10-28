import hydra
from omegaconf import OmegaConf, DictConfig
import os.path as osp
from PIL import Image
import numpy as np
import logging
from tqdm import tqdm
import wandb
import torch
import torchvision

from model import (
    load_model,
    save_model,
)

from data import (
    PennFudanDataset,
    center_pad,
    resize_with_long_edge,
)

from utils import (
    get_device,
    get_dtype,
    dice_loss,
    get_iou,
    seed_everything,
)

logger = logging.getLogger(__name__)

def validate(model, val_dataset, dtype):
    device = get_device()
    model.eval()
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
    )
    ious = []
    dice_scores = []
    with torch.no_grad():
        for images, masks in tqdm(val_loader, leave=False):
            images = images.to(device, dtype=dtype)
            masks = masks.to(device)
            outputs = model(images)
            pred_masks = outputs > 0
            ious.extend(get_iou(pred_masks, masks).tolist())
            dice_scores.append(1 - dice_loss(pred_masks, masks).tolist())
    ious = np.mean(ious)
    dice_scores = np.mean(dice_scores)
    # Add validation metrics here
    return ious, dice_scores

def run(cfg):
    wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True), name=cfg.wandb.name)
    seed_everything(cfg.train.seed, deterministic=True)
    model = load_model(model_cfg=cfg.model)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model has {num_params//1000:}K parameters')


    img_transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: resize_with_long_edge(x, cfg.train.img_size)),
        torchvision.transforms.Lambda(center_pad),
        torchvision.transforms.ToTensor(),
    ])
    mask_transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: resize_with_long_edge(x, cfg.train.img_size, Image.NEAREST)),
        torchvision.transforms.Lambda(center_pad),
        torchvision.transforms.Lambda(lambda x: torch.as_tensor(np.array(x).astype(bool))),
    ])

    train_dataset = PennFudanDataset(
        root=cfg.data.data_root,
        img_transforms=img_transform,
        mask_transforms=mask_transform,
        split='train',
    )

    val_dataset = PennFudanDataset(
        root=cfg.data.data_root,
        img_transforms=img_transform,
        mask_transforms=mask_transform,
        split='val',
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
    )
    optim = hydra.utils.instantiate(cfg.train.optimizer, model.parameters())
    if 'scheduler' in cfg.train:
        if getattr(cfg.train.scheduler, 'warmup_epochs') is None:
            cfg.train.scheduler.warmup_epochs = cfg.train.scheduler.total_epochs // 10
        if getattr(cfg.train.scheduler, 'min_lr') is None:
            cfg.train.scheduler.min_lr = cfg.train.scheduler.max_lr / 1e3
        scheduler = hydra.utils.instantiate(cfg.train.scheduler, optim)
    else:
        scheduler = None
    loss_fn = dice_loss
    device = get_device()
    dtype = get_dtype(cfg.train.dtype)
    model.to(device=device, dtype=dtype)
    for epoch in range(cfg.train.num_epochs):
        model.train()
        epoch_losses = []
        for images, masks in tqdm(dataloader, leave=False):
            images = images.to(device=device, dtype=dtype)
            masks = masks.to(device)
            optim.zero_grad()
            outputs = model(images)
            pred_mask = outputs.sigmoid()
            loss = loss_fn(pred_mask, masks)
            loss.backward()
            epoch_losses.append(loss.item())
            optim.step()

        if scheduler is not None:
            scheduler.step()
            cur_lr = scheduler.get_lr()[0]
        else:
            cur_lr = cfg.train.optimizer.lr    
        val_iou, val_dice_score = validate(model, val_dataset, dtype)

        save_model(osp.join(cfg.output_dir, f'epoch{epoch:05}'), cfg.model, model)
        logger.info(f"""Epoch [{epoch}/{cfg.train.num_epochs}],
                      Train Loss: {np.mean(epoch_losses):.4f},
                      Validation IoU: {val_iou:.4f},
                      Validation Dice Score: {val_dice_score:.4f},
                      Validation Dice Loss: {1 - val_dice_score:.4f},
                      Learning Rate: {cur_lr:.6f}""")
        wandb.log({
            'epoch_train_loss': np.mean(epoch_losses),
            'epoch_val_iou': val_iou,
            'epoch_val_dice_score': val_dice_score,
            'epoch_val_dice_loss': 1 - val_dice_score,
            'epoch_learning_rate': cur_lr,
        })

@hydra.main(version_base=None ,config_path="conf", config_name="config")
def main_run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    run(cfg)

if __name__ == "__main__":
    main_run()