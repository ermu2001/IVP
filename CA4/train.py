import hydra
from omegaconf import OmegaConf, DictConfig
import os.path as osp
from PIL import Image
import numpy as np
import logging
from tqdm import tqdm

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
    dice_loss,
    get_iou,
)

logger = logging.getLogger(__name__)

def validate(model, val_dataset):
    device = get_device()
    model.eval()
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
    )
    ious = []
    dice_losses = []
    with torch.no_grad():
        for images, masks in tqdm(val_loader, leave=False):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            pred_masks = outputs > 0
            ious.extend(get_iou(pred_masks, masks).tolist())
            dice_losses.append(dice_loss(pred_masks, masks).tolist())
    ious = np.mean(ious)
    dice_losses = np.mean(dice_losses)
    # Add validation metrics here
    logger.info(f'Validation IoU: {ious:.4f}, Dice Loss: {dice_losses:.4f}')
    model.train()


@hydra.main(version_base=None ,config_path="conf", config_name="config")
def main_run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    model = load_model(model_cfg=cfg.model)
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
    optim = hydra.utils.get_class(cfg.train.optimizer.cls)(model.parameters(), **cfg.train.optimizer.kwargs)
    scheduler = hydra.utils.get_class(cfg.train.scheduler.cls)(optim, **cfg.train.scheduler.kwargs)
    loss_fn = dice_loss
    device = get_device()
    model.to(device)
    for epoch in range(cfg.train.num_epochs):
        model.train()
        epoch_losses = []
        for images, masks in tqdm(dataloader, leave=False):
            images = images.to(device)
            masks = masks.to(device)
            optim.zero_grad()
            outputs = model(images)
            pred_mask = outputs.sigmoid()
            loss = loss_fn(pred_mask, masks)
            loss.backward()
            epoch_losses.append(loss.item())
            optim.step()
        scheduler.step()
        validate(model, val_dataset)
        save_model(osp.join(cfg.output_dir, f'epoch{epoch:05}'), cfg.model, model)

        logger.info(f'Epoch [{epoch}/{cfg.train.num_epochs}], Loss: {np.mean(epoch_losses):.4f}')


if __name__ == "__main__":
    main_run()