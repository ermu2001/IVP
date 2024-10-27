
import numpy as np
import requests
import torch
from torch import nn
import torchvision
import torchvision.transforms
import os



from PIL import Image
import torchvision.transforms.functional

def center_pad(img):
    # Get the size of the image
    w, h = img.size
    print(img.size)
    # Calculate the size of the new image
    new_w = max(w, h)
    new_h = new_w
    # Create a new image with a white background
    if img.mode == 'RGB':
        new_img = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    elif img.mode == 'L':
        new_img = Image.new("L", (new_w, new_h), (0)) 

    # Paste the original image onto the new image
    new_img.paste(img, ((new_w - w) // 2, (new_h - h) // 2))
    return new_img


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, img_transforms=None, mask_transforms=None):
        self.root = root
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        self.examine()        

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.img_transforms:
            image = self.img_transforms(image)
            mask = self.mask_transforms(mask)
        return image, mask

    def __len__(self):
        return len(self.imgs)

    def examine(self):
        imgs_from_png = set([img[:12] for img in self.imgs])
        imgs_from_mask = set([mask[:12] for mask in self.masks])
        if imgs_from_png != imgs_from_mask:
            print(f'imgs without masks: {imgs_from_png - imgs_from_mask}')
            print(f'masks without imgs: ', imgs_from_mask - imgs_from_png)
            raise ValueError("Mismatch between images and masks")


def get_dataset(data_root, img_transforms=None, mask_transforms=None):
    return PennFudanDataset(
        data_root,
        img_transforms=img_transforms,
        mask_transforms=mask_transforms
    )

if __name__ == "__main__":
    from PIL import ImagePalette
    data_root = "tmp/data/PennFudanPed"
    img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256, Image.BILINEAR),
        center_pad,
        torchvision.transforms.ToTensor(),
    ])
    mask_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256, interpolation=Image.NEAREST),
        center_pad,
        torchvision.transforms.Lambda(lambda x: torch.as_tensor(np.array(x).astype(np.int32))),
    ])


    dataset = get_dataset(
        data_root,
        img_transforms=img_transforms,
        mask_transforms=mask_transform
    )
    print(f"Loaded dataset with {len(dataset)} samples.")
    for img, mask in dataset:
        print("=" * 100)
        print("Image shape:", img.shape)
        # print(img.tolist())
        print(img)
        torchvision.transforms.functional.to_pil_image(img).save("test_img.png")
        print("=" * 100)
        print("Mask shape:", mask.shape)
        print(f'Mask unique values: {torch.unique(mask)}')
        # print(mask.tolist())
        print(mask)

        # display with lenna
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        # Define 10 distinct colors
        # Create a colormap
        cmap = plt.get_cmap('tab10', 10)
        colors = (cmap(np.arange(10))[:, :3] * 255).astype(np.uint8)  # RGB values scaled to 0-255



        mask = Image.fromarray(mask.numpy().astype(np.uint8)).convert("P")
        mask.putpalette(colors.flatten().tolist())
        mask.save("test_mask.png")
        break
