import torch
import random
import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image
from pathlib import Path
from pytorch_msssim import SSIM
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.imgs  = list(Path(path).rglob("*.jpg"))
        self.count = len(self.imgs)

    def __len__(self):
        return 999999999999

    def __getitem__(self, idx):
        index = random.randint(0, self.count-1)
        img = Image.open(str(self.imgs[index])).convert("RGB").resize((128,128))
        return TF.to_tensor(img)


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 1 - super(SSIM_Loss, self).forward(img1, img2)
    

def save_images(filename, *images):
    if not images: 
        return

    num_rows    , num_cols    = len(images), len(images[0])
    image_height, image_width = images[0].shape[-2], images[0].shape[-1]
    grid_height , grid_width  = num_rows * image_height, num_cols * image_width

    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for row in range(num_rows):
        batch = (images[row] * 255).permute(0, 2, 3, 1).to("cpu").to(torch.uint8).detach().numpy()
        for col in range(num_cols):
            # print(batch.shape, batch[col].shape)
            image = np.array(Image.fromarray(batch[col]).resize((image_width, image_height)))
            # print(image.shape)
            grid_image[row*image_height:(row+1)*image_height, col*image_width:(col+1)*image_width] = image
    Image.fromarray(grid_image).save(filename)