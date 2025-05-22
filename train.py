import time
import lpips
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader
from models import CannyEncoder, CannyDecoder
from utils import CustomDataset, SSIM_Loss, save_images
from adabelief import AdaBelief


def main():
    device = "cuda"
    dtype  = torch.float32
    batch_size = 16
    lr = 5e-5

    encoder = CannyEncoder(64).to(device=device)
    decoder = CannyDecoder(64).to(device=device)
    # encoder = torch.load("encoder_030.pth").to(device=device)
    # decoder = torch.load("decoder_030.pth").to(device=device)

    dataset   = CustomDataset(path="experiments/images")
    loader    = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    optimizer = AdaBelief([*encoder.parameters(), *decoder.parameters()], lr=lr, eps=1e-16, beta_1=0.9, beta_2=0.999, lr_dropout=1, lr_cos=0, clipnorm=0.0)

    ssim = SSIM_Loss(data_range=1.0, size_average=True, win_size=11, channel=3)
    loss_fn_alex = lpips.LPIPS(net='alex').to(device=device, dtype=dtype)

    sobel_x = torch.tensor([[-1,  0,  1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [ 0, 0, 0], [ 1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    total_steps = 100000
    steps = 0
    
    iter_start_time = time.time()

    while steps < total_steps:
        for img in loader:
            iter_time_ms = int((time.time() - iter_start_time) * 1000)
            iter_start_time = time.time()
            
            steps += 1
            if steps > total_steps:
                break

            c_img = TF.rgb_to_grayscale(img, num_output_channels=1)
            grad_x = F.conv2d(c_img, sobel_x, padding=1)
            grad_y = F.conv2d(c_img, sobel_y, padding=1)
            c_img = (torch.sqrt(grad_x**2 + grad_y**2) > 0.5).to(dtype=torch.float32)
            c_img = c_img.repeat(1, 3, 1, 1)

            img = img.to(device=device)

            o_img = decoder(encoder(c_img.to(device=device)))

            loss_ssim = 10 * ssim(o_img, img)
            loss_mse  = torch.mean(10 * torch.square(o_img - img))
            loss_alex = loss_fn_alex(o_img*2-1, img*2-1).mean()

            loss = loss_ssim + loss_mse + loss_alex

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if steps % 100 == 0:
                with torch.no_grad():
                    save_images(f"{steps:04d}.jpg", img[:8], c_img[:8], o_img[:8], img[8:16], c_img[8:16], o_img[8:16])
                    save_images(f"out.jpg", img[:8], c_img[:8], o_img[:8], img[8:16], c_img[8:16], o_img[8:16])

            if steps % 1000 == 0:
                torch.save(encoder, f"encoder_{steps//1000:03d}.pth")
                torch.save(decoder, f"decoder_{steps//1000:03d}.pth")

            print(f"\rSTEP: {steps:06d} | Iter time: {iter_time_ms:05d}ms | [{loss_ssim.item():06f}]+[{loss_mse.item():06f}]+[{loss_alex.item():06f}]=[{loss.item():06f}]", end="")


if __name__ == "__main__":
    main()