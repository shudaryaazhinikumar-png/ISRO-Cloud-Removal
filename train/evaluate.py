import torch
import numpy as np

from models.unet import UNet
from utils.dataset_loader import RiceDataset

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
    mean_squared_error
)

dataset = RiceDataset(
    r"C:\Users\AIML 25\CLOUD\dataset\test\cloudy",
    r"C:\Users\AIML 25\CLOUD\dataset\test\clear"
)

model = UNet()

model.load_state_dict(
    torch.load(
        r"C:\Users\AIML 25\CLOUD\models\unet_model.pth",
        map_location="cpu"
    )
)

model.eval()

cloudy, clear = dataset[0]

with torch.no_grad():
    pred = model(cloudy.unsqueeze(0))

pred = pred.squeeze(0).permute(1,2,0).numpy()
clear = clear.permute(1,2,0).numpy()

pred = np.clip(pred, 0, 1)

psnr = peak_signal_noise_ratio(clear, pred, data_range=1.0)

ssim = structural_similarity(
    clear,
    pred,
    channel_axis=2,
    data_range=1.0
)

rmse = np.sqrt(
    mean_squared_error(clear, pred)
)

print(f"PSNR : {psnr:.2f}")
print(f"SSIM : {ssim:.4f}")
print(f"RMSE : {rmse:.4f}")