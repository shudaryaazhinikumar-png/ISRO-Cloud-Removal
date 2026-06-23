import torch
import matplotlib.pyplot as plt

from models.unet import UNet
from utils.dataset_loader import RiceDataset

dataset = RiceDataset(
    r"C:\Users\AIML 25\CLOUD\dataset\test\cloudy",
    r"C:\Users\AIML 25\CLOUD\dataset\test\clear"
)

model = UNet()

model.load_state_dict(
    torch.load(
        r"C:\Users\AIML 25\CLOUD\models\unet_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

cloudy, clear = dataset[0]

with torch.no_grad():
    pred = model(cloudy.unsqueeze(0))

cloudy = cloudy.permute(1,2,0).numpy()
clear = clear.permute(1,2,0).numpy()
pred = pred.squeeze(0).permute(1,2,0).numpy()

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(cloudy)
plt.title("Cloudy")

plt.subplot(1,3,2)
plt.imshow(clear)
plt.title("Ground Truth")

plt.subplot(1,3,3)
plt.imshow(pred)
plt.title("Prediction")

plt.show()