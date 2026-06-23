import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.unet import UNet
from utils.dataset_loader import RiceDataset

dataset = RiceDataset(
    r"C:\Users\AIML 25\CLOUD\dataset\train\cloudy",
    r"C:\Users\AIML 25\CLOUD\dataset\train\clear"
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

model = UNet()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 2

for epoch in range(epochs):

    total_loss = 0

    for cloudy, clear in loader:

        output = model(cloudy)

        loss = criterion(output, clear)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    r"C:\Users\AIML 25\CLOUD\models\unet_model.pth"
)

print("Model Saved!")