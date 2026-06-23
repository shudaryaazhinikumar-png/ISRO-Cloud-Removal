from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class RiceDataset(Dataset):

    def __init__(self, cloudy_dir, clear_dir):

        self.cloudy_dir = cloudy_dir
        self.clear_dir = clear_dir

        self.images = sorted(os.listdir(cloudy_dir))

        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        file = self.images[idx]

        cloudy = Image.open(
            os.path.join(self.cloudy_dir, file)
        ).convert("RGB")

        clear = Image.open(
            os.path.join(self.clear_dir, file)
        ).convert("RGB")

        cloudy = self.transform(cloudy)
        clear = self.transform(clear)

        return cloudy, clear