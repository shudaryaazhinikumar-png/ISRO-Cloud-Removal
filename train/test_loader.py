from utils.dataset_loader import RiceDataset

dataset = RiceDataset(
    r"C:\Users\AIML 25\CLOUD\dataset\train\cloudy",
    r"C:\Users\AIML 25\CLOUD\dataset\train\clear"
)

print("Dataset Size:", len(dataset))

cloudy, clear = dataset[0]

print("Cloudy Shape:", cloudy.shape)
print("Clear Shape:", clear.shape)