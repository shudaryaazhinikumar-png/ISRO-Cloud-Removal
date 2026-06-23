import os
import shutil
import random

cloud_dir = r"C:\Users\AIML 25\Downloads\archive\RICE\RICE1\cloud"
label_dir = r"C:\Users\AIML 25\Downloads\archive\RICE\RICE1\label"

output_dir = r"C:\Users\AIML 25\CLOUD\dataset"

files = os.listdir(cloud_dir)
files.sort()

random.shuffle(files)

train_size = int(0.7 * len(files))
val_size = int(0.15 * len(files))

train_files = files[:train_size]
val_files = files[train_size:train_size + val_size]
test_files = files[train_size + val_size:]

def copy_files(file_list, split_name):
    for file in file_list:
        shutil.copy(
            os.path.join(cloud_dir, file),
            os.path.join(output_dir, split_name, "cloudy", file)
        )

        shutil.copy(
            os.path.join(label_dir, file),
            os.path.join(output_dir, split_name, "clear", file)
        )

copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("Dataset split completed!")
print("Train:", len(train_files))
print("Validation:", len(val_files))
print("Test:", len(test_files))