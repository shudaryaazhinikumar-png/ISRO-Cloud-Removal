import os

rice1_cloud = r"C:\Users\AIML 25\Downloads\archive\RICE\RICE1\cloud"
rice1_label = r"C:\Users\AIML 25\Downloads\archive\RICE\RICE1\label"

print("RICE1 Cloudy Images:", len(os.listdir(rice1_cloud)))
print("RICE1 Clear Images:", len(os.listdir(rice1_label)))