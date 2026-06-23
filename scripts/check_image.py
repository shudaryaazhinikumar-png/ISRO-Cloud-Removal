from PIL import Image
import matplotlib.pyplot as plt

cloudy = Image.open(
    r"C:\Users\AIML 25\Downloads\archive\RICE\RICE1\cloud\0.png"
)

clear = Image.open(
    r"C:\Users\AIML 25\Downloads\archive\RICE\RICE1\label\0.png"
)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(cloudy)
plt.title("Cloudy Image")

plt.subplot(1,2,2)
plt.imshow(clear)
plt.title("Clear Image")

plt.show()