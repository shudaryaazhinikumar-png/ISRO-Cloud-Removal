from PIL import Image

img = Image.open(
    r"C:\Users\AIML 25\Downloads\archive\RICE\RICE1\cloud\0.png"
)

print("Image Size:", img.size)
print("Image Mode:", img.mode)