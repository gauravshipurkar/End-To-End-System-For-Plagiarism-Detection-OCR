import os 
from pathlib import Path
from PIL import Image

print(Path('.').absolute())
os.chdir(r'C:\Users\gaura\OneDrive\Desktop\data')
print(Path('.').absolute())

count = 0
for img in os.listdir(r'C:\Users\gaura\OneDrive\Desktop\data'):
    image = Image.open(img)
    image.save('C:/Users/gaura/OneDrive/Desktop/data_jpg/'+str(count)+'.jpg')
    count = count +1
    print(img)

