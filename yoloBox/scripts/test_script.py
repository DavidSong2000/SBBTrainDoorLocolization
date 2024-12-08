from PIL import Image
from ultralytics import YOLO
from single_img_bounding import single_img_bounding


model = YOLO('../weights/best.pt')
img = Image.open('../test/IMG_3145_JPG.rf.cdbe2a4f1d6e46cabccd88c33326ab59.jpg').convert('RGB')

processed_img = single_img_bounding(img, model)

processed_img.save('output.png')
