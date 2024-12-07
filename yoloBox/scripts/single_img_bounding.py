import os
from ultralytics import YOLO

def single_img_bounding(img):
    model = YOLO('weights/best.pt')
    result = model(img)