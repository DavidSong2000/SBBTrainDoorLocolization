import os
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np


def single_img_bounding(img):
    """
    input:PIL Image(RGB格式)
    """

    original_img = img.convert("RGB")

    # 初始化模型（可在函数外初始化）
    model = YOLO('../weights/best.pt')
    results = model.predict(original_img, verbose=False)
    result = results[0]

    boxes = result.boxes.xyxy if hasattr(result.boxes, 'xyxy') else []

    rgba_img = original_img.convert("RGBA")
    width, height = rgba_img.size


    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # 在mask上绘制bbox区域为不透明
    for box in boxes:
        xmin, ymin, xmax, ymax = [int(v) for v in box]
        draw.rectangle([xmin, ymin, xmax, ymax], fill=255)

    # 将mask作为alpha通道
    r, g, b, _ = rgba_img.split()
    rgba_img = Image.merge("RGBA", (r, g, b, mask))

    return rgba_img
