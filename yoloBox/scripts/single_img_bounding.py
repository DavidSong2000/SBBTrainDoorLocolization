import os
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np


def single_img_bounding(img,model):
    """
    input:PIL Image(RGB格式)
    """

    original_img = img.convert("RGB")

    # 使用 YOLO 模型预测
    results = model.predict(original_img, verbose=False)
    result = results[0]

    # 提取预测框
    boxes = result.boxes.xyxy if hasattr(result.boxes, 'xyxy') else []

    # 创建一个黑色背景
    black_img = Image.new("RGB", original_img.size, (0, 0, 0))

    # 将原始图像上的框内区域复制到黑色背景上
    for box in boxes:
        xmin, ymin, xmax, ymax = [int(v) for v in box]
        cropped_region = original_img.crop((xmin, ymin, xmax, ymax))
        black_img.paste(cropped_region, (xmin, ymin))

    return black_img
