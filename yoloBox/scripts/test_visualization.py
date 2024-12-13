import os
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np


# 初始化 YOLO 模型
model = YOLO('weights/best.pt')

# 输入和输出路径
image_folder = 'test/SyntheticScene0/'  # 输入文件夹
output_folder = 'output/SyntheticScene0/'  # 输出文件夹
os.makedirs(output_folder, exist_ok=True)

def process_image_bounding_boxes(image_path, model, output_folder, idx):
    """
    处理单张图片并保存切割结果到输出文件夹。
    """
    original_img = Image.open(image_path).convert("RGB")

    # 使用 YOLO 模型预测
    results = model.predict(original_img, verbose=False)
    result = results[0]

    # 提取预测框
    boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, 'xyxy') else []

    # 创建一个黑色背景
    black_img = Image.new("RGB", original_img.size, (0, 0, 0))

    # 将原始图像上的框内区域复制到黑色背景上，并保存切割图片
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = [int(v) for v in box]
        cropped_region = original_img.crop((xmin, ymin, xmax, ymax))
        black_img.paste(cropped_region, (xmin, ymin))

        # # 保存切割后的单独区域
        # crop_output_path = os.path.join(output_folder, f"result_{idx}_box_{i}.jpg")
        # cropped_region.save(crop_output_path)

    # 保存整个黑色背景图像
    background_output_path = os.path.join(output_folder, f"result_{idx}_background.jpg")
    black_img.save(background_output_path)

# 遍历文件夹中的所有图片
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_folder, image_file)
    print(f"Processing image: {image_path}")
    process_image_bounding_boxes(image_path, model, output_folder, idx)

print(f"All processed images and bounding box results saved in {output_folder}")
