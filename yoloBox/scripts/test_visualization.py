import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import cv2

Crop = False


# 初始化 YOLO 模型
model = YOLO('runs/detect/train/weights/best.pt')

# 输入和输出路径
image_folder = 'test/iPad_scans/'  # 输入文件夹
output_folder = 'output/NewModel/iPad_scans/'  # 输出文件夹
os.makedirs(output_folder, exist_ok=True)

def process_image_bounding_boxes(image_path, model, output_folder, idx):
    """
    处理单张图片并保存切割结果到输出文件夹。
    """
    
    original_img = Image.open(image_path)
    original_img = ImageOps.exif_transpose(original_img).convert("RGB")

    # 使用 YOLO 模型预测
    results = model.predict(original_img, verbose=False)
    result = results[0]

    if Crop:
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
    
    else:
        annotated_image = result.plot()  

        # 转换颜色通道
        if isinstance(annotated_image, np.ndarray):
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            output_path = os.path.join(output_folder, f"result_{idx}.jpg")
            Image.fromarray(annotated_image).save(output_path)


# 遍历文件夹中的所有图片
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_folder, image_file)
    print(f"Processing image: {image_path}")
    process_image_bounding_boxes(image_path, model, output_folder, idx)

print(f"All processed images and bounding box results saved in {output_folder}")
