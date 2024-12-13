import cv2
import os

def images_to_video(image_folder, output_file, fps=30):
    # 获取图像文件列表并按文件名排序
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # 按文件名排序

    # 确保文件夹中有图像
    if not images:
        print("No images found in the folder.")
        return

    # 读取第一张图像以获取宽高
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 初始化视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 按顺序写入每张图像
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 释放资源
    video.release()
    print(f"Video saved as {output_file}")

# 设置输入文件夹和输出文件
image_folder = "output/SyntheticScene0"  # 替换为图像文件夹路径
output_file = "output/SyntheticScene0output.mp4"
fps = 30  # 设置帧率

images_to_video(image_folder, output_file, fps)
