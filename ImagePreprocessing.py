import os
from PIL import Image, ImageOps

# 输入和输出文件夹路径
input_folder = "../CombinedRawData/"  # 你的输入图片文件夹
output_folder = "../TransposedCombinedRawData/"  # 输出文件夹
os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在

def batch_fix_image_orientation(input_folder, output_folder):
    """
    批量调整图片方向并保存到输出文件夹。
    """
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('jpg', 'jpeg', 'png')):  # 支持的图片格式
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # 打开图片并自动调整方向
                with Image.open(input_path) as img:
                    fixed_img = ImageOps.exif_transpose(img)
                    fixed_img.save(output_path)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# 执行批量调整
batch_fix_image_orientation(input_folder, output_folder)

print("批量图片方向调整完成！")
