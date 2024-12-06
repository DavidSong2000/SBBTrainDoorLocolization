import os
from ultralytics import YOLO


model = YOLO('weights/best.pt')


image_path = 'test/'
output_folder = 'output/'
os.makedirs(output_folder, exist_ok=True)  

results = model(image_path)  


for idx, result in enumerate(results):

    annotated_image = result.plot()  

    output_path = os.path.join(output_folder, f"result_{idx}.jpg")
    from PIL import Image
    Image.fromarray(annotated_image).save(output_path)

print(f"all output images saved in {output_folder}")
