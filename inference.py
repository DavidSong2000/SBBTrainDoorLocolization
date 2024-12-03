import torch
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
from semantic_sam import prepare_image, plot_multi_results, build_semantic_sam, SemanticSAMPredictor
from semantic_sam.build_semantic_sam import select_mask

# get point position
# display the image with GUI and you can click on the image to get the point position
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load your image (replace 'path_to_your_image.jpg' with your actual image file path)
image_path = 'data/sbb_train_door/IMG_3116.JPG'
image = mpimg.imread(image_path)
img_h, img_w = image.shape[:2]

# Variable to store click coordinates
click_coordinates = None

# Function to handle click events and store coordinates
def on_click(event):
    global click_coordinates
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        click_coordinates = (x, y)
        print(f'Clicked at: ({x}, {y})')
        plt.close()  # Close the plot after the click

# Display the image and capture coordinates
def get_click_coordinates():
    global click_coordinates
    click_coordinates = None  # Reset coordinates before each click
    fig, ax = plt.subplots()
    ax.imshow(image)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    return [click_coordinates[0] / img_w, click_coordinates[1] / img_h]  # Return normalized coordinates

# Main part of the code
coordinates = get_click_coordinates()
print("Returned Coordinates:", coordinates)

original_image, input_image = prepare_image(image_pth='data/sbb_train_door/IMG_3116.JPG')  # change the image path to your image

with torch.no_grad():
    mask_generator = SemanticSAMPredictor(build_semantic_sam(model_type='L', ckpt='ckpt/swinl_only_sam_many2many.pth')) # model_type: 'L' / 'T', depends on your checkpint
    masks, ious, [iou_sort_masks, area_sort_masks] = mask_generator.predict_masks_(original_image, input_image, point=[[coordinates[0], coordinates[1]]]) # input point [[w, h]] relative location, i.e, [[0.5, 0.5]] is the center of the image
    select_mask(masks, iou_sort_masks, area_sort_masks, original_image, save_path='./vis/')  # results and original images will be saved at save_path