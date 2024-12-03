# --------------------------------------------------------
# Semantic-SAM: Segment and Recognize Anything at Any Granularity
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Feng Li (fliay@connect.ust.hk)
# --------------------------------------------------------

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import os

from utils.arguments import load_opt_from_config_file
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from tasks.automatic_mask_generator import SemanticSamAutomaticMaskGenerator
from tasks.interactive_idino_m2m_auto import show_anns
from tasks.interactive_predictor import SemanticSAMPredictor

# Variable to store click coordinates
click_coordinates = None

def prepare_image(image_pth):
    """
    apply transformation to the image. crop the image ot 640 short edge by default
    """
    image = Image.open(image_pth).convert('RGB')
    t = []
    t.append(transforms.Resize(640, interpolation=Image.BICUBIC))
    transform1 = transforms.Compose(t)
    image_ori = transform1(image)

    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()

    return image_ori, images


def build_semantic_sam(model_type, ckpt):
    """
    build model
    """
    cfgs={'T':"configs/semantic_sam_only_sa-1b_swinT.yaml",
          'L':"configs/semantic_sam_only_sa-1b_swinL.yaml"}

    sam_cfg=cfgs[model_type]
    opt = load_opt_from_config_file(sam_cfg)
    model_semantic_sam = BaseModel(opt, build_model(opt)).from_pretrained(ckpt).eval().cuda()
    return model_semantic_sam


def plot_results(outputs, image_ori, save_path='../vis/'):
    """
    plot input image and its reuslts
    """
    if os.path.isdir(save_path):
        image_ori_name = 'input.png'
        im_name = 'example.png'
    else:
        image_ori_name = os.path.basename(save_path).split('.')[0] + '_input.png'
        im_name = os.path.basename(save_path).split('.')[0]+ '_example.png'
        save_path = os.path.dirname(save_path)
        
    if not os.path.exists(save_path):
        os.mkdir(save_path)       
        
    fig = plt.figure()
    plt.imshow(image_ori)
    plt.savefig(os.path.join(save_path, image_ori_name))
    show_anns(outputs)
    fig.canvas.draw()
    im = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.savefig(os.path.join(save_path, im_name))
    return im

def plot_multi_results(iou_sort_masks, area_sort_masks, image_ori, save_path='../vis/'):
    """
    plot input image and its reuslts
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.imshow(image_ori)
    plt.savefig('../vis/input.png')
    def create_long_image(masks):
        ims = []
        for img in masks:
            ims.append(img)
        width, height = ims[0].size
        result = Image.new(ims[0].mode, (width * len(ims), height))
        for i, im in enumerate(ims):
            result.paste(im, box=(i * width, 0))
        return result
    create_long_image(iou_sort_masks).save('../vis/all_results_sort_by_iou.png')
    create_long_image(area_sort_masks).save('../vis/all_results_sort_by_areas.png')

# Function to handle click events and store coordinates
def on_click(event):
    global click_coordinates
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        click_coordinates = (x, y)
        print(f'Clicked at: ({x}, {y})')
        plt.close()  # Close the plot after the click

# Display the image and capture coordinates
def get_click_coordinates(image, img_w, img_h):
    global click_coordinates
    click_coordinates = None  # Reset coordinates before each click
    fig, ax = plt.subplots()
    ax.imshow(image)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    return [click_coordinates[0] / img_w, click_coordinates[1] / img_h]  # Return normalized coordinates

def select_mask(masks, iou_sort_masks, area_sort_masks, image_ori, save_path='../vis/'):
    """
    plot input image and its reuslts
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.imshow(image_ori)
    plt.savefig(os.path.join(save_path, 'input.png'))
    plt.close()
    def create_long_image(masks):
        ims = []
        for img in masks:
            ims.append(img)
        width, height = ims[0].size
        result = Image.new(ims[0].mode, (width * len(ims), height))
        for i, im in enumerate(ims):
            result.paste(im, box=(i * width, 0))
        return result
    
    save_dir = os.path.join(save_path, 'selected_masks.png')
    long_image = create_long_image(area_sort_masks)
    # display the image with GUI and you can click on the image to select the mask
    coordinates = get_click_coordinates(np.asarray(long_image), long_image.width, long_image.height)
    print("Returned Coordinates:", coordinates)
    length = len(area_sort_masks)
    mask_index = int(coordinates[0] * length)
    final_mask = area_sort_masks[mask_index]
    final_mask.save(save_dir)
    plt.imshow(final_mask)
    plt.show()
    # use_mask = (masks[mask_index] > 0.0).cpu().numpy()
    # # final_mask[final_mask > 0] = 1
    # blend_img = image_ori * use_mask.reshape(image_ori.shape[0], image_ori.shape[1], 1)
    # blend_img = blend_img.astype(np.uint8)
    # blend_img = Image.fromarray(blend_img)
    # blend_img.save(os.path.join(save_path, 'blend_img.png'))
