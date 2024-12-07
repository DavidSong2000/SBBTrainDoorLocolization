# Go through the images in PV and draw the bounding box based on HL2-test/pred_poses_interpolated.json

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def reproj(K, pose, pts_3d):
    """
    Reproj 3d points to 2d points
    @param K: [3, 3] or [3, 4]
    @param pose: [3, 4] or [4, 4]
    @param pts_3d: [n, 3]
    """
    assert K.shape == (3, 3) or K.shape == (3, 4)
    assert pose.shape == (3, 4) or pose.shape == (4, 4)

    if K.shape == (3, 3):
        K_homo = np.concatenate([K, np.zeros((3, 1))], axis=1)
    else:
        K_homo = K

    if pose.shape == (3, 4):
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    else:
        pose_homo = pose

    pts_3d = pts_3d.reshape(-1, 3)
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
    pts_3d_homo = pts_3d_homo.T

    reproj_points = K_homo @ pose_homo @ pts_3d_homo
    reproj_points = reproj_points[:] / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T
    return reproj_points  # [n, 2]

def draw_3d_box(image, corners_2d, linewidth=3, color="g"):
    """Draw 3d box corners
    @param corners_2d: [8, 2]
    """
    lines = np.array(
        [[0, 1, 5, 4, 2, 3, 7, 6, 0, 1, 5, 4], [1, 5, 4, 0, 3, 7, 6, 2, 3, 2, 6, 7]]
    ).T

    colors = {"g": (0, 255, 0), "r": (0, 0, 255), "b": (255, 0, 0)}
    if color not in colors.keys():
        color = (42, 97, 247)
    else:
        color = colors[color]

    for id, line in enumerate(lines):
        pt1 = corners_2d[line[0]].astype(int)
        pt2 = corners_2d[line[1]].astype(int)
        image = cv2.line(image, tuple(pt1), tuple(pt2), color, linewidth)

    return image

def save_demo_image(pose_pred, K, image_path, box3d, draw_box=True, save_path=None, interpolated=False):
    """ 
    Project 3D bbox by predicted pose and visualize
    """
    if isinstance(box3d, str):
        box3d = np.loadtxt(box3d)

    image_full = cv2.imread(image_path)

    # # draw a fixed box somewhere in the image
    # static_pose = np.array(
    #     [
    #         [
    #             -0.9702091993849774,
    #             -0.2407133606510095,
    #             -0.02740779875985519,
    #             0.00590981447415848
    #         ],
    #         [
    #             -0.00903316395727552,
    #             0.14899382481424361,
    #             -0.9887968659518215,
    #             0.6054960154337778
    #         ],
    #         [
    #             0.2421002093714185,
    #             -0.9590922365295845,
    #             -0.1467295827398014,
    #             3.230090323896229
    #         ],
    #         [
    #             0.0,
    #             0.0,
    #             0.0,
    #             1.0
    #         ]
    #     ])
    # reproj_box_static_2d = reproj(K, static_pose, box3d)
    # draw_3d_box(image_full, reproj_box_static_2d, color='g', linewidth=10)

    if draw_box:
        reproj_box_2d = reproj(K, pose_pred, box3d)
        if interpolated: draw_3d_box(image_full, reproj_box_2d, color='b', linewidth=10)
        else: draw_3d_box(image_full, reproj_box_2d, color='g', linewidth=10)
    
    if save_path is not None:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)

        cv2.imwrite(save_path, image_full)
    return image_full

def get_K(intrin_file):
    assert Path(intrin_file).exists()
    with open(intrin_file, 'r') as f:
        lines = f.readlines()
    intrin_data = [line.rstrip('\n').split(':')[1] for line in lines]
    fx, fy, cx, cy = list(map(float, intrin_data))

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    K_homo = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0,  0,  1, 0]
    ])
    return K, K_homo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='HL2-test')    
    args = parser.parse_args()


    with open(f'{args.source}/pred_poses_interpolated.json', 'r') as f:
        pred_poses_interp = json.load(f)

    for ts in pred_poses_interp:
        pred_pose, inliers, png_name, interpolated = pred_poses_interp[ts]
        pred_pose = np.array(pred_pose).reshape(4, 4)

        K, _ = get_K(f'{args.source}/intrinsics.txt')

        if args.source == 'HL2-test' or args.source == 'HL2-test-scaled':
            image = save_demo_image(pred_pose, K, 
                                    f'PV/{png_name}', f'{args.source}/box3d_corners.txt', 
                                    draw_box=True, save_path=f'PV_interp_{args.source}/{png_name}', interpolated=interpolated)
        elif args.source == 'scene_0-annotate':
            image = save_demo_image(pred_pose, K, 
                                    f'{args.source}_imgs/{png_name}', f'{args.source}/box3d_corners.txt', 
                                    draw_box=True, save_path=f'{args.source}_interp/{png_name}', interpolated=interpolated)