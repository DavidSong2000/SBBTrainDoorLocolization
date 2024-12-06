import os
import shutil
import pathlib
import json

import numpy as np
import cv2

from src.utils.data_utils import get_image_crop_resize, get_K_crop_resize


def main():

    obj_name = {
        2: "benchwise",
        4: "camera",
        5: "watering_can",
        8: "drill",
        10: "egg_carton",
    }

    lm_path = pathlib.Path("data/lm")
    outdir = pathlib.Path("data/datasets/lm_mask/")

    for obj_id in [2, 4, 5, 8, 10]:
        obj_path = lm_path / "test" / f"{obj_id:06}"

        image_seq_dir = obj_path / "rgb"
        image_mask_dir = obj_path / "mask"
        image_out_dir = (
            outdir
            / f"{(obj_id + 1000)}-{obj_name[obj_id]}-others"
            / f"{obj_name[obj_id]}-1"
            / "color",
            outdir
            / f"{(obj_id + 1000)}-{obj_name[obj_id]}-others"
            / f"{obj_name[obj_id]}-2"
            / "color",
        )
        intrin_out_dir = (
            outdir
            / f"{(obj_id + 1000)}-{obj_name[obj_id]}-others"
            / f"{obj_name[obj_id]}-1"
            / "intrin_ba",
            outdir
            / f"{(obj_id + 1000)}-{obj_name[obj_id]}-others"
            / f"{obj_name[obj_id]}-2"
            / "intrin_ba",
        )
        poses_out_dir = (
            outdir
            / f"{(obj_id + 1000)}-{obj_name[obj_id]}-others"
            / f"{obj_name[obj_id]}-1"
            / "poses_ba",
            outdir
            / f"{(obj_id + 1000)}-{obj_name[obj_id]}-others"
            / f"{obj_name[obj_id]}-2"
            / "poses_ba",
        )

        image_out_dir[0].mkdir(parents=True, exist_ok=True)
        image_out_dir[1].mkdir(parents=True, exist_ok=True)

        intrin_out_dir[0].mkdir(parents=True, exist_ok=True)
        intrin_out_dir[1].mkdir(parents=True, exist_ok=True)

        poses_out_dir[0].mkdir(parents=True, exist_ok=True)
        poses_out_dir[1].mkdir(parents=True, exist_ok=True)

        camera_json = obj_path / "scene_camera.json"
        camera_dict = json.load(open(camera_json, encoding="utf-8"))

        gt_json = obj_path / "scene_gt.json"
        gt_dict = json.load(open(gt_json, encoding="utf-8"))

        gt_info_json = obj_path / "scene_gt_info.json"
        gt_info_dict = json.load(open(gt_info_json, encoding="utf-8"))

        training_range_path = obj_path / "training_range.txt"
        with open(training_range_path, "r", encoding="utf-8") as file:
            train_ids = [int(line.strip()) for line in file]

        for image in os.listdir(image_seq_dir):

            image_num, image_extension = image.split(".")
            image_num = int(image_num)
            assert image_extension == "png"

            out_index = 0 if image_num in train_ids else 1

            i = cv2.imread(str(image_seq_dir / image))

            mask_path = image_mask_dir / f"{image_num:06}_000000.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            masked_image = cv2.bitwise_and(i, i, mask=mask)

            intin_matrix = np.array(camera_dict[str(image_num)]["cam_K"]).reshape(3, 3)

            bbox = gt_info_dict[str(image_num)][0]["bbox_obj"]
            x0, y0, w, h = bbox
            x1, y1 = x0 + w, y0 + h

            # Crop image by 2D visible bbox, and change K
            box = np.array([x0, y0, x1, y1])
            resize_shape = np.array([y1 - y0, x1 - x0])
            K_crop, K_crop_homo = get_K_crop_resize(box, intin_matrix, resize_shape)
            image_crop, _ = get_image_crop_resize(i, box, resize_shape)

            box_new = np.array([0, 0, x1 - x0, y1 - y0])
            resize_shape = np.array([256, 256])
            K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
            image_crop, _ = get_image_crop_resize(image_crop, box_new, resize_shape)

            cv2.imwrite(str(image_out_dir[out_index] / f"{image_num}.png"), image_crop)
            np.savetxt(
                os.path.join(intrin_out_dir[out_index], f"{image_num}.txt"),
                K_crop,
            )

            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = np.array(
                gt_dict[str(image_num)][0]["cam_R_m2c"]
            ).reshape(3, 3)
            pose_matrix[:3, -1] = (
                np.array(gt_dict[str(image_num)][0]["cam_t_m2c"]) / 1000
            )
            np.savetxt(
                os.path.join(poses_out_dir[out_index], f"{image_num}.txt"),
                pose_matrix,
            )

        # bbox
        models_info_json = lm_path / "models" / "models_info.json"
        models_info_dict = json.load(open(models_info_json, encoding="utf-8"))

        model_min_xyz = np.array(
            [
                models_info_dict[str(obj_id)]["min_x"],
                models_info_dict[str(obj_id)]["min_y"],
                models_info_dict[str(obj_id)]["min_z"],
            ]
        )
        model_size_xyz = np.array(
            [
                models_info_dict[str(obj_id)]["size_x"],
                models_info_dict[str(obj_id)]["size_y"],
                models_info_dict[str(obj_id)]["size_z"],
            ]
        )
        scale = model_size_xyz / 1000  # convert to m

        # Save 3D bbox:
        corner_in_cano = np.array(
            [
                [
                    -scale[0],
                    -scale[0],
                    -scale[0],
                    -scale[0],
                    scale[0],
                    scale[0],
                    scale[0],
                    scale[0],
                ],
                [
                    -scale[1],
                    -scale[1],
                    scale[1],
                    scale[1],
                    -scale[1],
                    -scale[1],
                    scale[1],
                    scale[1],
                ],
                [
                    -scale[2],
                    scale[2],
                    scale[2],
                    -scale[2],
                    -scale[2],
                    scale[2],
                    scale[2],
                    -scale[2],
                ],
            ]
        ).T
        corner_in_cano = corner_in_cano[:, :3] * 0.5
        np.savetxt(
            outdir
            / f"{(obj_id + 1000)}-{obj_name[obj_id]}-others"
            / "box3d_corners.txt",
            corner_in_cano,
        )

        diameter = models_info_dict[str(obj_id)]["diameter"] / 1000
        np.savetxt(
            outdir / f"{(obj_id + 1000)}-{obj_name[obj_id]}-others" / "diameter.txt",
            np.array([diameter]),
        )


if __name__ == "__main__":
    main()
