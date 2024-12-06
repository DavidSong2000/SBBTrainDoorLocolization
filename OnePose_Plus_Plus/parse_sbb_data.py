import os
import shutil
import pathlib
import json

import numpy as np


def main():
    obj_id = "0901"
    outdir = "data/datasets/sbb_data/"
    paths = [
        "data/sbb/test_coco_2_version_landschaft/scene_0-annotate/bop_data/",
        "data/sbb/test_coco_2_version_gebaeude/scene_0-annotate/bop_data/",
    ]

    train_ids = [
        85,
        102,
        38,
        100,
        114,
        101,
        108,
        51,
        17,
        76,
        139,
        42,
        2,
        60,
        29,
        119,
        70,
        148,
        16,
        120,
        73,
        53,
        130,
        127,
        43,
        135,
        4,
        96,
        31,
        123,
        14,
        106,
        107,
        145,
        36,
        40,
        18,
        78,
        68,
        59,
        129,
        99,
        81,
        22,
        137,
        12,
        65,
        98,
        37,
        110,
        25,
        140,
        6,
        54,
        141,
        109,
        61,
        146,
        86,
        112,
        136,
        21,
        77,
        20,
        104,
        84,
        34,
        147,
        15,
        92,
        30,
        124,
        122,
        69,
        133,
        132,
        46,
        55,
        117,
        80,
    ]

    query_ids = [
        102,
        38,
        100,
        101,
        108,
        17,
        76,
        139,
        42,
        119,
        16,
        120,
        53,
        127,
        43,
        135,
        4,
        96,
        31,
        106,
        145,
        36,
        40,
        18,
        68,
        59,
        137,
        12,
        37,
        110,
        140,
        6,
        54,
        109,
        146,
        86,
        112,
        21,
        77,
        104,
        34,
        147,
        92,
        124,
        122,
        69,
        133,
        132,
        117,
        80,
    ]

    id_filter = [train_ids, query_ids]

    for idx, path in enumerate(paths):
        image_seq_dir = os.path.join(path, "train_pbr", "000000", "rgb")
        image_out_dir = os.path.join(
            outdir, f"{obj_id}-ssb-others", f"sbb-{idx + 1}", "color"
        )
        intrin_out_dir = os.path.join(
            outdir, f"{obj_id}-ssb-others", f"sbb-{idx + 1}", "intrin_ba"
        )
        poses_out_dir = os.path.join(
            outdir, f"{obj_id}-ssb-others", f"sbb-{idx + 1}", "poses_ba"
        )

        pathlib.Path(image_out_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(intrin_out_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(poses_out_dir).mkdir(parents=True, exist_ok=True)

        camera_json = os.path.join(path, "train_pbr", "000000", "scene_camera.json")
        camera_dict = json.load(open(camera_json, encoding="utf-8"))

        gt_json = os.path.join(path, "train_pbr", "000000", "scene_gt.json")
        gt_dict = json.load(open(gt_json, encoding="utf-8"))

        for image in os.listdir(image_seq_dir):

            image_num, image_extension = image.split(".")
            image_num = int(image_num)
            assert image_extension == "png"

            if image_num not in id_filter[idx]:
                continue

            shutil.copy(
                os.path.join(image_seq_dir, image),
                os.path.join(image_out_dir, f"{image_num}.png"),
            )

            intin_matrix = np.array(camera_dict[str(image_num)]["cam_K"]).reshape(3, 3)
            np.savetxt(os.path.join(intrin_out_dir, f"{image_num}.txt"), intin_matrix)

            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = np.array(
                gt_dict[str(image_num)][0]["cam_R_m2c"]
            ).reshape(3, 3)
            pose_matrix[:3, -1] = (
                np.array(gt_dict[str(image_num)][0]["cam_t_m2c"]) / 1000
            )
            np.savetxt(os.path.join(poses_out_dir, f"{image_num}.txt"), pose_matrix)

    # bbox
    px, py, pz = -0.163014, -0.035823, 1.09251
    ex, ey, ez = 0.441565, 0.078606, 1.1143
    ex *= 2
    ey *= 2
    ez *= 2
    bbox_3d = np.array(
        [
            [px - ex, py - ey, pz - ez],  # back, left, down
            [px + ex, py - ey, pz - ez],  # front, left, down
            [px + ex, py - ey, pz + ez],  # front, left, up
            [px - ex, py - ey, pz + ez],  # back, left, up
            [px - ex, py + ey, pz - ez],  # back, right, down
            [px + ex, py + ey, pz - ez],  # front, right, down
            [px + ex, py + ey, pz + ez],  # front, right, up
            [px - ex, py + ey, pz + ez],  # back, right, up
        ]
    )
    np.savetxt(
        os.path.join(outdir, f"{obj_id}-ssb-others", "box3d_corners.txt"),
        bbox_3d,
    )


if __name__ == "__main__":
    main()
