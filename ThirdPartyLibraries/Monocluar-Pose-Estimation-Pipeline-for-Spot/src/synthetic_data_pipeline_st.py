import blenderproc as bproc
from blenderproc.python.loader.HavenEnvironmentLoader import (
    get_random_world_background_hdr_img_path_from_haven,
)

import numpy as np
import mathutils
import os
from pathlib import Path
import json
import random
# import loguru

######################## GLOBALS ########################
CONFIG = Path("config.json")
configs = json.load(open(CONFIG))
SCENE = configs.get("SCENE", 1)
DBG = bool(configs.get("DBG", 0))
COCO = bool(configs.get("COCO", 0))
RND_CAM = bool(configs.get("RND_CAM", 0))
N_FRAMES = int(configs.get("N_FRAMES", 1))
# how many different heights should be sampled
# for each z-level, N_FRAMES are generated
N_Z_LVLS = configs.get("N_Z_LVLS", 1)
DATA_DIR: Path = Path("output") / Path(configs.get("DATA_DIR", "data"))
OUTPUT_DIR: Path = DATA_DIR / f"scene_{SCENE}-annotate"
MODEL: str = configs.get("MODEL", "nerf")
assert MODEL in [
    "nerf",
    "urdf",
    "poly",
], "MODEL must be either 'nerf' or 'urdf' or 'poly'"
#########################################################


if DBG:
    import debugpy
    import warnings

    warnings.warn("Waiting for debugger Attach...", UserWarning)
    debugpy.listen(5678)
    debugpy.wait_for_client()

# init bproc & set scene
bproc.init()
bproc.world.set_world_background_hdr_img(
    get_random_world_background_hdr_img_path_from_haven("resources/haven/")
)


# load robot & set pose
if MODEL == "nerf":
    # robot = bproc.loader.load_obj(filepath="spot/nerf/sbbdoor-panelright-3(1).dae")
    # robot = bproc.loader.load_obj(filepath="spot/nerf/sbbdoor_double.dae")
    robot = bproc.loader.load_obj(filepath="spot/nerf/sbbdoor-panelright-4.dae")

    print(f'robot: {robot}')
    robot = robot[0]
    robot.set_local2world_mat(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # Set category id which will be used in the BopWriter
    robot.set_cp("category_id", 1)

elif MODEL == "poly":
    robot = bproc.loader.load_obj(filepath="spot/blender/poly_01.dae")
    robot = robot[0]
    robot.set_cp("category_id", 1)

elif MODEL == "urdf":
    # NOTE: the urdf contains spot's body twice, because the base link, i.e. the one w/o parent, has to be removed.
    #       now it's the first child of the base link and everything works properly
    robot = bproc.loader.load_urdf(urdf_file="spot/spot_basic.urdf")
    robot.remove_link_by_index(index=0)
    robot.set_ascending_category_ids()

poi = np.array([0, 0, 1.05])
# poi = np.array([0, -60, -10])
# poi = np.array([22.6, 20, 100])

if RND_CAM:
    # Add translational random walk on top of the POI
    poi_drift = bproc.sampler.random_walk(
        total_length=N_FRAMES,
        dims=3,
        step_magnitude=0.0005,
        window_size=10,
        interval=[-0.003, 0.003],
        distribution="uniform",
    )

    # Rotational camera shaking as a random walk: Sample an axis angle representation
    camera_shaking_rot_angle = bproc.sampler.random_walk(
        total_length=N_FRAMES,
        dims=1,
        step_magnitude=np.pi / 64,
        window_size=10,
        interval=[-np.pi / 12, np.pi / 12],
        distribution="uniform",
        order=2,
    )

    camera_shaking_rot_axis = bproc.sampler.random_walk(
        total_length=N_FRAMES, dims=3, window_size=10, distribution="normal"
    )

    camera_shaking_rot_axis /= np.linalg.norm(
        camera_shaking_rot_axis, axis=1, keepdims=True
    )

# +x is left [-4, 4]
# +y is backward [3, 5]
# +z is up [-3, 3]
# x_offset = -400 # x \in [-300, -500] forward to backward
x_offset = 4 # +y \in [400, -400] L to R
z_offset = 3 # z \in [-300, 300] down to up
# random initial position of camera
if RND_CAM:
    y_offset = random.uniform(-4, 4)
    z_offset = random.uniform(-3, 3)

for y in np.linspace(3, 5, N_Z_LVLS):
    for i in range(N_FRAMES):
        x = x_offset * np.cos(i / (1 * N_FRAMES) * 2 * np.pi)
        z = z_offset * np.sin(i / (1 * N_FRAMES) * 2 * np.pi)

        location_cam = np.array([x, y, z])
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location_cam)

        if RND_CAM:
            # Compute rotation based on vector going from location towards poi + drift
            rotation_matrix = bproc.camera.rotation_from_forward_vec(
                poi + poi_drift[i] - location_cam
            )

            # random walk axis-angle -> rotation matrix
            R_rand = np.array(
                mathutils.Matrix.Rotation(
                    camera_shaking_rot_angle[i], 3, camera_shaking_rot_axis[i]
                )
            )

            # Add the random walk to the camera rotation
            rotation_matrix = R_rand @ rotation_matrix

        cam2world_matrix = bproc.math.build_transformation_mat(
            location_cam, rotation_matrix
        )

        bproc.camera.set_resolution(512, 512)
        bproc.camera.add_camera_pose(cam2world_matrix)



# # draw a bounding box in blender to check if the pose is correct
# bbox_properties = {
#     # bounding box center coordinates in world coordinates
#     "px": -0.163014, #17.719, #1.6779,
#     "py": -0.035823, #25.761, #-15.825,
#     "pz": 1.09251, #108.62, #107.32,
#     # bounding box dimensions in world coordinates
#     "ex": 0.441565, #22.506, #6.382,
#     "ey": 0.078606, #84.020, #41.579,
#     "ez": 1.1143, #108.268, #110.115,
#     # bounding box orientation in world coordinates (quaternion)
#     "qw": 1.00,
#     "qx": 0.00,
#     "qy": 0.00,
#     "qz": 0.00,
# }

# # Create a cube
# cube = bproc.object.create_primitive('CUBE')
# cube.set_location([bbox_properties['px'], bbox_properties['py'], bbox_properties['pz']]) 
# cube.set_scale([bbox_properties['ex'], bbox_properties['ey'], bbox_properties['ez']])


# render results & save to disk
bproc.renderer.set_max_amount_of_samples(30)
bproc.renderer.enable_depth_output(True)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

data = bproc.renderer.render()
# bproc.writer.write_gif_animation(OUTPUT_DIR, data)

# if COCO:
#     loguru.logger.info("Writing COCO annotations...")
#     bproc.writer.write_coco_annotations(
#         os.path.join(OUTPUT_DIR, "coco_data"),
#         instance_segmaps=data["instance_segmaps"],
#         instance_attribute_maps=data["instance_attribute_maps"],
#         colors=data["colors"],
#         color_file_format="JPEG",
#     )

ignore_dist_thres = configs.get("IGNORE_DIST_THRES", 700.)

if MODEL in ["nerf", "poly"]:
    print(f'writing bop data to {os.path.join(OUTPUT_DIR, "bop_data")}')
    bproc.writer.write_bop(
        os.path.join(OUTPUT_DIR, "bop_data"),
        target_objects=[robot],
        depths=data["depth"],
        colors=data["colors"],
        m2mm=False,
        calc_mask_info_coco=False,
        ignore_dist_thres=ignore_dist_thres,
    )


elif MODEL == "urdf":
    bproc.writer.write_bop(
        os.path.join(OUTPUT_DIR, "bop_data"),
        target_objects=robot.links,
        depths=data["depth"],
        colors=data["colors"],
        m2mm=False,
        calc_mask_info_coco=False,
        ignore_dist_thres=ignore_dist_thres,
    )
