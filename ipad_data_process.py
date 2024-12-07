import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
import json

Datadir = "data/iPad_scans/2024_08_08_10_11_23"
NewDatadir = Datadir + "_processed"

NewFramesdir = NewDatadir + "/frames"
NewDepthdir = NewDatadir + "/depth"
os.makedirs(NewFramesdir, exist_ok=True)
os.makedirs(NewDepthdir, exist_ok=True)

framenames = [f for f in os.listdir(Datadir) if f.endswith('.jpg') and f.startswith('frame')]
framenames = sorted(framenames)

depthnames = []
for f in framenames:
    depthnames.append(f.replace("frame", "depth").replace(".jpg", ".png"))


# move to the directory where the frames are stored
new_camera_json = {}
idx = 0

for f in framenames:
    shutil.copy(Datadir + "/" + f, NewFramesdir + "/" + f)
    img_h, img_w, _ = cv2.imread(Datadir + "/" + f).shape
    # deal with camera.json
    camera_json_filename = f.replace(".jpg", ".json")
    camera_json_file = open(Datadir + "/" + camera_json_filename)
    camera_json = json.load(camera_json_file)
    new_camera_json[idx] = {
        "cam_K": camera_json["intrinsics"],
        "cam_w2c": camera_json["cameraPoseARFrame"],
        "depth_scale": 1.0,
        "height": img_h,
        "width": img_w,
        "framepath": NewFramesdir + "/" + f,
        "depthpath": NewDepthdir + "/" + depthnames[idx],
        "idx": int(f.replace("frame_", "").replace(".jpg", ""))
    }
    idx += 1

# write with format
new_camera_json_filename = NewDatadir + "/camera.json"
with open(new_camera_json_filename, 'w') as outfile:
    json.dump(new_camera_json, outfile, indent=4)

for f in depthnames:
    shutil.copy(Datadir + "/" + f, NewDepthdir + "/" + f)

