# go through HL2-test/pred_poses.json and interpolate the poses that have less than 25 inliers.
# interpolate using the timestamps from VLC LF_rig2world.txt
import json
import numpy as np
import scipy.interpolate as interp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import transforms3d as t3d
import ast
import argparse

def interpolate(rig_ts, rgb_ts, rig_transforms):
    '''
    interpolate so that rig_ts matches rgb_ts
    and separate the transform into translation and rotation
    use linear interpolation for translation
    use slerp for rotation
    combine the translation and rotation into one transform
    return a list of interpolated poses of dim [LEN, 4, 4]
    '''
    all_timestamps = sorted(list(set(rig_ts + rgb_ts)))
    assert(len(all_timestamps) == len(rig_ts) + len(rgb_ts)) # no duplicates

    rig_T_t, rig_T_r = {ts: pose[:3, 3] for ts, pose in rig_transforms.items()}, {ts: pose[:3, :3] for ts, pose in rig_transforms.items()}

    # interpolate translation
    interp_t = interp.interp1d(rig_ts, np.array([rig_T_t[ts] for ts in rig_ts]), axis=0)
    interp_t = interp_t(all_timestamps)
    

    # interpolate rotation with slerp
    slerp = Slerp(rig_ts, R.from_matrix([rot for rot in rig_T_r.values()]))
    interp_r = slerp(all_timestamps)

    # combine translation and rotation
    interp_transforms = []
    for i in range(len(all_timestamps)):
        transform = np.eye(4)
        transform[:3, 3] = interp_t[i]
        transform[:3, :3] = interp_r[i].as_matrix()
        interp_transforms.append(transform)

    assert(len(interp_transforms) == len(all_timestamps))
    assert(len(interp_transforms[0]) == 4)
    assert(len(interp_transforms[0][0]) == 4)
    return interp_transforms, all_timestamps

def extract_lines(lines):
    pairs = []
    for line in lines:
        line = line.strip().split(',')
        ts = int(line[0])
        transform = np.array([float(v) for v in line[1:]]).reshape(4, 4)
        pairs.append((ts, transform))
    return pairs

def calculate_door_poses(T_world_cams, pred_poses, use_pred=False, static_box=False, inliers_thresh=40, use_cherry_picked=False):
    last_good_pose_in_world = None
    last_good_ts = None

    new_pred_poses = {}

    # a transformation hardcoding
    hardcoded = np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ])

    if use_cherry_picked:
        # ts_as_static_in_world defined in main
        P_cam_door, inliers, png_name = [v for v in pred_poses.values() if v[2][:-4] == ts_as_static_in_world][0]
        P_cam_door = np.array(P_cam_door)
        P_cam_door = np.vstack((P_cam_door, np.array([0, 0, 0, 1])))
        T_world_cam = T_world_cams[int(ts_as_static_in_world)]
        last_good_pose_in_world = T_world_cam @ np.linalg.inv(hardcoded) @ P_cam_door
        last_good_ts = int(ts_as_static_in_world)

    if static_box:
        # first_pose is zero in world
        static_pose_in_world = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -5],
            [0, 0, 0, 1]
        ])
        for key in pred_poses:
            interpolated = False
            P_cam_door, inliers, png_name = pred_poses[key]
            ts = int(png_name[:-4])
            # Use first pose in world as static pose and calculate the current pose
            T_world_cam = T_world_cams[ts]
            
            # with hard coded rotation
            first_pose_in_cam = hardcoded @ np.linalg.inv(T_world_cam) @ static_pose_in_world #!!!!!!!!!!!!!!!!!
            # first_pose_in_cam = np.linalg.inv(T_world_cam) @ static_pose_in_world
            new_pred_poses[key] = [first_pose_in_cam.tolist(), inliers, png_name, interpolated]
        return new_pred_poses

    prev_pose = None
    prev_ts = None

    for key in pred_poses:
        interpolated = False
        P_cam_door, inliers, png_name = pred_poses[key]
        P_cam_door = np.array(P_cam_door)
        P_cam_door = np.vstack((P_cam_door, np.array([0, 0, 0, 1])))
        ts = int(png_name[:-4])

        if ts > 133410527694704003: # parlor trick # older 133410527664382925
            use_cherry_picked = False


        ####################################### CHECK IF POSITION JUMPED TOO FAR
        potential_jump = False
        if not use_cherry_picked:
            if prev_pose is None:
                prev_pose = P_cam_door
                prev_ts = ts
            else:
                # solve for translation distance between prev_pose and P_cam_door
                delta_t = np.linalg.norm(P_cam_door[:3, 3] - prev_pose[:3, 3])
                if delta_t > 0.6: # jumped between doors
                    potential_jump = True
                else:
                    prev_pose = P_cam_door
                    prev_ts = ts
        #######################################

        if use_pred:
            interpolated = False
        elif inliers >= inliers_thresh and not use_cherry_picked and not potential_jump:
            T_world_cam = T_world_cams[ts] 
            
            last_good_pose_in_world = T_world_cam @ np.linalg.inv(hardcoded) @ P_cam_door
            last_good_ts = ts
        elif last_good_pose_in_world is None: interpolated = False
        elif T_world_cams is None: interpolated = False
        else: # use the last good pose to calculate the current pose
            interpolated = True
            T_world_cam = T_world_cams[ts]

            P_cam_door = hardcoded @ np.linalg.inv(T_world_cam) @ last_good_pose_in_world

        new_pred_poses[key] = [P_cam_door.tolist(), inliers, png_name, interpolated]
    
    return new_pred_poses


def normalize_rotation_matrix(rot_mat):
    """
    Normalize a 3x3 rotation matrix.ßß

    Parameters:
    rot_mat (numpy.ndarray): A 3x3 rotation matrix

    Returns:
    numpy.ndarray: A normalized 3x3 rotation matrix
    """
    # Use Singular Value Decomposition (SVD) for normalization
    u, _, vh = np.linalg.svd(rot_mat)
    # Reconstruct the rotation matrix to ensure it is orthogonal
    return np.dot(u, vh)

def load_pv_data(csv_path):
    with open(csv_path) as f:
        lines = f.readlines()

    # The first line contains info about the intrinsics.
    # The following lines (one per frame) contain timestamp, focal length and transform PVtoWorld
    n_frames = len(lines) - 1
    frame_timestamps = np.zeros(n_frames, dtype=np.longlong)
    focal_lengths = np.zeros((n_frames, 2))
    pv2world_transforms = np.zeros((n_frames, 4, 4))

    intrinsics_ox, intrinsics_oy, \
        intrinsics_width, intrinsics_height = ast.literal_eval(lines[0])

    for i_frame, frame in enumerate(lines[1:]):
        # Row format is
        # timestamp, focal length (2), transform PVtoWorld (4x4)
        frame = frame.split(',')
        frame_timestamps[i_frame] = int(frame[0])
        focal_lengths[i_frame, 0] = float(frame[1])
        focal_lengths[i_frame, 1] = float(frame[2])
        pv2world_transforms[i_frame] = np.array(frame[3:20]).astype(float).reshape((4, 4))

        # * 1000
        # pv2world_transforms[i_frame, :3, 3] *= 1000

    return (frame_timestamps, focal_lengths, pv2world_transforms,
            intrinsics_ox, intrinsics_oy, intrinsics_width, intrinsics_height)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='HL2-test')    
    parser.add_argument('--use_pred', action='store_true')
    parser.add_argument('--static_box', action='store_true')
    parser.add_argument('--use_cherry_picked', action='store_true')
    args = parser.parse_args()

    with open(f'{args.source}/pred_poses.json', 'r') as f:
        pred_poses = json.load(f)

    print(f'len of pred_poses before removing the first pose: {len(pred_poses)}')

    # ts_as_static_in_world = '133410527672712892' # tracks the left panel well
    ts_as_static_in_world = '133410527336848446' # tracks the right panel well

    # [[[-0.9437829383180389, -0.3302695626177048, 0.01399218882374706, -0.5949360299246331], 
    # [-0.02144540958826427, 0.01893407800358715, -0.9995907137912726, 0.9009865022030388], 
    # [0.3298694586459332, -0.9436967291977144, -0.02495242581744228, 4.332067427560067]], 25, "133410527672712892.png"]

    thresh = 25

    # get rid of the first interpolated pose above 25 inliers because it's not good
    k_first = None
    for k in pred_poses:
        if pred_poses[k][1] >= thresh:
            k_first = k
            break
    pred_poses = {k: v for k, v in pred_poses.items() if int(k) >= int(k_first)}

    print(f'number of pred_poses with inlier above {thresh}: {len([v for v in pred_poses.values() if v[1] >= thresh])}')

    print(f'len of pred_poses after removing the first pose: {len(pred_poses)}')

    frame_timestamps, T_world_cams = None, None
    try:
        frame_timestamps, focal_lengths, T_world_cams, intrinsics_ox, intrinsics_oy, intrinsics_width, intrinsics_height = load_pv_data(f'{args.source}/odom.txt')
    except:
        print(f'could not load {args.source}/odom.txt')

    # with open('VLC_LF_rig2world.txt', 'r') as f:
        # lines = f.readlines()
    # rig_transforms = {ts: transform for ts, transform in extract_lines(lines)}

    # rig_ts = [int(ts) for ts in rig_transforms]           # number of timestamps for rig poses
    # timestamps = [v[2][:-4] for v in pred_poses.values()] # number of timestamps for RGB images
    # timestamps = [int(ts) for ts in timestamps if int(ts) < max(rig_ts) and int(ts) > min(rig_ts)] # cannot extrapolate
    
    # pred_poses = {v[2][:-4]: v for v in pred_poses.values()}    
    # pred_poses = {ts: pred_poses[str(ts)] for ts in timestamps} # only keep the poses that have timestamps

    print(f'len of pred_poses after removing the poses that do not have timestamps: {len(pred_poses)}')

    # # interpolate rig poses to match the timestamps of RGB images
    # T_LF_worlds, all_timestamps = interpolate(rig_ts, timestamps, rig_transforms)

    # T_cam_LF = np.array([[0, 1, 0, 0],
    #                     [-1, 0, 0, 0],
    #                     [0, 0, 1, 0],
    #                     [0, 0, 0, 1]])
    # T_LF_cam = np.linalg.inv(T_cam_LF) 

    # apply static transform to all transforms
    # T_world_cams = [np.linalg.inv(T_LF_world) @ T_LF_cam for T_LF_world in T_LF_worlds]

    # map T_world_cams with timestamps
    try:
        assert(len(frame_timestamps) == len(T_world_cams))
        T_world_cams = {ts: T_world_cams[i] for i, ts in enumerate(frame_timestamps)}
    except:
        print(f'frame_timestamps or T_world_cams is None')

    P_cam_doors = calculate_door_poses(T_world_cams, pred_poses, args.use_pred, args.static_box, thresh, use_cherry_picked=args.use_cherry_picked)

    with open(f'{args.source}/pred_poses_interpolated.json', 'w') as f:
        json.dump(P_cam_doors, f, indent=4)
