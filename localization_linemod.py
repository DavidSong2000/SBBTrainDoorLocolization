import os, sys
import numpy as np
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import limap.base as _base
import limap.pointsfm as _psfm
import limap.util.io as limapio
import limap.util.config as cfgutils
import limap.runners as _runners
import argparse
import logging
import pycolmap
import pickle
from pathlib import Path
import json
import limap
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import random

from hloc.utils.parsers import parse_retrieval
from runners.cambridge.utils import read_scene_visualsfm, get_scene_info, get_result_filenames, eval, undistort_and_resize, extract_features, localize_sfm, match_features, pairs_from_retrieval, create_query_list
from hloc.utils.viz import save_plot, plot_images, plot_keypoints, add_text
from hloc.utils.io import read_image
from hloc import visualization


#This script is adapted from runners/cambridge/localization.py
# Script that reads in data of an object from the LINEMOD dataset. The database-query images split (aka train test split) is 
# based on "training_range.txt" as this is the common split used for experiments with this dataset.
# Script performs point only and joint object pose detection on that data.

formatter = logging.Formatter(
    fmt='[%(asctime)s %(name)s %(levelname)s] %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger("JointLoc")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


# read in data of an object from the linemod dataset
#TODO: change the data_folder_path approriately to where you saved the linemod dataset
data_folder_path=Path("/cluster/courses/3dv/data/team-22/dataset/lm/test/000004/") #for different object, change the number 000004 to the correct one
masked_rgb_folder_path=data_folder_path / "masked_rgb"
db_path=masked_rgb_folder_path / "db"
query_path=masked_rgb_folder_path / "query"

rgb_folder_path=data_folder_path / "rgb"
mask_folder_path=data_folder_path / "mask"

#Lukas changed
#num_query_imgs=50
#num_db_imgs=80

def parse_config():
    arg_parser = argparse.ArgumentParser(description='run localization with point and lines')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/localization/cambridge.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/localization/default.yaml', help='default config file')
    arg_parser.add_argument('--nvm_file', type=str, default='reconstruction.nvm', help='nvm filename')
    arg_parser.add_argument('--info_path', type=str, default=None, help='load precomputed info')

    arg_parser.add_argument('--query_images', default=None, type=Path, help='Path to the file listing query images')
    arg_parser.add_argument('--eval', default=None, type=Path, help='Path to the result file')

    arg_parser.add_argument('--num_covis', type=int, default=20,
                        help='Number of image pairs for SfM, default: %(default)s')
    arg_parser.add_argument('--num_loc', type=int, default=10,
                        help='Number of image pairs for loc, default: %(default)s')

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    shortcuts['-nn'] = '--n_neighbors'
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg['nvm_file'] = args.nvm_file
    cfg['info_path'] = args.info_path
    cfg['n_neighbors'] = args.num_covis
    cfg['n_neighbors_loc'] = args.num_loc
    # Output path for LIMAP results (tmp)
    if cfg['output_dir'] is None:
        cfg['output_dir'] = 'tmp/cambridge/{}'.format(scene_id)
    # Output folder for LIMAP linetracks (in tmp)
    if cfg['output_folder'] is None:
        cfg['output_folder'] = 'finaltracks'
    return cfg, args


#read in the data of an object of the LINEMOD dataset, namely database and query images and their corresponding intrinsic and 
# extrinisic parameters
def read_scene_lm(cfg):
    print("read_scene_lm")

    gt = open(str(data_folder_path/"scene_gt.json"))
    gt_poses = json.load(gt)

    cameras, camimages = {}, {}

    cam_K_json_obj=open(str(data_folder_path/"scene_camera.json"))
    cam_K_json=json.load(cam_K_json_obj)
    K_vec=cam_K_json["0"]["cam_K"]
    K=np.array([K_vec[0:3],K_vec[3:6],K_vec[6:9]])
    print("loaded K")

    Path(cfg["output_dir"] + "/masked_rgb").mkdir(exist_ok=True, parents=True)
    id_to_origin_name = {}

    #read in database/train ids
    training_range_path = data_folder_path / "training_range.txt"

    num_tot_images = len([f for f in rgb_folder_path.iterdir() if f.is_file()])
    
    with open(training_range_path, 'r') as file:
        train_ids = [int(line.strip()) for line in file]
        
    all_ids = set(np.arange(0, num_tot_images, 1))
    query_ids = list(all_ids - set(train_ids))
    img_hw = []
    
    #go over all images
    for filename in rgb_folder_path.iterdir():
        # load groundtruth pose
        image_id = int(filename.stem)

        #where the processed input image will be saved
        if image_id in (train_ids + query_ids):
            curr_img_path = Path(cfg["output_dir"] + "/masked_rgb/image{0:08d}.png".format(image_id))
        else:
            continue
        id_to_origin_name[image_id] = str(curr_img_path)

        current_gt_pose = gt_poses[str(image_id)]
        R_vec = current_gt_pose[0]["cam_R_m2c"]
        R_matrix = np.array([R_vec[0:3], R_vec[3:6], R_vec[6:9]])
        t_vec = current_gt_pose[0]["cam_t_m2c"]
        t_matrix = np.array(t_vec[0:3])
        t_matrix = t_matrix/1000 #because we need numbers in m but they are given in mm
        pose = limap.base.CameraPose(R_matrix, t_matrix)

        #load image
        image = cv2.imread(str(filename))
        img_hw = [image.shape[0],image.shape[1]]
        mask_path = str(mask_folder_path / filename.stem) + "_000000.png" #get image mask
        mask1 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask as grayscale
        masked_image = cv2.bitwise_and(image, image, mask=mask1) #mask the image
        cv2.imwrite(str(curr_img_path), masked_image)
        h,w=masked_image.shape[0],masked_image.shape[1]

        if image_id in (train_ids + query_ids):
            #store image with its extrinsic parameters
            camimage = limap.base.CameraImage(0, pose, image_name=str(curr_img_path))
            print(str(curr_img_path))
            camimages[image_id] = camimage 

    cameras[0] = limap.base.Camera("PINHOLE", K, cam_id=0, hw=img_hw) #all images have the same intrinsic parameters

    #where all the intrinsics and extrinsics of all db and query images are stored/accessible
    imagecols = limap.base.ImageCollection(cameras, camimages)

    neighbors = None
    ranges = None

    print("train_ids")
    print(train_ids)
    print("query_ids")
    print(query_ids)

    return imagecols, neighbors, ranges, train_ids, query_ids, id_to_origin_name

#point-only object pose estimation
def run_hloc_lm(cfg, image_dir, imagecols, neighbors, train_ids, query_ids, id_to_origin_name,
                       results_file, num_loc=10, logger=None):
    feature_conf = {
        'output': 'feats-superpoint-n4096-r1024',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    }
    retrieval_conf = extract_features.confs['netvlad']
    matcher_conf = match_features.confs['superglue']

    results_dir = results_file.parent
    query_list = results_dir / 'query_list_with_intrinsics.txt'
    loc_pairs = results_dir / f'pairs-query-netvlad{num_loc}.txt'
    image_list = ['image{0:08d}.png'.format(img_id) for img_id in (train_ids + query_ids)]
    img_name_to_id = {'image{0:08d}.png'.format(id): id for id in (train_ids + query_ids)}

    imagecols_train = imagecols.subset_by_image_ids(train_ids)
    imagecols_query = imagecols.subset_by_image_ids(query_ids)

    # create query list
    create_query_list(imagecols_query, query_list)
    if logger: logger.info(f'Query list created at {query_list}')

    # pairs for retrieval
    if logger: logger.info('Extract features for image retrieval...')
    global_descriptors = extract_features.main(retrieval_conf, Path(cfg['output_dir']) / image_dir, results_dir, image_list=image_list)
    pairs_from_retrieval.main(
        global_descriptors, loc_pairs, num_loc,
        db_list=['image{0:08d}.png'.format(img_id) for img_id in train_ids],
        query_list=['image{0:08d}.png'.format(img_id) for img_id in query_ids])

    # feature extraction
    if logger: logger.info('Feature Extraction...')
    features = extract_features.main(
        feature_conf, Path(cfg['output_dir']) / image_dir, results_dir, as_half=True, image_list=image_list)
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf['output'], results_dir)

    # run reference sfm
    if logger: logger.info('Running COLMAP for 3D points...')
    neighbors_train=None

    ref_sfm_path = _psfm.run_colmap_sfm_with_known_poses(
        cfg['sfm'], imagecols_train, os.path.join(cfg['output_dir'], 'tmp_colmap'), neighbors=neighbors_train,
        map_to_original_image_names=False, skip_exists=cfg['skip_exists']
    )
    ref_sfm = pycolmap.Reconstruction(ref_sfm_path)

    if not (cfg['skip_exists'] or cfg['localization']['hloc']['skip_exists']) or not os.path.exists(results_file):
        # point only localization
        if logger: logger.info('Running Point-only localization...')
        localize_sfm.main(
            ref_sfm, query_list, loc_pairs, features, loc_matches, results_file, covisibility_clustering=False)

        # Read coarse poses
        with open(results_file, 'r') as f:
            lines = []
            for data in f.read().rstrip().split('\n'):
                data = data.split()
                name = data[0]
                q, t = np.split(np.array(data[1:], float), [4])
                img_id = img_name_to_id[name]
                line = ' '.join([id_to_origin_name[img_id]] + [str(x) for x in q] + [str(x) for x in t]) + '\n'
                lines.append(line)

        # Change image names back
        with open(results_file, 'w') as f:
            f.writelines(lines)

        if logger: logger.info(f'Coarse pose saved at {results_file}')
    else:
        if logger: logger.info(f'Point-only localization skipped.')


    #point-only visualizations
    logger.info(f'Save visualizations.')
    images = Path(cfg["output_dir"] + "/masked_rgb")
    vis_output_path = Path(cfg["output_dir"]) / "visualization_lm/point_only"
    vis_output_path.mkdir(exist_ok=True, parents=True)

    saved_train_ids = train_ids[0::10]
    saved_query_ids = query_ids[0::5]
    for id in saved_train_ids:
        selected = [id]
        visualize_sfm_2d(ref_sfm, images, color_by="visibility", selected=selected, n=1)
        save_plot(str(vis_output_path / f"hloc_reconstruction_viz_visibility_{id}.png"), dpi=1200)
        matplotlib.pyplot.close()
        visualize_sfm_2d(ref_sfm, images, color_by="track_length", selected=selected, n=1)
        save_plot(str(vis_output_path / f"hloc_reconstruction_viz_track_length_{id}.png"), dpi=1200)
        matplotlib.pyplot.close()
        visualize_sfm_2d(ref_sfm, images, color_by="depth", selected=selected, n=1)
        save_plot(str(vis_output_path / f"hloc_reconstruction_viz_depth_{id}.png"), dpi=1200)
        matplotlib.pyplot.close()
    
    selected = ['image{0:08d}.png'.format(id) for id in query_ids]
    visualization.visualize_loc(results_file, images, ref_sfm, n=1, top_k_db=3, selected=selected, seed=2)
    save_plot(str(vis_output_path / "hloc_reconstruction_viz_localization.png"), dpi=1200)
    matplotlib.pyplot.close()

    # Read coarse poses
    poses = {}
    with open(results_file, 'r') as f:
        lines = []
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:], float), [4])
            poses[name] = _base.CameraPose(q, t)
    if logger: logger.info(f'Coarse pose read from {results_file}')
    hloc_log_file = f'{results_file}_logs.pkl'

    return ref_sfm, poses, hloc_log_file


def plot_lines(line_segments, colors='orange', lw=1):
    """Plot lines for existing images.
    Args:
        line_segments: list of ndarrays of size (N, 2, 2), each containing line segments.
        colors: string or list of colors for the lines (one color per image).
        lw: line width.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(line_segments)
    
    axes = plt.gcf().axes
    for ax, segments, color in zip(axes, line_segments, colors):
        for segment in segments:
            ax.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color=color, linewidth=lw)


#Save some visualizations of reconstruction
def visualize_sfm_2d(reconstruction, image_dir, color_by='visibility',
                     selected=[], n=1, seed=0, dpi=75):
    assert image_dir.exists()
    if not isinstance(reconstruction, pycolmap.Reconstruction):
        reconstruction = pycolmap.Reconstruction(reconstruction)

    if not selected:
        image_ids = reconstruction.reg_image_ids()
        selected = random.Random(seed).sample(
                image_ids, min(n, len(image_ids)))

    for i in selected:
        image = reconstruction.images[i]
        keypoints = np.array([p.xy for p in image.points2D])
        visible = np.array([p.has_point3D() for p in image.points2D])

        if color_by == 'visibility':
            color = [(0, 0, 1) if v else (1, 0, 0) for v in visible]
            text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
        elif color_by == 'track_length':
            tl = np.array([reconstruction.points3D[p.point3D_id].track.length()
                           if p.has_point3D() else 1 for p in image.points2D])
            max_, med_ = np.max(tl), np.median(tl[tl > 1])
            tl = np.log(tl)
            color = cm.jet(tl / tl.max()).tolist()
            text = f'max/median track length: {max_}/{med_}'
        elif color_by == 'depth':
            p3ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
            z = np.array([image.transform_to_image(
                reconstruction.points3D[j].xyz)[-1] for j in p3ids])
            z -= z.min()
            color = cm.jet(z / np.percentile(z, 99.9))
            text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
            keypoints = keypoints[visible]
        elif color_by == 'all_same':
            color = [(0, 0, 1) for v in visible]
            text = ""
        else:
            raise NotImplementedError(f'Coloring not implemented: {color_by}.')

        name = image.name
        plot_images([read_image(image_dir / name)], dpi=dpi)
        plot_keypoints([keypoints], colors=[color], ps=4)
        add_text(0, text)
        add_text(0, name, pos=(0.01, 0.01), fs=5, lcolor=None, va='bottom')

def main():
    cfg, args = parse_config()
    cfg = _runners.setup(cfg)
    scene_id = os.path.basename(data_folder_path)

    # outputs is for localization-related results
    outputs = Path(cfg['output_dir']) / 'localization'
    outputs.mkdir(exist_ok=True, parents=True)

    logger.info(f'Working on scene "{scene_id}".')
    imagecols, neighbors, ranges, train_ids, query_ids, id_to_origin_name = read_scene_lm(cfg)

    # GT for queries
    poses_gt = {img_id: imagecols.camimage(img_id).pose for img_id in imagecols.get_img_ids()}

    if args.eval is not None:
        eval(args.eval, poses_gt, query_ids, id_to_origin_name, logger)
        return

    image_dir = cfg["output_dir"] + "/masked_rgb"

    imagecols_train = imagecols.subset_by_image_ids(train_ids)

    results_point, results_joint = get_result_filenames(cfg['localization'], args)
    results_point, results_joint = outputs / results_point, outputs / results_joint

    img_name_to_id = {"image{0:08d}.png".format(id): id for id in (train_ids + query_ids)}
    
    num_db_imgs = len(train_ids)
    num_query_imgs = len(query_ids)

    ##########################################################
    # [A] hloc point-based localization
    ##########################################################
    ref_sfm, poses, hloc_log_file = run_hloc_lm(
        cfg, image_dir, imagecols, neighbors, train_ids, query_ids, id_to_origin_name,
        results_point, args.num_loc, logger
    )
    eval(results_point, poses_gt, query_ids, id_to_origin_name, logger)

    # Some paths useful for LIMAP localization too
    loc_pairs = outputs / f'pairs-query-netvlad{args.num_loc}.txt'

    ##########################################################
    # [B] LIMAP triangulation/fitnmerge for database line tracks
    ##########################################################
    finaltracks_dir = os.path.join(cfg["output_dir"], "finaltracks")
    if not cfg['skip_exists'] or not os.path.exists(finaltracks_dir):
        logger.info("Running LIMAP triangulation...")
        linetracks_db = _runners.line_triangulation(cfg, imagecols_train, neighbors=neighbors, ranges=ranges)
    else:
        linetracks_db = limapio.read_folder_linetracks(finaltracks_dir)
        logger.info(f"Loaded LIMAP triangulation result from {finaltracks_dir}")


    ##########################################################
    # [C] Localization with points and lines
    ##########################################################
    _retrieval = parse_retrieval(loc_pairs)
    imagecols_query = imagecols.subset_by_image_ids(query_ids)

    retrieval = {}
    for name in _retrieval:
        qid = img_name_to_id[name]
        retrieval[id_to_origin_name[qid]] = [id_to_origin_name[img_name_to_id[n]] for n in _retrieval[name]]
    hloc_name_dict = {id: "image{0:08d}.png".format(id) for id in (train_ids + query_ids)}

     # Update coarse poses for epipolar methods
    if cfg['localization']['2d_matcher'] == 'epipolar' or cfg['localization']['epipolar_filter']:
        name_to_id = {hloc_name_dict[img_id]: img_id for img_id in query_ids}
        for qname in poses:
            qid = name_to_id[qname]
            imagecols_query.set_camera_pose(qid, poses[qname])

    with open(hloc_log_file, 'rb') as f:
        hloc_logs = pickle.load(f)
    point_correspondences = {}

    #Save 3d obj visualizations:
    logger.info("Save 3d obj visualizations")
    lines_only_obj = os.path.join(cfg["dir_save"], 'triangulated_lines_nv{0}.obj'.format(cfg["n_visible_views"]))
    points_only_obj = os.path.join(cfg["dir_save"], 'triangulated_points_only_nv{0}.obj'.format(cfg["n_visible_views"]))
    joint_obj = os.path.join(cfg["dir_save"], 'triangulated_lines_nv{0}_plus-points.obj'.format(cfg["n_visible_views"]))
    shutil.copy(lines_only_obj, joint_obj)

    formatted_points_string = ""
    for qid in query_ids:
        p2ds, p3ds, inliers = _runners.get_hloc_keypoints_from_log(hloc_logs, hloc_name_dict[qid], ref_sfm)
        point_correspondences[qid] = {'p2ds': p2ds, 'p3ds': p3ds, 'inliers': inliers}
        for pt in p3ds:
            formatted_points_string += "v "+str(pt[0])+" "+str(pt[1])+" "+str(pt[2])+"\n"
    #append points to end of copied line file to get joint 3d obj with points and lines
    with open(joint_obj, 'a') as file:
        file.write(formatted_points_string)
    # write points to points only file
    with open(points_only_obj, 'w') as file:
        file.write(formatted_points_string)
    

    final_poses = _runners.line_localization(
        cfg, imagecols_train, imagecols_query, point_correspondences, linetracks_db, retrieval, results_joint, img_name_dict=id_to_origin_name)

    #Frawa: save joint image visualizations
    logger.info("save joint image visualizations")
    vis_output_path = Path(cfg["output_dir"]) / "visualization_lm/joint"
    vis_output_path.mkdir(exist_ok=True, parents=True)
    images = Path(cfg["output_dir"] + "/masked_rgb")

    # get 2d line segments for all images
    basedir = os.path.join("line_detections", cfg["line2d"]["detector"]["method"])
    folder_load = os.path.join(cfg["dir_load"], basedir)
    all_2d_segs = limapio.read_all_segments_from_folder(os.path.join(folder_load, "segments"))
    all_2d_segs = {id: all_2d_segs[id] for id in imagecols.get_img_ids()}
    saved_train_ids = train_ids[0::10]
    for id in saved_train_ids:
        selected = [id]
        visualize_sfm_2d(ref_sfm, images, color_by="all_same", selected=selected, n=1)
        curr_2d_segs = all_2d_segs[id]
        curr_2d_segs = curr_2d_segs.reshape(-1, 2, 2)
        plot_lines([curr_2d_segs])
        save_plot(str(vis_output_path / f"joint_reconstruction_viz_visibility_{id}.png"), dpi=1200)
        matplotlib.pyplot.close()
    

    # Evaluate
    eval(results_joint, poses_gt, query_ids, id_to_origin_name, logger)

    #also compute and print proj2d error to compare to onepose++
    num_below_thresh_joint=0.0
    num_below_thresh_point=0.0

    for query in range(0, num_query_imgs):
        camview_point = _base.CameraView(imagecols.get_cameras()[0], poses[image_dir+"/"+hloc_name_dict[query_ids[query]]])
        camview_joint = _base.CameraView(imagecols.get_cameras()[0], final_poses[query_ids[query]])
        camview_gt = _base.CameraView(imagecols.get_cameras()[0], poses_gt[query_ids[query]])
        mean_proj_error_point=0.0
        mean_proj_error_joint=0.0
        for p3d in p3ds:
            pt_proj_error_joint=np.linalg.norm(camview_joint.projection(p3d)-camview_gt.projection(p3d))
            pt_proj_error_point=np.linalg.norm(camview_point.projection(p3d)-camview_gt.projection(p3d))
            mean_proj_error_joint+=pt_proj_error_joint
            mean_proj_error_point+=pt_proj_error_point
        print(str(mean_proj_error_joint/len(p3ds))+" joint vs point "+str(mean_proj_error_point/len(p3ds)))
        if((mean_proj_error_joint/len(p3ds))<=5):
            num_below_thresh_joint+=1.0
        if (mean_proj_error_point/len(p3ds) <= 5):
            num_below_thresh_point += 1.0
    logger.info("proj2D error computation:")
    logger.info("hloc proj2d error: "+str(num_below_thresh_point/num_query_imgs))
    logger.info("hybrid proj2d error: "+str(num_below_thresh_joint/num_query_imgs))

    
    #reprojection visualization
    for vis_id in range(0, num_query_imgs):#iterate over all query images
        img_path=str(image_dir+"/"+hloc_name_dict[query_ids[vis_id]])
        img_point= cv2.imread(img_path)
        img_joint= cv2.imread(img_path)
        img_match= cv2.imread(img_path)

        #compute arguments like in https://github.com/cvg/limap/blob/main/limap/runners/line_localization.py --> saved them in that code and read them in now
        data = np.load(cfg["output_dir"]+'/localization/line_corrs_'+str(query_ids[vis_id])+'.pkl', allow_pickle=True)
        l2ds = data['l2ds']
        l3ds = data['l3ds']
        l3d_ids = data['l3d_ids']
        #set poses for reprojection
        camview_joint = _base.CameraView(imagecols.get_cameras()[0], final_poses[query_ids[vis_id]])
        camview_point = _base.CameraView(imagecols.get_cameras()[0], poses[image_dir+"/"+hloc_name_dict[query_ids[vis_id]]])
        #reproject lines
        for l2d, l3d_id in zip(l2ds, l3d_ids):
            # choose random color for the matching visualization
            colorv = list(np.random.choice(range(256), size=3))
            color = (int(colorv[0]), int(colorv[1]), int(colorv[2]))
            l3d = l3ds[l3d_id]
            # draw detected lines
            img_joint = cv2.line(img_joint, l2d.start.astype(int), l2d.end.astype(int), color=[255, 0, 0])
            img_match = cv2.line(img_match, l2d.start.astype(int), l2d.end.astype(int), color=color)
            img_point = cv2.line(img_point, l2d.start.astype(int), l2d.end.astype(int), color=[255, 0, 0])
            # draw reprojected lines
            l2d_proj = l3d.projection(camview_joint)
            img_joint = cv2.line(img_joint, l2d_proj.start.astype(int), l2d_proj.end.astype(int), color=[0,0,255])
            img_match = cv2.line(img_match, l2d_proj.start.astype(int), l2d_proj.end.astype(int), color=color)
            l2d_proj_pt = l3d.projection(camview_point)
            img_point = cv2.line(img_point, l2d_proj_pt.start.astype(int), l2d_proj_pt.end.astype(int), color=[0,0,255])

        #reproject points
        p_corr=point_correspondences[query_ids[vis_id]]
        p2ds=p_corr['p2ds']
        p3ds=p_corr['p3ds']
        for p2d, p3d in zip(p2ds, p3ds):
            # draw detected points
            img_joint = cv2.circle(img_joint, p2d.astype(int), radius=1, color=[255, 0, 0])
            img_point = cv2.circle(img_point, p2d.astype(int), radius=1, color=[255, 0, 0])
            # draw reprojected points
            img_joint = cv2.circle(img_joint, camview_joint.projection(p3d).astype(int), radius=1, color=[0, 0, 255])
            img_point = cv2.circle(img_point, camview_point.projection(p3d).astype(int), radius=1, color=[0, 0, 255])

        #save image
        cv2.imwrite((cfg["output_dir"]+"/"+str(query_ids[vis_id])+"_joint.png"), img_joint)
        print((cfg["output_dir"]+"/"+str(query_ids[vis_id])+"_joint.png"))
        cv2.imwrite((cfg["output_dir"]+"/"+str(query_ids[vis_id])+"_lineMatch.png"), img_match)
        cv2.imwrite((cfg["output_dir"]+"/"+str(query_ids[vis_id])+"_point.png"), img_point)

if __name__ == '__main__':
    main()
