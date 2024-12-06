import os, sys
import numpy as np
import shutil
import glob
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import limap.base as _base
import limap.pointsfm as _psfm
import limap.util.io as limapio
import limap.util.config as cfgutils
import limap.runners as _runners
import limap.estimators as _estimators
import limap.line2d
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

import limap.merging as _mrg
import limap.triangulation as _tri
import limap.vplib as _vplib
import limap.optimize as _optim
import limap.visualize as limapvis



from hloc.utils.parsers import parse_retrieval
from runners.cambridge.utils import read_scene_visualsfm, get_scene_info, get_result_filenames, eval, undistort_and_resize, extract_features, localize_sfm, match_features, pairs_from_retrieval, create_query_list
from hloc.utils.viz import save_plot, plot_images, plot_keypoints, add_text
from hloc.utils.io import read_image
from hloc import visualization
from collections import defaultdict

#from limap.visualize import plot_lines

#This script is adapted from runners/cambridge/localization.py

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


# read in data of an object from the sbb dataset
#TODO: change the db_data_folder_path and the query2_data_folder_path approriately to where you saved the sbb-doors dataset
db_data_folder_path = Path("/cluster/courses/3dv/data/team-22/dataset/sbb_doors/train_pbr/000000")
db_imgs_path = db_data_folder_path / "rgb"

query2_data_folder_path = Path("/cluster/courses/3dv/data/team-22/dataset/sbb_doors/train_pbr/000001")
query2_imgs_path = query2_data_folder_path / "rgb"

num_query_imgs=50
num_db_imgs=80

def parse_config():
    arg_parser = argparse.ArgumentParser(description='run localization with point and lines')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/localization/cambridge.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/localization/default.yaml', help='default config file')   #Annika and Frawa changed
    #arg_parser.add_argument('-a', '--vsfm_path', type=str, required=True, help='visualsfm path')
    arg_parser.add_argument('--nvm_file', type=str, default='reconstruction.nvm', help='nvm filename')
    arg_parser.add_argument('--info_path', type=str, default=None, help='load precomputed info')

    arg_parser.add_argument('--query_images', default=None, type=Path, help='Path to the file listing query images')
    arg_parser.add_argument('--eval', default=None, type=Path, help='Path to the result file')

    # arg_parser.add_argument('--colmap_retriangulate', default=False, action='store_true')
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

    

    cfg['localization']['2d_matcher'] = 'superglue_endpoints'
    cfg['line2d']['extractor']['method'] = 'superpoint_endpoints'
    cfg['line2d']['matcher']['method'] = 'superglue_endpoints'         #nn_endpoints
    cfg['localization']['ransac']['weight_point'] = float(0)    #line localization only

  

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
    print("read_scene_sbb_doors")

    cameras, camimages = {}, {}

    # Opening JSON file for object pose ground truth
    gt = open(str(db_data_folder_path/"scene_gt.json"))
    gt_poses = json.load(gt)

   
    cam_K_json_obj=open(str(db_data_folder_path/"scene_camera.json")) 
    cam_K_json=json.load(cam_K_json_obj)
    K_vec=cam_K_json["0"]["cam_K"] #K is the same for all images
    K=np.array([K_vec[0:3],K_vec[3:6],K_vec[6:9]])

   
    final_imgs_folder_path = Path(cfg["output_dir"] + "/combined_rgb")
    final_imgs_folder_path.mkdir(exist_ok=True, parents=True)
    id_to_origin_name = {}

    num_tot_images = len([f for f in db_imgs_path.iterdir()])
    random.seed(40)
    sample = random.sample(list(np.arange(0,num_tot_images,1)), num_db_imgs)
    train_ids = []
    img_hw = []
    
    for i, filename in enumerate(db_imgs_path.iterdir()):
        # load groundtruth pose
        image_id = int(filename.stem)


        curr_img_path = final_imgs_folder_path /"image{0:08d}.png".format(image_id)

        id_to_origin_name[image_id] = str(curr_img_path)


        current_gt_pose = gt_poses[str(image_id)]
        R_vec = current_gt_pose[0]["cam_R_m2c"]
        R_matrix = np.array([R_vec[0:3], R_vec[3:6], R_vec[6:9]])
        t_vec = current_gt_pose[0]["cam_t_m2c"]
        t_matrix = np.array(t_vec[0:3])
        t_matrix = t_matrix/1000 #TODO: check metric for sbb_doors (is is meters, cm,mm)??

        pose = limap.base.CameraPose(R_matrix, t_matrix)


        image = cv2.imread(str(filename))


        cv2.imwrite(str(curr_img_path), image) #write to freshly created folder where both db and query imgs will live


        if i in sample:
            train_ids.append(image_id)
            camimage = limap.base.CameraImage(0, pose, image_name=str(curr_img_path))
            print(str(curr_img_path))
            camimages[image_id] = camimage

    cameras[0] = limap.base.Camera("PINHOLE", K, cam_id=0, hw=img_hw)

    gt = open(str(query2_data_folder_path/"scene_gt.json"))
    gt_poses = json.load(gt)

    cam_K_json_obj=open(str(query2_data_folder_path/"scene_camera.json")) 
    cam_K_json=json.load(cam_K_json_obj)
    K_vec=cam_K_json["0"]["cam_K"] #K is the same for all images
    K=np.array([K_vec[0:3],K_vec[3:6],K_vec[6:9]])

    num_tot_images = len([f for f in query2_imgs_path.iterdir()])

    random.seed(40)
    sample = random.sample(list(np.arange(0,num_tot_images,1)), num_query_imgs)

    query2_ids = []
    img_hw = []
    #go over all images
    for i, filename in enumerate(query2_imgs_path.iterdir()):
        # load groundtruth pose
        image_id = int(filename.stem)

        image_id = image_id + 1000

        curr_img_path = final_imgs_folder_path /"image{0:08d}.png".format(image_id)

        id_to_origin_name[image_id] = str(curr_img_path)


        current_gt_pose = gt_poses[str(image_id-1000)]
        R_vec = current_gt_pose[0]["cam_R_m2c"]
        R_matrix = np.array([R_vec[0:3], R_vec[3:6], R_vec[6:9]])
        t_vec = current_gt_pose[0]["cam_t_m2c"]
        t_matrix = np.array(t_vec[0:3])
        t_matrix = t_matrix/1000 #TODO: check metric for sbb_doors (is is meters, cm,mm)??

        pose = limap.base.CameraPose(R_matrix, t_matrix)

        #load image
        image = cv2.imread(str(filename))
        img_hw = [image.shape[0],image.shape[1]]

        cv2.imwrite(str(curr_img_path), image) #write to freshly created folder where both db and query imgs will live

        #collect images
        if i in sample:
 
            query2_ids.append(image_id)

            camimage = limap.base.CameraImage(1, pose, image_name=str(curr_img_path))
            print(str(curr_img_path))
            camimages[image_id] = camimage


    imagecols = limap.base.ImageCollection(cameras, camimages)


    neighbors = None
    ranges = None

    print("train_ids")
    print(train_ids)

    print("query_ids")

    print(query2_ids)
    print("read scene finished")

    return imagecols, neighbors, ranges, train_ids, query2_ids, id_to_origin_name, final_imgs_folder_path

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


    logger.info(f'Frawa: Save visualizations.')
    vis_output_path = Path(cfg["output_dir"]) / "visualization_lm/point_only"
    vis_output_path.mkdir(exist_ok=True, parents=True)

    saved_train_ids = train_ids[0::30]
    saved_query_ids = query_ids[0::5]
    for id in saved_train_ids: #+ saved_query_ids:
        selected = [id]

    selected = ['image{0:08d}.png'.format(id) for id in query_ids]
    visualization.visualize_loc(results_file, image_dir, ref_sfm, n=1, top_k_db=1, selected=selected, seed=2)
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

#reads out line segements from the 3d reconstruction     
def read_track_info(textfile):
    track_info = []
    with open(textfile, 'r') as file:
        num_tracks = int(file.readline().strip())

        
        for track_number in range(num_tracks):
            # Skip irrelevant rows
            for _ in range(3):
                file.readline()
            # Read image IDs and line IDs
            image_ids = [int(id_) for id_ in file.readline().strip().split()]
            line_ids = [int(id_) for id_ in file.readline().strip().split()]
            track_info.append((track_number, image_ids, line_ids))
            
            
            
            
    return track_info

#reads out descriptors from db images for each line segment in the 3d reconstruction
def filter_descinfo(descinfo_folder, image_id, line_id):
    # Load the descinfo file corresponding to the given image_id
    filename = f"descinfo_{image_id}.npz"
    descinfo = np.load(f"{descinfo_folder}/{filename}")
    
    # Get the indices of the lines that match the specified line_id
    line_indices = line_id
    
    x = descinfo['image_shape']
    y = descinfo['lines']
    w = descinfo['lines_score']
    z = descinfo['endpoints_desc']

    y = y[line_indices*2:line_indices*2+2]
    w = w[line_indices]
    z = z[:, line_indices*2:line_indices*2+2]



    
    return w, x, y, z

#calculates the average descriptors
def calculate_average(filtered_descinfo_folder, averagetrack_output, track_number):
    file_pattern = os.path.join(filtered_descinfo_folder, 'descinfo_*.npz')
    file_list = glob.glob(file_pattern)
    first_array = 0
    for file_path in file_list:

        descinfo = np.load(file_path)

        x = descinfo['image_shape']
        y = descinfo['lines']
        w = descinfo['lines_score']
        z = descinfo['endpoints_desc']


        num_lines = int(np.shape(w)[0])
    

        x_values = y[:, 0]  # First column contains x values
        y_values = y[:, 1]  # Second column contains y values


        x_values_reshaped = x_values.reshape(num_lines, 2)
        y_values_reshaped = y_values.reshape(num_lines, 2)

        # Calculate the mean along axis 0 for x and y values separately
        x1_avg = np.mean(x_values_reshaped[:, 0])
        y1_avg = np.mean(y_values_reshaped[:, 0])
        x2_avg = np.mean(x_values_reshaped[:, 1])
        y2_avg = np.mean(y_values_reshaped[:, 1])

        # Create a new array with the averaged values
        y = np.array([[x1_avg, y1_avg], [x2_avg, y2_avg]])

        # Reshape the array to (256, 2, 160/2)
        reshaped_data = z.reshape(256, 2, -1)

        # Calculate the mean along the last axis (which represents the pairs of values)
        z = np.mean(reshaped_data, axis=2)

        #lines_score and image_shape
        x = np.array([512, 512])             #np.mean(image_shape)
        w = np.array([np.mean(w)])      #np.mean(lines_score)

        if first_array == 0:
               

               image_shape = x
               lines = y
               lines_score = w
               endpoints_desc = z
               first_array = 1

        else:
               a = x
               b = y
               c = w
               d = z
               image_shape = np.array([512, 512])                           #image_shape = np.concatenate((image_shape, a), axis=0)
               lines = np.concatenate((lines, b), axis=0)
               lines_score = np.append(lines_score, c)
               endpoints_desc = np.concatenate((endpoints_desc, d), axis=1)



    output_filename = f"{averagetrack_output}/descinfo_direct_average.npz"
    np.savez(output_filename, image_shape = image_shape, lines = lines, lines_score = lines_score, endpoints_desc = endpoints_desc) 

#combines all averaged descriptors to one
def average(alltracks, descinfo_folder, filtered_descinfo_folder, averagetrack_output):

    track_info = read_track_info(alltracks)
    total_track_number = 0
    for track_number, image_ids, line_ids in track_info:
        print("Track Number:", track_number)
        first_array = 0
        total_track_number += 1

        for image_id, line_id in zip(image_ids, line_ids):
           w, x, y, z = filter_descinfo(descinfo_folder, image_id, line_id)

           if first_array == 0:
               
               image_shape = x
               lines = y
               lines_score = w
               endpoints_desc = z
               first_array = 1

           else:
               a = x
               b = y
               c = w
               d = z
               image_shape = np.array([512, 512])                           #image_shape = np.concatenate((image_shape, a), axis=0)
               lines = np.concatenate((lines, b), axis=0)
               lines_score = np.append(lines_score, c)
               endpoints_desc = np.concatenate((endpoints_desc, d), axis=1)

        
        print(image_shape.shape, lines.shape, lines_score.shape, endpoints_desc.shape)
        output_filename = f"{filtered_descinfo_folder}/descinfo_{track_number}.npz"
        np.savez(output_filename, image_shape = image_shape, lines = lines, lines_score = lines_score, endpoints_desc = endpoints_desc) 
    calculate_average(filtered_descinfo_folder, averagetrack_output, track_number)
    print("total_track_number ", total_track_number)
    return total_track_number

def line_localization(cfg, imagecols_db, imagecols_query, point_corresp, linemap_db, retrieval, results_path, img_name_dict=None, logger=None):
    """
    Run visual localization on query images with `imagecols`, it takes 2D-3D point correspondences from HLoc;
    runs line matching using 2D line matcher ("epipolar" for Gao et al. "Pose Refinement with Joint Optimization of Visual Points and Lines");
    calls :func:`~limap.estimators.absolute_pose.pl_estimate_absolute_pose` to estimate the absolute camera pose for all query images,
    and writes results in results file in `results_path`.

    Args:
        cfg (dict): Configuration, fields refer to :file:`cfgs/localization/default.yaml`
        imagecols_db (:class:`limap.base.ImageCollection`): Image collection of database images, with triangulated camera poses
        imagecols_query (:class:`limap.base.ImageCollection`): Image collection of query images, camera poses only used for epipolar matcher/filter as coarse poses, can be left uninitialized otherwise
        linemap_db (list[:class:`limap.base.LineTrack`]): LIMAP triangulated/fitted line tracks
        retrieval (dict): Mapping of query image file path to list of neighbor image file paths, e.g. returned from :func:`hloc.utils.parsers.parse_retrieval`
        results_path (str | Path): File path to write the localization results
        img_name_dict(dict, optional): Mapping of query image IDs to the image file path, by default the image names from `imagecols`
        logger (:class:`logging.Logger`, optional): Logger to print logs for information

    Returns:
        Dict[int -> :class:`limap.base.CameraPose`]: Mapping of query image IDs to the localized camera poses for all query images.
    """ 

    if cfg['localization']['2d_matcher'] not in ['epipolar', 'sold2', 'superglue_endpoints', 'gluestick', 'linetr', 'lbd', 'l2d2']:
        raise ValueError("Unknown 2d line matcher: {}".format(cfg['localization']['2d_matcher']))

    train_ids = imagecols_db.get_img_ids()
    query_ids = imagecols_query.get_img_ids()

    if img_name_dict is None:
        img_name_dict = {img_id: imagecols_db.image_name(img_id) for img_id in train_ids}
        img_name_dict.update({img_id: imagecols_query.image_name(img_id) for img_id in query_ids})
    id_to_name = img_name_dict
    name_to_id = {img_name_dict[img_id]: img_id for img_id in train_ids + query_ids}


    poses_db = {img_id: imagecols_db.camimage(img_id).pose for img_id in train_ids}

    all_query_segs, _ = _runners.compute_2d_segs(cfg, imagecols_query, compute_descinfo=False)

    all_query_lines = _base.get_all_lines_2d(all_query_segs)


    # Do matches for query images and retrieved neighbors for superglue endpoints matcher
    if cfg['localization']['2d_matcher'] != 'epipolar':
        weight_path = None if "weight_path" not in cfg else cfg["weight_path"]
        if cfg['localization']['2d_matcher'] == 'superglue_endpoints':
            extractor_name = 'superpoint_endpoints'
            matcher_name = 'superglue_endpoints'
        else:
            extractor_name = matcher_name = cfg['localization']['2d_matcher']
        #when topk:0 then it does mutual nearest neighbor matching
        ex_cfg = {"method": extractor_name, "topk": 0, "n_jobs": cfg["n_jobs"]}
        ma_cfg = {"method": matcher_name, "topk": 0, "n_jobs": cfg["n_jobs"], "superglue": {"weights": cfg["line2d"]["matcher"]["superglue"]["weights"]}}
        basedir = os.path.join("line_detections", cfg["line2d"]["detector"]["method"])
        folder_save = os.path.join(cfg["dir_save"], basedir)
        extractor = limap.line2d.get_extractor(ex_cfg, weight_path=weight_path)  
        averagetrack_output = Path(cfg["output_dir"] + "/averagetrack")
        descinfo = np.load(f"{averagetrack_output}/descinfo_direct_average.npz")
        descinfo_folder = extractor.get_descinfo_folder(folder_save)
        extractor.save_descinfo(descinfo_folder, "direct_3d", descinfo)

        se_descinfo_dir = extractor.extract_all_images(folder_save, imagecols_query, all_query_segs, skip_exists=True)
        basedir = os.path.join("line_matchings", cfg["line2d"]["detector"]["method"], "feats_{0}".format(matcher_name))
        matcher = limap.line2d.get_matcher(ma_cfg, extractor, n_neighbors=cfg['n_neighbors_loc'], weight_path=weight_path)
        folder_save = os.path.join(cfg["dir_save"], basedir)
        retrieved_neighbors = {qid: ["direct_3d"] for qid in query_ids}
        print("retrieved neighbors")
        print(retrieved_neighbors)
        se_matches_dir = matcher.match_all_neighbors(folder_save, query_ids, retrieved_neighbors, se_descinfo_dir, skip_exists=False)

    # Localization
    print("[LOG] Starting localization with points+lines...")
    final_poses = {}
    pose_dir = results_path.parent / 'poses_{}'.format(cfg['localization']['2d_matcher'])
    for qid in tqdm(query_ids):
        if cfg['localization']['skip_exists']:
            limapio.check_makedirs(str(pose_dir))
            if os.path.exists(os.path.join(pose_dir, f'{qid}.txt')):
                with open(os.path.join(pose_dir, f'{qid}.txt'), 'r') as f:
                    data = f.read().rstrip().split('\n')[0].split()
                    q, t = np.split(np.array(data[1:], float), [4])
                    final_poses[qid] = _base.CameraPose(q, t)
                    continue
        if logger:
            logger.info(f"Query Image ID: {qid}")

        query_lines = all_query_lines[qid] #query_lines is list[:class:`~limap.base.Line2d`]
        qname = id_to_name[qid]
        query_pose = imagecols_query.get_camera_pose(qid)
        query_cam = imagecols_query.cam(imagecols_query.camimage(qid).cam_id)
        query_camview = _base.CameraView(query_cam, query_pose)


        if cfg['localization']['2d_matcher'] != 'epipolar':
            # Read from the pre-computed matches
            all_line_pairs_2to2 = limapio.read_npy(os.path.join(se_matches_dir, "matches_{0}.npy".format(qid))).item()


        all_line_pairs_2to3 = defaultdict(list)
        for pair in all_line_pairs_2to2["direct_3d"]: #we need this because 1 query line can have mathces to several 3d lines/tracks
            query_line_id, track_id = pair
            all_line_pairs_2to3[query_line_id].append(track_id)

        print("Frawa: all_line_pairs_2to3")
        print(all_line_pairs_2to3)

        # filter based on reprojection distance (to 1-1 correspondences), mainly for "OPPO method"
        if cfg['localization']['reprojection_filter'] is not None:
            line_matches_2to3 = reprojection_filter_matches_2to3(
                query_lines, query_camview, all_line_pairs_2to3, linemap_db, dist_thres=2,
                dist_func=get_reprojection_dist_func(cfg['localization']['reprojection_filter']))
        else:
            line_matches_2to3 = [(x, y) for x in all_line_pairs_2to3 for y in all_line_pairs_2to3[x]]

        num_matches_line = len(line_matches_2to3)
        if logger:
            logger.info(f'{num_matches_line} line matches found for {len(query_lines)} 2D lines')

        l3ds = [track.line for track in linemap_db]
        l2ds = [query_lines[pair[0]] for pair in line_matches_2to3]
        l3d_ids = [pair[1] for pair in line_matches_2to3]
        
        print("annika: saving l2ds, l3ds etc for "+str(qid))
        array_dict = {'l2ds': l2ds, 'l3ds': l3ds, 'l3d_ids': l3d_ids}
    
        # Serialize the dictionary using pickle
        with open(results_path.parent / ('line_corrs_'+str(qid)+'.pkl'), 'wb') as f:
            pickle.dump(array_dict, f)

        p3ds, p2ds = point_corresp[qid]['p3ds'], point_corresp[qid]['p2ds']
        inliers_point = point_corresp[qid].get('inliers') # default None
        final_pose, ransac_stats = _estimators.pl_estimate_absolute_pose(
                cfg['localization'], l3ds, l3d_ids, l2ds, p3ds, p2ds, query_cam, query_pose, # query_pose not used for ransac methods
                inliers_point=inliers_point, silent=True, logger=logger)

        if cfg['localization']['skip_exists']:
            with open(os.path.join(pose_dir, f'{qid}.txt'), 'w') as f:
                name = id_to_name[qid]
                fq, ft = final_pose.qvec, final_pose.tvec
                line = ' '.join([name] + [str(x) for x in fq] + [str(x) for x in ft]) + '\n'
                f.writelines([line])

        final_poses[qid] = final_pose

    lines = []
    for qid in query_ids:
        name = id_to_name[qid]
        fpose = final_poses[qid]
        fq, ft = fpose.qvec, fpose.tvec
        line = ' '.join([name] + [str(x) for x in fq] + [str(x) for x in ft]) + '\n'
        lines.append(line)

    # write results
    with open(results_path, 'w') as f:
        f.writelines(lines)
    
    

    return final_poses


def main():
    cfg, args = parse_config()
    cfg = _runners.setup(cfg)

    scene_id = os.path.basename(query2_data_folder_path)     #query_data_folder_path

    # outputs is for localization-related results
    outputs = Path(cfg['output_dir']) / 'localization'
    outputs.mkdir(exist_ok=True, parents=True)

    logger.info(f'Working on scene "{scene_id}".')

    imagecols, neighbors, ranges, train_ids, query_ids, id_to_origin_name, final_imgs_folder_path = read_scene_lm(cfg)

    poses_gt = {img_id: imagecols.camimage(img_id).pose for img_id in imagecols.get_img_ids()}

    if args.eval is not None:
        eval(args.eval, poses_gt, query_ids, id_to_origin_name, logger)
        return


    image_dir = final_imgs_folder_path

    imagecols_train = imagecols.subset_by_image_ids(train_ids)


    results_point, results_joint = get_result_filenames(cfg['localization'], args)
    results_point, results_joint = outputs / results_point, outputs / results_joint

    img_name_to_id = {"image{0:08d}.png".format(id): id for id in (train_ids + query_ids)}


    ##########################################################
    # [A] hloc point-based localization
    ##########################################################
    logger.info("Run hloc")
    ref_sfm, poses, hloc_log_file = run_hloc_lm(
        cfg, image_dir, imagecols, neighbors, train_ids, query_ids, id_to_origin_name,
        results_point, args.num_loc, logger
    )
    logger.info("Evaluation for hloc will be done next, but evaluation not meaningful because wrong ground truth")
    logger.info("POSES")
    logger.info("POSES finished")
    logger.info("results_point")
    logger.info(str(results_point))
    logger.info("results_point finished")

    eval(results_point, poses_gt, query_ids, id_to_origin_name, logger)
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
    # [C] Average descriptors
    ##########################################################
    alltracks =  Path(cfg["output_dir"] + "/alltracks.txt")
    descinfo_folder = Path(cfg["output_dir"] + "/line_detections/lsd/descinfos/superpoint_endpoints/")
    filtered_descinfo_folder = Path(cfg["output_dir"] + "/filtered_descinfo")       ##final_imgs_folder_path.mkdir(exist_ok=True, parents=True)
    averagetrack_output = Path(cfg["output_dir"] + "/averagetrack")
    #averagetrack_output = Path(cfg["output_dir"] + "/filtered_descinfo")
    if not os.path.exists(filtered_descinfo_folder):
        os.makedirs(filtered_descinfo_folder)
    if not os.path.exists(averagetrack_output):
        os.makedirs(averagetrack_output)

    total_track_number = average(alltracks, descinfo_folder, filtered_descinfo_folder, averagetrack_output)

    ##########################################################
    # [D] Localization with points and lines
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

    #append points to end of copied line file to get joint obj
    with open(joint_obj, 'a') as file:
        file.write(formatted_points_string)
    # write points to points only file
    with open(points_only_obj, 'w') as file:
        file.write(formatted_points_string)
    

    final_poses = line_localization(cfg, imagecols_train, imagecols_query, point_correspondences, linetracks_db, retrieval, results_joint, img_name_dict=id_to_origin_name)

    #save joint image visualizations
    logger.info("save joint image visualizations")
    vis_output_path = Path(cfg["output_dir"]) / "visualization_lm/joint"
    vis_output_path.mkdir(exist_ok=True, parents=True)

    # get 2d line segments for all images
    basedir = os.path.join("line_detections", cfg["line2d"]["detector"]["method"])
    folder_load = os.path.join(cfg["dir_load"], basedir)
    all_2d_segs = limapio.read_all_segments_from_folder(os.path.join(folder_load, "segments"))
    all_2d_segs = {id: all_2d_segs[id] for id in imagecols.get_img_ids()}
    saved_train_ids = train_ids[0::30]

    for id in saved_train_ids: #+ saved_query_ids:
        selected = [id]

        visualize_sfm_2d(ref_sfm, image_dir, color_by="all_same", selected=selected, n=5)
        curr_2d_segs = all_2d_segs[id]
        curr_2d_segs = curr_2d_segs.reshape(-1, 2, 2)
        plot_lines([curr_2d_segs])
        save_plot(str(vis_output_path / f"joint_reconstruction_viz_visibility_{id}.png"), dpi=1200)
        matplotlib.pyplot.close()
    
    logger.info("Evaluation for limap will be done next, but evaluation not meaningful because wrong ground truth")
    logger.info("POSES")
    logger.info(str(final_poses))
    logger.info("POSES finished")
    logger.info("results_joint")
    logger.info(str(results_joint))
    logger.info("results_joint finished")

    # Evaluate
    eval(results_joint, poses_gt, query_ids, id_to_origin_name, logger)

    for vis_id in range(0,50):
        img_path=str(image_dir / hloc_name_dict[query_ids[vis_id]])
        img_point= cv2.imread(img_path)
        img_joint= cv2.imread(img_path)

        #compute arguments like in https://github.com/cvg/limap/blob/main/limap/runners/line_localization.py
        data = np.load(cfg["output_dir"]+'/localization/line_corrs_'+str(query_ids[vis_id])+'.pkl', allow_pickle=True)
        l3ds = data['l3ds']
        l2ds = data['l2ds']
        l3d_ids = data['l3d_ids']
        camview_joint = _base.CameraView(imagecols.get_cameras()[0], final_poses[query_ids[vis_id]])

        camview_point = _base.CameraView(imagecols.get_cameras()[0], poses[str(image_dir / hloc_name_dict[query_ids[vis_id]])])
        #reproject lines
        for l2d, l3d_id in zip(l2ds, l3d_ids):
            l3d = l3ds[l3d_id]
            img_joint = cv2.line(img_joint, l2d.start.astype(int), l2d.end.astype(int), color=[255, 0, 0])
            img_point = cv2.line(img_point, l2d.start.astype(int), l2d.end.astype(int), color=[255, 0, 0])
            l2d_proj = l3d.projection(camview_joint)
            img_joint = cv2.line(img_joint, l2d_proj.start.astype(int), l2d_proj.end.astype(int), color=[0, 0, 255])
            l2d_proj_pt = l3d.projection(camview_point)
            img_point = cv2.line(img_point, l2d_proj_pt.start.astype(int), l2d_proj_pt.end.astype(int), color=[0, 0, 255])

        #reproject points
        p_corr=point_correspondences[query_ids[vis_id]]
        p2ds=p_corr['p2ds']
        p3ds=p_corr['p3ds']
        for p2d, p3d in zip(p2ds, p3ds):
            img_joint = cv2.circle(img_joint, p2d.astype(int), radius=1, color=[255, 0, 0])
            img_point = cv2.circle(img_point, p2d.astype(int), radius=1, color=[255, 0, 0])
            img_joint = cv2.circle(img_joint, camview_joint.projection(p3d).astype(int), radius=1, color=[0, 255, 0])
            img_point = cv2.circle(img_point, camview_point.projection(p3d).astype(int), radius=1, color=[0, 255, 0])

        #save image
        cv2.imwrite((cfg["output_dir"]+"/"+str(query_ids[vis_id])+"_joint.png"), img_joint)
        cv2.imwrite((cfg["output_dir"]+"/"+str(query_ids[vis_id])+"_point.png"), img_point)

if __name__ == '__main__':
    main()
