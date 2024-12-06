import os
import numpy as np
import pycolmap
import pickle
import limap.estimators as _estimators
import limap.runners as _runners
import limap.base as _base
import limap.util.io as limapio
import limap.line2d

from tqdm import tqdm
from collections import defaultdict
from hloc.utils.io import get_keypoints, get_matches
from limap.optimize.line_localization.functions import *
from limap.line2d.SOLD2.misc.geometry_utils import keypoints_to_grid
import torch
import torch.nn.functional as F
import yaml

#This script is adapted from /limap/limap/runners/line_localization.py, but so that instead of doing 2d-2d matching by which we can
#afterwards do indirect 2d-3d matching, we have code that does our direct 2d-3d matching approach that we call sold3dSP.

def get_hloc_keypoints(ref_sfm, features_path, matches_path, query_img_name, target_img_ids, logger=None):
    if ref_sfm is None or features_path is None or matches_path is None:
        if logger:
            logger.debug('Not retrieving keypoint correspondences because at least one parameter is not provided.')
        return np.array([]), np.array([])

    kpq = get_keypoints(features_path, query_img_name)
    kpq += 0.5  # COLMAP coordinates

    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches_point = 0

    for i, tgt_id in enumerate(target_img_ids):
        image = ref_sfm.images[tgt_id]
        if image.num_points3D() == 0:
            if logger:
                logger.debug(f'No 3D points found for {image.name}.')
                continue
        points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                for p in image.points2D])

        matches, _ = get_matches(matches_path, query_img_name, image.name)
        matches = matches[points3D_ids[matches[:, 1]] != -1]
        num_matches_point += len(matches)
        for idx, m in matches:
            id_3D = points3D_ids[m]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    p2ds = np.array([kpq[i] for i in idxs for j in kp_idx_to_3D[i]])
    p3ds = np.array([ref_sfm.points3D[j].xyz for i in idxs for j in kp_idx_to_3D[i]])

    return p2ds, p3ds

def get_hloc_keypoints_from_log(logs, query_img_name, ref_sfm=None, resize_scales=None):
    if ref_sfm is None: # inloc
        p2ds = logs['loc'][query_img_name]['keypoints_query']
        p3ds = logs['loc'][query_img_name]['3d_points']
    else:
        p2ds = logs['loc'][query_img_name]['keypoints_query']
        p3d_ids = logs['loc'][query_img_name]['points3D_ids']
        p3ds = [ref_sfm.points3D[j].xyz for j in p3d_ids]
    inliers = logs['loc'][query_img_name]['PnP_ret']['inliers']

    p2ds, p3ds = np.array(p2ds), np.array(p3ds)
    if resize_scales is not None and query_img_name in resize_scales:
        scale = resize_scales[query_img_name]
        p2ds = (p2ds + .5) * scale - .5

    return p2ds, p3ds, inliers


#very similar to limap/line2d/SOLD2/model/line_matching.sample_line_points(..) but sampling along 3d line instead of 2d line
def sample_3dline_points(line_seg, num_samples=5, min_dist_pts=0.01): #sample points on lines are at least 10 cm apart
    """
    Regularly sample points along each line segments, with a minimal
    distance between each point. Pad the remaining points.
    Inputs:
        line_seg: an Nx2x3 torch.Tensor.
    Outputs:
        line_points: an Nxnum_samplesx3 np.array.
        valid_points: a boolean Nxnum_samples np.array.
    """
    num_lines = len(line_seg)
    line_lengths = np.linalg.norm(line_seg[:, 0] - line_seg[:, 1], axis=1)

    # Sample the points separated by at least min_dist_pts along each line
    # The number of samples depends on the length of the line
    num_samples_lst = np.clip(line_lengths // min_dist_pts,
                                2, num_samples)
    line_points = np.empty((num_lines, num_samples, 3), dtype=float)
    valid_points = np.empty((num_lines, num_samples), dtype=bool)
    for n in np.arange(2, num_samples + 1):
        cur_mask = num_samples_lst == n
        cur_line_seg = line_seg[cur_mask]
        line_points_x = np.linspace(cur_line_seg[:, 0, 0],
                                    cur_line_seg[:, 1, 0],
                                    n, axis=-1)
        line_points_y = np.linspace(cur_line_seg[:, 0, 1],
                                    cur_line_seg[:, 1, 1],
                                    n, axis=-1)
        line_points_z = np.linspace(cur_line_seg[:, 0, 2],
                                    cur_line_seg[:, 1, 2],
                                    n, axis=-1)
        cur_line_points = np.stack([line_points_x, line_points_y, line_points_z], axis=-1)

        # Pad
        cur_num_lines = len(cur_line_seg)
        cur_valid_points = np.ones((cur_num_lines, num_samples),
                                    dtype=bool)
        cur_valid_points[:, n:] = False
        cur_line_points = np.concatenate([
            cur_line_points,
            np.zeros((cur_num_lines, num_samples - n, 3), dtype=float)],
            axis=1)

        line_points[cur_mask] = cur_line_points
        valid_points[cur_mask] = cur_valid_points

    return line_points, valid_points


#Compared to sold3d, we now also need custom code to extract the 2d line descriptors because eventhough we still describe lines
# by sampling points along the line as in sold2 (sold3d), we use superpoints descripors for the points instead of sold2 descriptors
def extract_all_images_sp_on_line(extractor_sold2, extractor_sp, output_folder, imagecols, all_2d_segs, skip_exists=True):
    """
    Perform line descriptor extraction on all images and save the descriptors.
    A 2d line descriptor consists of the sampled super point descriptors of points sampled along a 2d line.

    Args:
        output_folder (str): The output folder.
        imagecols (:class:`limap.base.ImageCollection`): The input image collection
        all_2d_segs (dict[int -> :class:`np.array`]): The line detection for each image indexed by the image id. Each segment is with shape (N, 5). Each row corresponds to x1, y1, x2, y2 and score. Computed from `detect_all_images`
        skip_exists (bool): Whether to skip already processed images.
    Returns:
        descinfo_folder (str): The path to the saved descriptors.
    """
    descinfo_folder = extractor_sold2.get_descinfo_folder(output_folder)
    if not skip_exists:
        limapio.delete_folder(descinfo_folder)
    limapio.check_makedirs(descinfo_folder)
    # per image
    for img_id in tqdm(imagecols.get_img_ids()):
        if skip_exists and os.path.exists(extractor_sold2.get_descinfo_fname(descinfo_folder, img_id)):
            continue
        camview = imagecols.camview(img_id)
        segs = all_2d_segs[img_id] #all detected line segments in image
        img = camview.read_image(set_gray=True)

        if len(segs) == 0:
            descinfo = []
        else:
            segs = extractor_sold2.detector.segstosold2segs(segs[:, :4]) #superpoint only needs x,y of start and end point of lin
            # Sample points regularly along each line in regular mode like in sold2
            segs, valid_points = extractor_sold2.detector.line_matcher.line_matcher.sample_line_points(segs)
            torch_img = {'image': torch.tensor(img.astype(np.float32) / 255,
                                dtype=torch.float,
                                device=extractor_sp.device)[None, None]}
            segs = torch.tensor(segs.reshape(1, -1, 2),
                                        dtype=torch.float, device=extractor_sp.device)
            with torch.no_grad():
                #compute the superpoint descriptors per sampled point on 2d line segment
                desc_sampled = extractor_sp.sp.sample_descriptors(torch_img, segs)['descriptors'][0].cpu().numpy()

            #line_points contains the sampled points along the line segments
            #valid_points tells us if the corresponding points are valid or if less than num_samples were sampled along line, these 
            # are marked als False (i.e. invalid)
            descinfo = [desc_sampled, valid_points]
        extractor_sold2.save_descinfo(descinfo_folder, img_id, descinfo)
    return descinfo_folder


def line_localization(cfg, imagecols_db, imagecols_query, point_corresp, linemap_db, retrieval, results_path, img_name_dict=None, logger=None):
    """
    Run visual localization on query images with `imagecols`, it takes 2D-3D point correspondences from HLoc;
    runs line matching using our custom direct 2d-3d line matching sold3dSP
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

     # GT for queries
    poses_db = {img_id: imagecols_db.camimage(img_id).pose for img_id in train_ids}

    # line detection of query images, fetch detection of db images (generally already be detected during triangulation)
    all_db_segs, _ = _runners.compute_2d_segs(cfg, imagecols_db, compute_descinfo=False)
    all_query_segs, _ = _runners.compute_2d_segs(cfg, imagecols_query, compute_descinfo=False)
    #get_all_lines_2d converts :class:`np.array` representations of 2D line segments to dict of :class:`~limap.base.Line2d`.
    all_db_lines = _base.get_all_lines_2d(all_db_segs)
    all_query_lines = _base.get_all_lines_2d(all_query_segs)
    line2track = _base.get_invert_idmap_from_linetracks(all_db_lines, linemap_db)

    #TODO: below in samples_per_l3d, you can choose the number of points sampled along a 3d line as we have done in our ablation experiment.
    samples_per_l3d = 5 #here we can choose how many points should be sampled along 3d line of 3d model/reconstruction
    min_dist_pts = 8 #here we can choose the min_dist (i.e. the minimal distance between sampled points) for sampling along 2d line for query images
    # Overwrite num_samples:5 with num_samples: samples_per_l3d in export_line_features.yaml and same for min_dist_pts
    limap_limap_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sold2_cfg_path = os.path.join(limap_limap_path, 'line2d/SOLD2/config/export_line_features.yaml')
    with open(sold2_cfg_path, 'r') as file:
        config = yaml.safe_load(file)
    # Modify the num_samples and min_dist_pts value
    config['line_matcher_cfg']['num_samples'] = samples_per_l3d
    config['line_matcher_cfg']['min_dist_pts'] = min_dist_pts
    # Save the modified YAML file
    with open(sold2_cfg_path, 'w') as file:
        yaml.safe_dump(config, file)


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

        #we want to use superpoint descriptors now and can therefore reuse part of the implementation from superpoint_endpoint to extract
        # superpoint descriptors for 2d points
        ex_cfg_sp = {"method": "superpoint_endpoints", "topk": 0, "n_jobs": cfg["n_jobs"]}
        extractor_sp = limap.line2d.get_extractor(ex_cfg_sp, weight_path=weight_path)

        if True:
            print("Our custom 2d-3d line matching called sold3dSP starts")
            n_tracks = len(linemap_db)
            #to store computed 3d descriptors for the sampled points along the 3d line
            l3d_descriptors_array = np.empty((256, 0)) #changed from 128 to 256 because superpoint descriptor has dim 256
            #Depending on 3d line length, we sample less than samples_per_l3d. l3d_valid_points_list marks these additional, 
            # padded points as invalid.
            l3d_valid_points_list = []
            #per track, i.e. per 3d line, we compute the 3d line descriptor based on our own approach named sold3dSP
            for track_id in tqdm(range(n_tracks)):
                track = linemap_db[track_id]
                l3d = track.line
                l3d_line_seg = np.array([[l3d.start, l3d.end]])
                #sample 3d points along 3d line
                l3d_line_points, l3d_valid_points = sample_3dline_points(l3d_line_seg, num_samples=samples_per_l3d)
                #l3d_line_points: an 1xnum_samplesx3 np.array.
                #l3d_valid_points: a boolean 1xnum_samples np.array.
                collection_descriptors_2d = [] # we get the 2d descriptors of the sampled 3d points when reprojected to the corresponding images
                for idx in range(track.count_lines()): #go over all images that contain lines that match to the 3d line
                    img_id = track.image_id_list[idx]
                    camview2 = imagecols_db.camview(img_id)
                    img = camview2.read_image(set_gray=True)

                    l2d_line_points = np.zeros((l3d_line_points.shape[0],l3d_line_points.shape[1], 2))
                    #we go over the sampled 3d point along the 3d line
                    for sample_id in range(l3d_valid_points.shape[1]):
                        if l3d_valid_points[0, sample_id]:
                            #we reproject the sampled 3d points back to the 2d image to get 2d point
                            proj_p2d = camview2.projection(l3d_line_points[0,sample_id])
                            l2d_line_points[0, sample_id] = proj_p2d
                    
                    #Extract the superpoint descriptors for each 2d point
                    #inspired by limap/line2d/endpoints/extractor
                    if len(l2d_line_points) == 0:
                        desc_sampled = np.zeros((256, 0))
                    else: 
                        torch_img = {'image': torch.tensor(img.astype(np.float32) / 255,
                                            dtype=torch.float,
                                            device=extractor_sp.device)[None, None]}
                        l2d_line_points = torch.tensor(l2d_line_points.reshape(1, -1, 2),
                                                    dtype=torch.float, device=extractor_sp.device)
                        with torch.no_grad():
                            desc_sampled = extractor_sp.sp.sample_descriptors(torch_img, l2d_line_points)['descriptors'][0].cpu().numpy()
                    #desc_sampled: an 256xnum_samples np.array.
                    collection_descriptors_2d.append(desc_sampled)
                
                #average features per sampled point
                collection_descriptors_2d = np.array(collection_descriptors_2d) #shape track.count_lines()x156xnum_samples
                l3d_descriptor = np.average(collection_descriptors_2d, axis=0) #shape 256xnum_samples
                l3d_descriptors_array = np.hstack((l3d_descriptors_array, l3d_descriptor)) #contains our 3d line descriptors
                l3d_valid_points_list.append(l3d_valid_points)

            l3d_valid_points_array = np.array(l3d_valid_points_list)
            l3d_valid_points_array = l3d_valid_points_array.squeeze(axis=1)
            descinfo = [l3d_descriptors_array, l3d_valid_points_array]
            descinfo_folder = extractor.get_descinfo_folder(folder_save)
            extractor.save_descinfo(descinfo_folder, "direct_3d", descinfo) #we save the 3d line descriptors

    	#2d descriptors for query images
        #inspired by extract_all_images but with superpoint descriptors instead of sold2 descriptors
        #we must set skip_exists=True below, otherwise the folder containing the direct_3d descinfo is deleted!
        se_descinfo_dir = extract_all_images_sp_on_line(extractor, extractor_sp, folder_save, imagecols_query, all_query_segs, skip_exists=True)
        basedir = os.path.join("line_matchings", cfg["line2d"]["detector"]["method"], "feats_{0}".format(matcher_name))
        matcher = limap.line2d.get_matcher(ma_cfg, extractor, n_neighbors=cfg['n_neighbors_loc'], weight_path=weight_path)
        folder_save = os.path.join(cfg["dir_save"], basedir)
        # we want to compare and the query image desriptors only to the direct 3d descriptors
        retrieved_neighbors = {qid: ["direct_3d"] for qid in query_ids}
        #we compute and save the direct 2d-3d matches
        se_matches_dir = matcher.match_all_neighbors(folder_save, query_ids, retrieved_neighbors, se_descinfo_dir, skip_exists=cfg['skip_exists'])

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
            #these are now the 2d 3d matches already
            all_line_pairs_2to2 = limapio.read_npy(os.path.join(se_matches_dir, "matches_{0}.npy".format(qid))).item()


        all_line_pairs_2to3 = defaultdict(list)
        for pair in all_line_pairs_2to2["direct_3d"]: #we need this because 1 query line can have mathces to several 3d lines/tracks
            query_line_id, track_id = pair
            all_line_pairs_2to3[query_line_id].append(track_id)
        

        # filter based on reprojection distance (to 1-1 correspondences), mainly for "OPPO method"
        if cfg['localization']['reprojection_filter'] is not None:
            line_matches_2to3 = reprojection_filter_matches_2to3(
                query_lines, query_camview, all_line_pairs_2to3, linemap_db, dist_thres=2,
                dist_func=get_reprojection_dist_func(cfg['localization']['reprojection_filter']))
        else: #we go into this option
            line_matches_2to3 = [(x, y) for x in all_line_pairs_2to3 for y in all_line_pairs_2to3[x]]

        num_matches_line = len(line_matches_2to3)
        if logger:
            logger.info(f'{num_matches_line} line matches found for {len(query_lines)} 2D lines')

        l3ds = [track.line for track in linemap_db]
        l2ds = [query_lines[pair[0]] for pair in line_matches_2to3]
        l3d_ids = [pair[1] for pair in line_matches_2to3]
        
        #saving 2d lines, 3d lines and 3d line ids to access later
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

    print("finished query pose computation")
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

    # write back num_samples:5 and min_dist_pts: 8 just so that we are consistent
    limap_limap_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sold2_cfg_path = os.path.join(limap_limap_path, 'line2d/SOLD2/config/export_line_features.yaml')
    print(sold2_cfg_path)
    with open(sold2_cfg_path, 'r') as file:
        config = yaml.safe_load(file)
    # Modify the num_samples and min_dist_pts value
    config['line_matcher_cfg']['num_samples'] = 5
    config['line_matcher_cfg']['min_dist_pts'] = 8
    # Save the modified YAML file
    with open(sold2_cfg_path, 'w') as file:
        yaml.safe_dump(config, file)
    

    return final_poses
