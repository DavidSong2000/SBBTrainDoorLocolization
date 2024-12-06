# Improving object pose estimation with line features

This repository contains the code for the 3D vision project. More specifically it contains pipelines to perform hybrid (point and lines) object pose estimation both with indirect and direct 2d-3d line matching on the LINEMOD dataset and the SBB doors dataset.

### Main contributions:
- Adapt localization to hybrid (points + lines) object pose estimation and evaluate and benchmark on various low-textured datasets
- Propose way to do direct 2d-3d line matching and evaluate
suitability for the task

## Installation

You can build your own docker image from the Dockerfile provided in this repo. We used the [Dockerfile from LIMAP](https://github.com/cvg/limap/blob/main/docker/README.md) and made some smaller changes to Dockerfile for it to work.

Our code builds on [LIMAP](https://github.com/cvg/limap). In this repo we only store the changed files from limap. Therefore, after cloning this repo, you need to add the LIMAP repo as a submodule.

In this repo, the datasets are not included, because they are too big. The datasets can be downloaded as described below.

## Usage

We provide examples of how to run object pose estimation on the LINEMOD (LM) dataset which you can download [here](https://bop.felk.cvut.cz/datasets/). We used the LINEMOD dataset for benchmarking.

To split the dataset into database and query images, we follow the same approach as in previous methods like Onepose++ and Onepose. The information to split in can be downloaded from [here](https://liuyuan-pal.github.io/Gen6D/). Copy the training_range.txt from LINEMOD/OBJECT/ to the corresponding object in lm/test/object/ from the previously downloaded dataset.

As you run the script in a docker container, you need to bind files that you want to use apart from the file you execute.

### Indirect 2d-3d line matching
Search for TODOs in localization_linemod.py to adapt data path and LINEMOD object.

Example command that will give you visualizations and results of object pose estimation for the points-only approach and hybrid approach with indirect 2d-3d line matching:
```bash
apptainer run --nv --bind /your_path_to_this_repo/3d_lines/line_localization_original_with_save.py:/limap/limap/runners/line_localization.py /your/path/to/dockerimage.sif /bin/bash -c 'cd /limap && python /your_path_to_this_repo/3d_lines/localization_linemod.py --output_dir /your/output/path'
```


### Direct 2d-3d line matching
If we want to run direct 2d-3d line matching (i.e. 3nnds, sold3d or sold3dSP approach), we need to bind the appropriate file.

The example commands below will give you visualizations and results of object pose estimation for the respective indirect 2d-3d line matching approach:

#### 3nnds
```bash
apptainer run --nv --bind /your_path_to_this_repo/3d_lines/limap/limap/point2d/superglue/weights:/limap/limap/point2d/superglue/weights /your/path/to/dockerimage.sif /bin/bash -c 'cd /limap && python /your_path_to_this_repo/3d_lines/localization_linemod_3nnds.py --output_dir /your/output/path'
```

#### sold3d
```bash
apptainer run --nv --bind /your_path_to_this_repo/line_localization_sold3d.py:/limap/limap/runners/line_localization.py --bind /your_path_to_this_repo/3d_lines/limap/limap/line2d/SOLD2/config/export_line_features.yaml:/limap/limap/line2d/SOLD2/config/export_line_features.yaml /your/path/to/dockerimage.sif /bin/bash -c 'cd /limap && python /your_path_to_this_repo/3d_lines/localization_linemod.py --output_dir /your/output/path -- config_file /your_path_to_this_repo/3d_lines/cfg_sold2_desc.yaml'
```

#### sold3d
```bash
apptainer run --nv --bind /your_path_to_this_repo/line_localization_sold3dSP.py:/limap/limap/runners/line_localization.py --bind /your_path_to_this_repo/3d_lines/limap/limap/line2d/SOLD2/config/export_line_features.yaml:/limap/limap/line2d/SOLD2/config/export_line_features.yaml /your/path/to/dockerimage.sif /bin/bash -c 'cd /limap && python /your_path_to_this_repo/3d_lines/localization_linemod.py --output_dir /your/output/path -- config_file /your_path_to_this_repo/3d_lines/cfg_sold2_desc.yaml'
```

#### Ablation experiments
For our ablation experiments on sold3d and sold3dSP, we have chosen different numbers of points sampled along the 3d line. You can change that parameter by searching for the TODO comments in `line_localization_sold3d.py` and `line_localization_sold3dSP.py`.

### Running on SBB doors dataset
We also run experiments on the private SBB doors dataset as the initial idea of the project was to also look at the performance on that dataset. 

The rendered images were generated using this [repo](https://github.com/jiaqchen/Monocluar-Pose-Estimation-Pipeline-for-Spot).

#### Running on SBB doors
You can reuse the commands from above, but replace `localization_linemod.py` with `localization_sbb_doors_rendered_only.py` or `localization_sbb_doors_rendered_real.py`.

`localization_sbb_doors_rendered_only.py` reads in rendered SBB door images as database images and query images.

`localization_sbb_doors_rendered_real.py` has rendered SBB door images as database images and real-life images as query images. However, the real-life images have no groundtruth and therefore only the visualizations, but not the quantitative results are relevant. 

For the 3nnds approach use the following command:

```bash
apptainer run --nv --bind /your_path_to_this_repo/3d_lines/limap/limap/point2d/superglue/weights:/limap/limap/point2d/superglue/weights /your/path/to/dockerimage.sif /bin/bash -c 'cd /limap && python /your_path_to_this_repo/3d_lines/localization_sbb_3nnds.py --output_dir /your/output/path'
```

## Running OnePose++ to compare results
To run OnePose++ we have forked the OnePose++ repository to adjust the code. The modified code can be found the this [repos](https://github.com/schmim/OnePose_Plus_Plus). Also, all the main adjusted files are copied into this repo and can be found in the `OnePose_Plus_Plus` subfolder.

## Acknowledgements
This code builds upon [LIMAP](https://github.com/cvg/limap).




