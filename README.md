# SBBTrainDoorLocolization

## running the localization_sbb_doors_rendered_only.py

this file can directly run the training dataset and test with query image with limap
to run it, you need to:
change the path to your local path where the training dataset and query image dataset locates

### if you want to run it with docker

docker run --rm -it \
 --volume your path to training dataset:/limap/data \
 --volume your path where this file localization_sbb_doors_rendered_only.py locates:/limap/code \
 --volume your path for storing the output:/limap/output \
 my_custom_image:v1 \
 /bin/bash -c "cd /limap && python /limap/code/localization_sbb_doors_rendered_only.py --output_dir /limap/output --num_loc 1 --num_covis 1"

you can also delete "--num_loc 1 --num_covis 1"

## for docker file

this docker file is used to build the environment in MacOS ARM64
you have to install these dependency manually after building your docker image:

> 需要手动装的依赖： - `pyvista` - `./third-party/pytlsd` - `./third-party/hawp` - `-e ./third-party/Hierarchical-Localization` - `-e ./third-party/DeepLSD` - `-e ./third-party/GlueStick`

    使用的 `pycolmap` 版本：[v0.4.0](https://github.com/colmap/pycolmap/releases/tag/v0.4.0)

if your system is not ARM64 you can add the dependencies above in your docker file so that you don't need to install them manually.
