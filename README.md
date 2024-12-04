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
