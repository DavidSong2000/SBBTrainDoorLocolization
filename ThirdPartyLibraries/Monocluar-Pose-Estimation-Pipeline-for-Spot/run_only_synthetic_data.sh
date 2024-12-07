# create some synthetic data
echo "Running the synthetic data pipeline"
blenderproc run src/synthetic_data_pipeline_st.py

# read params from config.json (`brew install jq` if necessary)
echo "Creating the animation gif and removing the separate frames"
DATA_DIR=$(jq -r '.DATA_DIR' config.json)
# DATA_DIR=output/$DATA_DIR # new default location of output --> output dir
SCENE=$(jq -r '.SCENE' config.json)
echo "DATA_DIR: $DATA_DIR"


BOP_DATA="output/$DATA_DIR/scene_$SCENE-annotate/bop_data/train_pbr/000000"
echo "BOP_DATA: $BOP_DATA"

echo "Done"