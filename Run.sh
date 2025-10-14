#!/bin/bash
USER_NAME=$(whoami)
export PROJECTS_DIR="$(dirname "$PWD")"
PROJECT_DIR="/home/hep/$USER_NAME/MuonShieldProject"
CONTAINER_PATH="/disk/users/lprate/containers/snoopy_geant_cuda.sif"
PYTHON_SCRIPT_DIR="$PROJECT_DIR/BlackBoxOptimization"
LOG_FILE="$PYTHON_SCRIPT_DIR/output_GAs.log"

nvidia-smi

read -p "Enter the CUDA device index to use (e.g. 0, 1, 2, ...): " CUDA_DEVICE
if [[ -z "$CUDA_DEVICE" ]]; then
    echo "Error: You must specify a CUDA device index."
    exit 1
fi

nohup apptainer exec --nv \
  -B /cvmfs \
  -B /disk/users/$USER_NAME \
  -B /home/hep/$USER_NAME \
  $CONTAINER_PATH \
  bash --login -i -c "
    cd $PROJECT_DIR/MuonsAndMatter
    source ./set_env.sh
    cd $PYTHON_SCRIPT_DIR
    python3 -V
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 run_optimization.py --optimization GA
  " > "$LOG_FILE" 2>&1 &
