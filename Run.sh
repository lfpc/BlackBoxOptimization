#!/bin/bash
USER_NAME=$(whoami)
export PROJECTS_DIR="$(dirname "$PWD")"
PROJECT_DIR="/home/hep/$USER_NAME/MuonShieldProject"
CONTAINER_PATH="/disk/users/lprate/containers/snoopy_geant_cuda.sif"
PYTHON_SCRIPT_DIR="$PROJECT_DIR/BlackBoxOptimization"

nvidia-smi

read -p "Select optimization method (GA/RL/CMAES/bayesian): " OPT_METHOD
if [[ "$OPT_METHOD" != "GA" && "$OPT_METHOD" != "RL" && "$OPT_METHOD" != "CMAES" && "$OPT_METHOD" != "bayesian" ]]; then
    echo "Error: Invalid choice. Must be 'GA', 'RL', 'CMAES' or 'bayesian'."
    exit 1
fi

if [[ "$OPT_METHOD" == "bayesian" ]]; then
    read -p "Enter the CUDA device index to use (e.g. 0, 1, 2, ...): " CUDA_DEVICE
    if [[ -z "$CUDA_DEVICE" ]]; then
        echo "Error: You must specify a CUDA device index."
        exit 1
    fi
else
    read -p "Enter the CUDA device indexes to use (comma-separated, e.g. 0,1,2): " CUDA_DEVICE
    if [[ -z "$CUDA_DEVICE" ]]; then
        echo "Error: You must specify at least one CUDA device index."
        exit 1
    fi
fi

read -p "Enter the name for the Results folder: " RESULTS_NAME
if [[ -z "$RESULTS_NAME" ]]; then
    echo "Error: You must specify a Results folder name."
    exit 1
fi
RESULTS_DIR="$PYTHON_SCRIPT_DIR/outputs/$RESULTS_NAME"
if [[ -d "$RESULTS_DIR" ]]; then
    echo "Error: The folder '$RESULTS_DIR' already exists. Please choose a different name."
    exit 1
fi
mkdir -p "$RESULTS_DIR"

LOG_FILE="$RESULTS_DIR/output_${OPT_METHOD}.log"

#Install locally the python libraries that are missing in the container:
LOCAL_LIBRARIES_DIR="/home/hep/$USER_NAME/MuonShieldProject/BlackBoxOptimization/local_python_libs"
mkdir -p "$LOCAL_LIBRARIES_DIR"
pip install --target "$LOCAL_LIBRARIES_DIR" --no-deps cma gymnasium # Install locally if missing

nohup apptainer exec --nv \
  -B /cvmfs \
  -B /disk/users/$USER_NAME \
  -B /home/hep/$USER_NAME \
  -B "$LOCAL_LIBRARIES_DIR" \
  $CONTAINER_PATH \
  bash --login -i -c "
    export PYTHONPATH=\$PYTHONPATH:$LOCAL_LIBRARIES_DIR
    cd $PROJECT_DIR/MuonsAndMatter
    source ./set_env.sh
    cd $PYTHON_SCRIPT_DIR
    python3 -V
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 run_optimization.py --optimization $OPT_METHOD --name $RESULTS_NAME
  " > "$LOG_FILE" 2>&1 &
