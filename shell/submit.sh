#!/bin/bash

# Set the root directory of your project
PROJECT_ROOT="devtools_scicomp_project_2025"

# Create the directory structure
mkdir -p $PROJECT_ROOT/src/pyclassify
mkdir -p $PROJECT_ROOT/scripts
mkdir -p $PROJECT_ROOT/test
mkdir -p $PROJECT_ROOT/shell
mkdir -p $PROJECT_ROOT/experiments

# Create the required files
touch $PROJECT_ROOT/src/pyclassify/__init__.py
touch $PROJECT_ROOT/src/pyclassify/utils.py
touch $PROJECT_ROOT/scripts/run.py
touch $PROJECT_ROOT/test/test_.py
touch $PROJECT_ROOT/shell/submit.sbatch
touch $PROJECT_ROOT/shell/submit.sh
touch $PROJECT_ROOT/experiments/config.yaml

# Activate the conda environment
source activate devtools_scicomp

# Generate requirements.txt
pip freeze > $PROJECT_ROOT/requirements.txt

# Deactivate the conda environment
conda deactivate

echo "Project setup complete."
