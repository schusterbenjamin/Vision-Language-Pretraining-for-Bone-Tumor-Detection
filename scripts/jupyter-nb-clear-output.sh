#!/bin/bash

# Description: This script clears the output and metadata from Jupyter notebooks.
# This is used by the pre-commit-hook to ensure notebooks do not contain output when committed.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate project

for nb in "$@"; do
  jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace "$nb"
done