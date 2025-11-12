#!/bin/bash

# Description: This script exports the current Conda environment to a YAML file
# and appends a specific PyTorch extra index URL to the pip section if it exists.
# This is used by the pre-commit-hook to ensure the environment file is up-to-date.

conda env export -n project > environment.yaml
# Add index URL only if it's not already there AND 'pip:' section exists
if grep -q '^  - pip:' environment.yaml && \
   ! grep -q -- "--extra-index-url https://download.pytorch.org/whl/cu118" environment.yaml; then

  awk '/^  - pip:/ {
    print;
    print "      - --extra-index-url https://download.pytorch.org/whl/cu118";
    next
  }
  1' environment.yaml > tmp.yml && mv tmp.yml environment.yaml
fi