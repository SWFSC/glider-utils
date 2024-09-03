#!/bin/bash

### Common conda commands, for reference
# https://docs.conda.io/projects/conda/en/latest/
# https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf


# Update conda env from file, removing (pruning) unspecified packages 
conda env update --file glider-utils/environment.yml --prune
