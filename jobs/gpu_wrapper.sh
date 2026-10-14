#!/bin/bash
# One GPU per rank: bind by node-local rank, then exec the command.
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
exec "$@"
