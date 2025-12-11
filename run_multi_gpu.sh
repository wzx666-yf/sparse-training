#!/bin/bash

# Set different CUDA_VISIBLE_DEVICES for each MPI rank
# Each process sees cuda:0 but uses different physical GPU

# Get MPI rank from environment variables
# OpenMPI uses OMPI_COMM_WORLD_RANK, MPICH uses PMI_RANK
if [ -n "$OMPI_COMM_WORLD_RANK" ]; then
    RANK=$OMPI_COMM_WORLD_RANK
elif [ -n "$PMI_RANK" ]; then
    RANK=$PMI_RANK
else
    RANK=0
fi

# Set CUDA_VISIBLE_DEVICES based on rank
# rank 0 -> GPU 0, rank 1 -> GPU 1, rank 2 -> GPU 2
export CUDA_VISIBLE_DEVICES=$RANK

# Run main program, pass all arguments
exec python main_trainer.py "$@"
