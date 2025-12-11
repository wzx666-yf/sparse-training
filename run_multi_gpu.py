#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

# Get MPI rank from environment variables before importing anything
# OpenMPI sets OMPI_COMM_WORLD_RANK, MPICH sets PMI_RANK
rank_str = os.environ.get('OMPI_COMM_WORLD_RANK') or os.environ.get('PMI_RANK', '0')
rank = int(rank_str)

# Set CUDA_VISIBLE_DEVICES based on rank
# This makes each process see only one GPU as cuda:0
os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)

# Now import and run main_trainer
if __name__ == '__main__':
    main_trainer_path = os.path.join(os.path.dirname(__file__), 'main_trainer.py')
    with open(main_trainer_path, 'r', encoding='utf-8', errors='ignore') as f:
        code = compile(f.read(), main_trainer_path, 'exec')
        exec(code, {'__name__': '__main__', '__file__': main_trainer_path})
