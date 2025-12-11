## Requirements

- Python 3.8.13
- torch 1.11.0+cu113
- torchvision 0.12.0+cu113
- MPI 3.3.2
- mpi4py 3.0.3
- numpy 1.21.5
- ...

## Models and Datasets

We use four deep learning models, VGG-16, VGG-19, ResNet-20, LSTM, and three datasets as four different learning tasks.

CIFAR-10 : Download from https://pytorch.org/vision/stable/datasets.html#cifar

CIFAR-100 : Download from https://pytorch.org/vision/stable/datasets.html#cifar

IMDB : Download from http://ai.stanford.edu/~amaas/data/sentiment/index.html

## Quick Start

To train and evaluate with SparDL on 2 workers, i.e., node0, node1

```
mpiexec -n 4 -host localhost:4 bash run_multi_gpu.sh \
  --dnn vgg16 \
  --dataset cifar10 \
  --max-epochs 50 \
  --batch-size 32 \
  --nworkers 4 \
  --nwpernode 1 \
  --data-dir ./vgg_data \
  --lr 0.1 \
  --compression \
  --density 0.01 \
  --compressor spardl
```

The meaning of the flags:

- `--batch-size`: Batch size.
- `--nworkers`: Number of workers.
- `--compression`: Compress gradients or not.
- `--compressor`: Specify the compressors if 'compression' is open.
- `--density`: Density for sparsification.
- `--dataset`: Specify the dataset for training. options: {imagenet,cifar10,cifar100,mnist,imdb}
- `--dnn`: Specify the neural network for training. options: {resnet50,resnet20,resnet56,resnet110,vgg19,vgg16,alexnet,lstm,lstmimdb}
- `--data-dir`: Specify the data root path.
- `--lr`: Default learning rate.
- `--max-epochs`: Default maximum epochs to train.
