# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import torch
import numpy as np
import math
import sys
import argparse, os
import settings
import utils
import logging
import distributed_optimizer as dopt
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm.Set_errhandler(MPI.ERRORS_RETURN)

from dl_trainer import DLTrainer, _support_datasets, _support_dnns
from compression import compressors
from settings import logger, formatter
from tensorboardX import SummaryWriter

writer = None
relative_path = None

def robust_ssgd(dnn,
                dataset,
                data_dir,
                nworkers,
                lr,
                batch_size,
                nsteps_update,
                max_epochs,
                compression=False,
                compressor='topk',
                nwpernode=1,
                sigma_scale=2.5,
                pretrain=None,
                density=0.01,
                prefix=None):
    global relative_path, args
    torch.cuda.set_device(dopt.rank() % nwpernode)
    rank = dopt.rank()
    if rank != 0:
        pretrain = None

    trainer = DLTrainer(rank,
                        nworkers,
                        dist=False,
                        batch_size=batch_size,
                        nsteps_update=nsteps_update,
                        is_weak_scaling=True,
                        ngpus=1,
                        data_dir=data_dir,
                        dataset=dataset,
                        dnn=dnn,
                        lr=lr,
                        nworkers=nworkers,
                        prefix=(prefix + '-ds%s' % str(density)) if prefix else None,
                        pretrain=pretrain,
                        tb_writer=writer)
    init_epoch = trainer.get_train_epoch()
    init_iter = trainer.get_train_iter()
    trainer.set_train_epoch(comm.bcast(init_epoch))
    trainer.set_train_iter(comm.bcast(init_iter))

    def _error_handler(new_num_workers, new_rank):
        logger.info('Error info catched by trainer')
        trainer.update_nworker(new_num_workers, new_rank)

    comp_name = compressor if compression else 'none'
    comp_cls = compressors[comp_name]

    # Set HggTopk params if selected
    if getattr(comp_cls, 'name', '') == 'HggTopk':
        try:
            comp_cls.B = getattr(args, 'hggtopk_bins', getattr(comp_cls, 'B', 1024))
            comp_cls.gamma = getattr(args, 'hggtopk_gamma', getattr(comp_cls, 'gamma', 1000.0))
            comp_cls.beta = getattr(args, 'hggtopk_beta', getattr(comp_cls, 'beta', 0.98))
            comp_cls.sample_ratio = getattr(args, 'hggtopk_sample_ratio', getattr(comp_cls, 'sample_ratio', 0.02))
            comp_cls.sample_min = getattr(args, 'hggtopk_sample_min', getattr(comp_cls, 'sample_min', 65536))
            comp_cls.warmup_steps = getattr(args, 'hggtopk_warmup_steps', getattr(comp_cls, 'warmup_steps', 0))
            comp_cls.warmup_enabled = (comp_cls.warmup_steps > 0)
            logger.info('HggTopk params: B=%s, gamma=%s, beta=%s, sample_ratio=%s, sample_min=%s, warmup_steps=%s, warmup_enabled=%s',
                        str(comp_cls.B), str(comp_cls.gamma), str(comp_cls.beta), str(comp_cls.sample_ratio), str(comp_cls.sample_min), str(comp_cls.warmup_steps), str(comp_cls.warmup_enabled))
        except Exception:
            pass

    logger.info('Broadcast parameters....')
    model_state = comm.bcast(trainer.net.state_dict(), root=0)
    trainer.net.load_state_dict(model_state)
    comm.Barrier()
    logger.info('Broadcast parameters finished....')

    norm_clip = None
    is_sparse = compression
    optimizer = dopt.DistributedOptimizer(trainer.optimizer,
                                        trainer.net.named_parameters(),
                                        compression=comp_cls,
                                        is_sparse=is_sparse,
                                        err_handler=_error_handler,
                                        layerwise_times=None,
                                        sigma_scale=sigma_scale,
                                        density=density,
                                        norm_clip=norm_clip,
                                        writer=writer)

    trainer.update_optimizer(optimizer)
    iters_per_epoch = max(1, trainer.get_num_of_training_samples() // (nworkers * batch_size * nsteps_update))
    iter_log_every = 20

    logger.info('Start training ....')
    training_start = time.time()
    for epoch in range(max_epochs):
        epoch_time = time.time()
        trainer.the_test_time = 0
        hidden = None
        epoch_compute_acc = 0.0
        epoch_compress_acc = 0.0
        epoch_comm_acc = 0.0
        epoch_loss_acc = 0.0
        acc_start_idx = len(getattr(trainer, 'train_acc_top1', []))
        if dnn == 'lstm':
            hidden = trainer.net.init_hidden()

        for i in range(iters_per_epoch):
            optimizer.zero_grad()
            iter_compute_start = time.time()
            for j in range(nsteps_update):
                optimizer.local = (j < nsteps_update - 1 and nsteps_update > 1)
                if dnn == 'lstm':
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
            iter_compute_sum = time.time() - iter_compute_start
            trainer.update_model()

            iter_stats = getattr(optimizer._allreducer, '_last_iter_stats', {'comm': 0.0, 'compress': 0.0, 'sparse': 0.0})
            iter_loss = float(getattr(trainer, 'loss', float('nan')))
            iter_ppl = (math.exp(iter_loss) if (iter_loss == iter_loss and iter_loss < 50.0) else float('nan'))  # guard overflow
            if dopt.rank() == 0 and ((i+1) % iter_log_every == 0):
                logger.info('[CODEX][ITER] compute: %.6f s, loss: %s, ppl: %s, compress: %.6f s, comm: %.6f s', iter_compute_sum, ('%.6f' % iter_loss if (iter_loss == iter_loss) else 'N/A'), ('%.2f' % iter_ppl if (iter_ppl == iter_ppl) else 'N/A'), float(iter_stats.get('compress', 0.0)), float(iter_stats.get('comm', 0.0)))
            epoch_compute_acc += iter_compute_sum
            try:
                optimizer._allreducer._codex_acc['compute'] = optimizer._allreducer._codex_acc.get('compute', 0.0) + float(iter_compute_sum)
                optimizer._allreducer._codex_acc['compute_count'] = int(optimizer._allreducer._codex_acc.get('compute_count', 0)) + 1
            except Exception:
                pass
            epoch_compress_acc += float(iter_stats.get('compress', 0.0))
            epoch_comm_acc += float(iter_stats.get('comm', 0.0))
            epoch_loss_acc += float(getattr(trainer, 'loss', 0.0))



        optimizer.add_train_epoch()
        epoch_dur = time.time() - epoch_time - trainer.the_test_time
        total_so_far = time.time() - training_start
        acc_end_idx = len(getattr(trainer, 'train_acc_top1', []))
        acc_vals = getattr(trainer, 'train_acc_top1', [])[acc_start_idx:acc_end_idx] if acc_end_idx > acc_start_idx else []
        acc_avg = float(np.mean(acc_vals)) if len(acc_vals) > 0 else float('nan')
        iters_cnt = iters_per_epoch
        loss_avg = (epoch_loss_acc / float(max(1, iters_cnt * nsteps_update)))
        ppl_avg = (math.exp(loss_avg) if (loss_avg == loss_avg and loss_avg < 50.0) else float('nan'))
        if dopt.rank() == 0:
            logger.info('[CODEX][EPOCH %d] compute: %.6f s, compress: %.6f s, comm: %.6f s, loss(avg): %.6f, ppl(avg): %s, acc(avg): %s', epoch, epoch_compute_acc, epoch_compress_acc, epoch_comm_acc, loss_avg, ('%.2f' % ppl_avg if (ppl_avg == ppl_avg) else 'N/A'), ('%.3f' % acc_avg if not np.isnan(acc_avg) else 'N/A'))
            logger.info('[CODEX][TIME] epoch: %.6f s, total: %.6f s', epoch_dur, total_so_far)
        # clear any profiling buffers (kept for compatibility)
        optimizer._allreducer._profiling_norms = []

    optimizer.stop()
    if dopt.rank() == 0:
        logger.info('[CODEX] Total training time: %f s', time.time() - training_start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AllReduce trainer")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nsteps-update', type=int, default=1)
    parser.add_argument('--nworkers', type=int, default=1,
                        help='Just for experiments, and it cannot be used in production')
    parser.add_argument('--nwpernode', type=int, default=1,
                        help='Number of workers per node')
    parser.add_argument('--compression', dest='compression', action='store_true')
    parser.add_argument('--compressor', type=str, default='topk', choices=compressors.keys(),
                        help="Specify the compressor if 'compression' is on")
    # HggTopk knobs (will be used if compressor is HggTopk)
    parser.add_argument('--hggtopk-bins', type=int, default=1024,
                        help='HggTopk: number of log bins (for legacy variant)')
    parser.add_argument('--hggtopk-gamma', type=float, default=1000.0,
                        help='HggTopk: log mapping gamma (for legacy variant)')
    parser.add_argument('--hggtopk-beta', type=float, default=0.98,
                        help='HggTopk: conservative interpolation factor (for legacy variant)')
    parser.add_argument('--hggtopk-sample-ratio', type=float, default=0.02,
                        help='HggTopk: sampling ratio')
    parser.add_argument('--hggtopk-sample-min', type=int, default=65536,
                        help='HggTopk: min sample size')
    parser.add_argument('--hggtopk-warmup-steps', type=int, default=0,
                        help='HggTopk: warmup steps (0=disable)')
    parser.add_argument('--sigma-scale', type=float, default=2.5,
                        help='Maximum sigma scaler for sparsification')
    parser.add_argument('--density', type=float, default=0.01,
                        help='Density for sparsification')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=_support_datasets,
                        help='Specify the dataset for training')
    parser.add_argument('--dnn', type=str, default='resnet50', choices=_support_dnns,
                        help='Specify the neural network for training')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Specify the data root path')
    parser.add_argument('--lr', type=float, default=0.1, help='Default learning rate')
    parser.add_argument('--max-epochs', type=int, default=90, help='Default maximum epochs to train')
    parser.add_argument('--pretrain', type=str, default=None, help='Specify the pretrain path')
    parser.set_defaults(compression=False)
    args = parser.parse_args()
    batch_size = args.batch_size * args.nsteps_update
    prefix = settings.PREFIX if hasattr(settings, 'PREFIX') else 'run'
    if args.compression:
        prefix = 'comp-' + args.compressor + '-' + prefix
    logdir = 'allreduce-%s/%s-n%d-bs%d-lr%.4f-ns%d-sg%.2f-ds%s' % (
        prefix, args.dnn, args.nworkers, batch_size, args.lr, args.nsteps_update, args.sigma_scale, str(args.density))
    relative_path = './log/%s' % logdir
    utils.create_path(relative_path)
    rank = dopt.rank()
    if rank == 0:
        tb_runs = './runs/%s' % logdir
        writer = SummaryWriter(tb_runs)
    logfile = os.path.join(relative_path, settings.hostname + '-' + str(rank) + '-seed1-nointer.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.info('Configurations: %s', args)
    logger.info('Interpreter: %s', sys.version)

    robust_ssgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr,
                args.batch_size, args.nsteps_update, args.max_epochs,
                args.compression, args.compressor, args.nwpernode,
                args.sigma_scale, args.pretrain, args.density, prefix)