# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import time
import psutil
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tdata
import torchvision
import torchvision.transforms as transforms

import torch.cuda as ct
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

import settings
from settings import logger, formatter
import models

# Optional HF stack
try:
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from models.hf_models import GPT2SmallLM, GPT2MediumLM, BertLargeSST2
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
    GPT2SmallLM = GPT2MediumLM = BertLargeSST2 = None

cudnn.benchmark = False
cudnn.deterministic = True

torch.manual_seed(1)
torch.set_num_threads(1)

_support_datasets = ['imagenet', 'cifar10', 'cifar100', 'mnist', 'imdb', 'wikitext2', 'sst2']
_support_dnns = ['resnet50','resnet20','resnet56','resnet110','vgg19','vgg16','alexnet','lstm','lstmimdb','gpt2_small','gpt2_medium','bert_large']

NUM_CPU_THREADS = 0
process = psutil.Process(os.getpid())


def init_processes(rank, size, backend='tcp', master='gpu10'):
    os.environ['MASTER_ADDR'] = master
    os.environ['MASTER_PORT'] = '5935'
    logger.info("initialized trainer rank: %d of %d......" % (rank, size))
    dist.init_process_group(backend=backend, rank=rank, world_size=size)
    logger.info("finished trainer rank: %d......" % rank)


class DLTrainer:
    def __init__(self,
                 rank,
                 size,
                 master='gpu10',
                 dist=True,
                 ngpus=1,
                 batch_size=32,
                 nsteps_update=1,
                 is_weak_scaling=True,
                 data_dir='./data',
                 dataset='cifar10',
                 dnn='resnet20',
                 lr=0.04,
                 nworkers=1,
                 prefix=None,
                 sparsity=0.95,
                 pretrain=None,
                 num_steps=35,
                 tb_writer=None):
        self.size = size
        self.rank = rank
        self.pretrain = pretrain
        self.dataset = dataset
        self.prefix = prefix
        self.num_steps = num_steps
        self.ngpus = ngpus
        self.writer = tb_writer
        self.nsteps_update = nsteps_update
        self.is_cuda = (self.ngpus > 0)
        self.batch_size = (batch_size * self.ngpus if is_weak_scaling and self.ngpus > 0 else batch_size)
        self.num_batches_per_epoch = -1
        self.nworkers = nworkers
        self.data_dir = data_dir
        self.lr = lr

        # classes
        if self.dataset in ['cifar10','mnist']: self.num_classes = 10
        elif self.dataset == 'cifar100': self.num_classes = 100
        elif self.dataset == 'imagenet': self.num_classes = 1000
        elif self.dataset in ['sst2','imdb']: self.num_classes = 2
        else: self.num_classes = 0

        # build model
        self.dnn = dnn
        self.net, self.ext = self.create_net(self.num_classes, self.dnn)
        if self.is_cuda:
            self.net = self.net.cuda()
        self.net.share_memory()

        if self.dataset == 'imdb':
            self.criterion = nn.NLLLoss().cuda() if self.is_cuda else nn.NLLLoss()
        else:
            self.criterion = nn.CrossEntropyLoss().cuda() if self.is_cuda else nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4, nesterov=False)

        self.train_epoch = 0
        self.train_iter = 0
        self.recved_counter = 0
        if dist:
            init_processes(rank, size, master=master)

        # state for logging
        self.sparsities = []
        self.compression_ratios = []
        self.communication_sizes = []
        self.avg_loss_per_epoch = 0.0
        self.timer = 0.0
        self.iotime = 0.0
        self.epochs_info = []
        self.train_acc_top1 = []
        self.comp_time = 0.0
        self.the_test_time = 0.0

        # build data
        self.data_prepare()

    # --- Model factory ---
    def create_net(self, num_classes, dnn='resnet20', **kwargs):
        if dnn in ['resnet20','resnet56','resnet110']:
            net = models.__dict__[dnn](num_classes=num_classes)
        elif dnn == 'resnet50':
            net = models.__dict__['resnet50'](num_classes=num_classes)
        elif dnn in ['vgg16','vgg19']:
            net = models.VGG(dnn.upper(), num_classes)
        elif dnn == 'alexnet':
            net = torchvision.models.alexnet()
        elif dnn == 'mnistnet':
            from model_builder import MnistNet as _MN
            net = _MN()
        elif dnn in ['gpt2_small','gpt2_medium']:
            if not HF_AVAILABLE or GPT2SmallLM is None:
                raise RuntimeError('Install transformers + datasets for GPT-2.')
            net = GPT2SmallLM() if dnn=='gpt2_small' else GPT2MediumLM()
        elif dnn == 'bert_large':
            if not HF_AVAILABLE or BertLargeSST2 is None:
                raise RuntimeError('Install transformers + datasets for BERT.')
            net = BertLargeSST2(num_labels=2)
        elif dnn in ['lstm','lstmimdb']:
            raise NotImplementedError('lstm/lstmimdb not supported in simplified trainer')
        else:
            raise RuntimeError('Unsupport neural network %s' % dnn)
        return net, None

    # --- Data ---
    def data_prepare(self):
        if self.dataset == 'cifar10':
            image_size = 32
            normalize = transforms.Normalize(mean=[0.491,0.482,0.447], std=[0.247,0.243,0.262])
            train_tf = transforms.Compose([transforms.RandomCrop(image_size, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
            test_tf  = transforms.Compose([transforms.ToTensor(), normalize])
            self.trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=train_tf)
            self.testset  = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=test_tf)
            collate = None
        elif self.dataset == 'cifar100':
            image_size = 32
            normalize = transforms.Normalize(mean=[0.507,0.486,0.441], std=[0.267,0.256,0.276])
            train_tf = transforms.Compose([transforms.RandomCrop(image_size, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
            test_tf  = transforms.Compose([transforms.ToTensor(), normalize])
            self.trainset = torchvision.datasets.CIFAR100(root=self.data_dir, train=True, download=True, transform=train_tf)
            self.testset  = torchvision.datasets.CIFAR100(root=self.data_dir, train=False, download=True, transform=test_tf)
            collate = None
        elif self.dataset == 'mnist':
            train_tf = transforms.Compose([transforms.ToTensor()])
            self.trainset = torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True, transform=train_tf)
            self.testset  = torchvision.datasets.MNIST(root=self.data_dir, train=False, download=True, transform=train_tf)
            collate = None
        elif self.dataset == 'wikitext2':
            if not HF_AVAILABLE: raise RuntimeError('transformers/datasets required for wikitext2')
            tok = AutoTokenizer.from_pretrained('gpt2')
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            ds = load_dataset('wikitext','wikitext-2-raw-v1')
            block_size = 1024
            def build_blocks(texts):
                ids = []
                for t in texts:
                    if t: ids.extend(tok.encode(t, add_special_tokens=False))
                n = len(ids) // block_size
                return [torch.tensor(ids[i*block_size:(i+1)*block_size], dtype=torch.long) for i in range(n)]
            class SimpleSeq(tdata.Dataset):
                def __init__(self, arr): self.arr = arr
                def __len__(self): return len(self.arr)
                def __getitem__(self, i): x = self.arr[i]; return x, x
            self.trainset = SimpleSeq(build_blocks(ds['train']['text']))
            self.testset  = SimpleSeq(build_blocks(ds['validation']['text']))
            collate = None
        elif self.dataset == 'sst2':
            if not HF_AVAILABLE: raise RuntimeError('transformers/datasets required for sst2')
            tok = AutoTokenizer.from_pretrained('bert-large-uncased')
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            ds = load_dataset('glue','sst2')
            def encode_batch(sents, labels, max_len=128):
                enc = tok(sents, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
                return enc['input_ids'], enc['attention_mask'], torch.tensor(labels, dtype=torch.long)
            X_ids,X_mask,y = encode_batch(ds['train']['sentence'], ds['train']['label'])
            V_ids,V_mask,vy= encode_batch(ds['validation']['sentence'], ds['validation']['label'])
            class Sst2Set(tdata.Dataset):
                def __init__(self, ids, mask, y): self.ids,self.mask,self.y=ids,mask,y
                def __len__(self): return self.y.size(0)
                def __getitem__(self, i): return self.ids[i], self.mask[i], int(self.y[i])
            self.trainset = Sst2Set(X_ids,X_mask,y)
            self.testset  = Sst2Set(V_ids,V_mask,vy)
            def collate_fn(batch):
                ids,mask,lab = zip(*batch)
                return torch.stack(ids,0), torch.stack(mask,0), torch.tensor(lab, dtype=torch.long)
            collate = collate_fn
        else:
            raise RuntimeError('Unsupport dataset: %s' % self.dataset)

        train_sampler = None; shuffle=True
        if self.nworkers > 1:
            train_sampler = tdata.distributed.DistributedSampler(self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0); shuffle=False
        self.train_sampler = train_sampler
        self.trainloader = tdata.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=shuffle, num_workers=NUM_CPU_THREADS, sampler=train_sampler, collate_fn=collate)
        self.testloader  = tdata.DataLoader(self.testset,  batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
        self.data_iterator = None
        self.num_batches_per_epoch = max(1, len(self.trainset) // (self.batch_size * max(1,self.nworkers) * max(1,self.nsteps_update)))

    # --- Public helpers used by main_trainer ---
    def get_num_of_training_samples(self): return len(self.trainset)
    def get_train_epoch(self): return self.train_epoch
    def get_train_iter(self): return self.train_iter
    def set_train_epoch(self, epoch): self.train_epoch = epoch
    def set_train_iter(self, it): self.train_iter = it
    def update_optimizer(self, optimizer): self.optimizer = optimizer

    # --- Training/Test ---
    def data_iter(self):
        if self.data_iterator is None:
            self.data_iterator = iter(self.trainloader)
        try:
            return next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.trainloader)
            return next(self.data_iterator)

    def cal_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).cpu().data.numpy())
        return res

    def train(self, num_of_iters=1, data=None, hidden=None):
        self.loss = 0.0
        s = time.time()
        for i in range(num_of_iters):
            if data is None:
                data = self.data_iter()
            if self.dataset == 'sst2':
                input_ids, attn_mask, labels = data
                if self.is_cuda:
                    input_ids = input_ids.cuda(non_blocking=True)
                    attn_mask = attn_mask.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                inputs = (input_ids, attn_mask)
            else:
                inputs, labels = data
                if self.is_cuda:
                    inputs = inputs.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

            self.iotime += (time.time() - s)
            com_time = time.time()

            if self.dnn in ['gpt2_small','gpt2_medium']:
                out = self.net(inputs, labels=inputs)
                loss = out.loss if hasattr(out,'loss') else out[0]
                loss.backward()
            elif self.dnn == 'bert_large':
                input_ids, attn_mask = inputs
                out = self.net(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss = out.loss if hasattr(out,'loss') else out[0]
                loss.backward()
            else:
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

            self.comp_time = time.time() - com_time
            self.loss += float(loss.item())
            self.avg_loss_per_epoch += float(loss.item())
            if self.dnn not in ['gpt2_small','gpt2_medium']:
                try:
                    acc1, = self.cal_accuracy(outputs, labels, topk=(1,))
                    self.train_acc_top1.append(acc1)
                except Exception:
                    pass
            self.train_iter += 1
        self.timer += time.time() - s
        return num_of_iters

    def test(self, epoch):
        self.net.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        total_iters = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.testloader):
                if self.dataset == 'sst2':
                    input_ids, attn_mask, labels = data
                    if self.is_cuda:
                        input_ids = input_ids.cuda(non_blocking=True)
                        attn_mask = attn_mask.cuda(non_blocking=True)
                        labels = labels.cuda(non_blocking=True)
                    out = self.net(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                    loss = out.loss if hasattr(out,'loss') else out[0]
                    logits = out.logits if hasattr(out,'logits') else out[1]
                    pred = logits.argmax(dim=1)
                    correct += pred.eq(labels).cpu().sum()
                    total += labels.size(0)
                    test_loss += float(loss.item())
                elif self.dnn in ['gpt2_small','gpt2_medium']:
                    ids, _ = data
                    if self.is_cuda: ids = ids.cuda(non_blocking=True)
                    out = self.net(ids, labels=ids)
                    loss = out.loss if hasattr(out,'loss') else out[0]
                    test_loss += float(loss.item())
                    total += ids.size(0)
                else:
                    inputs, labels = data
                    if self.is_cuda:
                        inputs = inputs.cuda(non_blocking=True)
                        labels = labels.cuda(non_blocking=True)
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    test_loss += float(loss.item())
                    _, predicted = torch.max(outputs.data, 1)
                    correct += predicted.eq(labels.data).cpu().sum()
                    total += labels.size(0)
                total_iters += 1
        test_loss /= max(1,total_iters)
        acc = float(correct)/max(1,total) if self.dnn not in ['gpt2_small','gpt2_medium'] else 0.0
        logger.info('Epoch %d, lr: %f, val loss: %f, val acc: %f' % (epoch, self.lr, test_loss, acc))
        self.net.train()
        return acc

    def update_model(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
