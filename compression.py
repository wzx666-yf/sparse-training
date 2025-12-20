# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
import time

import math
import utils
from scipy import stats


class NoneCompressor():
    name = 'dense'

    @staticmethod
    def compress(tensor, name=None, sigma_scale=None, ratio=None):
        return tensor, None
        #return tensor, tensor.dtype

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor
        return z


class TopKCompressor():
    """
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
    residuals = {}
    c = 0
    sparsities = []
    t = 0.
    zero_conditions = {}
    values = {}
    indexes = {}
    name = 'topk'

    @staticmethod
    def compress_org(tensor, name=None, sigma_scale=2.5, ratio=0.05):
        start = time.time()
        with torch.no_grad():
            if name not in TopKCompressor.residuals:
                TopKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.data.add_(TopKCompressor.residuals[name].data)

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]
            if name not in TopKCompressor.zero_conditions:
                TopKCompressor.zero_conditions[name] = torch.ones(
                    numel, dtype=torch.float32, device=tensor.device)
            zero_condition = TopKCompressor.zero_conditions[name]
            zero_condition.fill_(1.0)
            zero_condition[indexes] = 0.0

            TopKCompressor.residuals[name].data.fill_(0.)
            TopKCompressor.residuals[name].data = tensor.data * zero_condition
            tensor.data.sub_(TopKCompressor.residuals[name].data)

            TopKCompressor.values[name] = values
            TopKCompressor.indexes[name] = indexes
            return tensor, indexes

    @staticmethod
    def compress(tensor, name=None, sigma_scale=2.5, ratio=0.05):
        start = time.time()
        with torch.no_grad():
            if name not in TopKCompressor.residuals:
                TopKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.data.add_(TopKCompressor.residuals[name].data)

            values, indexes = torch.topk(torch.abs(tensor.data),
                                        k=k,
                                        sorted=False)

            TopKCompressor.residuals[name].data = tensor.data + 0.0
            TopKCompressor.residuals[name].data[indexes] = 0.0

            TopKCompressor.values[name] = values
            TopKCompressor.indexes[name] = indexes
            return tensor, indexes

    @staticmethod
    def ratio2threshold(tensor, name=None, ratio=0.05):
        with torch.no_grad():
            if name not in TopKCompressor.residuals:
                TopKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.data.add_(TopKCompressor.residuals[name].data)

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)

            TopKCompressor.residuals[name].data = tensor.data + 0.0
            TopKCompressor.residuals[name].data[indexes] = 0.0

            TopKCompressor.values[name] = values
            TopKCompressor.indexes[name] = indexes
            #print("local topk elements: ", torch.numel(values))

            threshold = float(values[values.numel() - 1].item())
            return threshold

    @staticmethod
    def ratio2globalthreshold(tensor, ratio=0.05):
        with torch.no_grad():
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)

            threshold = float(values[values.numel() - 1].item())
            print("global topk elements: ", torch.numel(values), "threshold: ",
                threshold)
            return threshold

    @staticmethod
    def compressbythreshold(tensor, thres=0.0):
        with torch.no_grad():
            abs_tensor = torch.abs(tensor)

            one_indexes = abs_tensor > thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            values = tensor.data[indexes]

            indexes = indexes.type(torch.IntTensor)
            return indexes, values

    @staticmethod
    def compressbythresholdlong(tensor, thres=0.0):
        with torch.no_grad():
            abs_tensor = torch.abs(tensor)

            one_indexes = abs_tensor > thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            values = tensor.data[indexes]

            return indexes, values

    @staticmethod
    def get_residuals(name, like_tensor):
        if name not in TopKCompressor.residuals:
            TopKCompressor.residuals[name] = torch.zeros_like(like_tensor.data)
        return TopKCompressor.residuals[name]

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            residuals = TopKCompressor.residuals[name]
            if type(included_indexes) is np.ndarray:
                indexes_t = torch.from_numpy(included_indexes).cuda(
                    residuals.device).long()
            else:
                indexes_t = included_indexes
            values = TopKCompressor.values[name]
            values.data[indexes_t] = 0.0
            residuals.data[TopKCompressor.indexes[name]] += values.data

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor
        return z


class GaussianCompressor():
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {}
    indexes = {}
    c = 0
    t = 0.
    name = 'gaussionk'
    #inc_factor = 0.02
    #dec_factor = 1.8

    counter = 0
    local_threshold = 0.0

    @staticmethod
    def clear():
        GaussianCompressor.residuals = {}
        GaussianCompressor.sparsities = []
        GaussianCompressor.zero_conditions = {}
        GaussianCompressor.values = {}
        GaussianCompressor.indexes = {}

    #@staticmethod
    #def compress(tensor, name=None, ratio=0.05, counter=-1, rank=-1):
    #    with torch.no_grad():
    #        if name not in GaussianCompressor.residuals:
    #            GaussianCompressor.residuals[name] = torch.zeros_like(tensor.data)
    #        numel = tensor.numel()
    #        k = max(int(numel * ratio), 1)

    #        tensor.add_(GaussianCompressor.residuals[name].data)

    #        std = torch.std(tensor)
    #        mean = torch.mean(tensor)
    #        left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
    #        abs_tensor = torch.abs(tensor)
    #        one_indexes = abs_tensor > right_thres
    #        indexes = one_indexes.nonzero().data.squeeze().view(-1)

    #        #one_indexes = abs_tensor > right_thres
    #        #indexes = one_indexes.nonzero().data.squeeze().view(-1)
    #        #indexes = indexes #[0:k]
    #        values = tensor.data[indexes]
    #        GaussianCompressor.residuals[name].data = tensor.data + 0.0
    #        GaussianCompressor.residuals[name].data[indexes] = 0.0

    #        indexes = indexes.type(torch.IntTensor)
    #        return indexes, values

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, counter=-1, rank=-1):
        with torch.no_grad():
            if name not in GaussianCompressor.residuals:
                GaussianCompressor.residuals[name] = torch.zeros_like(
                    tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(GaussianCompressor.residuals[name].data)

            std = torch.std(tensor)
            mean = torch.mean(tensor)
            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(
                1 - ratio, float(mean), float(std))
            abs_tensor = torch.abs(tensor)

            one_indexes = abs_tensor > right_thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            init_num = indexes.numel()

            if init_num < 3 * k // 4:
                loops = 0
                while loops < 20:
                    if indexes.numel() < 3 * k // 4:
                        right_thres /= 1.02
                    else:
                        break
                    one_indexes = abs_tensor > right_thres
                    indexes = one_indexes.nonzero().data.squeeze().view(-1)
                    loops += 1
            elif init_num > 5 * k // 4:
                loops = 0
                while loops < 20:
                    if indexes.numel() > 5 * k // 4:
                        right_thres *= 1.02
                    else:
                        break
                    one_indexes = abs_tensor > right_thres
                    indexes = one_indexes.nonzero().data.squeeze().view(-1)
                    loops += 1
            else:
                pass

            values = tensor.data[indexes]
            GaussianCompressor.residuals[name].data = tensor.data + 0.0
            GaussianCompressor.residuals[name].data[indexes] = 0.0

            indexes = indexes.type(torch.IntTensor)
            return indexes, values

    #@staticmethod
    #def compress(tensor, name=None, ratio=0.05, counter=-1, rank=-1):
    #    with torch.no_grad():
    #        if name not in GaussianCompressor.residuals:
    #            GaussianCompressor.residuals[name] = torch.zeros_like(tensor.data)
    #        numel = tensor.numel()
    #        k = max(int(numel * ratio), 1)

    #        tensor.add_(GaussianCompressor.residuals[name].data)

    #        std = torch.std(tensor)
    #        mean = torch.mean(tensor)
    #        left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
    #        abs_tensor = torch.abs(tensor)
    #        loops = 0
    #        while loops < 3:
    #            one_indexes = abs_tensor > right_thres
    #            indexes = one_indexes.nonzero().data.squeeze().view(-1)
    #            if indexes.numel() < 2*k/3:
    #                right_thres *= 0.5
    #            elif indexes.numel() > 4*k/3:
    #                right_thres *= 1.5
    #            else:
    #                break
    #            loops += 1

    #        #print("local mean: ", mean, "local std: ", std, "adapt loops: ", loops)
    #        #one_indexes = abs_tensor > right_thres
    #        #indexes = one_indexes.nonzero().data.squeeze().view(-1)
    #        #indexes = indexes #[0:k]
    #        values = tensor.data[indexes]
    #        #print('gaussion vs topk: ', indexes.numel(), k)
    #        GaussianCompressor.residuals[name].data = tensor.data + 0.0
    #        GaussianCompressor.residuals[name].data[indexes] = 0.0

    #        indexes = indexes.type(torch.IntTensor)
    #        return indexes, values

    @staticmethod
    def predictratio2threshold(tensor, name=None, ratio=0.05):
        with torch.no_grad():
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            std = torch.std(tensor)
            mean = torch.mean(tensor)
            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(
                1 - ratio, float(mean), float(std))
            abs_tensor = torch.abs(tensor)
            one_indexes = abs_tensor > right_thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            pre_topk = indexes.numel()

            return right_thres, pre_topk

    @staticmethod
    def compressbythreshold(tensor, thres=0.0):
        with torch.no_grad():
            abs_tensor = torch.abs(tensor)

            one_indexes = abs_tensor > thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            values = tensor.data[indexes]

            indexes = indexes.type(torch.IntTensor)
            return indexes, values

    @staticmethod
    def compressbythreshold_residual(tensor, name, thres=0.0):
        with torch.no_grad():
            abs_tensor = torch.abs(tensor)

            one_indexes = abs_tensor > thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)

            GaussianCompressor.residuals[name].data[indexes] = 0.0
            values = tensor.data[indexes]

            indexes = indexes.type(torch.IntTensor)
            return indexes, values

    @staticmethod
    def compressbythresholdlong(tensor, thres=0.0):
        with torch.no_grad():
            abs_tensor = torch.abs(tensor)

            one_indexes = abs_tensor > thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            return indexes

    #@staticmethod
    #def compressbythresholdlong(tensor, thres=0.0):
    #    with torch.no_grad():
    #        abs_tensor = torch.abs(tensor)

    #        one_indexes = abs_tensor > thres
    #        indexes = one_indexes.nonzero().data.squeeze().view(-1)
    #        values = tensor.data[indexes]

    #        return indexes, values

    @staticmethod
    def ratio2threshold(tensor, name=None, ratio=0.05):
        with torch.no_grad():
            if name not in GaussianCompressor.residuals:
                GaussianCompressor.residuals[name] = torch.zeros_like(
                    tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.data.add_(GaussianCompressor.residuals[name].data)
            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            GaussianCompressor.residuals[name].data = tensor.data + 0.0

        return float(values[values.numel() - 1].item())

    @staticmethod
    def add2residual(tensor=None, name=None, thrd=None, tk=None):
        with torch.no_grad():
            if name not in GaussianCompressor.residuals:
                GaussianCompressor.residuals[name] = torch.zeros_like(
                    tensor.data)

            tensor.data.add_(GaussianCompressor.residuals[name].data)
            GaussianCompressor.residuals[name].data = tensor.data + 0.0

            abs_tensor = torch.abs(tensor)
            loops = 0
            thres = thrd
            while loops < 5:
                one_indexes = abs_tensor > thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                if indexes.numel() > 4 * tk // 3:
                    thres *= 1.03
                else:
                    break
                loops += 1

            return thres

    @staticmethod
    def k2globalthreshold(tensor, k=0):
        numel = tensor.numel()
        kk = min(numel, k)
        with torch.no_grad():
            values, indexes = torch.topk(torch.abs(tensor.data), k=kk)
            global_threshold = float(values[values.numel() - 1].item())
            values = tensor[indexes]
            #indexes = indexes.type(torch.IntTensor)
        return values, indexes, global_threshold

    @staticmethod
    def ratio2thresholdresidual(tensor, name=None, ratio=0.05):
        with torch.no_grad():
            if name not in GaussianCompressor.residuals:
                GaussianCompressor.residuals[name] = torch.zeros_like(
                    tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(GaussianCompressor.residuals[name].data)

            std = torch.std(tensor)
            mean = torch.mean(tensor)
            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(
                1 - ratio, float(mean), float(std))
            abs_tensor = torch.abs(tensor)
            loops = 0
            while loops < 3:
                one_indexes = abs_tensor > right_thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                if indexes.numel() < 2 * k / 3:
                    right_thres *= 0.5
                elif indexes.numel() > 4 * k / 3:
                    right_thres *= 1.5
                else:
                    break
                loops += 1
            GaussianCompressor.residuals[name].data = tensor.data + 0.0
            GaussianCompressor.residuals[name].data[indexes] = 0.0
        return right_thres

    #@staticmethod
    #def globalratio2threshold(sparse_tensor, ratio=0.05, num_workers=1):
    #    with torch.no_grad():
    #        mean = torch.mean(sparse_tensor)*num_workers
    #        std = torch.std(sparse_tensor)*math.sqrt(num_workers)

    #        print("global mean: ", mean, "global std: ", std)
    #        left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
    #        return right_thres

    @staticmethod
    def globalratio2threshold(sparse_tensor, ratio=0.05, num_workers=1):
        with torch.no_grad():
            mean = torch.mean(sparse_tensor) * num_workers
            std = torch.std(sparse_tensor) * math.sqrt(num_workers)

            print("global mean: ", mean, "global std: ", std)
            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(
                1 - ratio, float(mean), float(std))
            return right_thres

    @staticmethod
    def update_residuals(involved_indexes, name):
        with torch.no_grad():
            #indexes_t = torch.from_numpy(involved_indexes).to(device=GaussianCompressor.residuals[name].device)
            indexes_t = torch.from_numpy(involved_indexes).to(
                device=GaussianCompressor.residuals[name].device).long()
            GaussianCompressor.residuals[name].data[indexes_t] = 0.0

    @staticmethod
    def get_residuals(name, like_tensor):
        if name not in GaussianCompressor.residuals:
            GaussianCompressor.residuals[name] = torch.zeros_like(
                like_tensor.data)
        return GaussianCompressor.residuals[name]

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor
        return z

class HggTopkCompressor():
    name = 'HggTopk'
    # Large-model friendly knobs
    residuals = {}
    prev_thr = {}
    steps = {}
    # per-layer adaptive states
    _stable_count = {}
    _over_scale = {}
    # defaults
    sample_ratio = 0.001      # 0.1% sampling for threshold estimate
    sample_min = 32768        # lower bound of sampled elements
    over_select = 1.20        # select ~20% more then trim to k
    thr_reuse_tol = 0.02      # 2% tolerance on sample for threshold reuse
    max_dense_ratio = 0.20    # fallback to dense if ratio is large
    refresh_interval = 4      # reuse previous threshold for 3 steps; sample on every 4th step
    small_topk_threshold = 2_000_000  # for smaller tensors, direct topk is often faster
    warmup_steps = 0          # disable heavy warmup by default
    warmup_enabled = False

    @staticmethod
    def clear():
        HggTopkCompressor.residuals = {}
        HggTopkCompressor.prev_thr = {}
        HggTopkCompressor.steps = {}
        HggTopkCompressor._stable_count = {}
        HggTopkCompressor._over_scale = {}

    @staticmethod
    def _get_over_scale(name):
        s = HggTopkCompressor._over_scale.get(name, HggTopkCompressor.over_select)
        if s < 1.0: s = 1.0
        if s > 2.0: s = 2.0
        HggTopkCompressor._over_scale[name] = s
        return s

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, **kwargs):
        with torch.no_grad():
            if name not in HggTopkCompressor.residuals:
                HggTopkCompressor.residuals[name] = torch.zeros_like(tensor.data)
            t = tensor.data.add(HggTopkCompressor.residuals[name].data)
            numel = t.numel()
            k = max(int(numel * float(ratio)), 1)

            # Dense fallback when sparsity is low
            if k >= int(numel * HggTopkCompressor.max_dense_ratio):
                HggTopkCompressor.residuals[name].data.zero_()
                return t, None

            # Fast path for smaller tensors: direct topk is efficient on GPU
            if numel <= HggTopkCompressor.small_topk_threshold:
                vals, idx = torch.topk(torch.abs(t), k, sorted=False)
                HggTopkCompressor.residuals[name].data = t + 0.0
                HggTopkCompressor.residuals[name].data[idx] = 0.0
                # store an approximate threshold for reuse bookkeeping
                HggTopkCompressor.prev_thr[name] = float(vals.min().item())
                return t, idx

            abs_t = torch.abs(t)

            # Decide whether to sample this step
            step = HggTopkCompressor.steps.get(name, 0) + 1
            HggTopkCompressor.steps[name] = step
            do_sample = (HggTopkCompressor.refresh_interval <= 1) or (step % HggTopkCompressor.refresh_interval == 0) or (name not in HggTopkCompressor.prev_thr)

            thr = None
            if not do_sample and name in HggTopkCompressor.prev_thr:
                # reuse threshold without sampling
                thr = float(HggTopkCompressor.prev_thr[name])
            else:
                # Sampling on large tensors
                sample_n = min(numel, max(int(numel * HggTopkCompressor.sample_ratio), HggTopkCompressor.sample_min))
                if sample_n < numel:
                    samp_idx = torch.randint(0, numel, (sample_n,), device=t.device)
                    abs_s = abs_t[samp_idx]
                else:
                    abs_s = abs_t
                    sample_n = numel


                over = HggTopkCompressor._get_over_scale(name)
                K_target = min(numel, int(math.ceil(k * over)))
                K_sample = max(1, int(round(K_target * (sample_n / float(numel)))))


                eps = max(1, int(HggTopkCompressor.thr_reuse_tol * K_sample))
                reused = False
                if name in HggTopkCompressor.prev_thr:
                    thr_prev = float(HggTopkCompressor.prev_thr[name])
                    cnt_prev = int((abs_s >= thr_prev).sum().item())
                    if abs(cnt_prev - K_sample) <= eps:
                        thr = thr_prev
                        reused = True
                        HggTopkCompressor._stable_count[name] = HggTopkCompressor._stable_count.get(name, 0) + 1
                    else:
                        HggTopkCompressor._stable_count[name] = 0
                if thr is None:
                    m = min(K_sample, abs_s.numel())
                    vals, _ = torch.topk(abs_s, m, sorted=False)
                    thr = float(vals.min().item())


            mask = abs_t >= thr
            sel_idx = mask.nonzero(as_tuple=False).view(-1)
            sel_before = sel_idx.numel()
            if sel_before > k:
                vals = abs_t[sel_idx]
                _, loc = torch.topk(vals, k, sorted=False)
                sel_idx = sel_idx[loc]

                if sel_before > 4 * k:
                    HggTopkCompressor._over_scale[name] = max(1.0, HggTopkCompressor._get_over_scale(name) * 0.95)
            elif sel_before < k:

                need = k - sel_before
                rem = abs_t.clone()
                if sel_before > 0:
                    rem.index_fill_(0, sel_idx, -1.0)
                _, extra = torch.topk(rem, need, sorted=False)
                sel_idx = torch.cat((sel_idx, extra))

                HggTopkCompressor._over_scale[name] = min(2.0, HggTopkCompressor._get_over_scale(name) * 1.10)

            # Residual compensation
            HggTopkCompressor.residuals[name].data = t + 0.0
            HggTopkCompressor.residuals[name].data[sel_idx] = 0.0
            HggTopkCompressor.prev_thr[name] = float(thr)
            return t, sel_idx

    @staticmethod
    def decompress(tensor, ctc, name=None):
        return tensor

class TopKACompressor(TopKCompressor):
    name = 'topkA'

class TopKACompressor2(TopKCompressor):
    name = 'topkA2'

class TopKSACompressor(TopKCompressor):
    name = 'topkSA'

class gTopKCompressor(TopKCompressor):
    name = 'gtopk'

class GaussianKCompressor(GaussianCompressor):
    name = 'gaussiank'

class GaussianKCCCompressor(GaussianCompressor):
    name = 'gaussiankconcat'

class GaussianKSACompressor(GaussianCompressor):
    name = 'gaussiankSA'

class OKTopKCompressor(GaussianCompressor):
    name = 'oktopk'

class TopKAoptCompressor(GaussianCompressor):
    name = 'topkAopt'

class SpardlCompressor(GaussianCompressor):
    name = 'spardl'


compressors = {
    'hggtopk': HggTopkCompressor,
    'HggTopk': HggTopkCompressor,
    'topkA': TopKACompressor,
    'topkAopt': TopKAoptCompressor,
    'topkA2': TopKACompressor2,
    'topkSA': TopKSACompressor,
    'gtopk': gTopKCompressor,
    'gaussiank': GaussianKCompressor,
    'gaussiankconcat': GaussianKCCCompressor,
    'gaussiankSA': GaussianKSACompressor,
    'oktopk': OKTopKCompressor,
    'spardl': SpardlCompressor,
    'none': NoneCompressor
}

