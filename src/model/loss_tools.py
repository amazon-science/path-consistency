import numpy as np
from collections import OrderedDict, defaultdict
import torch
from typing import List
from ..utils.data_tools import ObjectSample

def logsumexp(x, dim=0):
    c = x.max()
    return c + torch.log(torch.exp(x - c).sum(dim=dim))

_dp_dict = {}
def exhaustive_sample(L):
    if L == 1:
        return [[1]]
    
    if L in _dp_dict:
        return _dp_dict[L]
    
    combinations = [[L]]
    for i in range(1, L):
        steps = exhaustive_sample(L-i)
        for s in steps:
            s = s[:]
            s.insert(0, i)
            combinations.append(s)
    _dp_dict[L] = combinations
    return combinations

def sample_step_combination(L: int, min_size=1, max_size=64):
    start_idx = 0
    step_combination = []
    while start_idx < L-1:
        if L-start_idx-1 <= min_size:
            step = L-start_idx-1
        else:
            step = np.random.randint(min_size, min(max_size, L-start_idx-1)+1)
        step_combination.append(step)
        start_idx = start_idx + step
    
    assert sum(step_combination) == L-1, step_combination
    return tuple(step_combination)

def sample_func(path_length, G, max_step_size=64):
    L = path_length
    max_num_combination = 2**(L-1)
    if max_num_combination <= G:
        combinations = exhaustive_sample(L-1)
    else:
        combinations = set()
        for g in range(G):
            if g == 0:
                step_combination = tuple([ 1 ] * (L-1))
            else:
                step_combination = sample_step_combination(L, max_size=max_step_size)
                try_count = 1
                while step_combination in combinations:
                    step_combination = sample_step_combination(L, max_size=max_step_size) 
                    try_count += 1
                    if try_count == 10: # avoid dead loop
                        break
            combinations.add(step_combination)
    return combinations

def _default_M_gen(video, scale=2):
    t = 0
    for frame, dets in video.det_by_frame.items():
        if len(dets) > 0:
            t += 1

    m = np.sqrt(video.get_num_bbox()/t)
    m = max(1, int(m * scale))
    return m

class PathWalker():

    def __init__(self, sample: ObjectSample, M, frame2tensorloc):
        self.sample = sample
        self.pathloc_to_info = {}
        self.frame_indices = sample.frame_indices
        self.frame2tensorloc = frame2tensorloc
        self.M = M

        for i, fidx in enumerate(self.frame_indices):
            bbox_ids = sample.fidx2bboxes[fidx]
            if i == 0:
                bbox_ids = [ bbox_ids[0]+1 ] # query object; offset for null object
            else:
                bbox_ids = self._spatial_constraint_masking(bbox_ids)

            assert len(bbox_ids) > 0, (fidx, sample.video)
            tloc = frame2tensorloc[fidx]
            self.pathloc_to_info[i] = [ fidx, tloc, bbox_ids ]

    def __len__(self):
        return len(self.sample)

    def compute_association_for_path(self, log_alignment_prob: torch.Tensor, path: list):
        l = 0 # current location
        output_logprob_history = {}
        internal_logprob_history = {}
        internal_logprob = None
        for i, step_size in enumerate(path): 
            internal_logprob, output_logprob = self.update_prob(l, l+step_size, log_alignment_prob, internal_logprob)
            fend = self.pathloc_to_info[l+step_size][0]
            internal_logprob_history[fend] = internal_logprob
            output_logprob_history[fend] = output_logprob

            l = l+step_size

        assert (l + 1) == len(self.sample)

        return output_logprob_history 

    def _spatial_constraint_masking(self, bbox_ids):
        bbox_ids = sorted([ b+1 for b in bbox_ids[:self.M] ]) 
        bbox_ids.insert(0, 0) # include null object
        return bbox_ids
    
    def update_prob(self, start, end, pairwise_align_logprob, track_logprob):
        fstart, tloc_start, start_object_bbox_ids = self.pathloc_to_info[start] # len N
        fend, tloc_end, end_object_bbox_ids = self.pathloc_to_info[end] # len M

        start_object_bbox_ids = [ int(i-1) for i in start_object_bbox_ids if i != 0 ] # remove the offset for null object
        indices = [ [], [], [], [] ] # T, N, T_, N_
        for bbox_i in start_object_bbox_ids:
            indices[1].extend([int(bbox_i)] * len(end_object_bbox_ids))
            indices[3].extend(end_object_bbox_ids)

        logprob = pairwise_align_logprob[tloc_start, indices[1], tloc_end, indices[3]]
        logprob = logprob.view(len(start_object_bbox_ids), len(end_object_bbox_ids)) # N, M

        if track_logprob is None:
            assert len(start_object_bbox_ids) == 1
            internal_track_logprob = logprob[0, 1:] # internal_track_logprob prevents association to null objects
            output_track_logprob = logprob[0] # output_track_logprob allows association to null objects
        else:
            track_logprob = track_logprob.unsqueeze(-1) # N, 1
            logprob = track_logprob + logprob # N, M
            track_logprob = logsumexp(logprob) # N, M -> M
            internal_track_logprob = track_logprob[1:]
            output_track_logprob = track_logprob

        return internal_track_logprob, output_track_logprob

def entropy_loss_from_logprob(logprob_dict):
    entropy_list = []
    for t, path_logprob in logprob_dict.items():
        if len(path_logprob) == 0:
            continue
        path_logprob = torch.stack(path_logprob, dim=0) # G, N+1
        c = path_logprob.max()
        path_logprob = c + torch.log(torch.exp(path_logprob - c).mean(dim=0))

        # compute path_prob and normalize it
        path_logprob_sum = logsumexp(path_logprob)
        path_logprob_normed = path_logprob - path_logprob_sum
        path_prob_normed = torch.exp(path_logprob_normed)

        entropy = - (path_prob_normed * path_logprob).sum()
        entropy_list.append(entropy)
    return entropy_list

def average_entropy(entropy_list, device):
    total_elist = []
    for e in entropy_list:
        total_elist.extend(e)
    if len(total_elist) == 0:
        return torch.tensor(0, device=device)
    else:
        return sum(total_elist) / len(total_elist)

def pcl(log_alignment_prob, samples, cfg, frame_indices):

    frame2tensorloc = { f:i for i, f in enumerate(frame_indices) }
    G = cfg.pcl.G
    interm_factor = cfg.pcl.ift # this will sample {ift} locations between start/end frames 
                                # and enforces consistent association probability at these locations
                                # e.g. ift=1 will require consistency at end frames, ift=2 requires consistency at the middle and end frames, ....
                                # a larger ift speeds up training.
    entropy_list = []


    for sample in samples:
        L = len(sample)
        m = _default_M_gen(sample.video)
        walker = PathWalker(sample, m, frame2tensorloc)
        path_list = sample_func(len(walker.sample), G, max_step_size=cfg.pcl.max_s)
        
        supervised_loc = []
        for i in range(1, min(interm_factor, L)):
            l = int(L * i / interm_factor)
            t, _ = walker.sample.get(pathloc=l) 
            supervised_loc.extend([t-1, t, t+1])
        t, _ = walker.sample.get(pathloc=L-1)
        supervised_loc.append(t)

        path_logprob_dict = defaultdict(list)
        for path in path_list:
            logp_dict = walker.compute_association_for_path(log_alignment_prob, path)
            for loc, v in logp_dict.items():
                if loc in supervised_loc:
                    path_logprob_dict[loc].append(v)

        elist = entropy_loss_from_logprob(path_logprob_dict)
        entropy_list.append(elist)

    return average_entropy(entropy_list, log_alignment_prob.device)

def mreg_loss(cfg, alignment_prob):
    num_matched = alignment_prob.sum(2) # B, T, N, T, N+1 -> B, T, T, N+1
    num_matched = torch.relu(num_matched - 1)

    # process null
    weight = num_matched.new_ones(num_matched.shape[-1])
    weight[0] = cfg.mreg.nullw
    num_matched = num_matched * weight

    num_matched = num_matched[num_matched>0]
    if num_matched.numel() > 0:
        l = torch.square(num_matched).mean()
    else:
        l = torch.tensor(0, device=alignment_prob.device)
    
    return l
