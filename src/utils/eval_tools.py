import numpy as np
import os
import json
import torch
from collections import defaultdict
from tqdm import tqdm
import pickle
import pandas as pd
import motmetrics as mm
import fnmatch
from pathlib import Path
from ..model import loss_tools
from ..model.model import Model
from .utils import generate_dist2diag_matrix, Video, SlidingWindow, already_finished, to_numpy
from .data_tools import get_input_batch
from .config_tools import load_cfg_json
from ..configs.default import get_cfg_defaults
from ..home import get_project_base

BASE = get_project_base()

def load_mot_gt_dataframe(vnames):
    import motmetrics as mm
    gt = { vname: mm.io.loadtxt(os.path.join(BASE, f'data/mot17/train/{vname}/gt/gt.txt'), fmt='mot15-2D', min_confidence=1) 
                for vname in vnames }
    return gt


def generate_label(video: Video, frame_indices, max_num_bbox):
    """
    (1) label is offsetted by null object
    (2) -1: FN object or padding object

    return: T x max_num_bbox x T
    """
    T = len(frame_indices)
    frame2loc = { f:i for i, f in enumerate(frame_indices) }
    label = np.zeros([T, max_num_bbox, T], dtype=int) - 1
    for track_id, dets in video.det_by_track.items():
        if track_id == -1:
            continue

        dets = [ d for d in dets if d['frame_idx'] in frame2loc ]
        # if len(dets) <= 1:
        #     continue
            
        # generate label of this track
        track = { d['frame_idx']: d['bbox_id'] for d in dets }
        # print(tid)
        # print(track)
        track_label = np.zeros(T, dtype=int)
        for loc, fidx in enumerate(frame_indices):
            if fidx in video._unlabeled_frames:
                track_label[loc] = -1
            elif fidx in track:
                track_label[loc] = track[fidx]+1
                
        # print(track_id, track_label.min())
        for fidx, bid in track.items():
            loc = frame2loc[fidx]
            label[loc, bid] = track_label
            label[loc, bid, loc] = -1
        
    return label

_DIST_GROUP = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 64), (1, 64)]
def eval_association(label, pred, dist_group=None, direction='both'):
    if dist_group is None:
        dist_group = _DIST_GROUP

    def eval(l, p, m):

        labelled_loc = l != -1
        pos_loc = l > 0

        null_gt = (l == 0)
        null_pd = (p == 0)

        null_m = np.logical_and(null_gt, null_pd).sum()
        null_prec =  null_m / (null_pd.sum() + 1e-5)
        null_reca =  null_m / (null_gt.sum() + 1e-5)
        null_f1 = (2 * null_prec * null_reca) / (null_reca + null_prec + 1e-5)

        metrics = {
            "acc": m[labelled_loc].mean(),
            "acc_true": m[pos_loc].mean(),
            "null_prec": null_prec,
            "null_reca": null_reca,
            "null_f1" : null_f1,
            "nullPnt_pred": (null_pd).mean(),
        }

        return metrics

    pred = np.transpose(pred, [0, 2, 1])
    label = np.transpose(label, [0, 2, 1])
    match = (pred == label)

    T = label.shape[0]
    ddm = generate_dist2diag_matrix(T)

    metrics = {}
    for s1, s2 in dist_group:
        if s2 > match.shape[0]:
            continue

        # assert s2 <= match.shape[0], (s1, s2, match.shape)
        dist_loc = np.logical_and( ddm >=s1, ddm <=s2 )
        if direction == 'fw':
            idx = np.tril_indices_from(dist_loc)
            dist_loc[idx[0], idx[1]] = False
        elif direction == 'bw':
            idx = np.triu_indices_from(dist_loc)
            dist_loc[idx[0], idx[1]] = False

        l = label[dist_loc]
        p = pred[dist_loc]
        m = match[dist_loc]
        metric = eval(l, p, m)

        metrics["%02d-%02d"%(s1, s2)] = metric

    
    # HACK
    # metrics['All'] = eval(label, pred, match)

    return metrics

def load_model(ckpt):
    logdir = os.path.dirname(os.path.dirname(ckpt))
    cfg_json = os.path.join(logdir, 'args.json')
    expname, cfg = load_cfg_json(cfg_json, update_from=get_cfg_defaults())

    ### create network #########################################################
    net = Model(cfg)
    state_dict = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(state_dict)
    net.eval()
    net.cuda()
    return cfg, net

def compute_sliding_window_alignment(vname, dataset, net, cfg, to_numpy=True, stride=1, save_logit=False):
    video : Video = dataset[vname][0]

    alignment_data = {}
    for frame_idx in range(1, len(video)+1, stride):
        if frame_idx == 1: # first frame
            continue
        else:
            frame_indices = frame_idx - np.arange(cfg.model.clip_len) * cfg.model.step_size
            frame_indices = frame_indices[::-1]

            data = get_input_batch(dataset, [[video, frame_indices]], include_confi=cfg.model.include_confi)
            if data is None:
                alignment_data[frame_idx] = None
                continue
            bbox_tensor, image_tensor, nbbox_tensor, _ = data
            bbox_tensor = bbox_tensor.cuda()
            image_tensor = image_tensor.cuda()
            nbbox_tensor = nbbox_tensor.cuda()

            prob = net(bbox_tensor, image_tensor, nbbox_tensor)[0] # B=1, T, N, T, N+1
            if save_logit:
                prob = net.alignment_score[0]
            if to_numpy:
                prob = prob.detach().cpu().numpy()
            N_latest = nbbox_tensor[0, -1].item()
            alignment_data[frame_idx] = [ frame_indices, prob, N_latest ] 
    
    return alignment_data

def _simplify_model_outputs(align_data):
    simplified_data = {}
    for t,  data in align_data.items():
        if data is None:
            simplified_data[t] = None
        else:
            (frame_indices, prob, N) = data
            fw = prob[:, :, -1, :N+1] # forward alignment (T, N, T, N+1) -> (T, N, n+1)
            bw = prob[-1, :N] # backward alignment (T, N, T, N+1) -> (n, T, N+1)
            simplified_data[t] = [ frame_indices, fw, bw, N ]
    return simplified_data


simple_metrics = ['idf1',
 'mota',
 'num_switches',
 'num_false_positives',
 'num_misses',
 'num_transfer']


def concat_name(*args):
    return "-".join(map(str, args))

class TrackBundle():

    MH = mm.metrics.create()

    @staticmethod
    def generate_mot_tbundle(vnames, gt=None):
        if gt is None:
            gt = load_mot_gt_dataframe(vnames)
        return TrackBundle(gt)

    def _mot_eval_func(self, gt, tracked):
        r = mm.utils.compare_to_groundtruth(gt, tracked.to_mm_dataframe(), 'iou', distth=0.5)
        return r

    def __init__(self, gt_dict=None, eval_func=None):
        self.gt_dict = gt_dict
        self._tracked_dict = {}
        self._r_dict = {}
        self.metric_df = None
        if eval_func is None:
            self._eval_func = self._mot_eval_func
        else:
            self._eval_func = eval_func

    def keys(self):
        return list(self._tracked_dict.keys())

    def has_tracked(self, name):
        if name in self._tracked_dict:
            return True
        else:
            return False

    def add(self, vname, tracked):
        self._tracked_dict[vname] = tracked
        if vname in self._r_dict:
            del self._r_dict[vname]

    def get_video(self, vname):
        return self._tracked_dict[vname]

    def get_r(self, vname, force=False):
        if force or (vname not in self._r_dict):
            tracked = self.get_video(vname)
            r = self._eval_func(self.gt_dict[vname], tracked)
            self._r_dict[vname] = r
        return self._r_dict[vname]

    def get_eval_dataframe(self, force=False, ignore_unlabel=False):
        if force or (self.metric_df is None):
            rs = []
            vnames = []
            for vname in self._tracked_dict:
                if (vname not in self.gt_dict) and ignore_unlabel:
                    continue
                vnames.append(vname)
                rs.append(self.get_r(vname, force=True))
            df = self.MH.compute_many(rs, names=vnames, metrics=simple_metrics, generate_overall=True)
            self.complete_metric_df = df
            self.metric_df = df.iloc[-1:]

        return self.metric_df

    def save(self, folder, vnames=None, metric_fname='metric'):
        if vnames is None:
            vnames = list(self._tracked_dict.keys())

        for vname in vnames:
            fname = os.path.join(folder, vname+'.txt')
            video = self.get_video(vname)
            video.to_mot(fname)
        
        if metric_fname:
            metric_csv = os.path.join(folder, metric_fname+'.csv')
            self.get_eval_dataframe()
            self.complete_metric_df.to_csv(metric_csv)

# ===================================
class ExpRun():

    def __init__(self, expfolder):
        self.expfolder = expfolder
        self.run_id = os.path.basename(expfolder)
        self.expname = os.path.basename(os.path.dirname(expfolder))
        saves = expfolder + '/saves/'
        save_jsons = [ saves+fname for fname in os.listdir(saves) if fname.endswith('.json') ]
        self.asso_acc = {}
        self.iterations = []
        for fn in save_jsons:
            iteration = int(fn.split('-')[-1][:-5])
            with open(fn) as fp:
                metrics = json.load(fp)
            self.asso_acc[iteration] = metrics
            self.iterations.append(iteration)
        self.iterations.sort()

    @property
    def finished(self):
        return already_finished(self.expfolder)

    def __str__(self):
        return f"{self.expname}, {self.run_id}"

    def __repr__(self):
        return str(self)

    def get_last_ckpt(self):
        i = max(self.iterations)
        return f"{self.expfolder}/ckpts/net-{i}.pth"
    
    def select(self, metric, return_, patience=0.0):
        """
        return: asso, ckpt, inference
        """
        i, m = select_max(self.asso_acc, metric, patience)
        if return_ =='accuracy':
            return m
        elif return_ == 'ckpt_path':
            return f"{self.expfolder}/ckpts/net-{i}.pth"
        else:
            raise ValueError(return_)

def select_max(iteration_metric_dict, key, patience=0.0):
    max_ = None
    max_iteration = None
    
    iterations = sorted(iteration_metric_dict)
    start_step = int( max(iterations) * patience )

    # iterations = [i for i in iterations if i >=start_step]
    for iteration, mdict in iteration_metric_dict.items():
        if iteration < start_step:
            continue
            
        if max_ is None or mdict[key] > max_:
            max_ = mdict[key]
            max_iteration = iteration
    
    return max_iteration, iteration_metric_dict[max_iteration]
