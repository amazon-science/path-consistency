import numpy as np
import json
import os
import sys
from collections import defaultdict, OrderedDict
import pandas as pd
from ..home import get_project_base

BASE = get_project_base()

def print_value_dict(d, switch_line=False, prefix=""):
    if prefix:
        string = prefix + " "
    else:
        string = ""
    sep = ", " if not switch_line else "\n"
    for k, v in d.items():
        string += "%s:%.3f%s" % (k, v, sep)
    string = string[:-len(sep)]
    print(string)

def already_finished(logdir):
    fulldir = os.path.join(BASE, logdir)
    if os.path.exists(fulldir) and os.path.exists(os.path.join(fulldir, "FINISH_PROOF")):
        return True
    else:
        return False

def resume_ckpt(cfg, logdir):
    """
    return global_step, ckpt_file
    """
    if cfg.aux.resume == "" or ( not os.path.exists(logdir) ):
        print("No resume, Train from Scratch")
        return 0, None

    elif cfg.aux.resume == "max":

        if already_finished(logdir) and cfg.aux.skip_finished:
            print('----------------------------------------')
            print("Exp %s %s already finished, Skip it!" % (cfg.aux.exp, cfg.aux.runid))
            print('----------------------------------------')
            sys.exit()

        # find the latest ckpt
        ckptdir = os.path.join(logdir, 'ckpts')
        network_ckpts = os.listdir(ckptdir)
        network_ckpts = [ os.path.join(ckptdir, f) for f in network_ckpts ]
        if len(network_ckpts) == 0:
            print("No resume, Train from Scratch")
            return 0, None

        iterations = [ int(os.path.basename(f)[:-4].split("-")[-1]) for f in network_ckpts ]
        load_iteration = max(iterations)
        ckpt_file = os.path.join(ckptdir, "net-%d.pth" % load_iteration )
        print("Resume from", ckpt_file)
        return load_iteration, ckpt_file

    else: # resume is a path to a network ckpt
        assert os.path.exists(cfg.aux.resume)
        assert cfg.split in cfg.aux.resume, (cfg.split , '||', cfg.aux.resume)

        load_iteration = os.path.basename(cfg.aux.resume)
        load_iteration = int(load_iteration.split('.')[0].split('-')[1])
        print("Resume from", cfg.aux.resume)
        return load_iteration, cfg.aux.resume

def resume_wandb_runid(logdir):
    from pathlib import Path

    # if has prev_cfg, try reading from it
    prev_cfg_file = os.path.join(logdir, 'args.json')
    if os.path.exists(prev_cfg_file):
        with open(prev_cfg_file) as fp:
            prev_cfg = json.load(fp)
            if "wandb_id" in prev_cfg['aux']:
                return prev_cfg['aux']['wandb_id']

    # if has wandb folder, try reading from it
    logdir = Path(logdir)
    latest = logdir / "wandb" / "latest-run"
    if latest.exists():
        latest = latest.resolve()
        runid = latest.name.split('-')[-1]
        return runid

    # if none above works
    return None


#====================================================
#====================================================

class Video():

    SEPERATOR = ','
    
    def __init__(self, vname, det_data=None, type_='mot'):
        self.vname = vname
        self.is_empty = True
        self.det_by_frame = {}
        self.det_by_track = {}
        self._unlabeled_frames = set()

        if det_data is not None:
            if type_=='personpath':
                self._load_from_personpath(det_data, allow_no_confidence=True)
            elif type_=='mot':
                self._load_from_mot(det_data)
            elif type_=='saved_anno':
                self._load_from_saved_annotation(det_data)
                self.generate_auxiliary_datastructure()
            elif type_ == 'kitti':
                self._load_from_kitti(det_data)
            else:
                raise ValueError(type_)

            self.is_empty = False

    def __str__(self):
        if not self.is_empty:
            return f"{self.vname}, T={self.T}, #bbox={self.get_num_bbox()}, #track={len(self.det_by_track)}"
        else:
            return f"{self.vname} EmptyObject"
    
    def __repr__(self):
        return str(self)

    def __len__(self):
        return self.T

    def _find_unlabelled_frames(self):
        for frame_idx, dets in self.det_by_frame.items():
            dets = [ d for d in dets if d['track_id'] != -1 ]
            if len(dets) == 0:
                self._unlabeled_frames.add(frame_idx)
        return self._unlabeled_frames
        
    def reorder_track(self, continuous=True):
        orig_ids = sorted(list(self.det_by_track.keys()))
        if continuous:
            self.det_by_track = { i+1: self.det_by_track[j] for i, j in enumerate(orig_ids) }
        else:
            self.det_by_track = { j-orig_ids[0]+1: self.det_by_track[j] for i, j in enumerate(orig_ids) }

    
    def generate_auxiliary_datastructure(self):
        self._generate_framedet_dict()
        self._generate_trackframe_dict()

    def create_empty_copy(self):
        new = Video(self.vname)
        if hasattr(self, 'metadata'):
            new.metadata = self.metadata
        if hasattr(self, '_pp_metadata'):
            new._pp_metadata = self._pp_metadata
        return new


    def _generate_framedet_dict(self):
        self.det_by_frame_bboxid = defaultdict(dict)
        for frame_idx, dets in self.det_by_frame.items():
            for det in dets:
                self.det_by_frame_bboxid[frame_idx][det['bbox_id']] = det

    def _generate_trackframe_dict(self):
        self.det_by_track_frame = defaultdict(dict)
        for track_id, dets in self.det_by_track.items():
            for det in dets:
                self.det_by_track_frame[track_id][det['frame_idx']] = det

    def _generate_bbox_id(self, overwrite=False):
        for frame, dets in self.det_by_frame.items():
            for i, det in enumerate(dets):
                if (det.get('bbox_id', None) is not None) and (not overwrite):
                    raise ValueError()
                det['bbox_id'] = i
    
    
    def add_det(self, dets):
        if not isinstance(dets, list):
            dets = [ dets ]
        
        if len(dets) == 0:
            return

        for det in dets:
            if det['frame_idx'] not in self.det_by_frame:
                self.det_by_frame[det['frame_idx']] = []
            self.det_by_frame[det['frame_idx']].append(det)

            track_id = det['track_id']
            if track_id not in self.det_by_track:
                self.det_by_track[track_id] = []
            self.det_by_track[track_id].append(det)

        self.is_empty = False
        self.T = max(self.det_by_frame)

    def update_track_dict(self):
        self.det_by_track = {}
        for frame_idx, dets in self.det_by_frame.items():
            for det in dets:
                track_id = det['track_id']
                if track_id == -1:
                    continue
                if track_id not in self.det_by_track:
                    self.det_by_track[track_id] = []
                self.det_by_track[track_id].append(det)

    def get_num_bbox(self):
        return sum([ len(dets) for dets in self.det_by_frame.values() ])

    def get_track(self, track_id, simplify=False):
        if not simplify:
            return self.det_by_track[track_id]
        else:
            track = self.det_by_track[track_id]
            return [  (det['frame_idx'], det['bbox_id']) for det in track ]
    
    def get_frame(self, frame_idx, default=[], as_arr=False):
        dets = self.det_by_frame.get(frame_idx, default)
        if as_arr:
            dets = dets2array(dets, keys=['left', 'top', 'right', 'bottom'])
        return dets
    
    def get_frame_det(self, frame_idx, bbox_id):
        return self.det_by_frame_bboxid[frame_idx][bbox_id]

    def get_track_frame(self, track_id, frame_idx):
        return self.det_by_track_frame[track_id][frame_idx]

    def iterbbox(self):
        frames = sorted(self.det_by_frame.keys())
        for f in frames:
            for det in self.det_by_frame[f]:
                yield f, det.get('bbox_id', None), det

    def _load_from_mot(self, seq_dets_fname):
        seq_dets = np.loadtxt(seq_dets_fname, delimiter=self.SEPERATOR)

        det_by_frame = {}
        det_by_track = {}

        if len(seq_dets) == 0:
            self.T = 0
            self.det_by_frame = det_by_frame
            self.det_by_track = det_by_track
            return 

        self.T = frame_num = int(seq_dets[:,0].max()) 
        for frame in range(frame_num):
            frame += 1 # detection and frame numbers begin at 1
            dets = seq_dets[seq_dets[:, 0]==frame]

            frame_detections = []
            for i, det in enumerate(dets):
                frame_idx = int(det[0])
                track_id = int(det[1])
                left = det[2]
                top = det[3]
                right = left + det[4]
                bottom = top + det[5]
                score = det[6]

                detect = {
                    'frame_idx': frame_idx, # frame_idx starts from 1
                    'track_id': track_id, # track_idx starts from 1
                    'left': left,
                    'top': top,
                    'right': right,
                    'bottom': bottom,
                    'width': right-left,
                    'height': bottom-top,
                    'score': score,
                } 

                frame_detections.append(detect)

                if track_id not in det_by_track:
                    det_by_track[track_id] = []
                det_by_track[track_id].append(detect)

            det_by_frame[frame] = frame_detections
        
        self.det_by_frame = det_by_frame
        self.det_by_track = det_by_track

    def _load_from_saved_annotation(self, annotation_or_fname):
        if isinstance(annotation_or_fname, str):
            with open(annotation_or_fname) as fp:
                annotation = json.load(fp)
        else:
            annotation = annotation_or_fname

        self.metadata = annotation['metadata']

        det_by_frame = { int(i): dets for i, dets in annotation['detection'].items() }
        det_by_track = {}

        self.T = max(det_by_frame.keys()) # json file includes frame 0, which is a empty frame

        for frame_idx, dets in det_by_frame.items():
            for i, d in enumerate(dets):
                d['bbox_id'] = int(d.get('bbox_id', -1)) # somehow it is float
                track_id = d['track_id']
                if track_id not in det_by_track:
                    det_by_track[track_id] = []
                det_by_track[track_id].append(d)
        
        self.det_by_frame = det_by_frame
        self.det_by_track = det_by_track

    def _load_from_kitti(self, txt_fname_or_anno):
        if isinstance(txt_fname_or_anno, str):
            with open(txt_fname_or_anno) as fp:
                anno = fp.read().split('\n')[:-1]
        else:
            anno = txt_fname_or_anno

        det_by_frame = defaultdict(list)
        det_by_track = defaultdict(list)
        for line in anno:
            line = line.split(' ')
            cls = line[2]
            # if cls not in ['Pedestrian', 'Car']:
            #     continue
            occluded = line[4] #  Integer (0,1,2,3) indicating occlusion state:
                               #  0 = fully visible, 1 = partly occluded
                               #  2 = largely occluded, 3 = unknown
            # if occluded in [2]:
            #     continue

            fidx = int(line[0]) + 1
            tid = int(line[1])
            b = list(map(float, line[6:10]))
            score = 1
            det = {
                'frame_idx': fidx,
                'track_id' : tid,
                'left': b[0],
                'top': b[1],
                'right': b[2],
                'bottom': b[3],
                'width': b[2]-b[0],
                'height': b[3]-b[1],
                'score': score,
                'class': cls,
                'occlusion': occluded,
            }

            det_by_frame[fidx].append(det)
            det_by_track[tid].append(det)

        self.det_by_frame = { k: v for k, v in det_by_frame.items() }
        self.det_by_track = { k: v for k, v in det_by_track.items() }
        self.T = max(self.det_by_frame)
        self.metadata = {}

    def _load_from_personpath(self, json_fname_or_anno, allow_no_confidence=False):
        if isinstance(json_fname_or_anno, str):
            with open(json_fname_or_anno, 'r') as fp:
                anno = json.load(fp)
        else:
            anno = json_fname_or_anno
        

        det_by_frame = {}
        det_by_track = {}
        self._pp_metadata = anno['metadata'] 

        metadata = anno['metadata']
        metadata = {
            'fps': metadata['fps'],
            'height': metadata['resolution']['height'],
            'width': metadata['resolution']['width'],
        }
        self.metadata = metadata

        for entity in anno['entities']:
            left, top, w, h = entity['bb']
            det = {
                'frame_idx': entity['blob']['frame_idx']+1,
                'track_id' : entity['id'],
                'left': left,
                'top': top,
                'right': left+w,
                'bottom': top+h,
                'width': w,
                'height': h,
            }
            if not allow_no_confidence:
                det['score'] = entity['confidence']
            else:
                det['score'] = entity.get('confidence', -1)
            if 'bbox_id' in entity['blob']:
                det['bbox_id'] = entity['blob']['bbox_id']
            if 'labels' in entity:
                det['_pp_labels'] = entity['labels']

            if det['frame_idx'] not in det_by_frame:
                det_by_frame[det['frame_idx']] = []
            det_by_frame[det['frame_idx']].append(det)

            track_id = det['track_id']
            if track_id not in det_by_track:
                det_by_track[track_id] = []
            det_by_track[track_id].append(det)
        
        self.det_by_frame = det_by_frame
        self.det_by_track = det_by_track
        self.T = max(self.det_by_frame)

        # assert anno['metadata']['number_of_frames'] == self.T, self
    
    def to_saved_anno(self, save_fname):
        data = { 'metadata': self.metadata, "detection": self.det_by_frame }
        with open(save_fname, 'w') as fp:
            json.dump(data, fp)

    def to_mot(self, save_fname=None, untracked='ignore', score=None, class_id=-1):
        string = ""
        frame_indices = list(self.det_by_frame.keys())
        frame_indices.sort()
        for fidx in frame_indices:
            dets = self.get_frame(fidx)
            for i, d in enumerate(dets):
                tid = d['track_id']
                if tid == -1:
                    if untracked == 'ignore':
                        continue
                    elif untracked == 'convert':
                        tid = i+1
                    elif untracked == "":
                        tid = -1
                if score is None:
                    s = d['score']
                else:
                    s = score
                line = "{},{},{},{},{},{},{},{},-1,-1\n".format(d['frame_idx'], tid, d['left'], d['top'], d['width'], d['height'], s, class_id)
                string += line
        # string = string[:-1]
        # string += '\n'
        if save_fname is not None:
            with open(save_fname, 'w') as fp:
                fp.write(string)
        else:
            return string

    def personpath_to_mot(self, save_fname=None):
        class_name_to_class_id = {'person': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4, 'motorbike': 5,
                                'non_mot_vehicle': 6, 'static_person': 7, 'distractor': 8, 'occluder': 9,
                                'occluder_on_ground': 10, 'occluder_full': 11, 'reflection': 12, 'crowd': 13}
        string = ""
        frame_indices = list(self.det_by_frame.keys())
        frame_indices.sort()
        for fidx in frame_indices:
            dets = self.get_frame(fidx)
            for i, d in enumerate(dets):
                tid = d['track_id']
                s = d['score']

                # HACK
                if 'crowd' in d['_pp_labels']:
                    li = 13
                else:
                    assert 'person' in d['_pp_labels'], d
                    li = 1 

                line = "{},{},{},{},{},{},{},{},-1,-1\n".format(d['frame_idx'], tid, d['left'], d['top'], d['width'], d['height'], s, li)
                string += line
        # string = string[:-1]
        # string += '\n'
        if save_fname is not None:
            with open(save_fname, 'w') as fp:
                fp.write(string)
        else:
            return string

    def to_kitti(self, save_fname=None, ignore_untracked=True):
        """
                #Values    Name      Description
        ----------------------------------------------------------------------------
        1    frame        Frame within the sequence where the object appearers
        1    track id     Unique tracking id of this object within this sequence
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                            'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                            'Misc' or 'DontCare'
        1    truncated    Integer (0,1,2) indicating the level of truncation.
                            Note that this is in contrast to the object detection
                            benchmark where truncation is a float in [0,1].
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                            0 = fully visible, 1 = partly occluded
                            2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based index):
                            contains left, top, right, bottom pixel coordinates
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1    score        Only for results: Float, indicating confidence in
                            detection, needed for p/r curves, higher is better.
        """

        _mappping = {
            "car" : "Car",
            "person" : "Pedestrian",
            "bicycle" : "Cyclist",
            "truck" : "Truck",
            "Car" : "Car",
            "Pedestrian" : "Pedestrian",
            "Cyclist" : "Cyclist",
            "Truck" : "Truck",
        }

        string = ""

        frame_indices = list(self.det_by_frame.keys())
        frame_indices.sort()
        for fidx in frame_indices:
            dets = self.get_frame(fidx)
            for i, d in enumerate(dets):
                tid = d['track_id']
                if tid == -1 and ignore_untracked:
                    continue

                cname = d['class']
                cname = _mappping.get(cname, 'DontCare')
                    
                line = f"{fidx-1} {tid} {cname} -1 -1 -1 {d['left']} {d['top']} {d['right']} {d['bottom']} -1 -1 -1 -1000 -1000 -1000 -10 {d['score']}\n"
                string += line
        
        if save_fname is not None:
            with open(save_fname, 'w') as fp:
                fp.write(string)
        else:
            return string

    def to_mm_dataframe(self, untracked='ignore'):
        from motmetrics.io import load_motchallenge
        from io import StringIO
        string = self.to_mot(untracked=untracked)
        string = StringIO(string)
        return load_motchallenge(string)

def clip_bbox(det, H, W):
    y1 = max(0, int(det['top']))
    y2 = min(H, int(det['bottom']))
    x1 = max(0, int(det['left']))
    x2 = min(W, int(det['right']))
    return x1, y1, x2, y2

def filter_small_bbox(video, min_side_len=5, crop_size=None):
    new_video = video.create_empty_copy()
    H = video.metadata['height']
    W = video.metadata['width']
    
    dets = []
    for f, b, det in video.iterbbox():
        x1 = min(H, max(0, int(det['top'])))
        x2 = min(H, max(0, int(det['bottom'])))
        y1 = min(W, max(0, int(det['left'])))
        y2 = min(W, max(0, int(det['right'])))
        
        if x2-x1 <= min_side_len:
            continue
        if y2-y1 <= min_side_len:
            continue
            
        if crop_size:
            h = x2-x1
            w = y2-y1
            resize_factor = min([float(crop_size) / h, float(crop_size) / w])
            resize_shape = [int(h * resize_factor), int(w * resize_factor)]
            if resize_shape[0] == 0 or resize_shape[1] == 0:
                continue
        
        dets.append(det)
    new_video.add_det(dets)
    
    return new_video


def dets2array_v2(dets, keys=['left', 'top', 'right', 'bottom', 'score'], sort_idx=None):
    return dets2array(dets, keys=keys, sort_idx=sort_idx)

def dets2array(dets, keys=['frame_idx', 'left', 'top', 'right', 'bottom'], sort_idx=None):
    if not isinstance(dets, list):
        dets = [ dets ]

    
        
    dets = np.array([[ det[x] for x in keys] for det in dets ])
    if sort_idx is not None:
        dets = dets[dets[:, sort_idx].argsort()]

    return dets
            
def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    o = np.nan_to_num(o, nan=0)
    return(o)  


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def compute_center(dets):
    return np.stack( [(dets[:, 2]-dets[:, 0])/2, (dets[:, 3]-dets[:, 1])/2], axis=1 )


class SlidingWindow():
    def __init__(self, seq_len, wsize, stride=None, exclude_last=False):
        assert not exclude_last

        if stride is None:
            stride = wsize
        self.seq_len = seq_len
        self.wsize = wsize
        self.stride = stride
        self.num_step = np.ceil( ((seq_len - wsize) / stride) + 1 )
        self.num_step = int(self.num_step)

        self.idx = 0

    def __len__(self):
        return self.num_step

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_step:
            self.idx = 0
            raise StopIteration
        
        start = self.idx * self.stride
        end = min(start + self.wsize, self.seq_len)
        self.idx += 1
        return start, end

def get_occlusion_length(track: list):
    f2d = { d['frame_idx']: d for d in track  }
    frames = sorted(f2d.keys())
    olens = []
    for i, f in enumerate(frames):
        if i == 0:
            continue
        if f-1 != frames[i-1]:
            olens.append(f - frames[i-1] - 1)
    
    return olens


def easy_reduce(scores, mode="mean", skip_nan=False):
    assert isinstance(scores, list), type(scores)

    if isinstance(scores[0], list):
        average = []
        L = len(scores[0])
        for i in range(L):
            average.append( easy_reduce([s[i] for s in scores ], mode=mode, skip_nan=skip_nan) )

    elif isinstance(scores[0], np.ndarray):
        assert len(scores[0].shape) == 1
        stack = np.stack(scores, axis=0)
        average = stack.mean(0)

    elif isinstance(scores[0], tuple):
        average = []
        L = len(scores[0])
        for i in range(L):
            average.append( easy_reduce([s[i] for s in scores ], mode=mode, skip_nan=skip_nan) )
        average = tuple(average)

    elif isinstance(scores[0], dict):
        average = {}
        for k in scores[0]:
            average[k] = easy_reduce([s[k] for s in scores], mode=mode, skip_nan=skip_nan)

    elif isinstance(scores[0], float) or isinstance(scores[0], int) or isinstance(scores[0], np.float32): 
        if skip_nan:
            scores = [ x for x in scores if not np.isnan(x) ]

        if mode == "mean":
            average = np.mean(scores)
        elif mode == "max":
            average = np.max(scores)
        elif mode == "median":
            average = np.median(scores)
        elif mode == "sum":
            average = np.sum(scores)
    else:
        raise TypeError("Unsupport Data Type %s" % type(scores[0]) )

    return average

DDM = None
DDM_torch = None
def _generate_dist2diag_matrix(N):
    M = np.zeros([N, N])
    dist = np.arange(N)
    dist = np.concatenate([dist[::-1][:-1], dist], axis=0)
    for i in range(N):
        start = N-1-i
        M[i] = dist[start:start+N] 
    return M

def generate_dist2diag_matrix(N, torch_tensor=False):
    global DDM, DDM_torch
    if torch_tensor:
        import torch
        if DDM_torch is None or DDM_torch.shape[0] < N:
            DDM = generate_dist2diag_matrix(N)
            DDM_torch = torch.FloatTensor(DDM)
            if torch.cuda.is_available():
                DDM_torch = DDM_torch.cuda()
            return DDM_torch
        else:
            return DDM_torch[:N, :N]

    else:
        if DDM is None or len(DDM) < N:
            DDM = _generate_dist2diag_matrix(N)
            return DDM
        else:
            return DDM[:N, :N]



def dicts2df(*args, **kwargs):
    if len(args) == 1:
        dicts = args[0]
    else:
        dicts = kwargs
    names = list(dicts.keys())
    keys = []
    for n, d in dicts.items():
        keys.extend(d.keys())
    keys = list(set(keys))
    keys = sorted(keys)
    # keys = dicts[names[0]].keys()
    # keys = []
    values = [ [dicts[n].get(k, None) for k in keys ] for n in names ]
    return pd.DataFrame(values, columns=keys, index=names)


def torch_safe_log(x, eps=1e-6):
    import torch
    x = torch.clamp_min(x, eps)
    log_x = torch.log(x)
    return log_x

def mot17_to_mot16(vname):
    return f'MOT16-{vname.split("-")[1]}'


def measure_track_occlusion(dets, annotated_frames):
    occlusion = []
    dets = [ d for d in dets if d['frame_idx'] in annotated_frames ]
    dets.sort(key=lambda x: x['frame_idx'])
    if len(dets) == 0:
        return occlusion
    for i in range(len(dets)-1):
        c = dets[i]
        n = dets[i+1]
        
        idx = annotated_frames.index(c['frame_idx'])
        if n['frame_idx'] != annotated_frames[idx+1]:
            occlusion.append( n['frame_idx'] - c['frame_idx'] )

    return occlusion

def measure_video_occlusion(video):
    frames = []
    for fidx, dets in video.det_by_frame.items():
        if len(dets) > 0:
            frames.append(fidx)
    frames.sort()

    olens = []
    for tid, dets in video.det_by_track.items():
        if tid == -1:
            continue
            
        o = measure_track_occlusion(dets, frames)
        
        olens.extend(o)

    return olens


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
