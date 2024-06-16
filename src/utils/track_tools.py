from typing import List
from collections import OrderedDict
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import time
from copy import deepcopy

from .utils import Video
from . import utils

from filterpy.kalman import KalmanFilter

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h        #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox,t,bbox_id):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.gt_bbox_path = { t: (bbox_id, bbox[:4]) }
        self.km_bbox_path_post = { t: self.get_state()[0] }
        self.km_bbox_path_pre = { t: self.get_state()[0] } 
        self.state = {}
        self._record_state(t)
        self.output_steps = []

    def _record_state(self, t):
        state = {
            'time_since_update': self.time_since_update,
            'hit_streak': self.hit_streak,
            'hits': self.hits
        }
        self.state[t] = state

    def update(self,bbox,t,bbox_id):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.gt_bbox_path[t] = (bbox_id, bbox[:4])
        self.km_bbox_path_post[t] = self.get_state()[0]
        self._record_state(t)

    def predict(self, t):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))

        self.km_bbox_path_pre[t] = self.get_state()[0]
        self._record_state(t)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)
    
class Tracklet():

    def __init__(self, track_id, frame_idx, bbox_id, bbox) -> None:
        self.track_id = track_id

        self.appearance_history = [ (frame_idx, bbox_id) ]
        self.track_history = [ (frame_idx, bbox_id) ]
        self.motion_model = KalmanBoxTracker(bbox, frame_idx, bbox_id)
        self.latest_motion_bbox = (frame_idx, bbox)

    def run_motion_tracker(self, frame_idx):
        assert frame_idx == self.latest_motion_bbox[0] + 1, (self.latest_motion_bbox[0], frame_idx)
        bbox = self.motion_model.predict(frame_idx)[0]
        self.latest_motion_bbox = (frame_idx, bbox)

    def get_motion_prediction(self, frame_idx):
        assert frame_idx == self.latest_motion_bbox[0]
        return self.latest_motion_bbox[1]

    def update(self, frame_idx, bbox_id, bbox, update_appearance=False):

        self.track_history.append((frame_idx, bbox_id))
        self.motion_model.update(bbox, frame_idx, bbox_id)   
        if update_appearance:
            self.appearance_history.append((frame_idx, bbox_id))

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
    return(o)  

def create_motion_tfilter(hist_len):

    def tfilter(frame_idx, track):
        thres = max(1, frame_idx - hist_len) # + 1 is hacky for backward compatibility
        fb_list = track.track_history
        fb_list = [ d for d in fb_list if d[0]>=thres ]
        if len(fb_list) == 0:
            return True

        return False

    return tfilter


def create_tfilter(hist_len, len2=None):
    """
    Track filter
    hist_len: remove a track from memory if it is not updated in last {hist_len} frames
    len2: for each track, only use the latest {len2} bboxes to compute similarity
    """

    def tfilter(frame_idx, tracks):
        new_tracks = OrderedDict()
        thres = max(1, frame_idx - hist_len) 
        for tid, track in tracks.items():
            fb_list = track.appearance_history
            fb_list = [ d for d in fb_list if d[0]>=thres ]
            if len(fb_list) == 0:
                continue
                
            if len2 is not None:
                dets = fb_list[-len2:]
                
            new_tracks[tid] = dets

        return new_tracks

    return tfilter

def create_ifilter(score):
    """
    Input filter: filter bbox by confidence score
    """
    ifilter = lambda x: x['score'] <= score
    return ifilter

def generate_weight_func(ratio):
    
    def gen_score(tracks, track_simi_dict):
        track_ids = []
        track_prob = []
        scale = []

        for track_id, p in track_simi_dict.items():
            t_indices = [ x[0] for x in tracks[track_id] ]
            max_ = max(t_indices)
            weight = np.array([ ratio** (max_-t) for t in t_indices ])[:, None]
            track_ids.append(track_id)
            track_prob.append( (weight*p).sum(0) )
            scale.append(weight.sum())
            # thres.append(weight.sum() * null_threshold)

        track_prob = np.stack(track_prob, axis=1) # N, N_\pi, 
        scale = np.array(scale)

        return track_ids, track_prob, scale

    return gen_score

def ct_time(func):
    def new_func(*args, **kwargs):
        self = args[0]
        t1 = time.perf_counter()
        out = func(*args, **kwargs)
        t2 = time.perf_counter()
        self.func_time_ct[func.__name__].append(t2-t1)
        return out
    return new_func

def _min_(fw_simi, bw_simi):
    if isinstance(fw_simi, dict):
        simi = { k: np.min(np.stack([fw_simi[k], bw_simi[k]], axis=0), axis=0) for k in fw_simi }
    else:
        simi = np.min(np.stack([fw_simi, bw_simi], axis=0), axis=0)
    return simi

def _max_(fw_simi, bw_simi):
    if isinstance(fw_simi, dict):
        simi = { k: np.max(np.stack([fw_simi[k], bw_simi[k]], axis=0), axis=0) for k in fw_simi }
    else:
        simi = np.max(np.stack([fw_simi, bw_simi], axis=0), axis=0)
    return simi

def _mean_(fw_simi, bw_simi):
    if isinstance(fw_simi, dict):
        simi = { k: np.mean(np.stack([fw_simi[k], bw_simi[k]], axis=0), axis=0) for k in fw_simi }
    else:
        simi = np.mean(np.stack([fw_simi, bw_simi], axis=0), axis=0)
    return simi

def _geomean_(fw_simi, bw_simi):
    if isinstance(fw_simi, dict):
        simi = { k: np.sqrt(fw_simi[k] * bw_simi[k]) for k in fw_simi }
    else:
        simi = np.sqrt(fw_simi*bw_simi)
    return simi

def _fw_(fw_simi, bw_simi):
    return fw_simi

def _bw_(fw_simi, bw_simi):
    return bw_simi

# different method to fuse temporal-forward and temporal-backward similarity
FuseFunction={ f.__name__[1:-1]: f for f in [ _min_, _max_, _mean_, _geomean_, _fw_, _bw_ ] }


class Tracker():

    def __init__(self, video: Video, 
                    null_threshold, # matching probability threshold to connect new bbox to tracklet
                    direction_fuse='geomean', # how to fuse temporal-forward and temporal-backward matching similarity
                    weight_func=1.0, # optionally give a different weight to each history bbox in a tracklet based on it timestamp
                    activate_thresh = 0.7, # start a new tracklet only if bbox confidence score is larger than the threshold
                    input_filter_func=None, # function to filter bbox from detectors
                    track_filter_func=None, # function to decide the tracklets for object matching with learned model
                    motion_track_filter_func=None, # function to decide the tracklets for motion-based tracker matching
                    motion_iou_thresh = 0.9, # iou threshold to connect new bbox to tracklet
                    mask_invalid_first=False, # if masking out unlikely matching (below threshold) before bipartite matching
                    appear_weight=1, # weight for object matching similarity
                    motion_weight=1, # weight for motion tracker similarity
                    confi_motion_weight=10, # if motion tracker matches a object with high confidence, give a high weight to it.
                    solver='scipy', 
                    ):
        self.video = video
        self.mask_invalid_first = mask_invalid_first

        if weight_func==1 or isinstance(weight_func, float):
            self.weight_func = generate_weight_func(weight_func)
        else:
            self.weight_func = weight_func

        self.tracked_video = video.create_empty_copy()

        self.input_filter_func = input_filter_func
        if input_filter_func is None:
            self.input_filter_func = lambda x: False
        self.activate_thresh = activate_thresh

        self.track_filter_func = track_filter_func
        self.null_threshold = null_threshold 
        if isinstance(direction_fuse, str):
            direction_fuse = FuseFunction[direction_fuse]
        self.direction_fuse = direction_fuse

        self.motion_iou_thresh = motion_iou_thresh
        self.motion_track_filter_func = motion_track_filter_func
        if motion_track_filter_func is None:
            self.motion_track_filter_func = lambda x: False

        self.motion_weight = motion_weight
        self.confi_motion_weight = confi_motion_weight
        self.appear_weight = appear_weight

        self.solver = solver
        if self.solver == 'scipy':
            self.match_func = linear_sum_assignment

        self.func_time_ct = defaultdict(list)
        self.ongoing_tracks = {}
        self.track_history = {}

        self.motion_tracker = {}

        self.match_history = {}

    def det2bbox(self, det):
        bbox = np.array([det['left'], det['top'], det['right'], det['bottom']])
        return bbox

    def compute_motion_simi(self, frame_idx, frame_dets: list, ongoing_tracks: List[Tracklet]):

        predicted_bbox = []
        valid_tracks = []
        motion_predictions = {}
        recent_tracks = []
        for tk in ongoing_tracks:
            if self.motion_track_filter_func(frame_idx, tk):
                continue
            bbox = tk.get_motion_prediction(frame_idx)
            if np.any(np.isnan(bbox)):
                continue

            valid_tracks.append(tk.track_id)
            predicted_bbox.append(bbox)
            motion_predictions[tk.track_id] = bbox

            if frame_idx - tk.track_history[-1][0] <= 4:
                recent_tracks.append(tk.track_id)

        if len(valid_tracks) == 0:
            return {}, {}

        predicted_bbox = np.stack(predicted_bbox, axis=0)
        input_bbox = utils.dets2array(frame_dets, keys=['left', 'top', 'right', 'bottom'])
        # input_bbox = self.det2bbox(frame_dets)
        iou_matrix = iou_batch(predicted_bbox, input_bbox)

        tk_simi_dict = {}
        for i, t in enumerate(valid_tracks):
            if t not in recent_tracks:
                tk_simi_dict[t] = iou_matrix[i]
            else:
                m = iou_matrix[i]
                m[m>0.8] *= self.confi_motion_weight
                tk_simi_dict[t] = m

        return motion_predictions, tk_simi_dict
    
    def compute_appearance_simi(self, frame_indices, frame_dets: dict, ongoing_tracks: dict, fw_prob, bw_prob):
        frame_idx = frame_indices[-1]
        bids = list(frame_dets.keys())
        bids.sort()

        if self.track_filter_func is not None:
            tracks = self.track_filter_func(frame_idx, ongoing_tracks)
        else:
            raise ValueError("Need track_filter_func")

        if len(tracks) == 0:
            return {}, {}

        tracks_reloc = OrderedDict()
        frame2loc = { f:i for i, f in enumerate(frame_indices) }
        for ti, track in tracks.items():
            track = [ (frame2loc[f], b) for f, b in track if f in frame2loc ]
            if len(track) == 0:
                continue
            tracks_reloc[ti] = track
        tracks = tracks_reloc

        fw_prob = fw_prob[..., bids] # T, all, N_latest
        fw_track_simi = self.compute_track_similarity(fw_prob, tracks)
        bw_prob = bw_prob[..., bids]
        bw_track_simi = self.compute_track_similarity(bw_prob, tracks)
        # for tid, tk in tracks.items():
        #     if len(tk) == 0:
        #         import ipdb; ipdb.set_trace()
        track_ids, fw_track_prob, scale1 = self.weight_func(tracks, fw_track_simi)
        track_ids, bw_track_prob, scale2 = self.weight_func(tracks, bw_track_simi)
        assert np.allclose(scale1, scale2)
        track_prob = self.direction_fuse(fw_track_prob, bw_track_prob) # nbbox, ntrack

        tk_simi_dict = { t:track_prob[:, i] for i, t in enumerate(track_ids) }
        scale = { t:scale1[i] for i, t in enumerate(track_ids) }

        return tk_simi_dict, scale



    @ct_time
    def save_track_update(self, frame_idx, frame_dets, match, uobj, window_size=64, ):
        """
        self.ongoing_tracks = { tid: end_det_idx }
        """
        if len(frame_dets) == 0:
            assert len(match) == 0 and len(uobj) == 0, (self.video, frame_idx)
            return 

        to_add = []
        for bi, ti, score in match:
            det = frame_dets[bi]
            det['track_id'] = ti
            update_appearance = score > self.null_threshold
            self.ongoing_tracks[ti].update(frame_idx, bi, self.det2bbox(det), update_appearance=update_appearance)

            to_add.append(det) # frame_dets[bi])

        # remove obsolete tracking
        obsolete_frame = frame_idx - window_size + 1
        keys = list(self.ongoing_tracks.keys())
        for ti in keys:
            track = self.ongoing_tracks[ti]
            start = None
            for i, (f, b) in enumerate(track.track_history):
                if f > obsolete_frame:
                    start = i
                    break
            if start is None:
                del self.ongoing_tracks[ti]
            else:
                track.track_history = track.track_history[i:]
        

        # create new tracklet
        new_det_ct = 0
        for i, bi in enumerate(uobj):
            det = frame_dets[bi]
            if det['score'] <= self.activate_thresh:
                continue
            ti = len(self.tracked_video.det_by_track) + new_det_ct + 1
            det['track_id'] = ti
            new_det_ct += 1

            self.ongoing_tracks[ti] = Tracklet(ti, frame_idx, bi, self.det2bbox(det))
            to_add.append(det)

            # mtk = KalmanBoxTracker(self.det2bbox(det), frame_idx, bi)
            # self.motion_tracker[ti] = mtk

        _tids = [ d['track_id'] for d in to_add ]
        assert len(set(_tids)) == len(_tids), (frame_idx, _tids)

        self.tracked_video.add_det(to_add)

    @ct_time
    def compute_track_similarity(self, prob, tracks):
        """
        prob: T, all, N
        """
        track_simi = {}
        for track_id, track in tracks.items():
            t_indices = [ x[0] for x in track ]
            b_indices = [ x[1] for x in track ]
            p = prob[t_indices, b_indices, :]
            track_simi[track_id] = p

        return track_simi

    @ct_time
    def bipartite_match(self, track_ids, track_prob, scale):
        """
        # track_prob: num_bbox, num_track
        track_ids: list, len = num_track
        """
        thres = self.null_threshold * scale
        if self.mask_invalid_first:
            mask = track_prob <= thres
            track_prob[mask] = -10000000
        
        match_result = []
        _matched_obj = set()
        _matched_tk = set()
        bbox_i, track_j = self.match_func(-track_prob)
        for bi, tj in zip(bbox_i, track_j):
            bi = int(bi)
            true_tj = int(track_ids[tj])
            if track_prob[bi, tj] > thres[tj]:
                match_result.append((bi, true_tj))
                _matched_obj.add(bi)
                _matched_tk.add(true_tj)
        
        N_object, N_track = track_prob.shape
        unmatched_objects = []
        unmatched_tracks = []
        for bi in range(N_object):
            if bi not in _matched_obj:
                unmatched_objects.append(bi)
        
        for i in range(N_track):
            if i not in _matched_tk:
                unmatched_tracks.append(track_ids[i])

        return match_result, unmatched_objects, unmatched_tracks, track_prob

    def _del(self, det, key):
        if key in det:
            del det[key]

    @ct_time
    def track_step(self, fw_prob, bw_prob, frame_indices):
        frame_idx = frame_indices[-1]
        frame_dets = {}

        for det in self.video.get_frame(frame_idx):
            if self.input_filter_func(det):
                continue
            det = det.copy()
            det['track_id'] = -1 # remove gt label
            frame_dets[det['bbox_id']] = det

        bids = list(frame_dets.keys())
        bids.sort()

        for tid, tk in self.ongoing_tracks.items():
            tk.run_motion_tracker(frame_idx) # ensure motion model is updated for each frame

        if len(frame_dets) == 0: # no bbox in this frame
            self.save_track_update(frame_idx, frame_dets, [], bids)
            return

        # get appearance prob
        appearance_simi, scale = self.compute_appearance_simi(frame_indices, frame_dets, self.ongoing_tracks,
                                                fw_prob, bw_prob)

        # get motion prob
        dets = [ frame_dets[b] for b in bids ]
        tracks = list(self.ongoing_tracks.values())
        motion_pred, motion_simi = self.compute_motion_simi(frame_idx, dets, tracks)

        ##################################### combine
        track_ids = list(appearance_simi.keys()) + list(motion_simi.keys())
        track_ids = list(set(track_ids))
        track_ids.sort()
        if len(track_ids) == 0:
            self.save_track_update(frame_idx, frame_dets, [], bids)
            return

        self.match_history[frame_idx] = [ frame_dets, deepcopy(appearance_simi), scale, deepcopy(motion_simi), motion_pred]

        # assert self.mask_invalid_first
        NEG_INF = - 10000000000

        combined_score = np.zeros([len(frame_dets), len(track_ids)])
        mask = np.zeros([len(track_ids)])
        for i, t in enumerate(track_ids):
            if t in appearance_simi:
                s = appearance_simi[t]
                combined_score[:, i] += s * self.appear_weight
                mask[i] = 1
            
            if t in motion_simi:
                s = motion_simi[t]
                combined_score[:, i] += s * scale.get(t, 1) * self.motion_weight
                mask[i] = 1
        combined_score[:, mask==0] = NEG_INF


        # track_prob = np.stack([appearance_simi[t] for t in track_ids], axis=1)
        def get_ascore(t, b):
            if t not in appearance_simi:
                return 0
            return appearance_simi[t][b]/scale[t]

        scale_arr = np.array([scale.get(t, 1) for t in track_ids])
        match, uobj, utk, p = self.bipartite_match(track_ids, combined_score, scale_arr)
        match = [ (bids[bi], ti,  get_ascore(ti, bi)) for bi, ti in match ]
        uobj = [ bids[bi] for bi in uobj ]

        match.sort(key=lambda x: x[0])

        self.save_track_update(frame_idx, frame_dets, match, uobj)

    def track_video(self, alignment_result):
        for frame_idx in range(1, self.video.T+1):
            if frame_idx == 1:
                self.track_step(None, None, [1])
                continue

            data = alignment_result[frame_idx]
            if data is None or (data[0] is None): # no bounding box in this frame
                assert len(self.video.get_frame(frame_idx)) == 0
                frame_indices = [ frame_idx ]
                fw_prob, bw_prob = None, None
            else:
                frame_indices, fw_prob, bw_prob, N_latest = data
                # fw_prob T, all, N+1
                # bw_prob, N, T, all+1
                fw_prob = fw_prob[..., 1:] # T, all, N; remove null object
                bw_prob = bw_prob[..., 1:] # N, T, all; remove null object
                bw_prob = np.transpose(bw_prob, (1, 2, 0))


            self.track_step(fw_prob, bw_prob, frame_indices)
        
        return self.tracked_video

def filter_track_result(tracked_video: Video, filter_short=None, filter_score=None):
    """
    following UNS20, remove tracklet shorter than certain length or bbox lower than certain score.
    """
    if (filter_short is not None) or (filter_score is not None):
        filtered_video = tracked_video.create_empty_copy()
        # if hasattr(tracked_video, 'metadata'):
        #     filtered_video.metadata = tracked_video.metadata
        for track_id, dets in tracked_video.det_by_track.items():
            if (filter_short is not None) and len(dets) < filter_short:
                continue
            for det in dets:
                if (filter_score is not None) and det['score'] < filter_score:
                    continue
                filtered_video.add_det(det)
        return filtered_video
    else:
        return tracked_video

def filter_frames(video: Video, ref):
    new = video.create_empty_copy()
    for f in ref.det_by_frame:
        dets = video.get_frame(f)
        if len(dets) == 0:
            continue
        new.add_det(dets)
    return new

def interpolate_video(video: Video, min_len=1, max_len=None, kitti=False):
    new = video.create_empty_copy()
    num_bbox = defaultdict(int)
    num_bbox.update({ f:len(dets) for f, dets in video.det_by_frame.items() })
    new_dets = []
    for tid, dets in video.det_by_track.items():
        f2det = { d['frame_idx']:d for d in dets }
        frame_indices = sorted(f2det.keys())
        for i, f in enumerate(frame_indices):
            new_dets.append(f2det[f])

            if i == len(frame_indices) - 1:
                continue
                
            fnext = frame_indices[i+1]
            if (min_len < fnext - f <= max_len):     
                jump = fnext - f
                prev_bbox = utils.dets2array(f2det[f], keys=['left', 'top', 'right', 'bottom', 'score'])
                next_bbox = utils.dets2array(f2det[fnext], keys=['left', 'top', 'right', 'bottom', 'score'])
                for j in range(1, jump):
                    prev_weight = float(jump-j) / float(jump)
                    next_weight = float(j) / float(jump)
                    interp_bbox = prev_bbox*prev_weight + next_bbox*next_weight
                    interp_bbox = interp_bbox[0]

                    d = {
                        'frame_idx': j+f,
                        'track_id': tid,
                        'left': interp_bbox[0],
                        'top': interp_bbox[1],
                        'right': interp_bbox[2],
                        'bottom': interp_bbox[3],
                        'width': interp_bbox[2] - interp_bbox[0],
                        'height': interp_bbox[3] - interp_bbox[1],
                        'bbox_id': num_bbox.get(j+f, 0),
                        'score': interp_bbox[4],
                    }

                    if kitti:
                        d['class'] = f2det[f]['class']

                    num_bbox[j+f] += 1
                    new_dets.append(d)

    if len(new_dets) != 0:
        new.add_det(new_dets)
            
    return new









                    



                    



