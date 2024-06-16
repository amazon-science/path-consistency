"""
        SORT: A Simple, Online and Realtime Tracker
        Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.    If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import pickle as pk
from .utils import Video

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


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


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

        self.tracker_hist = {}

    def update(self, frame_idx, bbox_ids, dets=np.empty((0, 5)), return_matched_gt_bbox=False, init_new_track=True):
        """
        Params:
            dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict(frame_idx)[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], frame_idx, bbox_ids[m[0]])

        # create and initialise new trackers for unmatched detections
        if init_new_track:
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i,:], frame_idx, bbox_ids[i])
                self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                if not return_matched_gt_bbox:
                    ret.append(np.concatenate((d,[trk.id+1,-1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
                else:
                    bbox_id, d_ = trk.gt_bbox_path[frame_idx]
                    ret.append(np.concatenate((d_,[trk.id+1,bbox_id])).reshape(1, -1))

                # if a track ever output result, add it to history
                if trk.id not in self.tracker_hist:
                    self.tracker_hist[trk.id] = trk
                trk.output_steps.append(frame_idx)

            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age): # time_since_update = 2
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,6))

def to_det(frame, bbox_id, d):
    return {
        'frame_idx': frame, # frame_idx starts from 1
        'track_id': int(d[4]), # track_idx starts from 1
        'left': d[0],
        'top': d[1],
        'right': d[2],
        'bottom': d[3],
        'width': d[2]-d[0],
        'height': d[3]-d[1],
        'score': 1,
        'bbox_id' : int(bbox_id),
    }

# def parse_out_all_tracking_data(vname, mot_tracker):
#     args = mot_tracker._args
#     def get_det(tkr, t):
#         if args.return_matched_bbox:
#             bbox_id, d = tkr.gt_bbox_path[t] 
#         else:
#             d = tkr.km_bbox_path_post[t]
#             bbox_id = None
        
#         return to_det(t, bbox_id, d)

#     include_minhit_video = Video(vname)
#     all_match_video = Video(vname)
#     for track_id, tkr in mot_tracker.tracker_hist.items():
#         steps = sorted(list(tkr.km_bbox_path_pre.keys()))
        
#         for i, t in enumerate(steps):
#             if (t in tkr.gt_bbox_path):
#                 all_match_video.add_det(get_det(tkr, t))
                

#             if t not in tkr.output_steps:
#                 continue
                        
#             include_minhit_video.add_det(get_det(tkr, t))
#             # if args.return_matched_bbox:
#             #     bbox_id, d = tkr.gt_bbox_path[t] 
#             # else:
#             #     d = tkr.km_bbox_path_post[t]
#             #     bbox_id = None
#             # include_minhit_video.add_det(to_det(t, bbox_id, d))

#             if (i!=0) and (t-1 not in tkr.output_steps): # discontinuous
#                 for j in range(1, 5): # 1, 2, 3, 4:
#                     t_ = t-j
#                     if (t_ in tkr.gt_bbox_path):
#                         assert tkr.state[t_]['hit_streak'] < 3
#                         include_minhit_video.add_det(get_det(tkr, t))
#                     else:
#                         break
#     return include_minhit_video, all_match_video

def _get_args():
    return parse_args([])

def track_one_video(video: Video, args=None, start_frame=None, init_new_track=True, return_matched_bbox=False):
    if args is None:
        args = parse_args([])

    mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    mot_tracker._args = args
    
    output_video = video.create_empty_copy()

    frames = list(video.det_by_frame.keys())
    frames.sort()

    if start_frame is not None:
        loc = frames.index(start_frame)
        frames = frames[loc:]
    else:
        start_frame = frames[0]


    # for frame in range(start_frame-1, video.T):
    for frame in frames:
        # frame += 1 #detection and frame numbers begin at 1
        orig_dets = dets = video.get_frame(frame)
        if len(dets) == 0:
            dets = np.empty((0, 5))
            bbox_ids = np.empty(0)
        else:
            get_ = lambda x : [ float(x[w]) for w in ['left', 'top', 'right', 'bottom', 'score', 'bbox_id'] ]  
            dets = np.array([ get_(det) for det in dets ])

            bbox_ids = dets[:, -1]
            dets = dets[:, :-1]

        init_ = init_new_track or (frame == start_frame)
        trackers = mot_tracker.update(frame, bbox_ids, dets, 
                    init_new_track=init_,
                    return_matched_gt_bbox=return_matched_bbox)

        assert trackers.shape[-1] == 6
        for d in trackers:
            det = to_det(frame, d[-1], d[:-1])
            if return_matched_bbox:
                orig_det = [ det for det in orig_dets if det['bbox_id'] == d[-1] ]
                assert len(orig_det) == 1
                orig_det = orig_det[0]
                det['score'] = orig_det['score']

                assert orig_det['top'] == det['top']
                assert orig_det['left'] == det['left']
                assert orig_det['bottom'] == det['bottom']
                assert orig_det['right'] == det['right']
                assert orig_det['frame_idx'] == det['frame_idx']

                if 'class' in orig_det:
                    det['class'] = orig_det['class']

            output_video.add_det(det)

        if len(mot_tracker.trackers) == 0 and not(init_new_track):
            break

    return output_video

def track_one_video_backward(video: Video, args=None, start_frame=1):
    if args is None:
        args = parse_args(['--return-matched-bbox'])

    mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    
    output_video = Video(video.vname)
    output_video.metadata = video.metadata
    fake_frame_id = 0
    for frame in range(video.T, start_frame-1, -1):
        fake_frame_id += 1
        orig_dets = dets = video.get_frame(frame)
        if len(dets) == 0:
            dets = np.empty((0, 5))
            bbox_ids = np.empty(0)
        else:
            get_ = lambda x : [ float(x[w]) for w in ['left', 'top', 'right', 'bottom', 'score', 'bbox_id'] ]  
            dets = np.array([ get_(det) for det in dets ])

            bbox_ids = dets[:, -1]
            dets = dets[:, :-1]

#         init_ = init_new_track or (frame == start_frame)
        trackers = mot_tracker.update(fake_frame_id, bbox_ids, dets, 
                    init_new_track=True,
                    return_matched_gt_bbox=args.return_matched_bbox)

        assert trackers.shape[-1] == 6
        for d in trackers:
            det = to_det(frame, d[-1], d[:-1])
            orig_det = [ det for det in orig_dets if det['bbox_id'] == d[-1] ]
            assert len(orig_det) == 1
            orig_det = orig_det[0]
            det['score'] = orig_det['score']
            output_video.add_det(det)

            assert orig_det['top'] == det['top']
            assert orig_det['left'] == det['left']
            assert orig_det['bottom'] == det['bottom']
            assert orig_det['right'] == det['right']
            assert orig_det['frame_idx'] == det['frame_idx']

        if len(mot_tracker.trackers) == 0 and not(True):
            break

    return mot_tracker, output_video

def parse_args(inputs=None):
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", help="Maximum number of frames to keep alive a track without associated detections.", 
                                            type=int, default=1)
    parser.add_argument("--min_hits", help="Minimum number of associated detections before track is initialised.", 
                                            type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    parser.add_argument("--result-dir", help="", type=str, default='output')
    # parser.add_argument("--return-matched-bbox", help="", action='store_true')
    if inputs is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(inputs)
    return args

if __name__ == '__main__':
    # all train
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3) #used only for display

    # if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, 'base'), exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, 'include_minhit'), exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, 'all_match'), exist_ok=True)

    pattern = os.path.join(args.seq_path, phase, '*-SDP', 'det', 'det.txt')
    # pattern = os.path.join(args.seq_path, phase, '*-SDP', 'det', 'det.txt')
    for seq_dets_fn in glob.glob(pattern):
        mot_tracker = Sort(max_age=args.max_age, 
                            min_hits=args.min_hits,
                            iou_threshold=args.iou_threshold) #create instance of the SORT tracker
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
        
        with open(os.path.join(args.result_dir, 'base', '%s.txt'%(seq)),'w') as out_file:
            print("Processing %s."%(seq))
            for frame in range(int(seq_dets[:,0].max())):
                frame += 1 #detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
                dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                start_time = time.time()
                trackers = mot_tracker.update(dets, return_matched_gt_bbox=args.return_matched_bbox)
                assert frame == mot_tracker.frame_count
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
                
        fp1 = open(os.path.join(args.result_dir, 'include_minhit', '%s.txt'%(seq)),'w')
        fp2 = open(os.path.join(args.result_dir, 'all_match', '%s.txt'%(seq)),'w')

        ret_include_minhit = []
        ret_all_match = []
        for track_id, tkr in mot_tracker.tracker_hist.items():
            steps = sorted(list(tkr.km_bbox_path_pre.keys()))
            
            for i, t in enumerate(steps):
                if (t in tkr.gt_bbox_path):
                    if args.return_matched_bbox:
                        d = tkr.gt_bbox_path[t] 
                    else:
                        d = tkr.km_bbox_path_post[t]
                    ret_all_match.append([t, tkr.id+1, d])
                    

                if t not in tkr.output_steps:
                    continue
                            
                if args.return_matched_bbox:
                    d = tkr.gt_bbox_path[t] 
                else:
                    d = tkr.km_bbox_path_post[t]
                ret_include_minhit.append([t, tkr.id+1, d])

                if (i!=0) and (t-1 not in tkr.output_steps): # discontinuous
                    for j in range(1, 5): # 1, 2, 3, 4:
                        t_ = t-j
                        if (t_ in tkr.gt_bbox_path):
                            assert tkr.state[t_]['hit_streak'] < 3
                            if args.return_matched_bbox:
                                d = tkr.gt_bbox_path[t_] 
                            else:
                                d = tkr.km_bbox_path_post[t_]
                            ret_include_minhit.append([t_, tkr.id+1, d])
                        else:
                                break
                        
        ret_include_minhit.sort(key=lambda x: x[0])
        for frame, tkr_id, d in ret_include_minhit:
            fp1.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n'%(frame,tkr_id,d[0],d[1],d[2]-d[0],d[3]-d[1]))
        ret_all_match.sort(key=lambda x: x[0])
        for frame, tkr_id, d in ret_all_match:
            fp2.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n'%(frame,tkr_id,d[0],d[1],d[2]-d[0],d[3]-d[1]))

        fp1.close()
        fp2.close()

        with open(os.path.join(args.result_dir, '%s.pk'%(seq)), 'wb') as out_file:
            pk.dump(mot_tracker, out_file)

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))


