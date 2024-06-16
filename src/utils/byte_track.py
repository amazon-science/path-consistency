import numpy as np
import pickle as pk
from .utils import Video
from . import video_names
from .byte_track_utils import BYTETracker
import argparse

def det2arr(d):
    return np.array([d[k] for k in ['left', 'top', 'right', 'bottom', 'score']])

def track_one_video(video: Video, args=None, return_matched_bbox=False):
    if args is None:
        args = _get_args()
    tracker = BYTETracker(args)
    # results = []
    tracked_dets = []
    
    vname = video.vname
    frames = list(video.det_by_frame.keys())
    frames.sort()

    # for frame in range(start_frame-1, video.T):
    for frame_idx in frames:
        # print(frame_idx)
        _dets = video.get_frame(frame_idx)
        if len(_dets) > 0:
            dets = [ det2arr(d) for d in _dets ]
            dets = np.stack(dets, axis=0)
        else:
            dets = np.zeros([0, 5])
        if return_matched_bbox:
            bbox_ids = np.array([ d['bbox_id'] for d in _dets ], dtype=int)
        else:
            bbox_ids = None

        info_imgs = [ 1, 1, frame_idx, vname, None ]
        img_size = (1, 1)

        online_targets = tracker.update(dets, info_imgs, img_size, bbox_ids=bbox_ids)
        for t in online_targets:
            if (frame_idx != 1) and return_matched_bbox:
                tlwh = t.orig_bbox.tlwh
                bid = t.orig_bbox.bid
            else:
                tlwh = t.tlwh
                bid = t.bid
               
            tid = t.track_id
            # print(tid)
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                if tid < 0:
                    tid = -1
                
                x1, y1, w, h = tlwh
                x1=round(x1, 1)
                y1=round(y1, 1) 
                w=round(w, 1)
                h=round(h, 1) 
                score=round(t.score, 2)

                det = {
                        'frame_idx': frame_idx,
                        'track_id': tid, 
                        'left': x1,
                        'top': y1,
                        'right': x1+w,
                        'bottom': y1+h,
                        'width': w,
                        'height': h,
                        'score': score,
                        'bbox_id': int(bid),
                    }
                tracked_dets.append(det)
    
    tracked = video.create_empty_copy()
    tracked.add_det(tracked_dets)

    return tracked

def interpolate(video, n_min=5, n_dti=20):
    new_dets = []
    for track_id, dets in video.det_by_track.items():
        dets = sorted(dets, key=lambda x: x['frame_idx']) 
        frames = [ d['frame_idx'] for d in dets ]
        new_dets.extend(dets)

        if len(dets) <= n_min:
            continue

        for i in range(0, len(frames)):
            right_frame = frames[i]
            if i > 0:
                left_frame = frames[i - 1]
            else:
                left_frame = frames[i]
            # disconnected track interpolation
            if 1 < right_frame - left_frame < n_dti:
                num_bi = int(right_frame - left_frame - 1)
                right_bbox = det2arr(dets[i])
                left_bbox = det2arr(dets[i-1])
                # right_bbox = tracklet[i, 2:6]
                # left_bbox = tracklet[i - 1, 2:6]
                for j in range(1, num_bi + 1):
                    curr_frame = j + left_frame
                    curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / \
                                (right_frame - left_frame) + left_bbox
                    d = {
                            'frame_idx': curr_frame,
                            'track_id': track_id, 
                            'left': curr_bbox[0],
                            'top': curr_bbox[1],
                            'right': curr_bbox[2],
                            'bottom': curr_bbox[3],
                            'width': curr_bbox[2]-curr_bbox[0],
                            'height': curr_bbox[3]-curr_bbox[1],
                            'score': curr_bbox[-1],
                        }
                    new_dets.append(d)

    new_video = video.create_empty_copy()
    new_video.add_det(new_dets)
    return new_video

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # det args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

_byte_args = None
def _get_args():
    return make_parser().parse_args()
    # global _byte_args
    # if _byte_args is None:
    #     from ..home import get_project_base
    #     BASE = get_project_base()
        
    #     with open(f'{BASE}/libs/ByteTrack/args.pk', 'rb') as fp:
    #         _byte_args = pk.load(fp)

    # return argparse.Namespace(**vars(_byte_args))

if __name__ == '__main__':
    with open('/home/cvsci/zilu/ByteTrack/args.pk', 'rb') as fp:
        args = pk.load(fp)

    for vname in video_names.TRAIN_MOT_VIDEOS:
        fname = f'./data/mot17/train/{vname}/det/det.txt'
        video = Video(vname, fname)
        print(vname)
        tracked = track_one_video(video, args)
        tracked = interpolate(tracked, n_min=5, n_dti=20)
        tracked.to_mot(f'./test/{vname}.txt')


