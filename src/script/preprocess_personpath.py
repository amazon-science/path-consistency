import json
from scipy.optimize import linear_sum_assignment
import os
import skimage
import pickle
import numpy as np
import multiprocessing
import argparse
from tqdm import tqdm
import pickle as pk
from collections import defaultdict
import shutil
import configparser

from ..utils.utils import Video, dets2array, iou_batch, compute_center, filter_small_bbox
from ..utils import video_names 
from ..utils.data_augmentation import crop_and_resize_image
from ..utils.track_tools import filter_track_result


CROP_SIZE = 64


def get_frame_fname(frame_idx):
    s = str(frame_idx)
    while len(s) < 7:
        s = '0' + s
    return s + '.jpg'

def extract_image_patch(video, img_path, save_path, scale):
    image_dict = {}
    for frame_idx, dets in video.det_by_frame.items():
        if len(dets) == 0:
            continue

        img = os.path.join(img_path, get_frame_fname(frame_idx))
        img = skimage.io.imread(img)

        image_dict[frame_idx] = {}
        for det in dets:
            image_dict[frame_idx][det['bbox_id']] = crop_and_resize_image(img, det, scale=scale)

    with open(save_path, 'wb') as f:
        pickle.dump(image_dict, f)

def compute_detdistance(video):
    N = max([ len(dets) for dets in video.det_by_frame.values() ])
    dist_rank_matrix = np.zeros([len(video), N, N]) - 1
    dist_rank_matrix = dist_rank_matrix.astype(int)
    for frame_idx in range(1, len(video)+1):
        dets = video.get_frame(frame_idx)
        if len(dets) == 0:
            continue
        dets.sort(key=lambda x: x['bbox_id'])

        detarr = dets2array(dets)[:, 1:]
        centers = compute_center(detarr) # n, 2
        distance = (centers[:, None] - centers[None, :]) ** 2
        distance = distance.sum(-1) # N, N

        dist_rank = np.argsort(distance, axis=1)
        n = len(dets)
        dist_rank_matrix[frame_idx-1][:n, :n] = dist_rank
    
    return dist_rank_matrix

def add_label(video: Video, gt: Video):
    
    for frame_idx, dets in video.det_by_frame.items():
        if len(dets) == 0:
            continue
        detsarr = dets2array(dets)[:, 1:]
        gtdets = gt.get_frame(frame_idx)
        if len(gtdets) == 0:
            continue
        gtsarr = dets2array(gtdets)[:, 1:]
        iou = iou_batch(detsarr, gtsarr)
        det_idx, gt_idx = linear_sum_assignment(-iou)

        for d, g in zip(det_idx, gt_idx):
            score = iou[d, g]
            if score < 0.5:
                dets[d]['track_id'] = -1
            else:
                dets[d]['track_id'] = gtdets[g]['track_id']
            
    det_by_track = {}
    for frame_idx, dets in video.det_by_frame.items():
        for det in dets:
            track_id = det['track_id']
            if track_id not in det_by_track:
                det_by_track[track_id] = []
            det_by_track[track_id].append(det)
    video.det_by_track = det_by_track

    return video

def preprocess_video(vname, scales=[1, 2]):
    print(vname)
    mot_fname = os.path.join(BASE_PATH, f'{vname}.mp4.json')
    gt_file = f'data/personpath/annotation/anno_amodal_2022/{vname}.mp4.json'
    save_folder = SAVE_PATH

    save_fname = os.path.join(save_folder, f'{vname}.json')
    if os.path.exists(save_fname):
        video = Video(vname, save_fname, 'saved_anno')
    else:
        video = Video(vname, mot_fname, 'personpath')

        video = filter_track_result(video, filter_score=0.5)
        video = filter_small_bbox(video, crop_size=CROP_SIZE)
        video._generate_bbox_id(overwrite=True)
        
        gt = Video(vname, gt_file, 'personpath')
        video = add_label(video, gt)
        video.to_saved_anno(save_fname)

    distM = compute_detdistance(video)
    np.save(os.path.join(save_folder, f'{vname}_dist_rank_matrix.npy'), distM)

    img_path = f'data/personpath/frames/{vname}/'
    for s in scales:
        save_fname = os.path.join(save_folder, f'{vname}-scale{s}.pk')
        if not os.path.exists(save_fname):
            extract_image_patch(video, img_path, save_fname, scale=s)

def combine_pkl_to_array(dataset_full_name, all_files, N, save_folder, scale):

    for I in range(N):
        save_fname = '%s-partition%02d-scale%d' % (dataset_full_name, I, scale)
        print(save_fname)
        step = int((len(all_files) + N - 1 ) / N)

        files = all_files[I*step:I*step+step]
        array = []
        image_loc = {}
        for fname in tqdm(files):
            with open(fname, 'rb') as fp:
                images = pk.load(fp)

            vname = os.path.basename(fname)[:-3] # remove extension
            vname = '-'.join(vname.split('-')[:-1])
            print(vname)
            image_loc[vname] = defaultdict(dict)
            
            frame_indices = list(images.keys())
            frame_indices.sort()

            for frame_idx in frame_indices:
                bbox_ids = list(images[frame_idx].keys())
                bbox_ids = sorted(bbox_ids)
                for bbox_id in bbox_ids:
                    img = images[frame_idx][bbox_id] # H, W, C
                    img = np.transpose(img, (2, 0, 1)) # C, H, W
                    array.append(img)
                    image_loc[vname][frame_idx][bbox_id] = (save_fname, len(array)-1)

        array = np.stack(array, axis=0)

        nmp_meta = {
            'shape' : array.shape,
            'dtype' : str(array.dtype),
            'fname' : save_fname,
            'image_loc': image_loc,
        }

        mm_obj = np.memmap(os.path.join(save_folder, f'{save_fname}.nmp'), 
                        dtype=str(array.dtype), mode="w+", shape=array.shape)
        mm_obj[:] = array[:]

        with open(os.path.join(save_folder, f'{save_fname}.nmp_meta'), 'w') as fp:
            json.dump(nmp_meta, fp)

BASE_PATH = None
SAVE_PATH = None

def compute_naive_path(vname):
    from ..utils import byte_track
    args = byte_track._get_args()

    print(vname)
    label = os.path.join(SAVE_PATH, f'{vname}.json')
    video = Video(vname, label, type_='saved_anno')

    tracked_video = byte_track.track_one_video(video, args, return_matched_bbox=True)

    savefname = os.path.join(SAVE_PATH, f"{vname}.naive_path")
    tracks = []
    for track_id, dets in tracked_video.det_by_track.items():
        tracks.append({ d['frame_idx']:int(d['bbox_id']) for d in dets })
    with open(savefname, 'w') as fp:
        json.dump(tracks, fp)

if __name__ == '__main__':

    BASE_PATH = './data/personpath22/fcos_det/amazon_format'
    SAVE_PATH = './data/personpath22/fcos_processed'
    os.makedirs(SAVE_PATH, exist_ok=True)

    print(BASE_PATH)
    print(SAVE_PATH)

    vnames = video_names.PERSONPATH_TRAIN_VIDEOS + video_names.PERSONPATH_TEST_VIDEOS

    p = multiprocessing.Pool(54)
    p.map(preprocess_video, vnames)
    p.close()

    p = multiprocessing.Pool(48)
    p.map(compute_naive_path, vnames)
    p.close()

    for scale in [1, 2]:
        pks = [ os.path.join(SAVE_PATH, f'{vname}-scale{scale}.pk') for vname in vnames ]
        combine_pkl_to_array('personpath22', pks, 10, SAVE_PATH, scale)

    print('You can remove all data/personpath22/fcos_processed/{vname}-scale{x}.pk files to save storage')
