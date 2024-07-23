import argparse
import os
import torch
from tqdm import tqdm

from .home import get_project_base
from .utils.data_tools import Dataset
from .utils import eval_tools 
from .utils import track_tools
from .utils import video_names
from .utils import utils
import pickle
from .utils import trackeval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, type=str)
    parser.add_argument('--fdr', default='fcos_processed', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--dst', default='output', type=str)
    args = parser.parse_args()

    BASE = get_project_base()

    try:
        torch.cuda.set_device('cuda:%s'%args.gpu)
    except Exception as e:
        print(e)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    os.makedirs(os.path.join(args.dst, 'PersonPath22'), exist_ok=True)

    expfolder = args.exp
    exp = eval_tools.ExpRun(expfolder)
    metric = '01-08.acc' # select a checkpoint based on association accuracy on training data
    ckpt = exp.select(metric, 'ckpt_path', patience=0.3) # patience: avoid selecting any of first 30% checkpoints
    cfg, net = eval_tools.load_model(ckpt)
    net.eval()

    vnames = video_names.PERSONPATH_TEST_VIDEOS
    dataset = Dataset('personpath22', args.fdr, '', vnames, training=False)

    trackeval_dataset = trackeval.create_personpath_dataset(vnames)
    eval_bundle = trackeval.PersonPath_TrackBundle(trackeval_dataset)

    with torch.no_grad():
        for vname in tqdm(vnames):
            data = eval_tools.compute_sliding_window_alignment(vname, dataset, net, cfg, save_logit=False)
            data = eval_tools._simplify_model_outputs(data)

            video = dataset.get_video(vname)
            tracker = track_tools.Tracker(video, 0.6, 
                                                activate_thresh=0.7, 
                                                direction_fuse='geomean', 
                                                input_filter_func=track_tools.create_ifilter(0.0),
                                                weight_func=1,
                                                track_filter_func=track_tools.create_tfilter(47, len2=8),
                                                motion_track_filter_func=track_tools.create_motion_tfilter(30),
                                                motion_iou_thresh=0.8,
                                                motion_weight=1, 
                                                confi_motion_weight=10)

            tracked = tracker.track_video(data)
            tracked = track_tools.filter_track_result(tracked, filter_short=20)
            eval_bundle.add(vname, tracked)

            tracked.to_mot(os.path.join(args.dst, 'PersonPath22', vname + '.txt'))

    print(eval_bundle.get_res())



