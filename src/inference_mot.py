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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, type=str)
    parser.add_argument('--fdr', default='public', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--dst', default='output', type=str)
    args = parser.parse_args()

    BASE = get_project_base()

    try:
        torch.cuda.set_device('cuda:%s'%args.gpu)
    except Exception as e:
        print(e)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    expfolder = args.exp
    exp = eval_tools.ExpRun(expfolder)
    metric = '01-08.acc' # select a checkpoint based on association accuracy on training data
    ckpt = exp.select(metric, 'ckpt_path', patience=0.3) # patience: avoid selecting any of first 30% checkpoints
    cfg, net = eval_tools.load_model(ckpt)
    net.eval()

    # SDP
    DET = 'SDP'
    vnames = video_names.get_MOT_ALL(DET)
    dataset = Dataset('mot17', args.fdr, DET, vnames, training=False)

    for vname in tqdm(vnames):
        with torch.no_grad():
            align_data = eval_tools.compute_sliding_window_alignment(vname, dataset, net, cfg, save_logit=False)
            align_data = eval_tools._simplify_model_outputs(align_data)

        video = dataset.get_video(vname)
        tracker = track_tools.Tracker(video, 0.3, 
                                                activate_thresh=0.0, 
                                                direction_fuse='geomean', 
                                                mask_invalid_first=False, 
                                                input_filter_func=None,
                                                weight_func=1,
                                                track_filter_func=track_tools.create_tfilter(47, len2=4),
                                                motion_track_filter_func=track_tools.create_motion_tfilter(30),
                                                motion_iou_thresh=0.8,
                                                motion_weight=1,
                                                confi_motion_weight=10
                                                )

        tracked1 = tracker.track_video(align_data)
        tracked1 = track_tools.filter_track_result(tracked1, filter_short=10)
        tracked1 = track_tools.interpolate_video(tracked1, max_len=20)

        tracked1.to_mot(os.path.join(args.dst, 'MOT17', vname + '.txt'))

    # FRCNN
    DET = 'FRCNN'
    vnames = video_names.get_MOT_ALL(DET)
    dataset = Dataset('mot17', args.fdr, DET, vnames, training=False)

    for vname in tqdm(vnames):
        with torch.no_grad():
            align_data = eval_tools.compute_sliding_window_alignment(vname, dataset, net, cfg, save_logit=False)
            align_data = eval_tools._simplify_model_outputs(align_data)

        video = dataset.get_video(vname)
        tracker = track_tools.Tracker(video, 0.5, 
                                                activate_thresh=0.0, 
                                                direction_fuse='geomean', 
                                                mask_invalid_first=False, 
                                                input_filter_func=track_tools.create_ifilter(0.6),
                                                weight_func=1,
                                                track_filter_func=track_tools.create_tfilter(47, len2=4),
                                                motion_track_filter_func=track_tools.create_motion_tfilter(30),
                                                motion_iou_thresh=0.8,
                                                motion_weight=1,
                                                confi_motion_weight=10
                                                )
       
        tracked1 = tracker.track_video(align_data)
        tracked1 = track_tools.filter_track_result(tracked1, filter_short=10)
        tracked1 = track_tools.interpolate_video(tracked1, max_len=20)

        tracked1.to_mot(os.path.join(args.dst, 'MOT17', vname + '.txt'))

    # DPM
    DET = 'DPM'
    vnames = video_names.get_MOT_ALL(DET)
    dataset = Dataset('mot17', args.fdr, DET, vnames, training=False)

    for vname in tqdm(vnames):
        with torch.no_grad():
            align_data = eval_tools.compute_sliding_window_alignment(vname, dataset, net, cfg, save_logit=False)
            align_data = eval_tools._simplify_model_outputs(align_data)

        video = dataset.get_video(vname)
        tracker = track_tools.Tracker(video, 0.3, 
                                                activate_thresh=0.0, 
                                                direction_fuse='geomean', 
                                                mask_invalid_first=False, 
                                                input_filter_func=track_tools.create_ifilter(0.6),
                                                weight_func=1,
                                                track_filter_func=track_tools.create_tfilter(47, len2=4),
                                                motion_track_filter_func=track_tools.create_motion_tfilter(30),
                                                motion_iou_thresh=0.8,
                                                motion_weight=1,
                                                confi_motion_weight=10
                                                )
        
        tracked1 = tracker.track_video(align_data)
        tracked1 = track_tools.filter_track_result(tracked1, filter_short=10)
        tracked1 = track_tools.interpolate_video(tracked1, max_len=20)

        tracked1.to_mot(os.path.join(args.dst, 'MOT17', vname + '.txt'))


    # Evaluate training data
    train_videos = video_names.get_MOT_TRAIN('ALL')
    eval_bundle = eval_tools.TrackBundle.generate_mot_tbundle(train_videos)
    for vname in tqdm(train_videos):
        video = utils.Video(vname, os.path.join(args.dst, 'MOT17', vname + '.txt'), 'mot')
        eval_bundle.add(vname, video)

    print(eval_bundle.get_eval_dataframe())



                    



                    


