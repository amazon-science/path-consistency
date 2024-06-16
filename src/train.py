import numpy as np
import logging
import argparse
import os
import json
from torch import optim
import torch
import time
from tqdm import tqdm, trange
import wandb
import shutil

from .home import get_project_base
from .utils.config_tools import setup_cfg, cfg2flatdict
from .configs.default import get_cfg_defaults
from .utils.utils import resume_ckpt, resume_wandb_runid
from .utils import eval_tools 
from .utils import video_names
from .model.model import Model
from .utils import utils
from .utils.data_tools import get_input_batch, create_dataset, Dataset

_LABEL_DICT = {}
def eval_pred(video, frame_indices, prob, dist_group=None):
    global _LABEL_DICT
    pred = prob.argmax(-1)

    key = (video.vname, frame_indices[0]) 
    if key not in _LABEL_DICT:
        _LABEL_DICT[key] = eval_tools.generate_label(video, frame_indices, prob.shape[1])
    
    label = _LABEL_DICT[key]
    # dist_group = [ (1, 8), (1, 16), (1, 32) ]
    dist_group = [ (1, 8), (9, 16), (17, 32) ]
    m = eval_tools.eval_association(label, pred, dist_group=dist_group)

    return m

def eval_model(cfg, dataset, video_data, net):
    if cfg.model.clip_len > 2*cfg.TNET.w:
        stride = cfg.model.clip_len - 2 * cfg.TNET.w 
    else:
        stride = max(cfg.TNET.w // 2, 1)

    video_metrics_list = []
    for vname in tqdm(video_data):
        video = dataset[vname][0]
        start_frame, T = video_data[vname]
        frame_idx = int(cfg.model.clip_len) * cfg.model.step_size + start_frame - 1
        num_batch = np.ceil((T - cfg.model.clip_len) / stride + 1)
        num_batch = int(num_batch)

        metric_list = []
        with torch.no_grad():
            for batch_idx in range(num_batch):
                frame_indices = frame_idx - np.arange(cfg.model.clip_len) * cfg.model.step_size
                frame_indices = frame_indices[::-1]

                batch_data = get_input_batch(dataset, [(video, frame_indices)], include_confi=cfg.model.include_confi)
                if batch_data is None:
                    # print(video, frame_indices)
                    frame_idx += stride
                    continue
                bbox_tensor, image_tensor, nbbox_tensor, transformed_data = batch_data

                bbox_tensor = bbox_tensor.cuda()
                image_tensor = image_tensor.cuda()
                nbbox_tensor = nbbox_tensor.cuda()

                prob = net(bbox_tensor, image_tensor, nbbox_tensor, transformed_data=None)
                prob = prob[0] # T, N, T, N

                prob = prob.detach().cpu().numpy()
                m = eval_pred(video, frame_indices, prob)

                metric_list.append(m)

                frame_idx += stride

        metrics = utils.easy_reduce(metric_list, skip_nan=True)
        video_metrics_list.append(metrics)
    
    metrics = utils.easy_reduce(video_metrics_list, skip_nan=True)
    return metrics

def print_metrics(metrics):
    for k, v in metrics.items():
        if isinstance(v, dict):
            string = k + " "
            sep = ", " 
            for k_, v_ in v.items():
                string += "%s:%.3f%s" % (k_, v_, sep)
            string = string[:-len(sep)]
            print(string)
        else:
            print(k, "%.3f" % v)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="cfg_file", nargs="*",
                            help="optional config file", default=[])
    parser.add_argument("--set", dest="set_cfgs",
            help="set config keys", default=None, nargs=argparse.REMAINDER,)

    args = parser.parse_args()
    BASE = get_project_base()

    ### initialize experiment #########################################################
    cfg = get_cfg_defaults()
    cfg = setup_cfg(cfg, args.cfg_file, args.set_cfgs, log_name='log')

    try:
        torch.cuda.set_device('cuda:%d'%cfg.aux.gpu)
    except Exception as e:
        print(e)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.aux.gpu)

    print('============')
    print(cfg)
    print('============')

    if cfg.aux.debug:
        seed = 1 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    logdir = os.path.join(BASE, cfg.aux.logdir)
    os.makedirs(logdir, exist_ok=True)
    print('Saving log at', logdir)

    wandb_runid = resume_wandb_runid(logdir)
    run = wandb.init(
                project=cfg.aux.wandb_project, entity="",
                dir=cfg.aux.logdir,
                group=cfg.aux.exp, id=wandb_runid, resume="allow",
                config=cfg2flatdict(cfg),
                reinit=True, save_code=False,
                mode="offline",
                notes="log_dir: " + logdir,
                )
    cfg.aux.wandb_id = run.id

    ckptdir = os.path.join(logdir, 'ckpts')
    savedir = os.path.join(logdir, 'saves')
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)

    argSaveFile = os.path.join(logdir, 'args.json')
    with open(argSaveFile, 'w') as f:
        json.dump(cfg, f, indent=True)

    global_step = 0
    global_step, ckpt_file = resume_ckpt(cfg, logdir)

    ### load dataset #########################################################
    dataset, trainloader, test_vnames, LABELLED = create_dataset(cfg) 

    print('Train dataset', dataset)
    num_save = len(trainloader) * cfg.epoch // cfg.aux.eval_every
    print(f">>>>>>>> Iteration Per Epoch {len(trainloader)}, Total Iteration {len(trainloader)*cfg.epoch}")
    print(f">>>>>>>> Num Saves {num_save}")
    if num_save > 50 or num_save < 10:
        logging.warning("Too many or too less saves!")

    ### create network #########################################################
    net = Model(cfg)

    if ckpt_file is not None:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        net.load_state_dict(ckpt, strict=False)

    net.cuda()
    print('Number of Training parameters', utils.count_parameters(net)/1e3)

    if cfg.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(),
                            lr=cfg.lr,
                            momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)

    elif cfg.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(),
                               lr=cfg.lr,
                               weight_decay=cfg.weight_decay)

    start_epoch = global_step // len(trainloader)
    training_metrics = []
    training_losses = []
    print('Start Training...')
    for e in range(start_epoch, cfg.epoch):
        for batch_idx, (bbox, image, nbbox, transformed_data, frame_indices, samples, video) in enumerate(trainloader):
            optimizer.zero_grad()
            bbox = bbox.cuda()
            image = image.cuda()
            nbbox = nbbox.cuda()
            vnames = [ v.vname for v in video ]
            if transformed_data is not None:
                transformed_data = [ x.cuda() for x in transformed_data ]
            loss, loss_dict = net.forward_and_loss(vnames, bbox, image, nbbox, samples, frame_indices, global_step, transformed_data=transformed_data)

            loss.backward()
            if cfg.model.null_offset:
                net.null_offset.data.clamp_(min=0)

            optimizer.step()

            B = bbox.shape[0]
            for b in range(B):
                if video[b].vname in LABELLED:
                    prob = net.alignment_prob[b].detach().cpu().numpy()
                    m = eval_pred(video[b], frame_indices[b], prob, dist_group=[(1, 16), (17, 32), (33, 64)])
                    training_metrics.append(m)
            training_losses.append(loss_dict)


            if global_step % cfg.aux.print_every == 0:
                print(f"[{global_step}]>>")

                l = utils.easy_reduce(training_losses)
                utils.print_value_dict(l)
                log_dict = {'training_loss/'+k: v for k,v in l.items()}
                run.log(log_dict, step=global_step)

                if len(training_metrics) > 0:
                    metrics = utils.easy_reduce(training_metrics, skip_nan=True)
                    print_metrics(metrics)
                    print(f'G{cfg.aux.gpu}, {cfg.split}', cfg.aux.exp, cfg.aux.runid)

                    # reduce log times
                    if global_step % (cfg.aux.print_every * 5) == 0:
                        metrics = cfg2flatdict(metrics, type_convert=False) # Hacky use
                        log_dict = {'training_metric/'+k : v for k,v in metrics.items()}
                        run.log(log_dict, step=global_step)
                    
                
                training_metrics = []
                training_losses = []
            
            if (global_step % cfg.aux.eval_every == 0):
                net.eval()
                acc_list = []
                print(f"[Eval-{global_step}]>> ")
                metrics = eval_model(cfg, dataset, test_vnames, net)
                print_metrics(metrics)
                metrics = cfg2flatdict(metrics, type_convert=False) # Hacky use
                log_dict = {'test_metric/'+k : v for k,v in metrics.items()}
                run.log(log_dict, step=global_step)

                with open(os.path.join(savedir, f'metrics-{global_step}.json'), 'w') as fp:
                    json.dump(metrics, fp)

                net.save_model(os.path.join(ckptdir, f'net-{global_step}.pth'))
                net.train()
                
                print(cfg.aux.exp)
                print()



            global_step += 1

    run.finish()
    open(os.path.join(logdir, 'FINISH_PROOF'), 'w').close()
