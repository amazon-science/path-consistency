import numpy as np
import json
from .utils import Video, SlidingWindow
from . import video_names 
import os
import torch
import pickle as pk
from tqdm import tqdm
from . import data_augmentation as da
from ..home import get_project_base

BASE = get_project_base()

def get_frame_fname(frame_idx):
	s = str(frame_idx)
	while len(s) < 6:
		s = '0' + s
	return s + '.jpg'

def get_loc(det, WIDTH, HEIGHT):
    cx = (det['left'] + det['right']) / 2
    cy = (det['top'] + det['bottom']) / 2
    cx = float(cx) / WIDTH
    cy = float(cy) / HEIGHT
    w = det['width'] / WIDTH
    h = det['height'] / HEIGHT
    return [cx, cy, w, h]


class ObjectSample():
    """
    Object Sample class for path consistency loss
    This specifies query object and starting/intermediate/ending frames
    """

    def __init__(self, video, frame_indices, fidx2bboxes: dict):
        self.video = video
        self.L = len(fidx2bboxes)
        self.frame_indices = frame_indices # starting/intermediate/ending frames
        self.fidx2bboxes = fidx2bboxes
        self.start_time = frame_indices[0] # start frame
        self.end_time = frame_indices[-1] # end frame

        self.frame2loc = { f:i for i, f in enumerate(frame_indices) } # map between absolute timestamp to relative postion between start, end frames

    def __len__(self):
        return self.L

    def get(self, frame_idx=None, pathloc=None):
        assert (frame_idx is not None) or (pathloc is not None)
        if pathloc is not None:
            frame_idx = self.frame_indices[pathloc]
        return (frame_idx, self.fidx2bboxes[frame_idx])

num_partition_dict = {
    "mot17": 1,
    "personpath22": 10,
    'kitti': 1,
}

class Dataset():
    def __init__(self, dataset, fdr, detection, vnames, 
                training=True, 
                augmentation_scale=None):
        self.dataset_name = dataset
        self.fdr = fdr
        self.detection = detection
        self.files = vnames
        self.videos = {}
        self.num_bbox = 0

        DATASET_PATH = f'{BASE.rstrip("/")}/data/{dataset}/{fdr}'
        print(f'Loading from {DATASET_PATH}')
        for vname in tqdm(vnames):
            video = Video(vname, f'{DATASET_PATH}/{vname}.json', "saved_anno")

            if dataset == 'personpath22':
                video._find_unlabelled_frames()
            
            if training:
                with open(f'{DATASET_PATH}/{vname}.naive_path', 'r') as fp:
                    naive_paths = json.load(fp)
                for i, p in enumerate(naive_paths):
                    naive_paths[i] = { int(f):int(b) for f, b in p.items() }

                dist_rank_matrix = np.load(f'{DATASET_PATH}/{vname}_dist_rank_matrix.npy')
            else:
                naive_paths = dist_rank_matrix = None

            self.videos[vname] = [ video, naive_paths, dist_rank_matrix ]
            self.num_bbox += video.get_num_bbox()
        
        self.nmp_dict = {}
        self.image_loc_dict = {}
        num_partition = num_partition_dict[dataset]
        for i in range(num_partition):
            if len(detection) > 0:
                partition_name = '%s-%s-partition%02d-scale1'% (dataset, detection, i)
            else:
                partition_name = '%s-partition%02d-scale1' % (dataset, i)
            nmp_fname = f'{DATASET_PATH}/{partition_name}.nmp'
            meta_json_fname =  f'{DATASET_PATH}/{partition_name}.nmp_meta'

            with open(meta_json_fname) as fp:
                meta = json.load(fp)
            mm_obj = np.memmap(nmp_fname, dtype=meta['dtype'], mode="r", shape=tuple(meta['shape']))
            self.nmp_dict[partition_name] = mm_obj 
            self.image_loc_dict.update(meta['image_loc'])
        
        if augmentation_scale is not None:
            self.da_nmp_dict = {}
            self.da_image_loc_dict = {}
            num_partition = num_partition_dict[dataset]
            for i in range(num_partition):
                if len(detection) > 0:
                    partition_name = '%s-%s-partition%02d-scale%d' % (dataset, detection, i, augmentation_scale)
                else:
                    partition_name = '%s-partition%02d-scale%d' % (dataset, i, augmentation_scale)
                nmp_fname = f'{DATASET_PATH}/{partition_name}.nmp'
                meta_json_fname =  f'{DATASET_PATH}/{partition_name}.nmp_meta'

                with open(meta_json_fname) as fp:
                    meta = json.load(fp)
                mm_obj = np.memmap(nmp_fname, dtype=meta['dtype'], mode="r", shape=tuple(meta['shape']))
                self.da_nmp_dict[partition_name] = mm_obj 
                self.da_image_loc_dict.update(meta['image_loc'])
            
        self._data_aug_ready = False

    def init_data_augmentation_from_cfg(self, da_cfg):
        self.da_cfg = da_cfg
        assert self.da_cfg.cs_min <= self.da_cfg.cs
        self._data_aug_ready = True

    def _sample(self, n, train):
        if self.da_cfg.cs == 0 or (not train):
            cs_ratio = [( 0, 0 )]*n
        else:
            cs_ratio = np.random.uniform(self.da_cfg.cs_min, self.da_cfg.cs, 2*n).tolist()
            sign = np.random.choice([1, -1], 2*n, replace=True)
            cs_ratio = cs_ratio * sign
            cs_ratio = cs_ratio.reshape(n, 2)

        if self.da_cfg.zm == 0 or (not train):
            zm_ratio = [(1, 1)]*n
        else:
            raise NotImplementedError('Does not support zoom augmentation for now')
            zm_ratio = np.random.uniform(1-zm_ratio, 1+zm_ratio, 2).tolist()
            # print(zm_ratio)
        return cs_ratio, zm_ratio

    def merge(self, dataset):
        self.files = self.files + dataset.files
        self.videos.update(dataset.videos)
        self.num_bbox += dataset.num_bbox
        self.nmp_dict.update(dataset.nmp_dict)
        self.image_loc_dict.update(dataset.image_loc_dict)

        self.da_nmp_dict.update(dataset.da_nmp_dict)
        self.da_image_loc_dict.update(dataset.da_image_loc_dict)
        return self

    def __getitem__(self, vname):
        return self.videos[vname]

    def __len__(self):
        return len(self.videos)

    def __repr__(self) -> str:
        return f"#Video-{len(self.videos)}, #BBox-{self.num_bbox}"

    def __str__(self) -> str:
        return repr(self)

    def get_video(self, vname: str):
        return self.videos[vname][0]

    def get_images(self, vname: str, frame_idx: int, bbox_ids: list, transpose_to_normal=False):
        d = self.image_loc_dict[vname][str(frame_idx)]
        locs = [ d[str(b)] for b in bbox_ids ]
        partition_name = [ l[0] for l in locs ]
        row_idx = [ l[1] for l in locs ]
        assert len(set(partition_name)) == 1
        partition_name = partition_name[0]
        # if 'scale1' in partition_name:
        #     partition_name = partition_name[:-len('-scale1')] # HACK
        img_arr = self.nmp_dict[partition_name][row_idx]
        if not transpose_to_normal:
            return img_arr
        else:
            return np.transpose(img_arr, [0, 2, 3, 1])
    
    def get_transformed_images(self, vname: str, frame_idx: int, bbox_ids: list, transpose_to_normal=False):
        assert self._data_aug_ready
        #################
        # get images
        d = self.da_image_loc_dict[vname][str(frame_idx)]
        locs = [ d[str(b)] for b in bbox_ids ]
        partition_name = [ l[0] for l in locs ]
        row_idx = [ l[1] for l in locs ]
        assert len(set(partition_name)) == 1
        img_arr = self.da_nmp_dict[partition_name[0]][row_idx]

        ##################
        # perform transformation
        flip = False # turn off flip
        video = self.get_video(vname)
        dets = [ video.get_frame_det(frame_idx, b) for b in bbox_ids ]

        data = []
        H = video.metadata['height']
        W = video.metadata['width']
        clist, zlist = self._sample(len(dets), train=True)
        for i, det in enumerate(dets):
            c, z = clist[i], zlist[i]
            # print(c)
            tlr = da.Translator.create(H, W, det, img_arr[i], self.da_cfg.scale, self.da_cfg.size)
            data.append((self.da_cfg.size, c, z, flip, tlr))

        outputs = [ process_one_object(d) for d in data ]
        bbox = [ o[0] for o in outputs ]
        images = [ o[1] for o in outputs ]

        # bbox = np.array(bbox)
        images = np.stack(images, axis=0)
        if transpose_to_normal:
            images = np.transpose(images, (0, 2, 3, 1)) # B, C, H, W
        return bbox, images
        

def process_one_object(data):
    crop_size, cs_ratio, zm_ratio, flip, tlr = data
    
    img = tlr.get_image(cs_ratio, zm_ratio, flip, frame_size=crop_size) 
    return img
    

def get_input_batch(dataset: Dataset, video_and_frame_indices_list: list, include_confi=False, transform=False):
    
    batch_data = []
    num_bbox = []
    transformed_batch_data = []
    for (video, frame_indices) in video_and_frame_indices_list:
        # flip = np.random.rand() < 0.5

        bbox_and_img = []
        transformed_bbox_and_img = []
        for i in frame_indices:
            dets = video.get_frame(i)
            if len(dets) == 0:
                bbox_and_img.append([])
                continue

            dets.sort(key=lambda x : x['bbox_id'])
            bbox_coord = torch.FloatTensor([ get_loc(d, video.metadata['width'], video.metadata['height']) for d in dets])
            bbox_id = [ d['bbox_id'] for d in dets ]
            bbox_confi = torch.FloatTensor([ d['score'] for d in dets])

            img_patches = dataset.get_images(video.vname, i, bbox_id)
            img_patches = torch.FloatTensor(img_patches)
            num_bbox.append(len(dets))
            bbox_and_img.append([bbox_coord, bbox_confi, img_patches])

            if transform:
                new_bbox_coord, new_img_patches = dataset.get_transformed_images(video.vname, i, bbox_id)
                new_bbox_coord = torch.FloatTensor(new_bbox_coord)
                new_bbox_coord[:, 0] = new_bbox_coord[:, 0] / video.metadata['width']
                new_bbox_coord[:, 2] = new_bbox_coord[:, 2] / video.metadata['width']
                new_bbox_coord[:, 1] = new_bbox_coord[:, 1] / video.metadata['height']
                new_bbox_coord[:, 3] = new_bbox_coord[:, 3] / video.metadata['height']
                new_img_patches = torch.FloatTensor(new_img_patches)
                transformed_bbox_and_img.append([new_bbox_coord, bbox_confi, new_img_patches])

        batch_data.append(bbox_and_img)
        transformed_batch_data.append(transformed_bbox_and_img)

    if len(num_bbox)==0:
        # import ipdb; ipdb.set_trace()
        return None

    N = max(num_bbox)
    C, H, W = img_patches[0].shape
    B = len(video_and_frame_indices_list)
    T = len(frame_indices)

    def data_to_tensor(batch_data_list):
        bbox_tensor = torch.zeros([B, T, N, 4]) if not include_confi else torch.zeros([B, T, N, 5]) 
        image_tensor = torch.zeros([B, T, N, C, H, W], dtype=torch.float)
        nbbox_tensor = torch.zeros([B, T], dtype=torch.long)
        for i, video_data in enumerate(batch_data_list):
            for t, data in enumerate(video_data):
                if len(data) == 0:
                    continue
                (bbox_coord, bbox_confi, img_patches) = data
                n = len(bbox_coord)
                bbox_tensor[i, t, :n, :4] = bbox_coord
                image_tensor[i, t, :n] = img_patches
                nbbox_tensor[i, t] = n
                if include_confi:
                    bbox_tensor[i, t, :n, -1] = bbox_confi
        return bbox_tensor, image_tensor, nbbox_tensor
    
    bbox_tensor, image_tensor, nbbox_tensor = data_to_tensor(batch_data)
    if transform:
        transformed_tensor = data_to_tensor(transformed_batch_data)
    else:
        transformed_tensor = None

    return bbox_tensor, image_tensor, nbbox_tensor, transformed_tensor


def generate_samples_for_pcl(video, naive_paths, window_frame_indices, dist_rank_matrix):
    """
    naive_paths: we use motion tracker to obtain naive object tracktories and use them to select query objects and start/intermediate/end frames
    """
    track_list = []

    for p in naive_paths:

        # find the overlap between frame window and p
        frame_indices = [ f for f in window_frame_indices if f in p ]
        if len(frame_indices) < 2:
            continue

        # generate data sample
        fidx2bboxes = {}
        for i, fidx in enumerate(frame_indices):

            if i == 0:
                fidx2bboxes[fidx] = [ p[fidx] ] # query object

            n = len(video.get_frame(fidx))
            assert n > 0
            bidx = p[fidx]
            # organize bbox by spatial distances
            bbox_ids = dist_rank_matrix[fidx-1, bidx, :n]
            assert -1 not in bbox_ids
            bbox_ids = bbox_ids.astype(int).tolist()
            if bidx != bbox_ids[0]:
                if bidx not in bbox_ids:
                    bbox_ids.insert(0, bidx)
                    bbox_ids = bbox_ids[:-1]
                else:
                    bbox_ids.remove(bidx)
                    bbox_ids.insert(0, bidx)

            fidx2bboxes[fidx] = bbox_ids

        p = ObjectSample(video, frame_indices, fidx2bboxes)
        track_list.append(p)
    
    return track_list

class TrainDataLoader():

    def __init__(self, dataset: Dataset, video_data: dict, cfg, 
                    batch_size: int, clip_len: int, step_size: int=1):
        self.video_data = video_data
        self.dataset = dataset
        self.batch_size = batch_size
        # self.video_length = { vname: len(self.dataset[vname][0]) for vname in self.video_list }
        self.clip_len = clip_len
        self.step_size = step_size
        self.cfg = cfg

        total_frame = sum([ x[1] for x in self.video_data.values()])
        # batch_frame = batch_size * clip_len
        batch_frame = batch_size * 64
        self.num_batch = int((total_frame + batch_frame - 1) / batch_frame)

        self.count = 0

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.num_batch:
            self.count = 0
            raise StopIteration

        self.count += 1

        input_data = None
        all_vnames = list(self.video_data.keys())
        while input_data is None:
            vnames = np.random.choice(all_vnames, size=self.batch_size, replace=False)
            video_and_frame_indices = []
            for vname in vnames:
                start_frame, T = self.video_data[vname]
                T = T - (self.clip_len * self.step_size)
                assert T > 0, (vname, T)
                start_frame = np.random.choice(T) + start_frame
                frame_indices = np.arange(self.clip_len) * self.step_size + start_frame

                video = self.dataset.get_video(vname)
                video_and_frame_indices.append([video, frame_indices])
            
            input_data = get_input_batch(self.dataset, video_and_frame_indices, 
                            include_confi=self.cfg.model.include_confi,
                            transform=self.cfg.da.enable)

        batch_samples = []
        batch_videos = []
        batch_frame_idx = []
        for video, frame_indices in video_and_frame_indices:
            _, naive_paths, dist_rank_matrix = self.dataset[video.vname]
            
            samples = generate_samples_for_pcl(video, naive_paths, frame_indices, dist_rank_matrix)
            
            batch_samples.append(samples)
            batch_videos.append(video)
            batch_frame_idx.append(frame_indices.tolist())

        bbox_tensor, image_tensor, nbbox_tensor, transformed_data = input_data


        return bbox_tensor, image_tensor, nbbox_tensor, transformed_data, batch_frame_idx, batch_samples, batch_videos

        
#===================================================================================
#===================================================================================

def create_dataset(cfg):
    TRAIN_MOT_VIDEOS = video_names.get_MOT_TRAIN(cfg.det)
    TEST_MOT_VIDEOS = video_names.get_MOT_TEST(cfg.det)

    train_video_data = test_video_data = None

    if cfg.dataset == 'kitti':
        if cfg.da.enable:
            scale = cfg.da.scale
        else:
            scale = None
        if cfg.split == 'all':
            train_vnames = video_names.KITTI_TRAIN2 + video_names.KITTI_TEST2
            test_vnames = video_names.KITTI_TRAIN2
            dataset = Dataset('kitti', cfg.fdr, cfg.det, train_vnames, augmentation_scale=scale)


    elif cfg.dataset == 'mot17':
        # MOT_VIDEOS = TRAIN_MOT_VIDEOS + TEST_MOT_VIDEOS
        if cfg.split == 'train':
            test_vnames = TRAIN_MOT_VIDEOS
            train_vnames = TRAIN_MOT_VIDEOS
            if cfg.da.enable:
                scale = cfg.da.scale
            else:
                scale = None
            dataset = Dataset('mot17', cfg.fdr, cfg.det, train_vnames, augmentation_scale=scale)

            train_video_data = { vname: [1, len(dataset[vname][0])] for vname in test_vnames }
            test_video_data = { vname: [1, len(dataset[vname][0])] for vname in train_vnames }

        elif cfg.split == 'all':
            if cfg.da.enable:
                scale = cfg.da.scale
            else:
                scale = None

            if cfg.det == 'ALL':
                train_vnames = video_names.get_MOT_ALL('ALL')
                test_vnames = video_names.get_MOT_TRAIN('ALL')

                dataset = Dataset('mot17', cfg.fdr, 'SDP', video_names.get_MOT_ALL('SDP'), augmentation_scale=scale)
                d2 = Dataset('mot17', cfg.fdr, 'FRCNN', video_names.get_MOT_ALL('FRCNN'), augmentation_scale=scale)
                d3 = Dataset('mot17', cfg.fdr, 'DPM', video_names.get_MOT_ALL('DPM'), augmentation_scale=scale)
                dataset.merge(d2)
                dataset.merge(d3)

            else:
                train_vnames = video_names.get_MOT_ALL(cfg.det)
                test_vnames = video_names.get_MOT_TRAIN(cfg.det)
                dataset = Dataset('mot17', cfg.fdr, cfg.det, video_names.get_MOT_ALL(cfg.det), augmentation_scale=scale)

        elif cfg.split == 'test':
            train_vnames = TEST_MOT_VIDEOS
            test_vnames = TRAIN_MOT_VIDEOS
            dataset = Dataset('mot17', cfg.fdr, cfg.det, train_vnames + test_vnames, augmentation_scale=scale)

    elif cfg.dataset == 'personpath22':
        if cfg.da.enable:
            scale = cfg.da.scale
        else:
            scale = None
        if cfg.split == 'train':
            dataset = Dataset('personpath22', cfg.fdr, cfg.det, video_names.PERSONPATH_TRAIN_VIDEOS + video_names.PERSONPATH_TEST_VIDEOS, augmentation_scale=scale)
            train_vnames = video_names.PERSONPATH_TRAIN_VIDEOS
            test_vnames = video_names.PERSONPATH_TEST_VIDEOS
        elif cfg.split == 'all':
            train_vnames = video_names.PERSONPATH_TRAIN_VIDEOS + video_names.PERSONPATH_TEST_VIDEOS
            test_vnames = video_names.PERSONPATH_TEST_VIDEOS
            dataset = Dataset('personpath22', cfg.fdr, cfg.det, train_vnames, augmentation_scale=scale)


    if train_video_data is None and test_video_data is None:
        train_video_data = { vname: [1, len(dataset[vname][0])] for vname in train_vnames }
        test_video_data = { vname: [1, len(dataset[vname][0])] for vname in test_vnames }
            

    labelled_vnames = set(video_names.PERSONPATH_TRAIN_VIDEOS + video_names.PERSONPATH_TEST_VIDEOS + video_names.get_MOT_TRAIN('ALL') + video_names.KITTI_TRAIN2)

    if cfg.da.enable:
        dataset.init_data_augmentation_from_cfg(cfg.da)

    trainloader = TrainDataLoader(dataset, train_video_data, cfg, cfg.batch_size, cfg.model.clip_len, cfg.model.step_size)
    print()
    print(cfg.dataset, cfg.split, cfg.fdr, dataset)
    print('Number Training Videos', len(train_video_data), 'Number Test Videos', len(test_video_data))
    print()
    return dataset, trainloader, test_video_data, labelled_vnames







        



    

    



