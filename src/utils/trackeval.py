"""
This files contains evaluation tools for PersonPath and Kitti dataset
"""

import os
import csv
import configparser
import numpy as np
from scipy.optimize import linear_sum_assignment
import sys
from ..home import get_project_base
from ..utils.utils import Video, iou_batch, dets2array_v2
# from ..utils import utils
import json
import pandas as pd

BASE = get_project_base()
sys.path.insert(0, os.path.join(BASE, 'libs/TrackEval/'))
np.int = int # HACK, for backward compatibility

import trackeval
from trackeval.datasets._base_dataset import _BaseDataset
from trackeval import utils
from trackeval import _timing
from trackeval.utils import TrackEvalException


class PersonPath22(_BaseDataset):
    """Dataset class for MOT Challenge 2D bounding box tracking"""

    TEST_SEQ_LENGTH = {'uid_vid_00008': 798,
        'uid_vid_00009': 323,
        'uid_vid_00011': 536,
        'uid_vid_00013': 600,
        'uid_vid_00018': 409,
        'uid_vid_00019': 353,
        'uid_vid_00020': 360,
        'uid_vid_00024': 564,
        'uid_vid_00028': 593,
        'uid_vid_00030': 690,
        'uid_vid_00031': 274,
        'uid_vid_00035': 673,
        'uid_vid_00036': 248,
        'uid_vid_00038': 720,
        'uid_vid_00043': 408,
        'uid_vid_00045': 432,
        'uid_vid_00046': 587,
        'uid_vid_00048': 792,
        'uid_vid_00051': 456,
        'uid_vid_00056': 670,
        'uid_vid_00057': 679,
        'uid_vid_00063': 360,
        'uid_vid_00064': 499,
        'uid_vid_00066': 432,
        'uid_vid_00067': 288,
        'uid_vid_00068': 433,
        'uid_vid_00069': 241,
        'uid_vid_00071': 456,
        'uid_vid_00076': 681,
        'uid_vid_00078': 448,
        'uid_vid_00079': 240,
        'uid_vid_00080': 452,
        'uid_vid_00082': 447,
        'uid_vid_00085': 576,
        'uid_vid_00086': 719,
        'uid_vid_00087': 240,
        'uid_vid_00090': 607,
        'uid_vid_00092': 897,
        'uid_vid_00096': 272,
        'uid_vid_00098': 1224,
        'uid_vid_00099': 494,
        'uid_vid_00100': 293,
        'uid_vid_00102': 2700,
        'uid_vid_00105': 1801,
        'uid_vid_00107': 1803,
        'uid_vid_00109': 2700,
        'uid_vid_00113': 2700,
        'uid_vid_00114': 1800,
        'uid_vid_00117': 2700,
        'uid_vid_00144': 2242,
        'uid_vid_00147': 3889,
        'uid_vid_00149': 2063,
        'uid_vid_00150': 2553,
        'uid_vid_00118': 208,
        'uid_vid_00121': 174,
        'uid_vid_00122': 124,
        'uid_vid_00124': 449,
        'uid_vid_00125': 778,
        'uid_vid_00126': 418,
        'uid_vid_00127': 316,
        'uid_vid_00130': 191,
        'uid_vid_00133': 329,
        'uid_vid_00141': 375,
        'uid_vid_00153': 625,
        'uid_vid_00158': 916,
        'uid_vid_00161': 4203,
        'uid_vid_00163': 841,
        'uid_vid_00166': 704,
        'uid_vid_00167': 521,
        'uid_vid_00169': 1188,
        'uid_vid_00170': 1094,
        'uid_vid_00172': 571,
        'uid_vid_00173': 565,
        'uid_vid_00174': 769,
        'uid_vid_00175': 683,
        'uid_vid_00178': 447,
        'uid_vid_00179': 977,
        'uid_vid_00183': 520,
        'uid_vid_00189': 527,
        'uid_vid_00190': 691,
        'uid_vid_00191': 250,
        'uid_vid_00193': 515,
        'uid_vid_00198': 367,
        'uid_vid_00200': 689,
        'uid_vid_00201': 336,
        'uid_vid_00205': 518,
        'uid_vid_00207': 298,
        'uid_vid_00212': 852,
        'uid_vid_00218': 446,
        'uid_vid_00219': 4045,
        'uid_vid_00221': 3651,
        'uid_vid_00222': 349,
        'uid_vid_00226': 741,
        'uid_vid_00228': 780,
        'uid_vid_00230': 626,
        'uid_vid_00162': 772,
        'uid_vid_00234': 934,
        'uid_vid_00235': 902,
        'uid_vid_99999': 1057,
        'uid_vid_00000': 428,
        'uid_vid_00001': 504,
        'uid_vid_00002': 604,
        'uid_vid_00003': 456,
        'uid_vid_00004': 372,
        'uid_vid_00005': 792,
        'uid_vid_00006': 278,
        'uid_vid_00007': 396,
        'uid_vid_00010': 478,
        'uid_vid_00012': 371,
        'uid_vid_00014': 1034,
        'uid_vid_00015': 495,
        'uid_vid_00016': 656,
        'uid_vid_00017': 396,
        'uid_vid_00021': 404,
        'uid_vid_00022': 384,
        'uid_vid_00023': 552,
        'uid_vid_00025': 816,
        'uid_vid_00026': 252,
        'uid_vid_00027': 864,
        'uid_vid_00029': 288,
        'uid_vid_00032': 480,
        'uid_vid_00033': 660,
        'uid_vid_00034': 501,
        'uid_vid_00037': 252,
        'uid_vid_00039': 492,
        'uid_vid_00040': 244,
        'uid_vid_00041': 504,
        'uid_vid_00042': 600,
        'uid_vid_00044': 528,
        'uid_vid_00047': 798,
        'uid_vid_00049': 472,
        'uid_vid_00050': 1176,
        'uid_vid_00052': 246,
        'uid_vid_00053': 384,
        'uid_vid_00054': 894,
        'uid_vid_00055': 624,
        'uid_vid_00058': 870,
        'uid_vid_00059': 720,
        'uid_vid_00060': 312,
        'uid_vid_00061': 898,
        'uid_vid_00062': 408,
        'uid_vid_00065': 631,
        'uid_vid_00070': 432,
        'uid_vid_00072': 885,
        'uid_vid_00073': 899,
        'uid_vid_00074': 768,
        'uid_vid_00075': 293,
        'uid_vid_00077': 312,
        'uid_vid_00081': 744,
        'uid_vid_00083': 361,
        'uid_vid_00084': 456,
        'uid_vid_00088': 386,
        'uid_vid_00089': 422,
        'uid_vid_00091': 246,
        'uid_vid_00093': 709,
        'uid_vid_00094': 1200,
        'uid_vid_00095': 264,
        'uid_vid_00097': 264,
        'uid_vid_00101': 480,
        'uid_vid_00103': 1805,
        'uid_vid_00104': 1193,
        'uid_vid_00106': 1801,
        'uid_vid_00108': 2700,
        'uid_vid_00110': 1801,
        'uid_vid_00111': 1801,
        'uid_vid_00112': 2700,
        'uid_vid_00115': 2700,
        'uid_vid_00116': 1800,
        'uid_vid_00145': 3218,
        'uid_vid_00146': 1871,
        'uid_vid_00148': 1584,
        'uid_vid_00151': 3726,
        'uid_vid_00152': 2080,
        'uid_vid_00119': 181,
        'uid_vid_00120': 629,
        'uid_vid_00123': 144,
        'uid_vid_00128': 2277,
        'uid_vid_00129': 448,
        'uid_vid_00131': 199,
        'uid_vid_00132': 238,
        'uid_vid_00134': 448,
        'uid_vid_00135': 868,
        'uid_vid_00136': 349,
        'uid_vid_00137': 900,
        'uid_vid_00140': 749,
        'uid_vid_00142': 389,
        'uid_vid_00143': 275,
        'uid_vid_00154': 1590,
        'uid_vid_00155': 756,
        'uid_vid_00156': 786,
        'uid_vid_00157': 1022,
        'uid_vid_00159': 426,
        'uid_vid_00160': 532,
        'uid_vid_00164': 336,
        'uid_vid_00165': 6975,
        'uid_vid_00168': 849,
        'uid_vid_00171': 3600,
        'uid_vid_00176': 442,
        'uid_vid_00177': 794,
        'uid_vid_00180': 696,
        'uid_vid_00181': 1058,
        'uid_vid_00182': 393,
        'uid_vid_00184': 331,
        'uid_vid_00185': 694,
        'uid_vid_00186': 303,
        'uid_vid_00187': 657,
        'uid_vid_00188': 472,
        'uid_vid_00192': 480,
        'uid_vid_00194': 308,
        'uid_vid_00195': 636,
        'uid_vid_00196': 297,
        'uid_vid_00197': 2070,
        'uid_vid_00199': 607,
        'uid_vid_00202': 1132,
        'uid_vid_00203': 504,
        'uid_vid_00204': 326,
        'uid_vid_00206': 504,
        'uid_vid_00208': 551,
        'uid_vid_00209': 796,
        'uid_vid_00210': 589,
        'uid_vid_00211': 493,
        'uid_vid_00213': 796,
        'uid_vid_00214': 3780,
        'uid_vid_00215': 264,
        'uid_vid_00216': 546,
        'uid_vid_00217': 1594,
        'uid_vid_00220': 940,
        'uid_vid_00223': 398,
        'uid_vid_00224': 380,
        'uid_vid_00225': 405,
        'uid_vid_00227': 1014,
        'uid_vid_00229': 2098,
        'uid_vid_00231': 695,
        'uid_vid_00233': 384,
        'uid_vid_00236': 500}

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/person_path_22/'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/person_path_22/'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': ['tracker'],  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
            'BENCHMARK': 'person_path_22',  # Valid: 'person_path_22'
            'SPLIT_TO_EVAL': 'test',  # Valid: 'train', 'test', 'all'
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': False,  # Whether to print current config
            'DO_PREPROC': True,  # Whether to perform preprocessing
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
            'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
            # 'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',  # '{gt_folder}/{seq}/gt/gt.txt'
            'GT_LOC_FORMAT': '{gt_folder}/{seq}.txt',  # '{gt_folder}/{seq}/gt/gt.txt'
            'SKIP_SPLIT_FOL': False,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
                                      # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
                                      # If True, then the middle 'benchmark-split' folder is skipped for both.
        }
        return default_config

    def __init__(self, gt_video_dict={}, pred_video_dict={}, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()

        self.gt_video_dict = gt_video_dict
        self.pred_video_dict = pred_video_dict

        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())


        self.benchmark = self.config['BENCHMARK']
        gt_set = self.config['BENCHMARK'] + '-' + self.config['SPLIT_TO_EVAL']
        self.gt_set = gt_set
        if not self.config['SKIP_SPLIT_FOL']:
            split_fol = gt_set
        else:
            split_fol = ''
        self.gt_fol = os.path.join(self.config['GT_FOLDER'], split_fol)
        # self.tracker_fol = os.path.join(self.config['TRACKERS_FOLDER'], split_fol)
        self.should_classes_combine = False
        self.use_super_categories = False
        # self.data_is_zipped = self.config['INPUT_AS_ZIP']
        self.do_preproc = self.config['DO_PREPROC']

        # self.output_fol = self.config['OUTPUT_FOLDER']
        # if self.output_fol is None:
        #     self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']

        # Get classes to eval
        self.valid_classes = ['pedestrian']
        self.class_list = [cls.lower() if cls.lower() in self.valid_classes else None
                           for cls in self.config['CLASSES_TO_EVAL']]
        if not all(self.class_list):
            raise TrackEvalException('Attempted to evaluate an invalid class. Only pedestrian class is valid.')
        self.class_name_to_class_id = {'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4, 'motorbike': 5,
                                       'non_mot_vehicle': 6, 'static_person': 7, 'distractor': 8, 'occluder': 9,
                                       'occluder_on_ground': 10, 'occluder_full': 11, 'reflection': 12, 'crowd': 13}
        self.valid_class_numbers = list(self.class_name_to_class_id.values())

        # Get sequences to eval and check gt files exist
        self.seq_list = []
        self.seq_lengths = {}

        self.tracker_list = self.config['TRACKERS_TO_EVAL']
        self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))


    def set_pred_videos(self, pred_video_dict):
        self.pred_video_dict = pred_video_dict
        self.seq_list = list(self.pred_video_dict.keys())
        self.seq_lengths = { k: self.TEST_SEQ_LENGTH[k] for k in self.seq_list }

    def get_eval_info(self):
        """Return info about the dataset needed for the Evaluator"""
        return self.tracker_list, self.seq_list, self.class_list
    
    def get_output_fol(self, tracker):
        return tracker

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _get_seq_info(self):
        seq_list = list(self.pred_video_dict.keys())
        seq_lengths = { k: self.TEST_SEQ_LENGTH[k] for k in seq_list }
        return seq_list, seq_lengths

    def _load_simple_text_file_from_string(self, string, crowd_ignore_filter=None, 
                                        time_col=0, id_col=None, 
                                        remove_negative_ids=False, valid_filter=None, convert_filter=None):

        if convert_filter is None:
            convert_filter = {}
        if crowd_ignore_filter is None:
            crowd_ignore_filter = {}

        read_data = {}
        crowd_ignore_data = {}

        fake_fp = string.strip('\n').split('\n')
        reader = csv.reader(fake_fp)

                    
        for row in reader:
            try:
                if len(row) == 0:
                    continue

                # Deal with extra trailing spaces at the end of rows
                if row[-1] in '':
                    row = row[:-1]

                timestep = str(int(float(row[time_col])))
                # Read ignore regions separately.
                is_ignored = False
                for ignore_key, ignore_value in crowd_ignore_filter.items():
                    if row[ignore_key].lower() in ignore_value:
                        # Convert values in one column (e.g. string to id)
                        for convert_key, convert_value in convert_filter.items():
                            row[convert_key] = convert_value[row[convert_key].lower()]
                        # Save data separated by timestep.
                        if timestep in crowd_ignore_data.keys():
                            crowd_ignore_data[timestep].append(row)
                        else:
                            crowd_ignore_data[timestep] = [row]
                        is_ignored = True
                if is_ignored:  # if det is an ignore region, it cannot be a normal det.
                    continue
                # Exclude some dets if not valid.
                if valid_filter is not None:
                    for key, value in valid_filter.items():
                        if row[key].lower() not in value:
                            continue
                if remove_negative_ids:
                    if int(float(row[id_col])) < 0:
                        continue
                # Convert values in one column (e.g. string to id)
                for convert_key, convert_value in convert_filter.items():
                    row[convert_key] = convert_value[row[convert_key].lower()]
                # Save data separated by timestep.
                if timestep in read_data.keys():
                    read_data[timestep].append(row)
                else:
                    read_data[timestep] = [row]
            except Exception as e:
                print(e)
                print(row)
                
        return read_data, crowd_ignore_data

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the MOT Challenge 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """

        # Ignore regions
        if is_gt:
            crowd_ignore_filter = {} # {7: ['13']}
        else:
            crowd_ignore_filter = None

        # Load raw data from text file
        # read_data, ignore_data = self._load_simple_text_file(file, is_zipped=self.data_is_zipped, zip_file=zip_file, crowd_ignore_filter=crowd_ignore_filter)
        if is_gt:
            seq_video = self.gt_video_dict[seq] 
            if not isinstance(seq_video, str):
                seq_video = seq_video.personpath_to_mot()
        else:
            seq_video = self.pred_video_dict[seq]
            if not isinstance(seq_video, str):
                seq_video = seq_video.to_mot()


        read_data, ignore_data = self._load_simple_text_file_from_string(seq_video, crowd_ignore_filter=crowd_ignore_filter)

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets']
        if is_gt:
            data_keys += ['gt_crowd_ignore_regions', 'gt_extras']
        else:
            data_keys += ['tracker_confidences']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        # Check for any extra time keys
        current_time_keys = [str( t+ 1) for t in range(num_timesteps)]
        extra_time_keys = [x for x in read_data.keys() if x not in current_time_keys]
        if len(extra_time_keys) > 0:
            if is_gt:
                text = 'Ground-truth'
            else:
                text = 'Tracking'
            raise TrackEvalException(
                text + ' data contains the following invalid timesteps in seq %s: ' % seq + ', '.join(
                    [str(x) + ', ' for x in extra_time_keys]))

        for t in range(num_timesteps):
            time_key = str(t+1)
            if time_key in read_data.keys():
                try:
                    time_data = np.asarray(read_data[time_key], dtype=float)
                except ValueError:
                    if is_gt:
                        raise TrackEvalException(
                            'Cannot convert gt data for sequence %s to float. Is data corrupted?' % seq)
                    else:
                        raise TrackEvalException(
                            'Cannot convert tracking data from tracker %s, sequence %s to float. Is data corrupted?' % (
                                tracker, seq))
                try:
                    raw_data['dets'][t] = np.atleast_2d(time_data[:, 2:6])
                    raw_data['ids'][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                except IndexError:
                    if is_gt:
                        err = 'Cannot load gt data from sequence %s, because there is not enough ' \
                              'columns in the data.' % seq
                        raise TrackEvalException(err)
                    else:
                        err = 'Cannot load tracker data from tracker %s, sequence %s, because there is not enough ' \
                              'columns in the data.' % (tracker, seq)
                        raise TrackEvalException(err)
                if time_data.shape[1] >= 8:
                    raw_data['classes'][t] = np.atleast_1d(time_data[:, 7]).astype(int)
                else:
                    if not is_gt:
                        raw_data['classes'][t] = np.ones_like(raw_data['ids'][t])
                    else:
                        raise TrackEvalException(
                            'GT data is not in a valid format, there is not enough rows in seq %s, timestep %i.' % (
                                seq, t))
                if is_gt:
                    gt_extras_dict = {'zero_marked': np.atleast_1d(time_data[:, 6].astype(int))}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.atleast_1d(time_data[:, 6])
            else:
                raw_data['dets'][t] = np.empty((0, 4))
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {'zero_marked': np.empty(0)}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.empty(0)
            if is_gt:
                if time_key in ignore_data.keys():
                    time_ignore = np.asarray(ignore_data[time_key], dtype=float)
                    raw_data['gt_crowd_ignore_regions'][t] = np.atleast_2d(time_ignore[:, 2:6])
                else:
                    raw_data['gt_crowd_ignore_regions'][t] = np.empty((0, 4))

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'classes': 'gt_classes',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        MOT Challenge:
            In MOT Challenge, the 4 preproc steps are as follow:
                1) There is only one class (pedestrian) to be evaluated, but all other classes are used for preproc.
                2) Predictions are matched against all gt boxes (regardless of class), those matching with distractor
                    objects are removed.
                3) There is no crowd ignore regions.
                4) All gt dets except pedestrian are removed, also removes pedestrian gt dets marked with zero_marked.
        """
        # Check that input data has unique ids
        self._check_unique_ids(raw_data)

        # self.distractor_class_names = ['person_on_vehicle', 'static_person', 'distractor', 'reflection']
        # if self.benchmark == 'MOT20':
        #     self.distractor_class_names.append('non_mot_vehicle')
        # distractor_class_names += ['bicycle', 'motorbike', 'non_mot_vehicle', 'occluder_full' ]
        # distractor_classes = [self.class_name_to_class_id[x] for x in self.distractor_class_names]
        distractor_classes = []
        cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):

            # Get all data
            gt_ids = raw_data['gt_ids'][t]
            gt_dets = raw_data['gt_dets'][t]
            gt_classes = raw_data['gt_classes'][t]
            gt_zero_marked = raw_data['gt_extras'][t]['zero_marked']

            tracker_ids = raw_data['tracker_ids'][t]
            tracker_dets = raw_data['tracker_dets'][t]
            tracker_classes = raw_data['tracker_classes'][t]
            tracker_confidences = raw_data['tracker_confidences'][t]
            similarity_scores = raw_data['similarity_scores'][t]
            crowd_ignore_regions = raw_data['gt_crowd_ignore_regions'][t]

            # Evaluation is ONLY valid for pedestrian class
            if len(tracker_classes) > 0 and np.max(tracker_classes) > 1:
                raise TrackEvalException(
                    'Evaluation is only valid for pedestrian class. Non pedestrian class (%i) found in sequence %s at '
                    'timestep %i.' % (np.max(tracker_classes), raw_data['seq'], t))

            # Match tracker and gt dets (with hungarian algorithm) and remove tracker dets which match with gt dets
            # which are labeled as belonging to a distractor class.
            to_remove_tracker = np.array([], int)
            if self.do_preproc and self.benchmark != 'MOT15' and (gt_ids.shape[0] > 0 or len(crowd_ignore_regions) > 0) and tracker_ids.shape[0] > 0:

                # Check all classes are valid:
                invalid_classes = np.setdiff1d(np.unique(gt_classes), self.valid_class_numbers)
                if len(invalid_classes) > 0:
                    print(' '.join([str(x) for x in invalid_classes]))
                    raise(TrackEvalException('Attempting to evaluate using invalid gt classes. '
                                             'This warning only triggers if preprocessing is performed, '
                                             'e.g. not for MOT15 or where prepropressing is explicitly disabled. '
                                             'Please either check your gt data, or disable preprocessing. '
                                             'The following invalid classes were found in timestep ' + str(t) + ': ' +
                                             ' '.join([str(x) for x in invalid_classes])))

                # matching_scores = similarity_scores.copy()
                # matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                # match_rows, match_cols = linear_sum_assignment(-matching_scores)
                # actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                # match_rows = match_rows[actually_matched_mask]
                # match_cols = match_cols[actually_matched_mask]

                # is_distractor_class = np.isin(gt_classes[match_rows], distractor_classes)
                # to_remove_tracker = match_cols[is_distractor_class]

                # # remove bounding boxes that overlap with crowd ignore region.
                # intersection_with_ignore_region = self._calculate_box_ious(tracker_dets, crowd_ignore_regions, box_format='xywh', do_ioa=True)
                # is_within_crowd_ignore_region = np.any(intersection_with_ignore_region > 0.95 + np.finfo('float').eps, axis=1)
                # to_remove_tracker = np.unique(np.concatenate([to_remove_tracker, np.where(is_within_crowd_ignore_region)[0]]))

            if self.do_preproc and self.benchmark != 'MOT15' and gt_ids.shape[0] == 0 and tracker_ids.shape[0] > 0:
                to_remove_tracker = np.arange(tracker_ids.shape[0])

            # import ipdb; ipdb.set_trace()

            # Apply preprocessing to remove all unwanted tracker dets.
            data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            data['tracker_confidences'][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            # Remove gt detections marked as to remove (zero marked), and also remove gt detections not in pedestrian
            # class (not applicable for MOT15)
            if self.do_preproc and self.benchmark != 'MOT15':
                gt_to_keep_mask = (np.not_equal(gt_zero_marked, 0)) & \
                                  (np.equal(gt_classes, cls_id))
            else:
                # There are no classes for MOT15
                gt_to_keep_mask = np.not_equal(gt_zero_marked, 0)
            data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
            data['gt_dets'][t] = gt_dets[gt_to_keep_mask, :]
            data['similarity_scores'][t] = similarity_scores[gt_to_keep_mask]

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']

        # remove tracker_dets for empty_frames
        # ngt_dets, ntk_dets, ngt_ids, ntk_ids = [], [], [], []

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='xywh')
        return similarity_scores


evaluator_config = {
 'USE_PARALLEL': False,
 'NUM_PARALLEL_CORES': 8,
 'BREAK_ON_ERROR': True,
 'RETURN_ON_ERROR': False,
 'LOG_ON_ERROR': os.path.join(BASE, 'libs/TrackEval/error_log.txt'),
 'PRINT_RESULTS': False,
 'PRINT_ONLY_COMBINED': False,
 'PRINT_CONFIG': False,
 'TIME_PROGRESS': False,
 'DISPLAY_LESS_PROGRESS': True,
 'OUTPUT_SUMMARY': False,
 'OUTPUT_EMPTY_CLASSES': True,
 'OUTPUT_DETAILED': False,
 'PLOT_CURVES': False
 }

metrics_config = {'METRICS': ['CLEAR', 'Identity'],
 'THRESHOLD': 0.5,
 'PRINT_CONFIG': False}

_distractor_classes = [
'person_on_open_vehicle', 'person_in_vehicle',  
'severly_occluded_person', 
'person_in_background', 
'reflection', 'fully_occluded', 'noise'
]

def filter_personpath_gt(video, distractor_class=None, min_d=20, crowded_iou_thres=0.95):
    if distractor_class is None:
        distractor_class = _distractor_classes

    keep = video.create_empty_copy()
    removed = video.create_empty_copy()
    for f, dets in video.det_by_frame.items():
        
        crowded = [ d for d in dets if 'crowd' in d['_pp_labels'] ]    
        if len(crowded) > 0:
            crowded_arr = dets2array_v2(crowded, sort_idx=None)[:, :-1]
            all_arr = dets2array_v2(dets, sort_idx=None)[:, :-1]
        
            simi = iou_batch(all_arr, crowded_arr)
            simi = simi.max(1)
            where = np.where(simi < crowded_iou_thres)[0]
            removed.add_det([dets[i] for i in range(len(dets)) if i not in where])
            dets = [dets[i] for i in where]

        if len(dets) == 0:
            continue
            
        for d in dets:
            remove = False
            if d['width'] < min_d or d['height'] < min_d:
                remove = True
            elif 'person' not in d['_pp_labels']:
                remove = True
            else:
                for c in distractor_class:
                    if c in d['_pp_labels']:
                        remove = True
                        break
    
            if remove:        
                removed.add_det(d)
            else:
                keep.add_det(d)

    # print(ct)
    return keep, removed

def clean_detections(video, gt_keep, gt_removed, thres=0.6):
    cleaned = video.create_empty_copy()
    for f, dets in video.det_by_frame.items():
        # new_dets = []
        # for d in dets:
        #     # if d['width'] < 20 or d['height'] < 20:
        #     #     continue
        #     new_dets.append(d)
            
        # dets = new_dets
            
        if len(dets) == 0:
            continue
        if f not in gt_keep.det_by_frame:
            continue
        if (thres > 0) and (f in gt_removed.det_by_frame):
            p = dets2array_v2(dets, sort_idx=None)
            p = p[:, :-1]
    
            gt_dets = gt_removed.get_frame(f)
            g = dets2array_v2(gt_dets, sort_idx=None)[:, :-1]
    
            simi = iou_batch(p, g)
            simi = simi.max(1)
            where = np.where(simi <= thres)[0]
            dets = [dets[i] for i in where]

        if len(dets) > 0:
            cleaned.add_det(dets)

    return cleaned


def create_personpath_dataset(vnames, distractor_class=None):
    gt_keep, gt_remove = {}, {}
    for vname in vnames:
        gt = Video(vname, 
            os.path.join(BASE, f'data/personpath22/annotation/anno_amodal_2022/{vname}.mp4.json'), 'personpath')
        keep, remove = filter_personpath_gt(gt, distractor_class=distractor_class)
        gt_keep[vname] = keep
        gt_remove[vname] = remove

    dataset = PersonPath22(gt_keep)
    dataset.gt_remove = gt_remove
    # dataset.seq_list = [ v[:-4] for v in dataset.seq_list ] 
    # dataset.seq_lengths = { v[:-4]: l for v, l in dataset.seq_lengths.items() }

    return dataset

def parse_metrics(m):
    metrics = {
    'IDF1': m['pedestrian']['Identity']['IDF1'],
    'MOTA': m['pedestrian']['CLEAR']['MOTA'],
    'IDSW': m['pedestrian']['CLEAR']['IDSW'],
    'FP'  : m['pedestrian']['CLEAR']['CLR_FP'],
    'FN'  : m['pedestrian']['CLEAR']['CLR_FN'],
    }
    return metrics

def evaluate_personpath(tracked_videos: dict, dataset: PersonPath22, remove_iou_thres=0.6, parallel=0):
    global evaluator_config, metrics_config

    tracked_clean = {}
    for vname, tracked in tracked_videos.items():
        gt_keep = dataset.gt_video_dict[vname]
        gt_remove = dataset.gt_remove[vname]
        pred_cleaned = clean_detections(tracked, gt_keep, gt_remove, thres=remove_iou_thres)
        tracked_clean[vname] = pred_cleaned

    dataset.set_pred_videos(tracked_clean)

    if parallel > 0:
        evaluator_config = evaluator_config.copy()
        evaluator_config['USE_PARALLEL'] = True
        evaluator_config['NUM_PARALLEL_CORES'] = parallel

    metrics_list = [trackeval.metrics.CLEAR(metrics_config), trackeval.metrics.Identity(metrics_config)]
    evaluator = trackeval.Evaluator(evaluator_config)
    
    res, msg = evaluator.evaluate([dataset], metrics_list, show_progressbar=False)
    msg = msg['PersonPath22']['tracker']
    res = res['PersonPath22']['tracker']
    res = { k: parse_metrics(v) for k, v in res.items() }
    
    return res

class PersonPath_TrackBundle():

    def __init__(self, dataset):
        self.dataset = dataset
        self._tracked_dict = {}
        self.results = None
        self.metric = None

    def keys(self):
        return list(self._tracked_dict.keys())

    def has_tracked(self, name):
        if name in self._tracked_dict:
            return True
        else:
            return False

    def add(self, vname, tracked):
        self._tracked_dict[vname] = tracked

    def get_video(self, vname):
        return self._tracked_dict[vname]

    def get_res(self, force=False):
        if force or (self.metric is None):
            res = evaluate_personpath(self._tracked_dict, self.dataset)
            self.metric = res['COMBINED_SEQ']
            self.metric = pd.Series(self.metric).to_frame().T
            self.results = res

        return self.metric

###########################################################################################################
###########################################################################################################

class Kitti2DBox(_BaseDataset):
    """Dataset class for KITTI 2D bounding box tracking"""

    _TRAINING_SEQ_MAP = {'0000': 154,
                        '0001': 447,
                        '0002': 233,
                        '0003': 144,
                        '0004': 314,
                        '0005': 297,
                        '0006': 270,
                        '0007': 800,
                        '0008': 390,
                        '0009': 803,
                        '0010': 294,
                        '0011': 373,
                        '0012': 78,
                        '0013': 340,
                        '0014': 106,
                        '0015': 376,
                        '0016': 209,
                        '0017': 145,
                        '0018': 339,
                        '0019': 1059,
                        '0020': 837}
    
    _TESTING_SEQ_MAP = {'0000': 465,
                        '0001': 147,
                        '0002': 243,
                        '0003': 257,
                        '0004': 421,
                        '0005': 809,
                        '0006': 114,
                        '0007': 215,
                        '0008': 165,
                        '0009': 349,
                        '0010': 1176,
                        '0011': 774,
                        '0012': 694,
                        '0013': 152,
                        '0014': 850,
                        '0015': 701,
                        '0016': 510,
                        '0017': 305,
                        '0018': 180,
                        '0019': 404,
                        '0020': 173,
                        '0021': 203,
                        '0022': 436,
                        '0023': 430,
                        '0024': 316,
                        '0025': 176,
                        '0026': 170,
                        '0027': 85,
                        '0028': 175}

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/kitti/kitti_2d_box_train'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/kitti/kitti_2d_box_train/'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': ['tracker'],  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': ['car', 'pedestrian'],  # Valid: ['car', 'pedestrian']
            'SPLIT_TO_EVAL': 'training',  # Valid: 'training', 'val', 'training_minus_val', 'test'
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': False,  # Whether to print current config
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        }
        return default_config
    
    def __init__(self, gt_video_dict={}, pred_video_dict={}, config=None, training=True):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()

        self.gt_video_dict = gt_video_dict
        self.pred_video_dict = pred_video_dict

        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())
        self.gt_fol = self.config['GT_FOLDER']
        self.tracker_fol = self.config['TRACKERS_FOLDER']
        self.should_classes_combine = False
        self.use_super_categories = False
        self.data_is_zipped = self.config['INPUT_AS_ZIP']

        if training:
            self.config['SPLIT_TO_EVAL'] = 'training'
        else:
            self.config['SPLIT_TO_EVAL'] = 'testing'


        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']

        self.max_occlusion = 2
        self.max_truncation = 0
        self.min_height = 25

        # Get classes to eval
        self.valid_classes = ['car', 'pedestrian']
        self.class_list = [cls.lower() if cls.lower() in self.valid_classes else None
                           for cls in self.config['CLASSES_TO_EVAL']]
        if not all(self.class_list):
            raise TrackEvalException('Attempted to evaluate an invalid class. Only classes [car, pedestrian] are valid.')
        self.class_name_to_class_id = {'car': 1, 'van': 2, 'truck': 3, 'pedestrian': 4, 'person': 5,  # person sitting
                                       'cyclist': 6, 'tram': 7, 'misc': 8, 'dontcare': 9, 'car_2': 1}

        # Get sequences to eval and check gt files exist
        self.seq_list = []
        self.seq_lengths = {}
        self.full_seq_lengths = {}
        for k, v in self._TRAINING_SEQ_MAP.items():
            self.full_seq_lengths['training-'+k] = v
        for k, v in self._TESTING_SEQ_MAP.items():
            self.full_seq_lengths['testing-'+k] = v


        # Get trackers to eval
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']

        self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))

    def set_pred_videos(self, pred_video_dict):
        self.pred_video_dict = pred_video_dict
        self.seq_list = list(self.pred_video_dict.keys())
        self.seq_lengths = { k: self.full_seq_lengths[k] for k in self.seq_list }

    def _get_seq_info(self):
        seq_list = list(self.pred_video_dict.keys())
        seq_lengths = { k: self.full_seq_lengths[k] for k in seq_list }
        return seq_list, seq_lengths

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _load_simple_text_file_from_string(self, string, crowd_ignore_filter=None, 
                                        time_col=0, id_col=None, 
                                        remove_negative_ids=False, valid_filter=None, convert_filter=None):

        if convert_filter is None:
            convert_filter = {}
        if crowd_ignore_filter is None:
            crowd_ignore_filter = {}

        read_data = {}
        crowd_ignore_data = {}

        fake_fp = string.strip('\n').split('\n')
        reader = csv.reader(fake_fp, delimiter=" ")

                    
        for row in reader:
            try:
                if len(row) == 0:
                    continue

                # Deal with extra trailing spaces at the end of rows
                if row[-1] in '':
                    row = row[:-1]

                timestep = str(int(float(row[time_col])))
                # Read ignore regions separately.
                is_ignored = False
                for ignore_key, ignore_value in crowd_ignore_filter.items():
                    if row[ignore_key].lower() in ignore_value:
                        # Convert values in one column (e.g. string to id)
                        for convert_key, convert_value in convert_filter.items():
                            row[convert_key] = convert_value[row[convert_key].lower()]
                        # Save data separated by timestep.
                        if timestep in crowd_ignore_data.keys():
                            crowd_ignore_data[timestep].append(row)
                        else:
                            crowd_ignore_data[timestep] = [row]
                        is_ignored = True
                if is_ignored:  # if det is an ignore region, it cannot be a normal det.
                    continue
                # Exclude some dets if not valid.
                if valid_filter is not None:
                    for key, value in valid_filter.items():
                        if row[key].lower() not in value:
                            continue
                if remove_negative_ids:
                    if int(float(row[id_col])) < 0:
                        continue
                # Convert values in one column (e.g. string to id)
                for convert_key, convert_value in convert_filter.items():
                    row[convert_key] = convert_value[row[convert_key].lower()]
                # Save data separated by timestep.
                if timestep in read_data.keys():
                    read_data[timestep].append(row)
                else:
                    read_data[timestep] = [row]
            except Exception as e:
                print(e)
                print(row)
                
        return read_data, crowd_ignore_data

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the kitti 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        # File location
        # if self.data_is_zipped:
        #     if is_gt:
        #         zip_file = os.path.join(self.gt_fol, 'data.zip')
        #     else:
        #         zip_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
        #     file = seq + '.txt'
        # else:
        #     zip_file = None
        #     if is_gt:
        #         file = os.path.join(self.gt_fol, 'label_02', seq + '.txt')
        #     else:
        #         file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')

        # Ignore regions
        if is_gt:
            crowd_ignore_filter = {2: ['dontcare']}
        else:
            crowd_ignore_filter = None

        # Valid classes
        valid_filter = {2: [x for x in self.class_list]}
        if is_gt:
            if 'car' in self.class_list:
                valid_filter[2].append('van')
            if 'pedestrian' in self.class_list:
                valid_filter[2] += ['person']

        # Convert kitti class strings to class ids
        convert_filter = {2: self.class_name_to_class_id}

        if is_gt:
            seq_video = self.gt_video_dict[seq]
        else:
            seq_video = self.pred_video_dict[seq]

        if not isinstance(seq_video, str):
            seq_video = seq_video.to_kitti()

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file_from_string(seq_video, time_col=0, id_col=1, remove_negative_ids=True,
                                                             valid_filter=valid_filter,
                                                             crowd_ignore_filter=crowd_ignore_filter,
                                                             convert_filter=convert_filter)
        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets']
        if is_gt:
            data_keys += ['gt_crowd_ignore_regions', 'gt_extras']
        else:
            data_keys += ['tracker_confidences']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        # Check for any extra time keys
        current_time_keys = [str(t) for t in range(num_timesteps)]
        extra_time_keys = [x for x in read_data.keys() if x not in current_time_keys]
        if len(extra_time_keys) > 0:
            if is_gt:
                text = 'Ground-truth'
            else:
                text = 'Tracking'
            raise TrackEvalException(
                text + ' data contains the following invalid timesteps in seq %s: ' % seq + ', '.join(
                    [str(x) + ', ' for x in extra_time_keys]))

        for t in range(num_timesteps):
            time_key = str(t)
            if time_key in read_data.keys():
                time_data = np.asarray(read_data[time_key], dtype=float)
                raw_data['dets'][t] = np.atleast_2d(time_data[:, 6:10])
                raw_data['ids'][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                raw_data['classes'][t] = np.atleast_1d(time_data[:, 2]).astype(int)
                if is_gt:
                    gt_extras_dict = {'truncation': np.atleast_1d(time_data[:, 3].astype(int)),
                                      'occlusion': np.atleast_1d(time_data[:, 4].astype(int))}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    if time_data.shape[1] > 17:
                        raw_data['tracker_confidences'][t] = np.atleast_1d(time_data[:, 17])
                    else:
                        raw_data['tracker_confidences'][t] = np.ones(time_data.shape[0])
            else:
                raw_data['dets'][t] = np.empty((0, 4))
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {'truncation': np.empty(0),
                                      'occlusion': np.empty(0)}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.empty(0)
            if is_gt:
                if time_key in ignore_data.keys():
                    time_ignore = np.asarray(ignore_data[time_key], dtype=float)
                    raw_data['gt_crowd_ignore_regions'][t] = np.atleast_2d(time_ignore[:, 6:10])
                else:
                    raw_data['gt_crowd_ignore_regions'][t] = np.empty((0, 4))

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'classes': 'gt_classes',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        KITTI:
            In KITTI, the 4 preproc steps are as follow:
                1) There are two classes (pedestrian and car) which are evaluated separately.
                2) For the pedestrian class, the 'person' class is distractor objects (people sitting).
                    For the car class, the 'van' class are distractor objects.
                    GT boxes marked as having occlusion level > 2 or truncation level > 0 are also treated as
                        distractors.
                3) Crowd ignore regions are used to remove unmatched detections. Also unmatched detections with
                    height <= 25 pixels are removed.
                4) Distractor gt dets (including truncated and occluded) are removed.
        """
        if cls == 'pedestrian':
            distractor_classes = [self.class_name_to_class_id['person']]
        elif cls == 'car':
            distractor_classes = [self.class_name_to_class_id['van']]
        else:
            raise (TrackEvalException('Class %s is not evaluatable' % cls))
        cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):

            # Only extract relevant dets for this class for preproc and eval (cls + distractor classes)
            gt_class_mask = np.sum([raw_data['gt_classes'][t] == c for c in [cls_id] + distractor_classes], axis=0)
            gt_class_mask = gt_class_mask.astype(bool)
            gt_ids = raw_data['gt_ids'][t][gt_class_mask]
            gt_dets = raw_data['gt_dets'][t][gt_class_mask]
            gt_classes = raw_data['gt_classes'][t][gt_class_mask]
            gt_occlusion = raw_data['gt_extras'][t]['occlusion'][gt_class_mask]
            gt_truncation = raw_data['gt_extras'][t]['truncation'][gt_class_mask]

            tracker_class_mask = np.atleast_1d(raw_data['tracker_classes'][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(bool)
            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
            tracker_confidences = raw_data['tracker_confidences'][t][tracker_class_mask]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

            # Match tracker and gt dets (with hungarian algorithm) and remove tracker dets which match with gt dets
            # which are labeled as truncated, occluded, or belonging to a distractor class.
            to_remove_matched = np.array([], int)
            unmatched_indices = np.arange(tracker_ids.shape[0])
            if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                match_rows = match_rows[actually_matched_mask]
                match_cols = match_cols[actually_matched_mask]

                is_distractor_class = np.isin(gt_classes[match_rows], distractor_classes)
                is_occluded_or_truncated = np.logical_or(
                    gt_occlusion[match_rows] > self.max_occlusion + np.finfo('float').eps,
                    gt_truncation[match_rows] > self.max_truncation + np.finfo('float').eps)
                to_remove_matched = np.logical_or(is_distractor_class, is_occluded_or_truncated)
                to_remove_matched = match_cols[to_remove_matched]
                unmatched_indices = np.delete(unmatched_indices, match_cols, axis=0)

            # For unmatched tracker dets, also remove those smaller than a minimum height.
            unmatched_tracker_dets = tracker_dets[unmatched_indices, :]
            unmatched_heights = unmatched_tracker_dets[:, 3] - unmatched_tracker_dets[:, 1]
            is_too_small = unmatched_heights <= self.min_height + np.finfo('float').eps

            # For unmatched tracker dets, also remove those that are greater than 50% within a crowd ignore region.
            crowd_ignore_regions = raw_data['gt_crowd_ignore_regions'][t]
            intersection_with_ignore_region = self._calculate_box_ious(unmatched_tracker_dets, crowd_ignore_regions,
                                                                       box_format='x0y0x1y1', do_ioa=True)
            is_within_crowd_ignore_region = np.any(intersection_with_ignore_region > 0.5 + np.finfo('float').eps, axis=1)

            # Apply preprocessing to remove all unwanted tracker dets.
            to_remove_unmatched = unmatched_indices[np.logical_or(is_too_small, is_within_crowd_ignore_region)]
            to_remove_tracker = np.concatenate((to_remove_matched, to_remove_unmatched), axis=0)
            data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            data['tracker_confidences'][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            # Also remove gt dets that were only useful for preprocessing and are not needed for evaluation.
            # These are those that are occluded, truncated and from distractor objects.
            gt_to_keep_mask = (np.less_equal(gt_occlusion, self.max_occlusion)) & \
                              (np.less_equal(gt_truncation, self.max_truncation)) & \
                              (np.equal(gt_classes, cls_id))
            data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
            data['gt_dets'][t] = gt_dets[gt_to_keep_mask, :]
            data['similarity_scores'][t] = similarity_scores[gt_to_keep_mask]

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']

        # Ensure that ids are unique per timestep.
        self._check_unique_ids(data)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='x0y0x1y1')
        return similarity_scores

def create_kitti_dataset(vnames, training=True, gt_video_dict={}):
    if training and len(gt_video_dict) == 0:
        gt_video_dict = {}
        for vname in vnames:
            if '-' in vname:
                _vname = vname.split('-')[1]
            else:
                _vname = vname
            with open(os.path.join(BASE, f'data/kitti/training/label_02/{_vname}.txt')) as fp:
                content = fp.read()
            gt_video_dict[vname] = content

    dataset = Kitti2DBox(gt_video_dict, training=training)

    return dataset


        

def evaluate_kitti(tracked_videos: dict, dataset: Kitti2DBox, parallel=0, idf1=False):
    global evaluator_config, metrics_config

    dataset.set_pred_videos(tracked_videos) # = tracked_videos

    if parallel > 0:
        evaluator_config = evaluator_config.copy()
        evaluator_config['USE_PARALLEL'] = True
        evaluator_config['NUM_PARALLEL_CORES'] = parallel

    metrics_list = [trackeval.metrics.HOTA(metrics_config)] #, trackeval.metrics.CLEAR(metrics_config)]
    if idf1:
        metrics_list.append( trackeval.metrics.Identity(metrics_config) )

    evaluator = trackeval.Evaluator(evaluator_config)
    
    res, msg = evaluator.evaluate([dataset], metrics_list)
    res = res['Kitti2DBox']['tracker']

    def parse_metrics(m):
        metrics = {
        'HOTA': m['HOTA']['HOTA'].mean(),
        'DetA': m['HOTA']['DetA'].mean(),
        'AssA': m['HOTA']['AssA'].mean(),
        'DetRe': m['HOTA']['DetRe'].mean(),
        'DetPr': m['HOTA']['DetPr'].mean(),
        'AssRe': m['HOTA']['AssRe'].mean(),
        'AssPr': m['HOTA']['AssPr'].mean(),
        # 'MOTA': m['CLEAR']['MOTA'],
        # 'IDSW': m['CLEAR']['IDSW'],
        # 'FP'  : m['CLEAR']['CLR_FP'],
        # 'FN'  : m['CLEAR']['CLR_FN'],
        }
        if idf1:
            metrics['IDF1'] = m['Identity']['IDF1']
        return metrics

    new_res = {}
    for k, vdict in res.items():
        vdict = { category: parse_metrics(m) for category, m in vdict.items() }
        new_res[k] = vdict

    return new_res, res





class KITTI_TrackBundle():

    def __init__(self, dataset):
        self.dataset = dataset
        self._tracked_dict = {}
        self.results = None
        self.metric = None

    def keys(self):
        return list(self._tracked_dict.keys())

    def has_tracked(self, name):
        if name in self._tracked_dict:
            return True
        else:
            return False

    def add(self, vname, tracked):
        self._tracked_dict[vname] = tracked

    def get_video(self, vname):
        return self._tracked_dict[vname]

    def get_res(self, force=False):
        if force or (self.metric is None):
            res, full_res = evaluate_kitti(self._tracked_dict, self.dataset)
            self.metric = res['COMBINED_SEQ']['car']
            self.metric = pd.Series(self.metric).to_frame().T
            self.results = [res, full_res]

        return self.metric

def get_total_bbox(keep, remove):
    nbbox = {}
    for vname in keep:
        nbbox[vname] = keep[vname].get_num_bbox() + remove[vname].get_num_bbox()
    return nbbox