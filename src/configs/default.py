from yacs.config import CfgNode as CN

_C = CN()

_C.aux = CN()
_C.aux.eval_every = 5
_C.aux.print_every = 1
_C.aux.wandb_project = "MOT"
_C.aux.wandb_id = None
_C.aux.exp = None
_C.aux.mark = ""
_C.aux.debug = False
_C.aux.runid = 0
_C.aux.gpu = 0
_C.aux.resume = "max"
_C.aux.skip_finished = True

_C.batch_size = 1
_C.epoch = 100
_C.optimizer = 'Adam'
_C.lr = 0.00005
_C.weight_decay = 0.0
_C.dataset = 'mot17'
_C.det = 'SDP' # detection
_C.fdr = 'zilu' # feature directory
_C.split = 'train'

_C.model = CN()
_C.model.inet = 'cnn' # image network
_C.model.tnet = 'tsmr' # track network
_C.model.cnet = 'attn' # classification network
_C.model.clip_len = 64 # length of input video clip
_C.model.step_size = 1 # temporal down-sampling factor
_C.model.mask_inp = None # mask out certain input modality
_C.model.include_confi = True # if include bbox confidence score as model inputs
_C.model.null_offset = False # learn offset for null class


_C.pcl = CN() # config for path consistency loss
_C.pcl.w = 1.0 # loss weight
_C.pcl.G = 25 # number of path to sample
_C.pcl.max_s = 64 # max frame skipping length
_C.pcl.ift = 1 # if applying supervision in the middle of paths
_C.pcl.M = 'sqrt' # spatial constraint masking


_C.tcl = CN() # bidrectional consistency loss
_C.tcl.w = 0.0

_C.mreg = CN() # one-to-one matching loss
_C.mreg.w = 0.0
_C.mreg.nullw = 0.5

_C.INET = CN()
_C.INET.layers = 6
_C.INET.hdim = 64
_C.INET.stride = 2
_C.INET.dropout = 0.0

_C.TNET = CN()
_C.TNET.hdim = 128
_C.TNET.ffndim = 512
_C.TNET.dropout = 0.0
_C.TNET.attn_dropout = 0.0
_C.TNET.salayers = 2
_C.TNET.calayers = 2
_C.TNET.ca_pos = True
_C.TNET.nhead = 8
_C.TNET.w = 4

_C.CNET = CN()
_C.CNET.layers = 2
_C.CNET.hdim = 64
_C.CNET.dropout = 0.0
_C.CNET.cmp = 'c2s'
_C.CNET.normalize = False 
_C.CNET.skq = False

_C.da = CN() # configurations for data augmentation (paper supplementary)
_C.da.enable = False
_C.da.size = 64
_C.da.scale = 2
_C.da.cs = 0.0 # center shift
_C.da.cs_min = 0.0
_C.da.zm = 0.0 # zoom in/out (not used)
_C.da.flip = False # image flip (not used)
_C.da.clw = 0.0 

def get_cfg_defaults():
    return _C.clone()

RENAME_KEYS = {}