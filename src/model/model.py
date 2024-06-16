import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
from .layers import ATTWeightHead
from . import loss_tools
from ..utils.utils import generate_dist2diag_matrix

class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.model.inet == 'cnn':
            layers = []
            for i in range(cfg.INET.layers):
                layers.append(nn.Dropout(cfg.INET.dropout))
                in_channel = 3 if i==0 else cfg.INET.hdim
                layers.append(nn.Conv2d(in_channel, cfg.INET.hdim, 3, cfg.INET.stride, padding=1))
                if i != cfg.INET.layers-1:
                    layers.append(nn.ReLU())
            self.inet = nn.Sequential(*layers)

        if cfg.model.tnet == 'tsmr':
            bbox_ndim = 5 if cfg.model.include_confi else 4
            self.null_object_feature = nn.Parameter(torch.randn(1, 1, 1, cfg.INET.hdim+bbox_ndim)) # b, t, n, h
            self.channel_map = nn.Linear(cfg.INET.hdim+bbox_ndim, cfg.TNET.hdim)
            salayers = []
            for i in range(cfg.TNET.salayers):
                salayers.append(AttentionLayer(cfg.TNET.hdim, cfg.TNET.nhead, dim_feedforward=cfg.TNET.ffndim, 
                                dropout=cfg.TNET.dropout, attn_dropout=cfg.TNET.attn_dropout))
            self.salayers = nn.ModuleList(salayers)

            calayers = []
            for i in range(cfg.TNET.calayers):
                calayers.append(AttentionLayer(cfg.TNET.hdim, cfg.TNET.nhead, dim_feedforward=cfg.TNET.ffndim, 
                                dropout=cfg.TNET.dropout, attn_dropout=cfg.TNET.attn_dropout, attn_window=cfg.TNET.w))
            self.calayers = nn.ModuleList(calayers)

            self.frame_pe = nn.Parameter(torch.randn(cfg.model.clip_len, cfg.TNET.hdim))
        
        if cfg.model.cnet == 'attn':
            self.cnet = ATTWeightHead(cfg.TNET.hdim, cfg.CNET.hdim, cfg.CNET.layers, cfg.CNET.dropout, 
                                        share_kq=cfg.CNET.skq)


        if cfg.model.null_offset:
            self.null_offset = nn.Parameter(torch.zeros(cfg.model.clip_len))
            self.distM = generate_dist2diag_matrix(cfg.model.clip_len, torch_tensor=True).long().cuda() # T, T

        # self.buffered_clips = []
        # self.buffered_vnames = []
        # self.buffer_size = cfg.bnl.bsize

    def _iou(self, bbox_tensor):
        """
        From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
        bbox_tensor: B, T, N, 4  --> center_x, center_y, w, h
        """

        B, T, N, _ = bbox_tensor.shape
        bbox_tensor = bbox_tensor[..., :4]
        bbox_tensor = bbox_tensor.view(B, T * N, 4)
        half_w = bbox_tensor[:, :, -2] / 2
        half_h = bbox_tensor[:, :, -1] / 2
        ltrb = [ bbox_tensor[:, :, 0] - half_w, bbox_tensor[:, :, 1] - half_h, 
                    bbox_tensor[:, :, 0] + half_w, bbox_tensor[:, :, 1] + half_h   ]
        
        ltrb_1 = [ x.unsqueeze(1) for x in ltrb ]
        ltrb_2 = [ x.unsqueeze(2) for x in ltrb ]

        xx1 = torch.maximum(ltrb_1[0], ltrb_2[0])
        yy1 = torch.maximum(ltrb_1[1], ltrb_2[1])
        xx2 = torch.minimum(ltrb_1[2], ltrb_2[2])
        yy2 = torch.minimum(ltrb_1[3], ltrb_2[3])
        w = torch.clamp_min(xx2 - xx1, 0)
        h = torch.clamp_min(yy2 - yy1, 0)
        wh = w * h
        o = wh / ((ltrb_2[2] - ltrb_2[0]) * (ltrb_2[3] - ltrb_2[1])                                      
        + (ltrb_1[2] - ltrb_1[0]) * (ltrb_1[3] - ltrb_1[1]) - wh)                                              
        o = torch.nan_to_num(o, nan=0)
        o = o.view(B, T, N, T, N)

        return(o) 

    def forward(self, bbox_tensor, image_tensor, nbbox, transformed_data=False):
        """
        bbox_tensor:
        """
        if transformed_data:
            origin_B = image_tensor.shape[0]
            origin_data = [ bbox_tensor, image_tensor, nbbox ]
            b, i, n = transformed_data

            assert i.shape == image_tensor.shape
            bbox_tensor = torch.cat([bbox_tensor, b], dim=0)
            image_tensor = torch.cat([image_tensor, i], dim=0)
            nbbox = torch.cat([nbbox, n], dim=0)


        B, T, N, C, H, W = image_tensor.shape


        if self.cfg.model.mask_inp is None:
            pass
        elif self.cfg.model.mask_inp == 'visual':
            image_tensor = torch.zeros_like(image_tensor)
        elif self.cfg.model.mask_inp == 'spatial':
            bbox_tensor = torch.zeros_like(bbox_tensor)
        else:
            raise ValueError(self.cfg.model.mask_inp)

        # INET
        image_tensor = image_tensor.view(B*T*N, C, H, W)
        image_feature = self.inet(image_tensor)
        image_feature = image_feature.squeeze(-1).squeeze(-1).view(B, T, N, -1) # B, T, N, D
        self.raw_cnn_feature = image_feature

        # TNET
        null_object = self.null_object_feature.expand(B, T, 1, -1)
        object_feature = torch.cat([image_feature, bbox_tensor], dim=-1) # B, T, N, D+4
        object_feature = torch.cat([null_object, object_feature], dim=-2) # B, T, N+1, D+4
        key_padding_mask = torch.ones((B, T, N+1), dtype=torch.bool).to(object_feature.device)
        for b in range(B):
            for t in range(T):
                key_padding_mask[b, t, :nbbox[b, t]+1] = 0
        self.key_padding_mask = key_padding_mask

        ## self attention
        object_feature = object_feature.view(B*T, N+1, -1)
        object_feature = self.channel_map(object_feature)
        self.cnn_feature = object_feature.view(B, T, N+1, -1)
        mask = key_padding_mask.view(B*T, N+1)
        for SA in self.salayers:
            object_feature = SA(object_feature, object_feature, object_feature,
                                    key_padding_mask=mask)
        self.all_sa_feature = self.sa_feature = object_feature = object_feature.view(B, T, N+1, -1) # B, T, N+1, D
        
        ## cross attention
        if self.cfg.TNET.ca_pos:
            pos = self.frame_pe.unsqueeze(0).unsqueeze(2).expand(B, T, N+1, -1)
        else:
            pos = None
        for CA in self.calayers:
            object_feature = CA(object_feature, object_feature, object_feature,
                                query_pos=pos, key_pos=pos, 
                                key_padding_mask=key_padding_mask)
        
        self.all_ca_feature = self.ca_feature = object_feature #[:, :, 1:] # B, T, N, D

        # CNET
        if transformed_data:

            fw_da_alignment_score = self.cnet.forward_for_da(
                self.all_ca_feature[:origin_B, :, 1:], self.all_sa_feature[origin_B:]) # B, T, N, N+1
            bw_da_alignment_score = self.cnet.forward_for_da(
                self.all_ca_feature[origin_B:, :, 1:], self.all_sa_feature[:origin_B]) # B, T, N, N+1
            da_alignment_score = torch.stack([fw_da_alignment_score, bw_da_alignment_score], dim=1) # B, 2, T, N, N+1

            mask = key_padding_mask[origin_B:].unsqueeze(2).unsqueeze(1) # B, T, N+1 -> B, 1, T, 1, N+1
            da_alignment_score.masked_fill_(mask, -float('inf'))

            self.da_alignment_score = da_alignment_score
            # if self.cfg.model.null_zero:
            #     self.da_alignment_score[..., 0] = 0
            self.da_key_padding_mask = key_padding_mask[origin_B:]

            self.ca_feature = self.ca_feature[:origin_B]
            self.sa_feature = self.sa_feature[:origin_B]
            self.key_padding_mask = key_padding_mask = key_padding_mask[:origin_B] # B, T, N+1
            bbox_tensor = origin_data[0]

            # print(self.all_ca_feature[0, 0, 1])
            # print(self.all_ca_feature[1, 0, 1])
            # import ipdb; ipdb.set_trace()
            # print(image_tensor]

        if self.cfg.CNET.cmp == 'c2s':
            alignment_score = self.cnet(self.ca_feature[:, :, 1:], self.sa_feature) # B, T, N, T_, N+1, 
        elif self.cfg.CNET.cmp == 'c2c':
            alignment_score = self.cnet(self.ca_feature[:, :, 1:], self.ca_feature)
    
        # if self.cfg.model.null_zero:
        #     alignment_score[..., 0] = 0

        if self.cfg.model.null_offset:
            offset = torch.cumsum(self.null_offset, dim=0)
            offset = offset[self.distM] # T x T
            T = offset.shape[0]
            offset = offset.view(1, T, 1, T)
            alignment_score[..., 0] = alignment_score[..., 0] + offset

        mask = key_padding_mask.unsqueeze(1).unsqueeze(1) # B, 1, 1, T_, N+1
        self.alignment_score_mask = mask
        alignment_score.masked_fill_(mask, -float('inf'))
        if self.cfg.CNET.normalize:
            alignment_score = alignment_score / math.sqrt(self.ca_feature.shape[-1])

        alignment_prob = torch.softmax(alignment_score, dim=-1)

        self.alignment_score = alignment_score
        self.alignment_prob = alignment_prob

        return alignment_prob

    def _compute_log_prob(self, alignment_prob):
        updated_prob = torch.clamp_min(alignment_prob, min=1e-6)
        log_alignment_prob = torch.log(updated_prob)
        self.log_alignment_prob = log_alignment_prob
        return log_alignment_prob

    def compute_da_loss(self):
        """
        self.da_alignment_prob # B, 2, T, N, N+1
        self.da_key_padding_mask # B, T, N+1
        """

        logprob = torch.log_softmax(self.da_alignment_score, dim=-1)
        match = torch.diagonal(logprob[..., 1:], dim1=-2, dim2=-1) # B, T, N
        mask = ~self.da_key_padding_mask[:, None, ..., 1:].expand(-1, 2, -1, -1)
        loss1 = -match[mask].mean()

        return loss1

    def forward_and_loss(self, vnames, bbox_tensor, image_tensor, nbbox, batch_object_samples, batch_frame_idx, global_iteration, transformed_data=None):
        alignment_prob = self.forward(bbox_tensor, image_tensor, nbbox, transformed_data)
        log_alignment_prob = self._compute_log_prob(alignment_prob)

        cfg = self.cfg
        loss_dict = OrderedDict()

        if cfg.da.enable and cfg.da.clw > 0:
            assert transformed_data is not None
            loss1 = self.compute_da_loss()
            loss_dict['dacl'] = [ cfg.da.clw, loss1 ]

        if cfg.pcl.w > 0:
            B = alignment_prob.shape[0]
            cumu_loss = []
            for b in range(B):
                l = loss_tools.pcl(log_alignment_prob[b], batch_object_samples[b], cfg, batch_frame_idx[b])

                cumu_loss.append(l)
            cumu_loss = sum(cumu_loss) / len(cumu_loss)

            loss_dict['pcl'] = [ cfg.pcl.w, cumu_loss ]

        if cfg.tcl.w > 0:
            fw_prob = alignment_prob[..., 1:]
            bw_prob = fw_prob.permute(0, 3, 4, 1, 2)
            tcl = ((fw_prob - bw_prob)**2).mean()
            loss_dict['tcl'] = [ cfg.tcl.w, tcl]
        
        if cfg.mreg.w > 0:
            l = loss_tools.mreg_loss(cfg, alignment_prob)
            loss_dict['mreg'] = [ cfg.mreg.w, l ]

        loss = torch.tensor(0, dtype=torch.float32, device=log_alignment_prob.device)
        for k, (w, l) in loss_dict.items():
            loss += w * l
        
        self.loss_dict = loss_dict
        loss_dict = { k: l.item() for k, (w, l) in loss_dict.items() }
        loss_dict['overall'] = loss.item()
        return loss, loss_dict

    def save_model(self, fname):
        torch.save(self.state_dict(), fname)

#===============================================
#===============================================

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def generate_mask(key_padding_mask, w):
    """
    Input -> key_padding_mask: B, T, N
    Return -> B, TxN, W
    """
    B, T, N = key_padding_mask.shape
    # mask = key_padding_mask.view(B, T*N)
    W = 2*w+1

    ref_mask2 = torch.ones(B, W, T, N).to(key_padding_mask.device).bool()
    for i in range(W):
        start1 = max(w-i, 0)
        start2 = max(0, i-w)
        len_ = T- max(start1, start2)
        ref_mask2[:, i, start1:start1+len_] = key_padding_mask[:, start2:start2+len_]
    
    # ref_mask2 = ref_mask2.transpose(2, 1)
    return ref_mask2

def generate_mask_reference(key_padding_mask, w):
    """
    key_padding_mask: B, T, N
    """
    B, T, N = key_padding_mask.shape
    # mask = key_padding_mask.view(B, T*N)
    W = 2*w+1

    ref_mask = torch.ones(B, T, W, N).to(key_padding_mask.device)
    for t in range(T):
        for i in range(W):
            if t+i-w<0:
                continue
            if t+i-w>=T:
                continue
            ref_mask[:, t, i] = key_padding_mask[:, t+i-w]

    return ref_mask

def windowed_attention_reference(q, k, w):
    B, T, N, nhead, D = q.shape
    W = 2*w+1
    window_key = torch.zeros([B, T, W, N, nhead, D]).to(q.device)
    for t in range(T):
        for i in range(W):
            if t+i-w<0:
                continue
            if t+i-w>=T:
                continue

            window_key[:, t, i] = k[:, t+i-w]
    
    logit = torch.einsum('btnhd,btwmhd->btnhwm', q, window_key)
    logit = logit.reshape(B, T*N, nhead, W*N) # B, TxN, nhead, wxN

    return window_key, logit
        
def windowed_attention_qk(q, k, w):
    """
    Input: B, T, N, nhead, D
    Return: B, T, N, W, N, nhead
    """
    B, T, N, nhead, D = q.shape
    W = 2*w+1
    window_key = torch.zeros([B, W, T, N, nhead, D]).to(q.device)
    for t in range(W):
        start1 = max(w-t, 0)
        start2 = max(0, t-w)
        len_ = T - max(start1, start2)
        # wk: B, W, T, N, h, D              k: B, T, N, h, D
        window_key[:, t, start1:start1+len_] = k[:, start2:start2+len_]

    logit = torch.einsum('btnhd,bwtmhd->btnwmh', q, window_key)

    return window_key, logit

def windowed_attention_pv(attn, v):
    """
    Input: 
        attn_weights: B, T, N, WxN, nhead
        v: B, T, N, nhead, D
    Return: 
        v' = B, T, N, nhead, D
    """
    B, T, N, nhead, D = v.shape
    W = attn.shape[3] // N
    w = (W-1) // 2

    window_value = torch.zeros([B, W, T, N, nhead, D]).to(v.device)
    for t in range(W):
        start1 = max(w-t, 0)
        start2 = max(0, t-w)
        len_ = T - max(start1, start2)
        # wk: B, W, T, N, h, D              k: B, T, N, h, D
        window_value[:, t, start1:start1+len_] = v[:, start2:start2+len_]
    window_value = window_value.transpose(1, 2).reshape(B, T, W*N, nhead, D)

    logit = torch.einsum('btnmh,btmhd->btnhd', attn, window_value) # B, T, N, nhead, D

    return window_value, logit

class WindowedSelfAttention(nn.Module):
    def __init__(self, embed_dim, nhead, attn_window, attn_dilation=1, kdim=None, vdim=None, dropout=0.5, batch_first=True):
        super().__init__()
        assert batch_first
        assert attn_dilation == 1
        self.num_heads = nhead
        self.embed_dim = embed_dim
        self.head_dim = int(embed_dim / nhead)
        if kdim is None:
            kdim = embed_dim
        if vdim is None:
            vdim = embed_dim

        self.query = nn.Linear(embed_dim, self.embed_dim)
        self.key = nn.Linear(kdim, self.embed_dim)
        self.value = nn.Linear(vdim, self.embed_dim)

        self.dropout = dropout

        self.attention_window = attn_window
        self.attention_dilation = attn_dilation

    def forward(
        self,
        query, key, value,
        key_padding_mask=None,
        attn_mask=None,
        average_attn_weights=False,
    ):
        '''
        query: B, T, N, H
        attn_mask: B, TxN, W
        '''
        assert attn_mask is None
        assert not average_attn_weights

        B, T, N, embed_dim = query.shape
        W = 2 * self.attention_window + 1
        assert embed_dim == self.embed_dim
        q = self.query(query).view(B, T, N, self.num_heads, self.head_dim)
        k = self.key(key).view(B, T, N, self.num_heads, self.head_dim)
        v = self.value(value).view(B, T, N, self.num_heads, self.head_dim)
        q /= math.sqrt(self.head_dim)

        _, attn_weights = windowed_attention_qk(q, k, self.attention_window) # B, T, N, W, N_, nhead
        mask = generate_mask(key_padding_mask, self.attention_window) # B, W, T, N_
        mask = mask.transpose(1, 2) # B, T, W, N_
        attn_weights.masked_fill_( mask.unsqueeze(2).unsqueeze(-1), -float('inf') )
        self.attn_weights = attn_weights

        attn_weights = attn_weights.view(B, T, N, W*N, self.num_heads)
        attn = torch.softmax(attn_weights, dim=-2)
        _, updated_query = windowed_attention_pv(attn, v)
        updated_query = updated_query.reshape(B, T, N, embed_dim)
        return updated_query, attn

class AttentionLayer(nn.Module):
    """
    self or cross attention
    """

    def __init__(self, q_dim, nhead, dim_feedforward=2048, kv_dim=None,
                 dropout=0.1, attn_dropout=0.1,
                 attn_window=None,
                 activation="relu"):
        super().__init__()


        kv_dim = q_dim if kv_dim is None else kv_dim
        if attn_window is None:
            self.multihead_attn = nn.MultiheadAttention(q_dim, nhead, 
                    kdim=kv_dim, vdim=kv_dim, dropout=attn_dropout, batch_first=True)
        else:
            self.multihead_attn = WindowedSelfAttention(q_dim, nhead, attn_window, 
                    kdim=kv_dim, vdim=kv_dim, dropout=attn_dropout, batch_first=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(q_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, q_dim)

        self.norm1 = nn.LayerNorm(q_dim)
        self.norm2 = nn.LayerNorm(q_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.q_dim = q_dim
        self.kv_dim=kv_dim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.attn_window = attn_window

    def __str__(self) -> str:
        if self.attn_window is None:
            return f"AttentionLayer(SA: {self.q_dim}x{self.kv_dim}->{self.q_dim}, {self.nhead}, FFN: {self.dim_feedforward})"
        else:
            return f"AttentionLayer(CA{self.attn_window}: {self.q_dim}x{self.kv_dim}->{self.q_dim}, {self.nhead}, FFN: {self.dim_feedforward})"
    
    def __repr__(self):
        return str(self)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, value, 
            query_pos = None,
            key_pos = None,
            value_pos = None,
            key_padding_mask = None,
            attn_mask=None,):

        query=self.with_pos_embed(query, query_pos)
        key=self.with_pos_embed(key, key_pos)
        value=self.with_pos_embed(value, value_pos)

        # attn
        query2, self.attn = self.multihead_attn(query, key, value, 
                    key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                    average_attn_weights=False) # attn: nhead, batch, q, k
        self.attn_feature = query2 
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # ffn
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        return query
