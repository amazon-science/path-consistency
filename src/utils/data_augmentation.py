import numpy as np
import skimage
from multiprocessing import Pool

def crop_and_resize_image(im, det, scale=1, CROP_SIZE=64, embed=True):
    C = CROP_SIZE * scale
    H, W, _ = im.shape
    fix_crop = np.zeros((C, C, 3), dtype='uint8')
    center_x = (det['left'] + det['right']) / 2
    center_y = (det['top'] + det['bottom']) / 2

    w = det['width'] * scale / 2
    h = det['height'] * scale / 2
    x1 = max(0, int(center_x - w))
    x2 = min(W, int(center_x + w))
    y1 = max(0, int(center_y - h))
    y2 = min(H, int(center_y + h))

    w = det['width'] / 2
    h = det['height'] / 2
    ix1 = min(W, max(0, int(center_x - w)))
    ix2 = min(W, max(0, int(center_x + w)))
    iy1 = min(H, max(0, int(center_y - h)))
    iy2 = min(H, max(0, int(center_y + h)))

    h_ = iy2 - iy1
    w_ = ix2 - ix1
    resize_factor = min([float(CROP_SIZE) / h_, float(CROP_SIZE) / w_])

    # center_x = (det['left'] + det['right']) / 2
    # center_y = (det['top'] + det['bottom']) / 2
    # w = det['width'] * scale / 2
    # h = det['height'] * scale / 2
    # x1 = center_x - w
    # x2 = center_x + w
    # y1 = center_y - h
    # y2 = center_y + h
    
    # x1 = max(0, int(x1))
    # x2 = min(W, int(x2))
    # y1 = max(0, int(y1))
    # y2 = min(H, int(y2))
    
    crop = im[y1:y2, x1:x2]
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        return fix_crop

    # resize_factor = min([float(C) / crop.shape[0], float(C) / crop.shape[1]])
    resize_shape = [int(crop.shape[0] * resize_factor), int(crop.shape[1] * resize_factor)]
    if resize_shape[0] > C:
        r = C / resize_shape[0]
        resize_shape = [ resize_shape[0] * r, resize_shape[1] * r ]
    if resize_shape[1] > C:
        r = C / resize_shape[1]
        resize_shape = [ resize_shape[0] * r, resize_shape[1] * r ]
        
    # resize_shape = [ min(CROP_SIZE, resize_shape[0]), min(CROP_SIZE, resize_shape[1]) ]
    if resize_shape[0] == 0 or resize_shape[1] == 0:
        print(det) 
        assert False
    # if resize_shape[0] == 0 or resize_shape[1] == 0:
    #     return fix_crop

    crop = (skimage.transform.resize(crop, resize_shape)*255).astype('uint8')

    h, w, c = crop.shape
    h_head = (C - h) // 2
    w_head = (C - w) // 2
    # try:
    fix_crop[h_head:h_head+h, w_head:w_head+w, :] = crop
    # except Exception as e:
    #     import ipdb; ipdb.set_trace()

    if embed:
        return fix_crop
    else:
        return crop

class Translator():

    def __init__(self, img, x1, y1, x2, y2, 
                    ix1, iy1, ix2, iy2):

        self.img = img
        
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.cx = (ix1 + ix2) / 2
        self.cy = (iy1 + iy2) / 2

        self.inner_w = ix2 - ix1
        self.inner_h = iy2 - iy1 

        self.ix1 = ix1
        self.ix2 = ix2
        self.iy1 = iy1
        self.iy2 = iy2

    def get_shifted_center(self, ratio):
        cx = self.cx + ratio[0] * self.inner_w
        cy = self.cy + ratio[1] * self.inner_h
        return cx, cy
    
    def get_zoomed_wh(self, ratio):
        w = self.inner_w * ratio[0]
        h = self.inner_h * ratio[1]
        if not (ratio[0] == 1 and ratio[1] == 1):
            assert False # current cx, cy is not the real center of bbox, due to image clip, 
                        # thus zoom in and out can be incorrect
        return w, h

    def get_corner_xy(self, cx, cy, w, h):
        x1 = max(self.x1, int(cx - w / 2))
        x2 = min(self.x2, int(cx + w / 2))
        y1 = max(self.y1, int(cy - h / 2))
        y2 = min(self.y2, int(cy + h / 2))

        return x1, x2, y1, y2

    def rescale_if_necessary(self, crop, C):
        resize_factor = min([float(C) / crop.shape[0], float(C) / crop.shape[1]])
        assert resize_factor == 1
        if resize_factor == 1:
            return crop

        resize_shape = [int(crop.shape[0] * resize_factor), int(crop.shape[1] * resize_factor)]
        crop = (skimage.transform.resize_local_mean(crop, resize_shape)*255).astype('uint8')
        return crop
    
    def get_image(self, cs_ratio, zm_ratio, flip=False, frame_size=None):
        assert zm_ratio == (1, 1), zm_ratio
        cx, cy = self.get_shifted_center(cs_ratio)
        w, h = self.get_zoomed_wh(zm_ratio)
        x1, x2, y1, y2 = self.get_corner_xy(cx, cy, w, h) 
        p = self.img[:, y1:y2, x1:x2]
        if p.shape[0] == 0 or p.shape[1] == 0:
            p = self.img[:, int(self.iy1):int(self.iy2), int(self.ix1):int(self.ix2)]

        if flip:
            p = np.flip(p, axis=2)

        if frame_size is not None: 
            frame = np.zeros((3, frame_size, frame_size), dtype=self.img.dtype)
            w, h = p.shape[2], p.shape[1]
            h_head = (frame_size - h) // 2
            w_head = (frame_size - w) // 2
            frame[:, h_head:h_head+h, w_head:w_head+w] = p
            
            p = frame
        
        # get new bbox
        cx = (self.det['left'] + self.det['right']) / 2
        cy = (self.det['top'] + self.det['bottom']) / 2
        cx = float(cx)
        cy = float(cy)
        w = self.det['width'] 
        h = self.det['height']
        assert zm_ratio[0] == 1 and zm_ratio[1] == 1
        cx = cx + cs_ratio[0] * w
        cy = cy + cs_ratio[1] * h

        return [cx, cy, w, h ], p

    def __str__(self) -> str:
        return f"{self.x1},{self.y1},{self.x2},{self.y2} - {self.inner_w},{self.inner_h}"

    def __repr__(self) -> str:
        return str(self)


    @staticmethod
    def create(H, W, det, image, scale=1, CROP_SIZE=64):
        C = CROP_SIZE * scale

        center_x = (det['left'] + det['right']) / 2
        center_y = (det['top'] + det['bottom']) / 2

        w = det['width'] * scale / 2
        h = det['height'] * scale / 2
        x1 = max(0, int(center_x - w))
        x2 = min(W, int(center_x + w))
        y1 = max(0, int(center_y - h))
        y2 = min(H, int(center_y + h))

        w = det['width'] / 2
        h = det['height'] / 2
        ix1 = max(0, center_x - w)
        ix2 = min(W, center_x + w)
        iy1 = max(0, center_y - h)
        iy2 = min(H, center_y + h)

        offset = [ix1-x1, iy1-y1, x2-ix2, y2-iy2]

        h_ = iy2 - iy1
        w_ = ix2 - ix1
        resize_factor = min([float(CROP_SIZE) / h_, float(CROP_SIZE) / w_])

        h = int(resize_factor * (y2-y1))
        w = int(resize_factor * (x2-x1))
        if h > C:
            resize_factor = (C / h) * resize_factor
            h = int(resize_factor * (y2-y1))
            w = int(resize_factor * (x2-x1))
        if w > C:
            resize_factor = (C / w) * resize_factor
            h = int(resize_factor * (y2-y1))
            w = int(resize_factor * (x2-x1))

        offset = [ o * resize_factor for o in offset ]

        x1 = (C - w) // 2
        y1 = (C - h) // 2
        x2 = x1 + w
        y2 = y1 + h


        ix1 = x1+offset[0]
        iy1 = y1+offset[1]
        ix2 = x2-offset[2]
        iy2 = y2-offset[3]

        assert x1 >= 0 and y1 >= 0

        tlr = Translator(image, x1, y1, x2, y2, ix1, iy1, ix2, iy2)

        tlr.det = det

        return tlr

