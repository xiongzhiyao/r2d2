# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use


import os, pdb
from PIL import Image
import numpy as np
import torch

from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *

from adalam import AdalamFilter
import cv2
import matplotlib.pyplot as plt

import json

def show_matches(img1, img2, k1, k2, target_dim=800.):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    def resize_horizontal(h1, w1, h2, w2, target_height):
        scale_to_align = float(h1) / h2
        current_width = w1 + w2 * scale_to_align
        scale_to_fit = target_height / h1
        target_w1 = int(w1 * scale_to_fit)
        target_w2 = int(w2 * scale_to_align * scale_to_fit)
        target_h = int(target_height)
        return (target_w1, target_h), (target_w2, target_h), scale_to_fit, scale_to_fit * scale_to_align, [target_w1, 0]

    target_1, target_2, scale1, scale2, offset = resize_horizontal(h1, w1, h2, w2, target_dim)

    im1 = cv2.resize(img1, target_1, interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(img2, target_2, interpolation=cv2.INTER_AREA)

    h1, w1 = target_1[::-1]
    h2, w2 = target_2[::-1]

    vis = np.ones((max(h1, h2), w1 + w2, 3), np.uint8) * 255
    vis[:h1, :w1] = im1
    vis[:h2, w1:w1 + w2] = im2

    p1 = [np.int32(k * scale1) for k in k1]
    p2 = [np.int32(k * scale2 + offset) for k in k2]

    for (x1, y1), (x2, y2) in zip(p1, p2):
        cv2.line(vis, (x1, y1), (x2, y2), [0, 255, 0], 1)

    visrgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20,10))
    plt.imshow(visrgb)
    plt.show()

def load_network(model_fn): 
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net']) 
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()


class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr
    
    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability   >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


def extract_multiscale( net, img, detector, scale_f=2**0.25, 
                        min_scale=0.0, max_scale=1, 
                        min_size=256, max_size=1024, 
                        verbose=False):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    assert max_scale <= 1
    s = 1.0 # current scale factor
    
    X,Y,S,C,Q,D = [],[],[],[],[],[]
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])
                
            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y,x = detector(**res) # nms
            c = reliability[0,0,y,x]
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S) # scale
    scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
    XYS = torch.stack([X,Y,S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores


def extract_keypoints(args, img_path):
    iscuda = common.torch_set_gpu(args.gpu)

    # load the network...
    net = load_network(args.model)
    if iscuda: net = net.cuda()

    # create the non-maxima detector
    detector = NonMaxSuppression(
    rel_thr = args.reliability_thr, 
    rep_thr = args.repeatability_thr)
    
    print(f"\nExtracting features for {img_path}")
    img = Image.open(img_path).convert('RGB')
    W, H = img.size
    img = norm_RGB(img)[None] 
    if iscuda: img = img.cuda()
    
    # extract keypoints/descriptors for a single image
    xys, desc, scores = extract_multiscale(net, img, detector,
        scale_f   = args.scale_f, 
        min_scale = args.min_scale, 
        max_scale = args.max_scale,
        min_size  = args.min_size, 
        max_size  = args.max_size, 
        verbose = True)

    xys = xys.cpu().numpy()
    desc = desc.cpu().numpy()
    scores = scores.cpu().numpy()
    idxs = scores.argsort()[-args.top_k or None:]
  
    imsize = (W,H)
    keypoints = xys[idxs]
    descriptors = desc[idxs]
    scores = scores[idxs]

    return imsize, keypoints, descriptors, scores

def extract_pair(args, img1, img2):
    size1, k1_raw, d1, s1 = extract_keypoints(args, img1)
    size2, k2_raw, d2, s2 = extract_keypoints(args, img2)
    matcher = AdalamFilter(custom_config={
    'scale_rate_threshold':None,
    'orientation_difference_threshold':None
    })
    k1 = k1_raw[:,:2]
    k2 = k2_raw[:,:2]
    matches = matcher.match_and_filter(k1=k1, k2=k2,
                                      d1=d1, d2=d2,
                                      im1shape=size1,
                                      im2shape=size2).cpu().numpy()
    # canvas0 = cv2.imread(img1)
    # canvas1 = cv2.imread(img2)
    #show_matches(canvas0, canvas1, k1=k1[matches[:, 0]], k2=k2[matches[:, 1]]) 

    result = {}
    left_kps = [[int(kp[0]), int(kp[1])] for kp in k1]
    right_kps = [[int(kp[0]), int(kp[1])] for kp in k2]
    left_based_match = np.ones(len(left_kps), dtype=np.int)
    left_based_match *= -1
    for i in range(len(matches)):
        left_based_match[matches[i, 0]] = matches[i, 1]

    result['keypoints0'] = left_kps
    result['keypoints1'] = right_kps
    result['matches'] = left_based_match

    return result 

class NPToJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NPToJSONEncoder, self).default(obj)
   
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Extract keypoints for a given image")
    parser.add_argument("--model", type=str, default='models/r2d2_WASF_N16.pt', help='model path')
    
    parser.add_argument("--tag", type=str, default='r2d2', help='output file tag')
    
    parser.add_argument("--top-k", type=int, default=5000, help='number of keypoints')

    parser.add_argument("--scale-f", type=float, default=2**0.25)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    
    parser.add_argument("--reliability-thr", type=float, default=0.7)
    parser.add_argument("--repeatability-thr", type=float, default=0.7)

    parser.add_argument("--gpu", type=int, nargs='+', default=-1, help='use -1 for CPU')
    args = parser.parse_args()

    img1 = './imgs/image_1883020_order_225988.jpg'
    img2 = './imgs/image_1883037_order_225988.jpg'
    result = extract_pair(args, img1, img2)
    save_file = f'test.json'
    with open(save_file, 'w') as outfile:
        json.dump(result, outfile, cls=NPToJSONEncoder)
        print(f"saved {save_file}")

