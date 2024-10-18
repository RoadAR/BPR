import argparse
import os 
import os.path as osp
import torch 
import torch.nn.functional as F 
import numpy as np
import cv2
import mmcv
from mmcv.ops.nms import nms
from mmcv.ops.roi_align import roi_align
from tqdm import tqdm
from functools import partial
from torch.utils.data import Dataset, DataLoader

from mmengine.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmcv.parallel import MMDataParallel, DataContainer, collate



def find_float_boundary(maskdt, width):
    """Find the boundaries.

    Args:
        maskdt (tensor): shape N, H, W
        width (int): boundary width.

    Returns:
        tensor: shape N, H, W
    """
    N, H, W = maskdt.shape
    maskdt = maskdt.view(N, 1, H, W)
    boundary_finder = maskdt.new_ones((1, 1, width, width))
    boundary_mask = F.conv2d(maskdt, boundary_finder, 
                    stride=1, padding=width//2)
    bml = torch.abs(boundary_mask - width*width)
    bms = torch.abs(boundary_mask)
    fbmask = torch.min(bml, bms) / (width*width/2)
    return fbmask.view(N, H, W)


def _force_move_back(sdets, H, W, patch_size):
    # force the out of range patches to move back
    s = sdets[:, 0] < 0 
    sdets[s, 0] = 0
    sdets[s, 2] = patch_size

    s = sdets[:, 1] < 0 
    sdets[s, 1] = 0
    sdets[s, 3] = patch_size

    s = sdets[:, 2] >= W 
    sdets[s, 0] = W - 1 - patch_size
    sdets[s, 2] = W - 1

    s = sdets[:, 3] >= H 
    sdets[s, 1] = H - 1 - patch_size
    sdets[s, 3] = H - 1
    return sdets


def get_dets(fbmask, patch_size, iou_thresh=0.3):
    """boundaries of coarse mask -> patch bboxs

    Args:
        fbmask (tensor): H,W, float boundary mask
        patch_size (int): [description]
        iou_thresh (float, optional): useful for nms. Defaults to 0.3.

    Returns:
        tensor: filtered bboxs. x1, y1, x2, y2, score
    """
    ys, xs = torch.nonzero(fbmask, as_tuple=True)
    scores = fbmask[ys,xs]
    ys = ys.float()
    xs = xs.float()
    dets = torch.stack([xs-patch_size//2, ys-patch_size//2, 
            xs+patch_size//2, ys+patch_size//2, scores]).T
    _, inds = nms(dets[:,:4].contiguous(), 
        dets[:,4].contiguous(), iou_thresh)
    sdets = dets[inds]

    H, W = fbmask.shape
    return _force_move_back(sdets, H, W, patch_size)


class PatchDataset(Dataset):
    def __init__(self, imgs_cap, masks_cap, target_class, device, out_size=(128,128)):
        self.device = device
        self.out_size = out_size
        self.img_mean = np.array([123.675, 116.28, 103.53]).reshape(1,1,3)
        self.img_std = np.array([58.395, 57.12, 57.375]).reshape(1,1,3)
        self.imgs_cap = imgs_cap
        self.masks_cap = masks_cap
        self.target_class = target_class

    def __len__(self):
        return int(min(self.imgs_cap.get(cv2.CAP_PROP_FRAME_COUNT), self.masks_cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    def __getitem__(self, i):
        ret_img, img = self.imgs_cap.read()
        ret_mask, mask = self.masks_cap.read()
        if not ret_img or not ret_mask:
            return None
        img = img[:,:,::-1]     # BGR -> RGB
        img = np.ascontiguousarray(img)
        img = (img - self.img_mean) / self.img_std

        if len(mask.shape) > 2:
            mask = mask[:,:,2] # Red channel (for LSA segmentation)
        if self.target_class is None:
            mask = mask > 0
        else:
            mask = mask == self.target_class

        return DataContainer([
                torch.tensor(img, dtype=torch.float), \
                torch.tensor(mask, dtype=torch.float)
            ])


def _build_dataloader(imgs_cap, masks_cap, target_class, device):
    dataset = PatchDataset(imgs_cap, masks_cap, target_class, device)
    return DataLoader(dataset, pin_memory=True, collate_fn=collate)


def _build_model(cfg, ckpt, patch_size=64):
    # build the model and load checkpoint
    cfg = mmcv.Config.fromfile(cfg)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    img_meta = [dict(
        ori_shape=(patch_size, patch_size),
        flip=False)]
    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    checkpoint = load_checkpoint(model, ckpt, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    return partial(model.module.inference, img_meta=img_meta, rescale=False)


def _to_rois(xyxys):
    inds = xyxys.new_zeros((xyxys.size(0), 1))
    return torch.cat([inds, xyxys], dim=1).float().contiguous()


def split(img, maskdts, boundary_width=3, iou_thresh=0.25, patch_size=64, out_size=128):
    # maskdts: N, H, W
    fbmasks = find_float_boundary(maskdts, boundary_width)

    detss = []
    for i in range(fbmasks.size(0)):
        dets = get_dets(fbmasks[i], patch_size, iou_thresh=iou_thresh)[:,:4]
        detss.append(dets)

    all_dets = torch.cat(detss, dim=0)
    img = img.permute(2,0,1).unsqueeze(0).float().contiguous()   # 1,3,H,W
    img_patches = roi_align(img, _to_rois(all_dets), patch_size)

    _detss = [torch.cat([i*_.new_ones((_.size(0), 1)), _], dim=1) for i,_ in enumerate(detss)]
    _detss = torch.cat(_detss)
    dt_patches = roi_align(maskdts[:,None,:,:], _detss, patch_size)

    img_patches = F.interpolate(img_patches, (out_size, out_size), mode='bilinear')
    dt_patches = F.interpolate(dt_patches, (out_size, out_size), mode='nearest')
    return detss, torch.cat([img_patches, 2*dt_patches-1], dim=1)


def merge(maskdts, detss, maskss, patch_size=64):
    # detss: list of dets (Ni,4), x1,y1,x2,y2 format, len K
    # maskdts: (K, H, W)
    # maskss (sum_i Ni, 128, 128)
    out = []

    K, H, W = maskdts.shape
    maskdts = maskdts.bool()
    maskss = F.interpolate(maskss.unsqueeze(0), (patch_size, patch_size), 
            mode='bilinear').squeeze(0)
    dt_refined = torch.zeros_like(maskdts[0], dtype=torch.float32)  # H, W
    dt_count = torch.zeros_like(maskdts[0], dtype=torch.float32)    # H, W

    p = 0
    for k in range(K):
        dets = detss[k]
        dets = dets[:, :4].int()    # Ni, 4
        maskdt = maskdts[k]         # H, W
        q = p + dets.size(0)
        masks = maskss[p:q]         # Ni, 64, 64
        p = q

        dt_refined.zero_()
        dt_count.zero_()
        for i in range(dets.size(0)):
            x1, y1, x2, y2 = dets[i]
            dt_refined[y1:y2, x1:x2] += masks[i]
            dt_count[y1:y2, x1:x2] += 1

        s = dt_count > 0
        dt_refined[s] /= dt_count[s]
        maskdt[s] = dt_refined[s] > 0.5

        out.append(maskdt)
    return out


def inference(cfg, ckpt, video_path, masks_video_path, out_video_path, target_class = None):
    model = _build_model(cfg, ckpt)
    imgs_cap = cv2.VideoCapture(video_path)
    masks_cap = cv2.VideoCapture(masks_video_path)
    assert imgs_cap.isOpened()
    assert masks_cap.isOpened()
    dataloader = _build_dataloader(imgs_cap, masks_cap, target_class, device=torch.device('cuda:0'))
    width  = int(masks_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(masks_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = masks_cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    def _inference_one(img, maskdts, curr_mask, target_class, out_writer): # to save GPU memory
        if maskdts is not None:
            dets, patches = split(img, maskdts.unsqueeze(0))
            masks = model(patches)[:,1,:,:]         # N, 128, 128
            refined = merge(maskdts.unsqueeze(0), dets, masks)[0]
            refined = torch.max(curr_mask, refined)
        else:
            refined = curr_mask

        refined = refined.cpu().numpy().astype(np.uint8)
        kernel = np.ones((11, 11), dtype=np.uint8)
        out_mask = np.zeros((height, width, 3), dtype=np.uint8)
        out_mask[:,:,2] = cv2.dilate(refined * target_class, kernel) # out like LSA segmentation
        out_writer.write(out_mask)
        refined = cv2.erode(refined, kernel)
        return torch.tensor(refined, dtype=torch.float, device='cuda')

    if target_class is None:
        target_class = 255
    # inference on each image
    with tqdm(dataloader) as tloader:
        maskdts = None
        for dc in tloader:
            if dc is None:
                break
            img, curr_mask = dc.data[0][0]
            img = img.cuda()             # 3, H, W
            curr_mask = curr_mask.cuda()     # H, W

            maskdts = _inference_one(img, maskdts, curr_mask, target_class, out_writer)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=False, default='../configs/bpr/hrnet48_256.py', help='path to the config')
    parser.add_argument('--ckpt', required=True, help='path to the checkpoint')
    parser.add_argument('--video', required=True, help='path to the rgb video')
    parser.add_argument('--masks_video', required=True, help='path to the video with masks')
    parser.add_argument('--out_video', required=True, help='path to the output video with masks')
    parser.add_argument('--target_class', required=False, default=None, type=int, help='mark of the target semantic class')

    args = parser.parse_args()

    inference(args.cfg, args.ckpt, args.video, args.masks_video, args.out_video, args.target_class)
