import json
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

def get_shape_agnostic_mask(mask, kh, kw, x, y, h, w):
    """
    mask : torch.Tensor
        (h, w)
    """
    H, W = mask.shape
    mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, h, w)
    block_h = int(np.ceil(h / kh))
    block_w = int(np.ceil(w / kw))
    
    foreground_mask = mask[:, :, x:min(x + kh*block_h, H), y:min(y + kw*block_w, W)]
    _, _, fh, fw = foreground_mask.shape
    patch = torch.nn.functional.max_pool2d(foreground_mask, (int(block_h), int(block_w))) # (f, kh, kw)
    agnostic_mask = torch.nn.functional.interpolate(patch, (fh, fw), mode="nearest") # repeat_interleaveでもよい

    # paste agnostic_mask to mask
    mask[:, :, x:x + fh, y:y + fw] = agnostic_mask[:, :, :fh, :fw]

    return mask

def get_bbox_from_mask(mask):
    """
    mask : torch.Tensor
        (h, w)
    """
    h, w = mask.shape
    # print(mask.sum(dim=0))
    # print(mask.sum(dim=1))
    vert_mask = mask.sum(dim=0) > 0 # w 
    hori_mask = mask.sum(dim=1) > 0 # h

    vert_mask_ids = torch.nonzero(vert_mask)
    hori_mask_ids = torch.nonzero(hori_mask)
    if len(vert_mask_ids) == 0 or len(hori_mask_ids) == 0:
        return 0, 0, h, w
    ymin, ymax = torch.min(vert_mask_ids), torch.max(vert_mask_ids)
    xmin, xmax = torch.min(hori_mask_ids), torch.max(hori_mask_ids)
    return int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)

def environment_formulation(images, masks):
    """
    images : torch.Tensor
        (f, c, h, w)
    masks : torch.Tensor
        (f, 1, h, w) or (f, h, w)
        1 for foreground, 0 for background
    """
    env_image_list = []
    for image, mask in zip(images, masks):
        x, y, h, w = get_bbox_from_mask(mask)
        # sample number of block kh, kw from kh in (1, h) and kw in (1, w)
        h = max(h, 2)
        w = max(w, 2)
        kh = random.randint(1, h)
        kw = random.randint(1, w)
        agnostic_mask = get_shape_agnostic_mask(mask, kh, kw, x, y, h, w)
        agnostic_mask = agnostic_mask.squeeze(0).squeeze(0)
        environment_image = image * (1 - agnostic_mask)
        env_image_list.append(environment_image)

    environment_images = torch.stack(env_image_list, dim=0)
    return environment_images

def environment_formulation_inf(images, masks):
    """
    images : torch.Tensor
        (f, c, h, w)
    masks : torch.Tensor
        (f, 1, h, w) or (f, h, w)
        1 for foreground, 0 for background
    """
    env_image_list = []
    for image, mask in zip(images, masks):
        x, y, h, w = get_bbox_from_mask(mask)

        # sample number of block kh, kw from kh in (1, h) and kw in (1, w)
        kh = h//10
        kw = w//10
        agnostic_mask = get_shape_agnostic_mask(mask, kh, kw, x, y, h, w)
        agnostic_mask = agnostic_mask.squeeze(0).squeeze(0)
        environment_image = image * (1 - agnostic_mask)
        env_image_list.append(environment_image)
    environment_images = torch.stack(env_image_list, dim=0)
    return environment_images


class HumanDanceVideoDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fashion_meta.json"],
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]

        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)

        assert len(video_reader) == len(
            kps_reader
        ), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"

        video_length = len(video_reader)

        clip_length = min(
            video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        # start_idx = 60
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()

        # read frames and kps
        vid_pil_image_list = []
        pose_pil_image_list = []
        for index in batch_index:
            img = video_reader[index]
            vid_pil_image_list.append(Image.fromarray(img.asnumpy()))
            img = kps_reader[index]
            pose_pil_image_list.append(Image.fromarray(img.asnumpy()))

        ref_img_idx = random.randint(0, video_length - 1)
        ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())

        # transform
        state = torch.get_rng_state()
        pixel_values_vid = self.augmentation(
            vid_pil_image_list, self.pixel_transform, state
        )
        pixel_values_pose = self.augmentation(
            pose_pil_image_list, self.cond_transform, state
        )
        pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
        clip_ref_img = self.clip_image_processor(
            images=ref_img, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
            video_dir=video_path,
            pixel_values_vid=pixel_values_vid,
            pixel_values_pose=pixel_values_pose,
            pixel_values_ref_img=pixel_values_ref_img,
            clip_ref_img=clip_ref_img,
        )

        return sample

    def __len__(self):
        return len(self.vid_meta)



class HumanVideoDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fashion_meta.json"],
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.ref_transform = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.ref_transform_nonorm = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.PILToTensor(),
            ]
        )        

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.cond_transform_nonorm = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.PILToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]
        depth_path = video_meta["depth_path"]
        seg_path = video_meta["seg_path"]

        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)
        depth_reader = VideoReader(depth_path)
        seg_reader = VideoReader(seg_path)

        assert len(video_reader) == len(kps_reader) == len(depth_reader) == len(seg_reader), \
            f"Video {len(video_reader)}, Pose {len(kps_reader)}, Depth {len(depth_reader)}, Mask {len(seg_reader)} lengths do not match in {video_path}"

        video_length = len(video_reader)

        clip_length = min(
            video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()

        # read frames and kps
        vid_pil_image_list = []
        pose_pil_image_list = []
        depth_pil_image_list = []
        seg_pil_image_list = []
        for index in batch_index:
            img = video_reader[index]
            vid_pil_image_list.append(Image.fromarray(img.asnumpy()))
            img = kps_reader[index]
            pose_pil_image_list.append(Image.fromarray(img.asnumpy()))
            img = depth_reader[index]
            depth_pil_image_list.append(Image.fromarray(img.asnumpy()))
            # print(img.asnumpy().shape)
            # img = seg_reader[index]
            img = depth_reader[index]
            seg_pil_image_list.append(Image.fromarray(img.asnumpy()))
            # print(img.asnumpy())

        ref_img_idx = random.randint(0, video_length - 1)
        ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())
        ref_seg = Image.fromarray(seg_reader[ref_img_idx].asnumpy())

        # transform
        state = torch.get_rng_state()
        pixel_values_vid = self.augmentation(
            vid_pil_image_list, self.pixel_transform, state
        )
        pixel_values_pose = self.augmentation(
            pose_pil_image_list, self.cond_transform, state
        )
        pixel_values_depth = self.augmentation(
            depth_pil_image_list, self.cond_transform, state
        )
        pixel_values_seg = self.augmentation(
            seg_pil_image_list, self.cond_transform_nonorm, state
        )

        # seg to mask
        # print(pixel_values_seg)
        pixel_values_mask = (pixel_values_seg != 100).any(dim=1).float() # f, h, w
        # print(pixel_values_mask)
        try:
            pixel_values_env = environment_formulation(pixel_values_vid, pixel_values_mask)
        except Exception as e:
            print(video_path)
            raise e

        pixel_values_ref_img = self.augmentation(ref_img, self.ref_transform)
        pixel_values_ref_seg = self.augmentation(ref_seg, self.ref_transform_nonorm)
        pixel_values_ref_mask = (pixel_values_ref_seg != 100).any(dim=0, keepdim=True).float() # 1, h, w
        # print(pixel_values_ref_mask)
        # masking by sky blue
        # ここdebugが必要
        blue_bg = torch.tensor([0.0, 1.0, 1.0]).view(3, 1, 1)
        pixel_values_ref_masked_img = pixel_values_ref_img * pixel_values_ref_mask + blue_bg * (1 - pixel_values_ref_mask)
        
        sample = dict(
            video_dir=video_path,
            pixel_values_vid=pixel_values_vid,
            pixel_values_pose=pixel_values_pose,
            pixel_values_depth=pixel_values_depth,
            pixel_values_env=pixel_values_env,
            pixel_values_ref_img=pixel_values_ref_masked_img,
        )

        return sample

    def __len__(self):
        return len(self.vid_meta)
