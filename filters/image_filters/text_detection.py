import os
from tqdm import tqdm
import requests
import pandas as pd
from PIL import Image
import random
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler, DataLoader

import cv2
from filters.craft_pytorch import craft_utils, imgproc, file_utils, CRAFT
from collections import OrderedDict

from utils import init_logger, FP16Module, read_image_rgb

from huggingface_hub import hf_hub_url, cached_download

MODELS = {
    'CRAFT-MLT': dict(
        repo_id='boomb0om/dataset-filters',
        filename='craft_mlt_25k.pth',
    ),
}

def get_text_detection_model(name, device='cuda:0', fp16=True, cache_dir='/tmp/datasets_utils'):
    assert name in MODELS
    config = MODELS[name]
    model = CRAFT()
    
    config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
    cached_download(config_file_url, cache_dir=cache_dir, force_filename=config['filename'])
    weights = copyStateDict(torch.load(os.path.join(cache_dir, config['filename'])))
    model.load_state_dict(weights)
    
    if fp16:
        model = FP16Module(model)
        
    model.eval()
    model = model.to(device)
    
    return model

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def preprocess_image(image):
    canvas_size = 1280
    mag_ratio = 1
    
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [b, c, h, w]
    return x, ratio_w, ratio_h


class ImageDataset(Dataset):
    
    def __init__(self, paths, resize_to):
        self.paths = paths
        self.resize_to = resize_to
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        pil_img = read_image_rgb(path)
        pil_img = pil_img.resize(self.resize_to)
        
        preproc_img, _, _ = preprocess_image(np.array(pil_img))
        
        return preproc_img
    
    
class FastCRAFTPredictor:
    
    def __init__(self, craft_model, save_parquets_dir=None,
                 save_parquets=True, device='cuda:0', workers=16, bs=128,
                 log_filename=None, logging_dir='./logs/', logger=None):
        
        self.save_parquets = save_parquets
        
        self.num_workers = workers
        self.bs = bs
        self.device = device
        self.text_threshold = 0.7
        self.resize_to = (512,512)
        
        logfile = f'log_fast_text_detection.log' if log_filename is None else log_filename
        self.logger = logger or init_logger(logfile, logging_dir=logging_dir)
        
        if self.save_parquets:
            self.save_parquets_dir = save_parquets_dir.rstrip('/')
            os.makedirs(self.save_parquets_dir, exist_ok=True)
            self.logger.info(f'Saving dataframes to: {self.save_parquets_dir}')
        
        self.logger.info(f'Using device {self.device}')
        self.model = craft_model
        
    def run(self, task_name, files):
        self.logger.info(f'Starting task {task_name}')
        self.logger.info(f'Files to process: {len(files)}')
        
        eval_dataset = ImageDataset(files, self.resize_to)
        loader = DataLoader(
            eval_dataset,
            sampler=torch.utils.data.SequentialSampler(eval_dataset),
            batch_size=self.bs,
            drop_last=False,
            num_workers=self.num_workers
        )
        
        df_labels = {
            'text_area': [],
        }
        
        for batch in tqdm(loader):
            with torch.no_grad():
                batch = batch.to(self.device)
                y, feature = self.model(batch)
                score_text = y[:,:,:,0].cpu()

                heatmap = torchvision.transforms.functional.resize(score_text, size=(512, 512))
                heatmap *= 2.5
                heatmap[heatmap<self.text_threshold] = 0
                batch_areas = heatmap.view(heatmap.shape[0], -1).mean(1).data.numpy()

                df_labels['text_area'].extend(batch_areas)
        
        df = pd.DataFrame(df_labels)
        df['path'] = files
        self.logger.info(f'Processing task {task_name} finished')
        
        if self.save_parquets:
            parquet_path = f'{self.save_parquets_dir}/{task_name}.parquet'
            df.to_parquet(
                parquet_path,
                index=False
            )
            self.logger.info(f'Parquet saved to {parquet_path}')
        
        return df