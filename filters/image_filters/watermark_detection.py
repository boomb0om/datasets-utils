import os
from tqdm import tqdm
import requests
import pandas as pd
from PIL import Image
import random
import time
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler, DataLoader

from utils import init_logger, FP16Module, read_image_rgb

from huggingface_hub import hf_hub_url, cached_download

MODELS = {
    'resnext101_32x8d-large': dict(
        resnet=models.resnext101_32x8d,
        repo_id='boomb0om/dataset-filters',
        filename='watermark_classifier-resnext101_32x8d-input_size320-4epochs_c097_w082.pth',
    ),
    'resnext50_32x4d-small': dict(
        resnet=models.resnext50_32x4d,
        repo_id='boomb0om/dataset-filters',
        filename='watermark_classifier-resnext50_32x4d-input_size320-4epochs_c082_w078.pth',
    )
}

def get_watermarks_detection_model(name, device='cuda:0', fp16=True, cache_dir='/tmp/datasets_utils'):
    assert name in MODELS
    config = MODELS[name]
    model_ft = config['resnet'](pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    
    config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
    cached_download(config_file_url, cache_dir=cache_dir, force_filename=config['filename'])
    weights = torch.load(os.path.join(cache_dir, config['filename']), device)
    model_ft.load_state_dict(weights)
    
    if fp16:
        model_ft = FP16Module(model_ft)
        
    model_ft.eval()
    model_ft = model_ft.to(device)
    
    return model_ft


class ImageDataset(Dataset):
    
    def __init__(self, paths):
        self.paths = paths
        self.resnet_transforms = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        pil_img = read_image_rgb(path)
        
        resnet_img = self.resnet_transforms(pil_img).float()
        
        return resnet_img
    
    
class ResnetWatermarksPredictor:
    
    def __init__(self, resnet_model, save_parquets_dir=None,
                 save_parquets=False, device='cuda:0', workers=16, bs=128,
                 log_filename=None, logging_dir='./logs/', logger=None):
        
        self.save_parquets = save_parquets
        
        self.num_workers = workers
        self.bs = bs
        self.device = device
        
        logfile = f'log_watermark_predictor.log' if log_filename is None else log_filename
        self.logger = logger or init_logger(logfile, logging_dir=logging_dir)
        if self.save_parquets:
            assert save_parquets_dir is not None and type(save_parquets_dir) == str
            self.save_parquets_dir = save_parquets_dir.rstrip('/')
            os.makedirs(self.save_parquets_dir, exist_ok=True)
            self.logger.info(f'Saving dataframes to: {self.save_parquets_dir}')
        
        self.logger.info(f'Using device {self.device}')
        self.resnet_model = resnet_model
        
    def run(self, task_name, files):
        self.logger.info(f'Starting task {task_name}')
        self.logger.info(f'Files to process: {len(files)}')
        
        eval_dataset = ImageDataset(files)
        loader = DataLoader(
            eval_dataset,
            sampler=torch.utils.data.SequentialSampler(eval_dataset),
            batch_size=self.bs,
            drop_last=False,
            num_workers=self.num_workers
        )
        
        df_labels = {
            'watermark': [],
        }
        
        for batch in tqdm(loader):
            with torch.no_grad():
                outputs = self.resnet_model(batch.to(self.device))
                df_labels['watermark'].extend(torch.max(outputs, 1)[1].cpu().reshape(-1).tolist())
        
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