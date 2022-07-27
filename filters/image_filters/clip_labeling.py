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
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler, DataLoader

import clip

from utils import init_logger, read_image_rgb


class ImageDataset(Dataset):
    
    def __init__(self, paths, clip_processor):
        self.paths = paths
        self.clip_processor = clip_processor
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        pil_img = read_image_rgb(path)
        
        clip_pixel_values = self.clip_processor(pil_img)
        
        return clip_pixel_values
    
    
class CLIPPredictor:
    
    def __init__(self, clip_model, clip_processor, save_parquets_dir=None,
                 save_parquets=False, device='cuda:0', workers=16, bs=128, 
                 templates=['{}', 'photo of a {}'],
                 log_filename=None, logging_dir='./logs/', logger=None):
        
        self.save_parquets = save_parquets
        
        self.num_workers = workers
        self.bs = bs
        self.device = device
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.templates = templates
        
        logfile = f'log_clip_predictor.log' if log_filename is None else log_filename
        self.logger = logger or init_logger(logfile, logging_dir=logging_dir)
        if self.save_parquets:
            assert save_parquets_dir is not None and type(save_parquets_dir) == str
            self.save_parquets_dir = save_parquets_dir.rstrip('/')
            os.makedirs(self.save_parquets_dir, exist_ok=True)
            self.logger.info(f'Saving dataframes to: {self.save_parquets_dir}')
        self.logger.info(f'CLIP templates: {self.templates}')
        
    def get_text_features(self, labels):
        text_features = []
        for template in self.templates:
            texts = clip.tokenize([template.format(class_label.lower().strip()) for class_label in labels]).to(self.device)
            text_features.append(self.clip_model.encode_text(texts))
        text_features = torch.stack(text_features).mean(0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
        
    def run(self, task_name, files, labels):
        self.logger.info(f'Starting task {task_name}')
        self.logger.info(f'Files to process: {len(files)}')
        
        eval_dataset = ImageDataset(files, self.clip_processor)
        loader = DataLoader(
            eval_dataset,
            sampler=torch.utils.data.SequentialSampler(eval_dataset),
            batch_size=self.bs,
            drop_last=False,
            num_workers=self.num_workers
        )
        
        df_labels = {label: [] for label in labels}
        self.logger.info(f'Labels count: {len(df_labels)}')
        
        text_latents = self.get_text_features(labels)
        for batch in tqdm(loader):
            with torch.no_grad():
                image_latents = self.clip_model.encode_image(batch.to(self.device))
                image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)
                logits_per_image = torch.matmul(image_latents, text_latents.t())
                probs = logits_per_image.cpu().numpy().tolist()

                for c, label in enumerate(df_labels):
                    df_labels[label] += [i[c] for i in probs]
        
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