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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler, DataLoader

import ruclip

from utils import init_logger, read_image_rgb


def get_similarity(predictor, inputs, text_latents):
    with torch.no_grad():
        logit_scale = predictor.clip_model.logit_scale.exp()
        image_latents = predictor.clip_model.encode_image(inputs['pixel_values'])
        image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)
        logits_per_text = torch.matmul(text_latents.to(predictor.device), image_latents.t()) * logit_scale
        logits_per_text = logits_per_text.cpu().numpy()
    
    return np.diag(logits_per_text)


class ImageDataset(Dataset):
    
    def __init__(self, paths, texts, ruclip_processor):
        self.paths = paths
        self.texts = texts
        self.ruclip_processor = ruclip_processor
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        text = self.texts[idx]
        pil_img = read_image_rgb(path)
        
        img_tensor = self.ruclip_processor.image_transform(pil_img)
        
        return img_tensor, text
    
    
class RuCLIPPredictor:
    
    def __init__(self, ruclip_model, ruclip_processor, save_parquets_dir=None,
                 save_parquets=False, device='cuda:0', workers=16, bs=128,
                 templates=['{}', 'изображение с {}', 'фото с {}'],
                 log_filename=None, logging_dir='./logs/', logger=None):
        
        self.save_parquets = save_parquets
        
        self.num_workers = workers
        self.bs = bs
        self.device = device
        self.templates = templates
        
        logfile = f'log_ruclip_similarity.log' if log_filename is None else log_filename
        self.logger = logger or init_logger(logfile, logging_dir=logging_dir)
        
        if self.save_parquets:
            assert save_parquets_dir is not None and type(save_parquets_dir) == str
            self.save_parquets_dir = save_parquets_dir.rstrip('/')
            os.makedirs(self.save_parquets_dir, exist_ok=True)
            self.logger.info(f'Saving dataframes to: {self.save_parquets_dir}')
            
        self.ruclip_model = ruclip_model
        self.ruclip_processor = ruclip_processor
        self.ruclip_predictor = ruclip.Predictor(self.ruclip_model, self.ruclip_processor, device, bs=self.bs, templates=self.templates)
        
    def run(self, task_name, files, texts):
        self.logger.info(f'Starting task {task_name}')
        self.logger.info(f'Files to process: {len(files)}')
        
        assert len(files) == len(texts)
        
        eval_dataset = ImageDataset(files, texts, self.ruclip_processor)
        loader = DataLoader(
            eval_dataset,
            sampler=torch.utils.data.SequentialSampler(eval_dataset),
            batch_size=self.bs,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: x
        )
        
        df_labels = {
            'ruclip_similarity': [],
        }
        
        for data in tqdm(loader):
            image_tensors, batch_labels = list(zip(*data))
            image_tensors = [t.to(self.device) for t in image_tensors]
            with torch.no_grad():
                inputs = {}
                inputs['pixel_values'] = pad_sequence(image_tensors, batch_first=True)
                
                text_latents = self.ruclip_predictor.get_text_latents(batch_labels)
                batch_similarity = get_similarity(self.ruclip_predictor, inputs, text_latents).tolist()
                df_labels['ruclip_similarity'].extend(batch_similarity)
        
        df = pd.DataFrame(df_labels)
        df['path'] = files
        df['caption'] = texts
        self.logger.info(f'Processing task {task_name} finished')
        
        if self.save_parquets:
            parquet_path = f'{self.save_parquets_dir}/{task_name}.parquet'
            df.to_parquet(
                parquet_path,
                index=False
            )
            self.logger.info(f'Parquet saved to {parquet_path}')
        
        return df