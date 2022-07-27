import os
from tqdm import tqdm
import requests
import pandas as pd
from PIL import Image
import random
import numpy as np
from argparse import ArgumentParser
import multiprocessing as mp

from torch.utils.data import Dataset
from torch.utils.data import BatchSampler, DataLoader

from utils import init_logger, read_image_rgb


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        is_correct = True
        width, height, channels = None, None, None
        
        try:
            pil_img = read_image_rgb(path)
            
            arr = np.array(pil_img)
            
            width = pil_img.width
            height = pil_img.height
            if len(arr.shape) == 2:
                channels = 1
            else:
                channels = arr.shape[2]
        except Exception as err:
            is_correct = False
            
        return path, is_correct, width, height, channels
    
    
class ImagesInfoGatherer:
    
    def __init__(self, save_parquets_dir, save_parquets=True, workers=16,
                 log_filename=None, logging_dir='./logs/', logger=None):
        
        self.save_parquets_dir = save_parquets_dir.rstrip('/')
        self.save_parquets = save_parquets
        self.num_workers = workers
        
        os.makedirs(self.save_parquets_dir, exist_ok=True)
        
        logfile = f'log_gather_images_info.log' if log_filename is None else log_filename
        self.logger = logger or init_logger(logfile, logging_dir=logging_dir)
        self.logger.info(f'Using {self.num_workers} workers')
        if self.save_parquets:
            self.logger.info(f'Saving dataframes to: {self.save_parquets_dir}')
        
    def run(self, task_name, files):
        self.logger.info(f'Starting task {task_name}')
        self.logger.info(f'Files to process: {len(files)}')
        
        eval_dataset = ImageDataset(files)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: x
        )
        
        df_labels = {
            'path': [], 
            'is_correct': [], 
            'width': [], 
            'height': [], 
            'channels': []
        }
        for data in tqdm(eval_loader):
            path, is_correct, width, height, channels = data[0]
            df_labels['path'].append(path)
            df_labels['is_correct'].append(is_correct)
            df_labels['width'].append(width)
            df_labels['height'].append(height)
            df_labels['channels'].append(channels)
        
        df = pd.DataFrame(df_labels)
        invalid_images_count = len(df[df['is_correct'] == False])
        self.logger.info(f'Task finished. Invalid images count: {invalid_images_count}')
        
        if self.save_parquets:
            df.to_parquet(
                f'{self.save_parquets_dir}/{task_name}.parquet',
                index=False
            )
        
        return df