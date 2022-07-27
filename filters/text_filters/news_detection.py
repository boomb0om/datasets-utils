import os
from tqdm import tqdm
import requests
import pandas as pd
from PIL import Image
import random
import time
import numpy as np
import math

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler, DataLoader

from transformers import pipeline

from utils import init_logger


def iterbatch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
    
    
class NewsPredictor:
    
    def __init__(self, save_parquets_dir, save_parquets=True, device='cpu', bs=64,
                 log_filename=None, logging_dir='../logs/news_detection/', logger=None):
        
        self.save_parquets_dir = None
        self.save_parquets = save_parquets
        if self.save_parquets:
            self.save_parquets_dir = save_parquets_dir.rstrip('/')
            os.makedirs(self.save_parquets_dir, exist_ok=True)
        
        self.bs = bs
        self.device = torch.device(device)
        
        logfile = f'log_news_detection.log' if log_filename is None else log_filename
        self.logger = logger or init_logger(logfile, logging_dir=logging_dir)
        if self.save_parquets:
            self.logger.info(f'Saving dataframes to: {self.save_parquets_dir}')
        
        model_name = 'cointegrated/rubert-base-cased-nli-twoway'
        
        assert self.device.type == 'cuda', "Device must be cuda for using this model"
        self.gpu_index = 0 if self.device.index is None else self.device.index
        self.logger.info(f'Loading model {model_name}, using device {self.device}')
        
        self.model = pipeline(
            task='zero-shot-classification', 
            model=model_name,
            device=self.gpu_index
        )
        self.candidate_labels = ["Это новость", "Это не новость"]
        self.label_to_column = {
            "Это новость": 'is_news',
            "Это не новость": 'is_not_news',
        }
        
    def run(self, task_name, texts):
        self.logger.info(f'Starting task {task_name}')
        self.logger.info(f'Texts to process: {len(texts)}')
        
        df_labels = {v: [] for k, v in self.label_to_column.items()}
        
        with self.model.device_placement():
            for batch in tqdm(iterbatch(texts, self.bs), total=math.ceil(len(texts)/self.bs)):
                results = self.model(
                  sequences=batch, 
                  candidate_labels=self.candidate_labels, 
                  hypothesis_template="{}."
                )
                for res in results:
                    for c, l in enumerate(res['labels']):
                        df_labels[self.label_to_column[l]].append(res['scores'][c])
        
        df = pd.DataFrame(df_labels)
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