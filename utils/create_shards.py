import os
from tqdm import tqdm
import time
import requests
import pandas as pd
from PIL import Image
import random
import numpy as np
import csv
import tarfile
from argparse import ArgumentParser
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map, thread_map

from torch.utils.data import Dataset
from torch.utils.data import BatchSampler, DataLoader

from utils import init_logger

def flush_chunk(samples, shard_path, additional_columns):
    tar_name = shard_path + '.tar'
    csv_name = shard_path + '.csv'
    tar = tarfile.open(tar_name, "w")
    with open(csv_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name']+additional_columns)
        for data in samples:
            image_path = data[0]
            image_name = os.path.basename(image_path)
            writer.writerow([image_name, *data[1:]])
            tar.add(image_path, arcname=image_name)
    tar.close()
    
def mp_flush_chunk(params):
    flush_chunk(*params)


class ImageShardsGenerator:
    
    def __init__(self, save_shards_dir, workers=16, samples_per_shard=1000,
                 log_filename=None, logging_dir='./logs/', logger=None):
        
        self.save_shards_dir = save_shards_dir.rstrip('/')
        self.num_workers = workers
        self.samples_per_shard = samples_per_shard
        
        os.makedirs(self.save_shards_dir, exist_ok=True)
        
        logfile = f'log_create_shards.log' if log_filename is None else log_filename
        self.logger = logger or init_logger(logfile, logging_dir=logging_dir)
        self.logger.info(f'Using {self.num_workers} workers')
        self.logger.info(f'Saving shards to: {self.save_shards_dir}')

    def run(self, task_name, df, image_path_column, additional_columns):
        task_shards_dir = os.path.join(self.save_shards_dir, task_name)
        os.makedirs(task_shards_dir, exist_ok=True)
        self.logger.info(f'Starting task: {task_name}')
        self.logger.info(f'Saving task shards to: {task_shards_dir}')
        self.logger.info(f'Dataframe length: {len(df)}')
        
        all_columns = [image_path_column]+additional_columns
        params = []
        total = 0
        for chunk_id, (a, b) in enumerate(zip(
            np.arange(0, df.shape[0], self.samples_per_shard),
            np.arange(self.samples_per_shard, df.shape[0]+self.samples_per_shard, self.samples_per_shard),
        )):
            chunk = df[a:b]
            shard_path = f'{task_shards_dir}/{task_name}_{chunk_id}'
            params.append((zip(*[chunk[col] for col in all_columns]), shard_path, additional_columns))
            total += len(chunk)
            print(f'Preparing params: {total}/{df.shape[0]}', end='\r')
        
        thread_map(mp_flush_chunk, params, max_workers=self.num_workers)
        self.logger.info(f'Finished creating shards for task: {task_name}')
        