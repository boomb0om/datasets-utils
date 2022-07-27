import os
from tqdm import tqdm
import pandas as pd
from PIL import Image
import random
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="unclosed file")

from filters.image_filters.imagededup.phashcustom import PHashCustom
from multiprocessing import Pool
from typing import Callable, Dict, List

from utils import init_logger


def parallelise(function: Callable, data: List, verbose: bool, num_workers=16) -> List:
    pool = Pool(processes=num_workers)
    results = list(
        tqdm(pool.imap(function, data, 100), total=len(data), disable=not verbose)
    )
    pool.close()
    pool.join()
    return results
    
    
class ImageDeduplicator:
    
    def __init__(self, chunk_size=50000, workers=16, max_distance_threshold=0,
                 log_filename=None, logging_dir='./logs/', logger=None):
        
        self.chunk_size = chunk_size
        self.max_distance_threshold = max_distance_threshold
        
        self.num_workers = workers
        
        logfile = f'log_deduplication.log' if log_filename is None else log_filename
        self.logger = logger or init_logger(logfile, logging_dir=logging_dir)
        
        self.logger.info(f'Using PHasher with chunk_size {self.chunk_size} and {self.num_workers} workers')
        
    def get_duplicates_dict(self, files):
        hasher = PHashCustom()
        self.logger.info('Encoding')
        hashes = parallelise(hasher.encode_image, files, hasher.verbose, num_workers=self.num_workers)
        hash_initial_dict = dict(zip(files, hashes))
        encodings = {
            k: v for k, v in hash_initial_dict.items() if v
        }

        self.logger.info('Finding duplicates')
        duplicates = hasher.find_duplicates(encoding_map=encodings, max_distance_threshold=self.max_distance_threshold,
                                            search_method='bktree')
        return duplicates
    
    def get_duplicates_to_remove_scored(self, hasher, encodings, path2score):
        duplicates = hasher.find_duplicates(encoding_map=encodings, max_distance_threshold=self.max_distance_threshold, 
                                            search_method='bktree')
        imgs_duplicates = {}
        for k, v in duplicates.items():
            if len(v) > 0:
                imgs_duplicates[k] = v
                
        to_remove = []
        visited = []
        for name in tqdm(list(imgs_duplicates.keys())):
            if name not in visited:
                all_dups = [name]+imgs_duplicates[name]
                dups_scores = [(dup, path2score[dup]) for dup in all_dups]
                dups_scores_sorted = sorted(dups_scores, key=lambda x: x[1], reverse=True)
                to_remove.extend([i[0] for i in dups_scores_sorted][1:])
                visited.extend(all_dups)

        to_remove = np.unique(to_remove).tolist()
        return to_remove
        
    def run(self, task_name, files, path2score=None, delete_duplicates=True):
        self.logger.info(f'Starting task {task_name}')
        if path2score is not None:
            self.logger.info(f'Finding duplicates based on scores')
        self.logger.info(f'Files to process: {len(files)}')
        self.logger.info(f'Delete duplicates = {delete_duplicates}')
        
        duplicates = []
        for c, i_start in enumerate(range(0, len(files), self.chunk_size)):
            files_chunk = files[i_start:i_start+self.chunk_size]
            
            hasher = PHashCustom()
            self.logger.info('Encoding')
            hashes = parallelise(hasher.encode_image, files_chunk, hasher.verbose, num_workers=self.num_workers)
            hash_initial_dict = dict(zip(files_chunk, hashes))
            encodings = {
                k: v for k, v in hash_initial_dict.items() if v
            }
            
            self.logger.info('Finding duplicates')
            if path2score is not None:
                duplicates_chunk = self.get_duplicates_to_remove_scored(hasher, encodings, path2score)
            else:
                duplicates_chunk = hasher.find_duplicates_to_remove(encoding_map=encodings, 
                                                                    max_distance_threshold=self.max_distance_threshold)
            duplicates.extend(duplicates_chunk)
            self.logger.info(f'Chunk: {c}, [{i_start}:{i_start+self.chunk_size}] {len(duplicates_chunk)} duplicates')
            
            if delete_duplicates:
                self.logger.info(f'Deleting duplicates in chunk')
                for path in tqdm(duplicates_chunk):
                    os.remove(path)
        
        self.logger.info(f'Found total {len(duplicates)} duplicates')
        self.logger.info(f'Processing task {task_name} finished')
        
        return duplicates