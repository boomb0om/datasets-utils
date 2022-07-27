from glob import glob, iglob
import os
from io import BytesIO
from tqdm import tqdm
import requests
import pandas as pd
from PIL import Image
import random
import tarfile

def vprint(*args, verbose=True):
    if verbose:
        print(*args)

def check_shards(shards_dir: str, csv_columns: list, image_name_col: str = "image_name", verbose=False) -> list:
    shards_dir = shards_dir.rstrip('/')
    files_tar = glob(f'{shards_dir}/*.tar')
    files_csv = glob(f'{shards_dir}/*.csv')
    files_csv_renamed = [i.replace('.tar', '.csv') for i in files_tar]
    
    assert len(files_csv) != 0, "Not found any .csv files"
    assert len(files_tar)==len(files_csv) and set(files_csv) == set(files_csv_renamed), \
            "Every .tar file should have .csv file with same filename"
    
    total_files = 0
    for csv_path in tqdm(files_csv):
        df = pd.read_csv(csv_path)
        assert all([col in df.columns for col in csv_columns]), ".csv file missing some columns"
        
        tar = tarfile.open(csv_path.replace('.csv', '.tar'), mode='r')
        images = {}
        for c, member in enumerate(tar):
            img = Image.open(BytesIO(tar.extractfile(member.name).read()))
            images[member.name] = img
        tar.close()
        
        image_names = df[image_name_col]
        assert set(image_names) == set(images.keys()), "filenames in tar do not match filenames in csv"
        
        total_files += len(image_names)
        vprint(f'Files in {os.path.basename(csv_path)}: {len(image_names)}', verbose=verbose)
    
    vprint(f'Total files: {total_files}', verbose=verbose)
    return True