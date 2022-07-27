import s3fs
import os
from tqdm import tqdm
from glob import glob
import time


def download_folder_from_s3(s3_client: s3fs.core.S3FileSystem, 
                            s3_path: str, target_path: str, max_retries: int = 3) -> list:
    s3_path = s3_path.rstrip('/')+'/'
    target_path = target_path.rstrip('/')+'/'
    
    print('Getting info about paths')
    paths_to_upload = []
    for path, folders, files in tqdm(s3_client.walk(s3_path)):
        for file in files:
            if file != '':
                paths_to_upload.append(os.path.join(path, file))
    
    print(f'Found {len(paths_to_upload)} files to download')
    
    error_files = []
    for path in tqdm(paths_to_upload):
        retries = 0
        ok = False
        while retries < max_retries:
            try:
                retries += 1
                _ = s3_client.get(
                    lpath=path.replace(s3_path, target_path),
                    rpath=path,
                    recursive=False,
                )
                ok = True
                break
            except Exception as err:
                time.sleep(5)
                print(f'Error while uploading file {path}: {err}. Retrying')
                
        if not ok:
            print(f"Can't download file {path}, skipping")
            error_files.append(path)
            
    return error_files


def upload_folder_to_s3(s3_client: s3fs.core.S3FileSystem, 
                        source_path: str, s3_path: str, max_retries: int = 3) -> list:
    s3_path = s3_path.rstrip('/')+'/'
    source_path = source_path.rstrip('/')+'/'
    
    paths_to_upload = []
    for path, folders, files in os.walk(source_path):
        for file in files:
            paths_to_upload.append(os.path.join(path, file))
    
    print(f'Found {len(paths_to_upload)} files to upload')
    
    error_files = []
    for path in tqdm(paths_to_upload):
        retries = 0
        ok = False
        while retries < max_retries:
            try:
                retries += 1
                _ = s3_client.upload(
                    path,
                    path.replace(source_path, s3_path),
                    recursive=False,
                )
                ok = True
                break
            except Exception as err:
                time.sleep(5)
                print(f'Error while uploading file {path}: {err}. Retrying')
                
        if not ok:
            print(f"Can't upload file {path}, skipping")
            error_files.append(path)
            
    return error_files


def upload_files_to_s3(s3_client: s3fs.core.S3FileSystem, 
                       source_files: list, s3_path: str, max_retries: int = 3) -> list:
    s3_path = s3_path.rstrip('/')
    print(f'Found {len(source_files)} files to upload')
    
    error_files = []
    for path in tqdm(source_files):
        source_path = os.path.dirname(path)
        
        retries = 0
        ok = False
        while retries < max_retries:
            try:
                retries += 1
                _ = s3_client.upload(
                    path,
                    path.replace(source_path, s3_path),
                    recursive=False,
                )
                ok = True
                break
            except Exception as err:
                time.sleep(5)
                print(f'Error while uploading file {path}: {err}. Retrying')
                
        if not ok:
            print(f"Can't upload file {path}, skipping")
            error_files.append(path)
            
    return error_files