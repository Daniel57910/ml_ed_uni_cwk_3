# import os
# import time
# import numpy as np
# from PIL import Image
# from torch.utils.data.dataset import Dataset
# from tqdm import tqdm
# from torchvision import transforms
# from torchvision import models
# import torch
# from torch.utils.tensorboard import SummaryWriter
# from sklearn.metrics import precision_score, recall_score, f1_score
# from torch import nn
# from torch.utils.data.dataloader import DataLoader
# from matplotlib import pyplot as plt
# from numpy import printoptions
# import requests
# import tarfile
# import random
# import json
# from shutil import copyfile


# torch.manual_seed(2020)
# torch.cuda.manual_seed(2020)
# np.random.seed(2020)
# random.seed(2020)
# torch.backends.cudnn.deterministic = True
import requests
import os
import tarfile
import time
import tqdm

img_folder = 'images'
chunk_count = 1
print("Running data implementation algorithm")
if not os.path.exists(img_folder):
    def download_file_from_google_drive(id, destination):
        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None

        def save_response_content(response, destination):
            CHUNK_SIZE = 32768
            with open(destination, "wb") as f:
                for chunk in tqdm.tqdm(iterable=response.iter_content(chunk_size=CHUNK_SIZE), unit='KB'):
                    if chunk:  # filter out keep-alive new chunk
                        f.write(chunk)
                        f.flush()

        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
        save_response_content(response, destination)

    file_id = '0B7IzDz-4yH_HMFdiSE44R1lselE'
    path_to_tar_file = str(time.time()) + '.tar.gz'
    download_file_from_google_drive(file_id, path_to_tar_file)
    print('Extraction')
    with tarfile.open(path_to_tar_file) as tar_ref:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner) 
            
        
        safe_extract(tar_ref, os.path.dirname(img_folder))
    os.remove(path_to_tar_file)
