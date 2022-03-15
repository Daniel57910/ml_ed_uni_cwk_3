import os
import json
import numpy as np
from PIL import Image
IMAGE_PATH = 'images'
META_PATH = 'nus_wide'
from torchvision import transforms, datasets
from pycocotools.coco import COCO
import pdb
import torch
import torch.utils.data as data
import os
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json

TOP_20_CLASSES = [
    "sky",
    "clouds",
    "person",
    "water",
    "animal",
    "grass",
    "buildings",
    "window",
    "plants",
    "lake",
    "ocean",
    "road",
    "flowers",
    "sunset",
    "reflection",
    "rocks",
    "vehicle",
    "tree",
    "snow",
    "beach"
]

class NusDataset:
    def __init__(self, data_path, anno_path, transforms):
        self.transforms = transforms
        with open(anno_path) as fp:
            json_data = json.load(fp)
        samples = json_data['samples']
        print('Loaded Samples')
        self.classes = json_data['labels']

        self.imgs = []
        self.annos = []
        self.data_path = data_path
        print('loading', anno_path)
        for sample in samples:
                intersection = [l for l in sample['image_labels'] if l in TOP_20_CLASSES]
                if intersection:
                    self.imgs.append(sample['image_name'])
                    self.annos.append(sample['image_labels'])
        for item_id in range(len(self.annos)):
            item = self.annos[item_id]
            vector = [cls in item for cls in self.classes]
            self.annos[item_id] = np.array(vector, dtype=float)

    def __getitem__(self, item):
        anno = self.annos[item]
        img_path = os.path.join(self.data_path, self.imgs[item])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((500, 500))
        convert_tensor = transforms.ToTensor()

        img = convert_tensor(img)
        if self.transforms is not None:
            img = self.transforms(img)

        if img.shape != (3, 180, 180):
            print(f"Data at {img_path} is corrupted, skipping")
            return None

        return img, anno

    def __len__(self):
        return len(self.imgs)
