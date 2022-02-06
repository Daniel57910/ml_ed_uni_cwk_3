import os
import json
import numpy as np
from PIL import Image
IMAGE_PATH = 'images'
META_PATH = 'nus_wide'
from torchvision import transforms

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
            self.imgs.append(sample['image_name'])
            self.annos.append(sample['image_labels'])
        for item_id in range(len(self.annos)):
            item = self.annos[item_id]
            vector = [cls in item for cls in self.classes]
            self.annos[item_id] = np.array(vector, dtype=float)

    def __getitem__(self, item):
        anno = self.annos[item]
        img_path = os.path.join(self.data_path, self.imgs[item])
        img = Image.open(img_path)
        img = img.resize((180, 180))
        convert_tensor = transforms.ToTensor()

        img = convert_tensor(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, anno

    def __len__(self):
        return len(self.imgs)
