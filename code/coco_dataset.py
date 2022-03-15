import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import torch
import numpy as np
import pdb
# modify for transformation for vit
# modfify wider crop-person images

COCO_INDEXES = [
    0,
    56,
    2,
    60,
    41,
    39,
    45,
    26,
    7,
    24,
    13,
    73,
    67,
    71,
    62,
    57,
    74,
    43,
    58,
    16
]

class DataSet(Dataset):
    def __init__(self,
                ann_files,
                augs,
                img_size,
                dataset,
                ):
        self.dataset = dataset
        self.ann_files = ann_files
        self.augment = self.augs_function(augs, img_size)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            ]
            # In this paper, we normalize the image data to [0, 1]
            # You can also use the so called 'ImageNet' Normalization method
        )
        self.anns = []
        self.load_anns()
        print(self.augment)

        # in wider dataset we use vit models
        # so transformation has been changed
        if self.dataset == "wider":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]
            )

    def augs_function(self, augs, img_size):
        t = []
        if 'randomflip' in augs:
            t.append(transforms.RandomHorizontalFlip())
        if 'ColorJitter' in augs:
            t.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0))
        if 'resizedcrop' in augs:
            t.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
        if 'RandAugment' in augs:
            t.append(RandAugment())

        t.append(transforms.Resize((img_size, img_size)))

        return transforms.Compose(t)

    def load_anns(self):
        self.anns = []
        for ann_file in self.ann_files:
            json_data = json.load(open(ann_file, "r"))
            self.anns += json_data

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        idx = idx % len(self)
        ann = self.anns[idx]
        img = Image.open(ann["img_path"]).convert("RGB")
        img = self.augment(img)
        img = self.transform(img)

        indexes_true = list(np.where(np.array(ann["target"]) == 1)[0])
        intersection = list(set(indexes_true) & set(COCO_INDEXES))
        if intersection:
            return torch.Tensor(img), torch.tensor(ann["target"])
        else:
            return None

