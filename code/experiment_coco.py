import json
from nis import cat
from operator import itemgetter
from unicodedata import category
import torchvision.datasets as dset
from coco_dataset import DataSet
CORE_PATH = "coco_annotations"
import pdb
import numpy as np
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

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


instances_path = "coco_annotations/instances_val2014.json"


with open("coco_annotations/annotations/instances_train2014.json") as f:
    coco = json.load(f)

# pdb.set_trace()

categories = [c['name'] for c in coco['categories']]
indexes = [c['id'] for c in coco['categories']]
print(categories)
print(indexes)
print(len(categories))
print(len(set(categories)))