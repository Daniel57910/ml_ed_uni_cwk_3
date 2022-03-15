import json
from operator import itemgetter
import torchvision.datasets as dset
from coco_dataset import DataSet
CORE_PATH = "coco_annotations"
import pdb
import numpy as np
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


instances_path = "coco_annotations/instances_val2014.json"

coco_dataset = DataSet(
    ["data/coco/train_coco2014.json"],
    [],
    448,
    "coco"
)
