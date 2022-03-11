import json
import torchvision.datasets as dset
from coco_dataset import DataSet
CORE_PATH = "coco_annotations"

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

for i, j in enumerate(coco_dataset):
    if i > 1:
        break
    print(j['img'])
    print(type(j['img']))
    print(len(j['target']))


