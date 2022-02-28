from dataset import NusDataset
import os
import numpy as np
IMAGE_PATH = 'images'
META_PATH = 'nus_wide'
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from attention_model import BaseModel
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)
torch.backends.cudnn.deterministic = True
import torch.optim as optim
from datetime import datetime
import pandas as pd
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch import optim
import torch.distributed.autograd as dist_autograd


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {
        'accuracy': accuracy_score(y_true=target, y_pred=pred),
        'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
        'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
        'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
        'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
        'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
        'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
    }

learning_rate = weight_decay = 1e-4 # Learning rate and weight decay
max_epoch_number = 35 # Number of epochs for training
dist.init_process_group(backend='nccl')
NUM_CLASSES = 81
BATCH_SIZE=60
save_path = 'chekpoints/'

dataset_val = NusDataset(
    IMAGE_PATH, os.path.join(META_PATH, 'test.json'), None)

dataset_train = NusDataset(
    IMAGE_PATH, os.path.join(META_PATH, 'train.json'), None)

sampler_train = DistributedSampler(dataset_train)
sampler_val = DistributedSampler(dataset_val)

train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, sampler=sampler_train, collate_fn=collate_fn)
test_dataloader = DataLoader(dataset_val, batch_size=BATCH_SIZE, sampler=sampler_val, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaseModel(
   NUM_CLASSES
)
model.to(device)

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Parralelising training across {device_count} GPU")
        model = DDP(
            model,
            find_unused_parameters=True
        )
        print(f'Use multi GPU', device)
    else:
        print('Use GPU', device)

criterion = nn.BCELoss()
optimizer = ZeroRedundancyOptimizer(
    model.parameters(),
    optimizer_class=torch.optim.Adam,
    lr=learning_rate,
    weight_decay=weight_decay
)

epoch = 0
iteration = 0
running_loss = 0
batch_losses = []
batch_losses_test = []
for i in range(0, max_epoch_number):

    """
    Run against training data
    """
    model.train()
    with tqdm(train_dataloader, unit="batch") as train_epoch:
        for imgs, targets in train_epoch:
            train_epoch.set_description(f"Epoch: {i}")
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()

            model_result = model(imgs)
            loss = criterion(model_result, targets)
            batch_loss_value = float(loss)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                result = calculate_metrics(
                    model_result.cpu().numpy(),
                    targets.cpu().numpy()
            )

            result['epoch'] = i
            result['losses'] = batch_loss_value
            batch_losses.append(result)
            train_epoch.set_postfix(train_loss=batch_loss_value, train_acc=result['accuracy'])
    """
    Run against test data
    """
    with tqdm(test_dataloader, unit="batch") as test_epoch:
        test_epoch.set_description(f"Epoch: {i}")
        with torch.no_grad():
            model.eval()
            for val_imgs, val_targets in test_epoch:
                val_imgs, val_targets = val_imgs.to(device), val_targets.to(device)
                val_result = model(val_imgs)
                val_losses = criterion(val_result, val_targets)
                val_metrics = calculate_metrics(
                    val_result.cpu().numpy(),
                    val_targets.cpu().numpy()
                )

                batch_loss_test = float(val_losses)

                val_metrics['epoch'] = i
                val_metrics['losses'] = batch_loss_test
                batch_losses_test.append(val_metrics)
                test_epoch.set_postfix(test_loss=batch_loss_test, test_acc=val_metrics['accuracy'])

    """
    Early stoppage if model overfitting
    """
    print(f"Batch training losses at {i} {batch_losses[-1]}")
    print(f"Batch validation losses at {i} {batch_losses_test[-1]}")
    learning_completed = [l['losses'] for l in batch_losses if l['losses'] < 0.01]
    if batch_losses[-1]['losses'] < 0.01 or len(learning_completed) > 1:
        print("Early stoppage implementation")
        break

time = datetime.now().strftime("%Y_%m_%d-%H:%M")
df = pd.DataFrame(batch_losses)
# df_val = pd.DataFrame(batch_losses_test)

df.to_csv(f"training_results_{time}.csv")
# df_val.to_csv(f"validation_results_{time}.csv")




