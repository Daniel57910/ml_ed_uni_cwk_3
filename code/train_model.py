from attention_model_v2 import AttModelV2
from attention_model_v1 import AttModelV1
from resnet_models import get_resnet_18
from dataset import NusDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, hamming_loss
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch import optim
import torch.distributed.autograd as dist_autograd
# from mean_average_precision import MetricBuilder
from torch.cuda.amp import GradScaler, autocast
import argparse

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
        "hamming_loss": hamming_loss(y_true=target, y_pred=pred)
    }

def load_data(train_path, test_path):
    dataset_val = NusDataset(
        IMAGE_PATH,
        os.path.join(META_PATH, test_path),
        None)

    dataset_train = NusDataset(
        IMAGE_PATH,
        os.path.join(META_PATH, train_path),
        None)

    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val)

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        sampler=sampler_train,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn)

    test_dataloader = DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        sampler=sampler_val,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn)

    return train_dataloader, test_dataloader

LEARNING_RATE = WEIGHT_DECAY = 1e-4
MAX_EPOCH_NUMBER = 70
NUM_CLASSES = 81
# BATCH_SIZE=110
BATCH_SIZE=30
IMAGE_PATH = 'images'
META_PATH = 'nus_wide'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", help="name of model", default="att_v2")
    parser.add_argument("--f_name", "-f", help="name of output file name", default="att_v2")
    args = parser.parse_args()
    return args.model_name, args.f_name

def main():

    # torch.cuda._lazy_init()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    np.random.seed(2020)
    random.seed(2020)
    dist.init_process_group(backend='nccl')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name, f_name = parse_args()
    print(f"found device {device}")
    print(f"Model to use {model_name}, csv output file {f_name}")

    if model_name == "att_v1":
        model = AttModelV1(
           NUM_CLASSES
        )
        find_params = True

    if model_name == "att_v2":
        model = AttModelV2(
            NUM_CLASSES
        )
    if model_name == "resnet_18":
        model = get_resnet_18(NUM_CLASSES)
        find_params = False

    model.to(device)

    model = DDP(
        model,
        find_unused_parameters=find_params
    )

    print(f"Attaching model {model_name} to device: {device}")
    print(f"Device count: {torch.cuda.device_count()}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.Adam,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scaler = GradScaler()
    print(f"Criterion {criterion} setup, optimizer {optimizer} setup")
    train_dataloader, test_dataloader = load_data('train.json', 'test.json')
    batch_losses = []
    batch_losses_test = []
    for i in range(0, 1):
        print(f"Training model at epoch {i}")
        model.train()
        with tqdm(train_dataloader, unit="batch") as train_epoch:
            train_epoch.set_description(f"Epoch: {i}")
            for imgs, targets in train_epoch:
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                with autocast():
                    model_result = model(imgs)
                    break
                    loss = criterion(model_result, targets)
                batch_loss_value = loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                with torch.no_grad():
                    result = calculate_metrics(
                    model_result.cpu().numpy(),
                    targets.cpu().numpy()
                )

                result['epoch'] = i
                result['losses'] = batch_loss_value
                batch_losses.append(result)
                train_epoch.set_postfix(train_loss=batch_loss_value, train_acc=result['accuracy'])
        break
        with tqdm(test_dataloader, unit="batch") as test_epoch:
            print(f"Running validation at epoch {i}")
            test_epoch.set_description(f"Epoch: {i}")
            with torch.no_grad():
                model.eval()
                for val_imgs, val_targets in test_epoch:
                    val_imgs, val_targets = val_imgs.to(device), val_targets.to(device)
                    with autocast():
                        val_result = model(val_imgs)
                        val_losses = criterion(val_result, val_targets)
                    val_metrics = calculate_metrics(
                        val_result.cpu().numpy(),
                        val_targets.cpu().numpy()
                    )

                    batch_loss_test = val_losses.item()

                    val_metrics['epoch'] = i
                    val_metrics['losses'] = batch_loss_test
                    batch_losses_test.append(val_metrics)
                    test_epoch.set_postfix(test_loss=batch_loss_test, test_acc=val_metrics['accuracy'])

    # time_in_hours = datetime.now().strftime("%Y_%m_%d_%H")
    # time_in_minutes = datetime.now().strftime("%H_%M")
    # df = pd.DataFrame(batch_losses)
    # df_val = pd.DataFrame(batch_losses_test)

    # results_directory = f"results_{time_in_hours}"

    # if not os.path.exists(results_directory):
    #     os.makedirs(results_directory)

    # print(f"Saving results to {results_directory}")
    # df.to_csv(f"{results_directory}/{f_name}_training_{time_in_minutes}.csv")
    # df_val.to_csv(f"{results_directory}/{f_name}_validation_{time_in_minutes}.csv")

if __name__ == "__main__":
    main()