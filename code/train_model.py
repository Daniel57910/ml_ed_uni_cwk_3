from dataset import NusDataset
import os
import json
import numpy as np
from PIL import Image
IMAGE_PATH = 'images'
META_PATH = 'nus_wide'
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from attention_model import BaseModel
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score

torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)
torch.backends.cudnn.deterministic = True
from functions import train_epoch, val_epoch
import torch.optim as optim
from datetime import datetime

# Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {
        'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
        'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
        'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
        'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
        'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
        'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
    }

learning_rate = weight_decay = 1e-4 # Learning rate and weight decay
max_epoch_number = 3 # Number of epochs for training

NUM_CLASSES = 27
save_path = 'chekpoints/'

dataset_val = NusDataset(
    IMAGE_PATH, os.path.join(META_PATH, 'small_test.json'), None)

dataset_train = NusDataset(
    IMAGE_PATH, os.path.join(META_PATH, 'small_train.json'), None)

train_dataloader = DataLoader(dataset_train, batch_size=60, shuffle=True)
test_dataloader = DataLoader(dataset_val, batch_size=60, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())

model = BaseModel(
   NUM_CLASSES
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
logger = SummaryWriter("runs/cnn_attention_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))

epoch = 0
iteration = 0
running_loss = 0
for i in range(0, max_epoch_number):
    batch_losses = []
    for imgs, targets in train_dataloader:
        # imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()

        model_result = model(imgs)
        loss = criterion(model_result, targets)
        batch_loss_value = loss.item()
        loss.backward()
        optimizer.step()
        logger.add_scalar('train_loss', batch_loss_value, iteration)
        batch_losses.append(batch_loss_value)

        with torch.no_grad():
            result = calculate_metrics(
                model_result,
                targets)

        print(f"Results at epoch {i}: loss = {batch_losses[-1]}")
        print(f"Accuracy stats {i}: {result}")



