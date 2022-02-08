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
from attention_model import Resnext50, BaseModel
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
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
            # 'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            # 'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            # 'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }

# Here is an auxiliary function for checkpoint saving.
def checkpoint_save(model, save_path, epoch):
    f = os.path.join(save_path, 'checkpoint-{:06d}.pth'.format(epoch))
    if 'module' in dir(model):
        torch.save(model.module.state_dict(), f)
    else:
        torch.save(model.state_dict(), f)
    print('saved checkpoint:', f)

# Initialize the training parameters.
num_workers = 8 # Number of CPU processes for data preprocessing
learning_rate = weight_decay = 1e-4 # Learning rate and weight decay
batch_size = 32
save_freq = 1 # Save checkpoint frequency (epochs)
test_freq = 200 # Test model frequency (iterations)
max_epoch_number = 3 # Number of epochs for training
# Note: on the small subset of data overfitting hap0ens after 30-35 epochs
NUM_CLASSES = 27
save_path = 'chekpoints/'

dataset_val = NusDataset(
    IMAGE_PATH, os.path.join(META_PATH, 'small_test.json'), None)

dataset_train = NusDataset(
    IMAGE_PATH, os.path.join(META_PATH, 'small_train.json'), None)

train_dataloader = DataLoader(dataset_train, batch_size=60, shuffle=True)
test_dataloader = DataLoader(dataset_val, batch_size=60, shuffle=True)

num_train_batches = int(np.ceil(len(dataset_train) / batch_size))
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
for i in range(0, max_epoch_number):
    batch_losses = []
    for imgs, targets in train_dataloader:
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()

        model_result = model(imgs)
        print(f"Model fitted on data {i}")
        print(model_result)
    #     loss = criterion(model_result, targets.type(torch.float))

    #     batch_loss_value = loss.item()
    #     loss.backward()
    #     optimizer.step()

    #     logger.add_scalar('train_loss', batch_loss_value, iteration)
    #     batch_losses.append(batch_loss_value)
    #     with torch.no_grad():
    #         result = calculate_metrics(model_result.cpu().numpy(), targets.cpu().numpy())
    #         for metric in result:
    #             logger.add_scalar('train/' + metric, result[metric], iteration)

    #     if iteration % test_freq == 0:
    #         model.eval()
    #         with torch.no_grad():
    #             model_result = []
    #             targets = []
    #             for imgs, batch_targets in test_dataloader:
    #                 imgs = imgs.to(device)
    #                 model_batch_result = model(imgs)
    #                 model_result.extend(model_batch_result.cpu().numpy())
    #                 targets.extend(batch_targets.cpu().numpy())

    #         result = calculate_metrics(np.array(model_result), np.array(targets))
    #         for metric in result:
    #             logger.add_scalar('test/' + metric, result[metric], iteration)

    #         print(f"epoch {epoch} iter {iteration} micro {result['micro/f1']} macro {result['macro/f1']}")
    #         model.train()
    #     iteration += 1

    # loss_value = np.mean(batch_losses)
    # print("epoch:{:2d} iter:{:3d} train: loss:{:.3f}".format(epoch, iteration, loss_value))
    # if epoch % save_freq == 0:
    #     checkpoint_save(model, save_path, epoch)
    # epoch += 1
    # if max_epoch_number < epoch:
    #     break