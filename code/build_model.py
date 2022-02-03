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
from attention_model import AttnVGG
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)
torch.backends.cudnn.deterministic = True
from functions import train_epoch, val_epoch
import torch.optim as optim
from datetime import datetime

# Initialize the training parameters.
num_workers = 8 # Number of CPU processes for data preprocessing
learning_rate = weight_decay = 1e-4 # Learning rate and weight decay
batch_size = 32
save_freq = 1 # Save checkpoint frequency (epochs)
test_freq = 200 # Test model frequency (iterations)
max_epoch_number = 35 # Number of epochs for training
# Note: on the small subset of data overfitting happens after 30-35 epochs
NUM_CLASSES = 27
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

dataset_val = NusDataset(
    IMAGE_PATH, os.path.join(META_PATH, 'small_test.json'), None)

dataset_train = NusDataset(
    IMAGE_PATH, os.path.join(META_PATH, 'small_train.json'), None)

train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=True)

num_train_batches = int(np.ceil(len(dataset_train) / batch_size))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attention_model = AttnVGG(
    num_train_batches,
    NUM_CLASSES
).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(attention_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
writer = SummaryWriter("runs/cnn_attention_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))

for epoch in range(35):
    train_epoch(attention_model, criterion, optimizer, train_dataloader, device, epoch, 10, writer)
    val_epoch(attention_model, criterion, test_dataloader, device, epoch, writer)