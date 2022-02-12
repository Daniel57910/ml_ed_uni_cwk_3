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
from sklearn.metrics import precision_score, recall_score, f1_score
torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)
torch.backends.cudnn.deterministic = True
import torch.optim as optim
from datetime import datetime
import pandas as pd

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
max_epoch_number = 35 # Number of epochs for training

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
)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

epoch = 0
iteration = 0
running_loss = 0
batch_losses = []
for i in range(0, max_epoch_number):
    for index, (imgs, targets) in enumerate(train_dataloader):
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()

        model_result = model(imgs)
        loss = criterion(model_result, targets)
        batch_loss_value = loss.item()
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

        # if i >= 4:
        #     break

        # print(f"Accuracy stats {i}: {result}")
    print(f"Barch losses at {i}: ")
    print(batch_losses[-1])

time = datetime.now().strftime("%Y_%m_%d-%H:%M")
df = pd.DataFrame(batch_losses)
print(df.tail(10))
df.to_csv(f"training_results_{time}.csv")




