from matplotlib.axis import Axis
import torch
import argparse
import os
import torch.distributed as dist
import torch.nn as nn
CORE_PATH = "results"
import pdb
from train_model import load_data_coco, load_data_nus
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import random
from torch.autograd import Variable
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from PIL import ImageDraw
import json


def integrated_grads(model, img, target, classes, n_steps=13):
    torch.cuda.empty_cache()
    ig = IntegratedGradients(model)
    targets, target_indexes = torch.topk(target, 1)
    attrs = ig.attribute(img.unsqueeze(0), target=target_indexes, n_steps=n_steps)
    class_names = [classes[trg_index] for trg_index in target_indexes]
    class_names = ",".join(class_names)
    to_pil = T.ToPILImage()
    img_to_save = to_pil(img)
    myFont = ImageFont.load_default()
    editable = ImageDraw.Draw(img_to_save)
    editable.text((100, 100), class_names, font=myFont, fill=(255, 0, 0))
    fig, axs = viz.visualize_image_attr(np.transpose(attrs.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(img.squeeze().cpu().detach().numpy(), (1,2,0)),
                             method='heat_map',
                            #  cmap=default_cmap,
                             show_colorbar=True,
                             sign='all',
                             title=f'Integrated Gradients: {class_names}',
                            )
    return fig, axs, img_to_save

def get_classes_coco():
    with open("coco_annotations/annotations/instances_val2014.json") as f:
        result = json.load(f)

    return [c['name'] for c in result['categories']]

def get_classes_nus():
    with open("nus_wide/train.json") as f:
        classes = json.load(f)
    return classes['labels']

def main():
    torch.cuda._lazy_init()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    np.random.seed(2020)
    random.seed(2020)
    dist.init_process_group(backend='nccl')
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-m', '--model', required=True)
    args = parser.parse_args()
    dataset, model = args.dataset, args.model

    model_path = os.path.join(CORE_PATH, dataset + "_models", model)
    loaded_model = torch.load(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model.to(device)
    if dataset == 'nus':
        train_data, val_data = load_data_nus('train.json', 'test.json')
        classes = get_classes_nus()
    else:
        train_data, val_data = load_data_coco()
        classes = get_classes_coco()

    correct_counter = 0
    incorrect_counter = 0
    loaded_model.eval()

    correct_results = []
    incorrect_results = []
    stop_eval = False
    with torch.no_grad():
        for image_batch, target_batch in val_data:
            if stop_eval:
                break
            image_batch, target_batch = image_batch.to(device), target_batch.to(device)
            with autocast():
                model_result = loaded_model(image_batch)
            for image, result, target in zip(image_batch, model_result, target_batch):
                result = result.cpu().numpy()
                result = np.where(result > .5, 1, 0)
                fig, ax, img_to_save = integrated_grads(loaded_model, torch.clone(image), target, classes)
                if np.array_equal(result, target.cpu().numpy()):
                    correct_counter +=1
                    if not os.path.exists(f"{dataset}/{model}"):
                        os.makedirs(f"{dataset}/{model}")
                    img_to_save.save(f"{dataset}/{model}/example_{correct_counter}.png")
                    plt.savefig(f"{model}/{dataset}/{correct_counter}.png")


                if correct_counter > 5:
                    stop_eval = True
                if stop_eval:
                    break



main()

