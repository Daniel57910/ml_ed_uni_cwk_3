import torch
import argparse
import os
import torch.distributed as dist
CORE_PATH = "results"
import pdb
def main():
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

    children = list(loaded_model.children())
    for c in children:
        print(c)


main()