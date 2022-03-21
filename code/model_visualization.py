import torch
import argparse
import os
import torch.distributed as dist
CORE_PATH = "results"
def main():
    dist.init_process_group(backend='nccl')
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-m', '--model', required=True)
    args = parser.parse_args()
    dataset, model = args.dataset, args.model

    model_path = os.path.join(CORE_PATH, dataset + "_models", model)
    loaded_model = torch.load(model_path)


main()