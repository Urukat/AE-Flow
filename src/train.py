import os.path as osp
import argparse
import torch

from model import ae_flow
from tqdm import tqdm
from HYPERPARAMETER import alpha, beta, batch_size
from dataloader import ChestXrayDataset

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ae_flow.AE_FLOW()
    dataset = ChestXrayDataset(root="./data", name='chest_xray', split='train', label='NORMAL')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train()