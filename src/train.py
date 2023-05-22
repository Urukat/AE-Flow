import os.path as osp
import argparse
import torch
from torch.utils.data import DataLoader
from model import ae_flow
from tqdm import tqdm
from HYPERPARAMETER import alpha, beta, batch_size, epochs
from dataloader import ChestXrayDataset

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ae_flow.AE_FLOW()
    dataset = ChestXrayDataset(root="./data", name='chest_xray', split='train', label='NORMAL')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i, (img, label) in tqdm(enumerate(dataloader)):
            img = img.to(device)
            label = label.to(device)
            rec_img = model(img)
            optimizer.zero_grad()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train()