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
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        anomaly_scores = []
        # label always is normal(0)
        for i, (img, _) in tqdm(enumerate(dataloader)):
            img = img.to(device)
            rec_img, z_hat, jac = model(img)
            recon_loss = torch.nn.functional.mse_loss(rec_img, img)
            flow_loss, log_z = model.flow_loss()
            loss = (1 - alpha) * recon_loss + alpha * flow_loss
            
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            anomaly_score = model.anomaly_score(beta, log_z, img)
            anomaly_scores.append(anomaly_score)
        # print(epoch)
        # print(anomaly_scores)
        # print(torch.sum(torch.stack(anomaly_scores)))
        # print(epoch_loss)
        print(f"Train: epoch {epoch}, anomaly_score : {torch.sum(torch.stack(anomaly_scores))} train loss = {epoch_loss}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train()