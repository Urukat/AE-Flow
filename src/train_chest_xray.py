import os.path as osp
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import ae_flow
import model.utils as ut
from tqdm import tqdm
import scipy
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from HYPERPARAMETER import alpha, beta, batch_size, log_frequency
from dataloader import ChestXrayDataset

def cal_given_threshold(thr, anomaly_scores, labels):
    # Generate predictions based on the proposed threshold
    pred = (anomaly_scores > thr).astype(int)
    # preds = [score > thr for score in anomaly_scores]
    # Calculate the F1 score
    # score = -f1_score(y_true=true_labels[0], y_pred=preds[0])
    # Print the threshold and corresponding F1 score
    # print(f"Threshold: {thr}, F1-score: {score}.")
    return -f1_score(y_true=labels, y_pred=pred)

@torch.no_grad()
def find_threshold(model, normal_loader, abnormal_loader):
    # find best threshold by calculating 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    anomaly_scores = []
    labels = []
    for i, (img, _) in tqdm(enumerate(normal_loader)):
        img = img.to(device)
        rec_img, z_hat, jac = model(img)
        flow_loss, log_z = model.flow_loss()
        anomaly_score = np.array(model.anomaly_score(beta, log_z, img).cpu())
        for score in anomaly_score:
            anomaly_scores.append(score)
            labels.append(0)
        # break
    
    for i, (img, _) in tqdm(enumerate(abnormal_loader)):
        img = img.to(device)
        rec_img, z_hat, jac = model(img)
        flow_loss, log_z = model.flow_loss()
        anomaly_score = np.array(model.anomaly_score(beta, log_z, img).cpu())
        for score in anomaly_score:
            anomaly_scores.append(score)
            labels.append(1)
        # break
    
    best_f1_threshold = scipy.optimize.fmin(cal_given_threshold, args=(anomaly_scores, labels), x0=np.mean(anomaly_scores))
    
    return best_f1_threshold[0]


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ae_flow.AE_FLOW(subnet=args.subnet)
    train_set_normal = ChestXrayDataset(root="./data", name='chest_xray', split='train', label='NORMAL')
    train_loader_normal = DataLoader(train_set_normal, batch_size=batch_size, shuffle=True)
    train_set_pneumonia = ChestXrayDataset(root="./data", name='chest_xray', split='train', label='PNEUMONIA')
    train_loader_pneumonia = DataLoader(train_set_pneumonia, batch_size=batch_size, shuffle=True)

    test_set_normal = ChestXrayDataset(root="./data", name='chest_xray', split='test', label='NORMAL')
    test_set_pneumonia = ChestXrayDataset(root="./data", name='chest_xray', split='test', label='PNEUMONIA')
    test_loader_normal = DataLoader(test_set_normal, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_pneumonia = DataLoader(test_set_pneumonia, batch_size=batch_size, shuffle=False, drop_last=True)

    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    model.to(device)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        anomaly_scores = []
        # label always is normal(0)
        for i, (img, _) in tqdm(enumerate(train_loader_normal)):
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
            # break
            
        model.eval()
            # do not know if this works
            # torch.cuda.empty_cache()
        # find threshold from training data
        optimal_threshold = find_threshold(model, train_loader_normal, train_loader_pneumonia)
        print(f"Optimal threshold: {optimal_threshold}")
        # get resutls of test set
        results = ut.get_test_results(model, beta, optimal_threshold, test_loader_normal, test_loader_pneumonia)
        
        print(f"Epoch {epoch}: {results}")
        # this is for test set
        # ut.plot_distribution(model, beta, test_loader_normal, test_loader_pneumonia, "chest_xray_test", epoch)
        # ut.plot_distribution(model, beta, train_loader_normal, train_loader_pneumonia, "chest_xray_train", epoch)

    torch.save(model, "./src/checkpoint/{}_{}.pt".format(args.subnet, args.epochs))              
    print(f"Train: epoch {epoch}, anomaly_score : {torch.sum(torch.stack(anomaly_scores))} train loss = {epoch_loss}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subnet', default='conv_type', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    args = parser.parse_args()
    train(args)