import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import ae_flow
from tqdm import tqdm
from HYPERPARAMETER import alpha, beta, batch_size, epochs
from dataloader import ChestXrayDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, auc, roc_curve
from model.utils import optimize_threshold


def metrics(true, anomaly_scores, threshold):
    results = {}

    # Calculate true positive (tp), false positive (fp), false negative (fn), true negative (tn)
    tp = sum(true[i] == 1 and anomaly_scores[i] >= threshold for i in range(len(true)))
    fp = sum(true[i] == 0 and anomaly_scores[i] >= threshold for i in range(len(true)))
    fn = sum(true[i] == 1 and anomaly_scores[i] < threshold for i in range(len(true)))
    tn = sum(true[i] == 0 and anomaly_scores[i] < threshold for i in range(len(true)))

    # Calculate true positive rate (sensitivity/recall)
    sen = tp / (tp + fn)
    # Calculate true negative rate (specificity)
    spe = tn / (tn + fp)
    # Calculate accuracy
    acc = (tp + tn) / (tp + fp + fn + tn)
    # Calculate F1 score
    f1 = (2 * tp) / (2 * tp + fp + fn)
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(true, anomaly_scores)
    # Calculate AUC
    auc_score = auc(fpr, tpr)

    # Store the results
    results['AUC'] = auc_score
    results['ACC'] = acc
    results['SEN'] = sen
    results['SPE'] = spe
    results['F1'] = f1

    return results

def plot_anomaly_score_distribution(true_labels, anomaly_scores):
    normal_scores = np.array([score for label, score in zip(true_labels, anomaly_scores) if label == 0])
    anomaly_scores = np.array([score for label, score in zip(true_labels, anomaly_scores) if label == 1])

    plt.figure(figsize=(12, 8))

    # Plot histogram for normal scores
    hist, bins = np.histogram(normal_scores, bins=100, density=True)
    plt.plot(bins[:-1], hist, color='blue', alpha=0.7, label='Normal')
    # Plot histogram for anomaly scores
    hist, bins = np.histogram(anomaly_scores, bins=50, density=True)
    plt.plot(bins[:-1], hist, color='red', alpha=0.7, label='Anomaly')

    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ae_flow.AE_FLOW()
    dataset = ChestXrayDataset(root="./data", name='chest_xray', split='train', label='NORMAL')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        anomaly_scores = []
        true_labels = []
        # label always is normal(0)
        for i, (img, lbl) in tqdm(enumerate(dataloader)):
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
            true_labels.append(lbl)

    true = [tensor.cpu().numpy() for tensor in true_labels]
    true = [item for sublist in true for item in sublist]

    anomaly_scores = [tensor.cpu().numpy() for tensor in anomaly_scores]
    anomaly_scores = [item for sublist in anomaly_scores for item in sublist]

    # mean_anomaly_score = torch.mean(torch.cat(anomaly_scores))
    # print(f"Mean anomaly score: {mean_anomaly_score}")

    # Plot anomaly score distribution
    plot_anomaly_score_distribution(true_labels, anomaly_scores)

    # Find optimal threshold
    optimal_threshold = optimize_threshold(anomaly_scores, true_labels)
    results = metrics(true, anomaly_scores, optimal_threshold)

    print(f"Optimal threshold: {optimal_threshold}")
    print(f"Epoch {epoch}: {results}")

    # TODO: best f1 score

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train()
