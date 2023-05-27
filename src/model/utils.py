import torch
import scipy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

def cal_given_threshold(thr, anomaly_scores, true_labels):
    # Generate predictions based on the proposed threshold
    preds = [score > thr for score in anomaly_scores]
    # Calculate the F1 score
    score = -f1_score(y_true=true_labels[0].cpu(), y_pred=preds[0].cpu())
    # Print the threshold and corresponding F1 score
    print(f"Threshold: {thr}, F1-score: {score}.")
    return score


def optimize_threshold(anomaly_scores, true_labels):
    # Flatten the anomaly scores list
    anomaly_scores_flat = [tensor.cpu().numpy() for tensor in anomaly_scores]
    anomaly_scores_flat = [i for sublist in anomaly_scores_flat for i in sublist]

    # Set the lower and upper bounds for the bisection search
    lower_bound = np.partition(anomaly_scores_flat, 10)[10]
    upper_bound = np.partition(anomaly_scores_flat, -10)[-10]

    # Perform bisection search to find the optimal threshold
    opt_threshold = scipy.optimize.bisect(f=cal_given_threshold, a=lower_bound, b=upper_bound,
                                          args=(anomaly_scores, true_labels))

    return opt_threshold

@torch.no_grad()
def plot_distribution(model, beta, test_normal_loader, test_abnormal_loader, dataset_name, epoch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    normal_anomaly_scores = []
    abnormal_anomaly_scores = []
    for i, (img, label) in tqdm(enumerate(test_normal_loader)):
        img = img.to(device)
        label = label.to(device)

        rec_img, z_hat, jac = model(img)
        flow_loss, log_z = model.flow_loss()
        anomaly_score = model.anomaly_score(beta, log_z, img)
        normal_anomaly_scores.append(anomaly_score.item())
        # break
    
    for i, (img, label) in tqdm(enumerate(test_abnormal_loader)):
        img = img.to(device)
        label = label.to(device)

        rec_img, z_hat, jac = model(img)
        flow_loss, log_z = model.flow_loss()
        anomaly_score = model.anomaly_score(beta, log_z, img)
        abnormal_anomaly_scores.append(anomaly_score.item())
        # break
    
    ana_min = min(normal_anomaly_scores + abnormal_anomaly_scores)
    fig, ax = plt.subplots()
    ax.hist(-normal_anomaly_scores / ana_min, color = 'green', alpha=0.5, label = 'normal')
    ax.hist(-abnormal_anomaly_scores / ana_min, color = 'red', alpha=0.5, label = 'abnormal')
    plt.savefig("./src/graphs/{}_{}.png".format(dataset_name, epoch))
    # for i, (img, label) in tqdm(enumerate(test_abnormal_loader)):
    #     print(label)
    #     return 


