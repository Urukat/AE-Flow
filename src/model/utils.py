import torch
import scipy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

@torch.no_grad()
def plot_distribution(model, beta, normal_loader, abnormal_loader, dataset_name, epoch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    normal_anomaly_scores = []
    abnormal_anomaly_scores = []
    for i, (img, label) in tqdm(enumerate(normal_loader)):
        img = img.to(device)
        label = label.to(device)

        rec_img, z_hat, jac = model(img)
        flow_loss, log_z = model.flow_loss()
        anomaly_score = np.array(model.anomaly_score(beta, log_z, img).cpu())
        for score in anomaly_score:
            normal_anomaly_scores.append(score)
        # break
    
    for i, (img, label) in tqdm(enumerate(abnormal_loader)):
        img = img.to(device)
        label = label.to(device)

        rec_img, z_hat, jac = model(img)
        flow_loss, log_z = model.flow_loss()
        anomaly_score = np.array(model.anomaly_score(beta, log_z, img).cpu())
        for score in anomaly_score:
            abnormal_anomaly_scores.append(score)
        # break
    
    ana_min = -min(normal_anomaly_scores + abnormal_anomaly_scores)
    # ana_max = -max(normal_anomaly_scores + abnormal_anomaly_scores)
    # normal_anomaly_scores = np.array(normal_anomaly_scores)
    # abnormal_anomaly_scores = np.array(abnormal_anomaly_scores)
    # normal_anomaly_scores += ana_max
    # abnormal_anomaly_scores += ana_max

    fig, ax = plt.subplots()
    ax.hist(np.array(normal_anomaly_scores) / ana_min, color = 'green', alpha=0.5, label = 'normal', bins=20)
    ax.hist(np.array(abnormal_anomaly_scores) / ana_min, color = 'red', alpha=0.5, label = 'abnormal', bins=20)
    plt.savefig("./src/graphs/{}_{}.png".format(dataset_name, epoch))
    # for i, (img, label) in tqdm(enumerate(test_abnormal_loader)):
    #     print(label)
    #     return 

def metrics(labels, anomaly_scores, threshold):
    results = {}
    # print(len())
    # Calculate true positive (tp), false positive (fp), false negative (fn), true negative (tn)
    pred = (anomaly_scores > threshold).astype(int)
    tp = sum(labels[i] == 1 and pred[i] == 1 for i in range(len(labels)))
    fp = sum(labels[i] == 0 and pred[i] == 1 for i in range(len(labels)))
    fn = sum(labels[i] == 1 and pred[i] == 0 for i in range(len(labels)))
    tn = sum(labels[i] == 0 and pred[i] == 0 for i in range(len(labels)))

    # # Calculate true positive rate (sensitivity/recall)
    sen = tp / (tp + fn)
    # # Calculate true negative rate (specificity)
    spe = tn / (tn + fp)
    # # Calculate accuracy
    # acc = (tp + tn) / (tp + fp + fn + tn)
    # # Calculate F1 score
    # f1 = (2 * tp) / (2 * tp + fp + fn)
    # # Calculate ROC curve
    fpr, tpr, _ = roc_curve(labels, anomaly_scores)
    # Calculate AUC
    auc_score = auc(fpr, tpr)

    # Store the results
    results['AUC'] = auc_score
    results['ACC'] = accuracy_score(y_pred=pred, y_true=labels)
    results['SEN'] = sen
    results['SPE'] = spe
    results['F1'] = f1_score(y_pred=pred, y_true=labels)

    return results

@torch.no_grad()
def get_test_results(model, beta, threshold, normal_loader, abnormal_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    anomaly_scores = []
    labels = []
    for i, (img, label) in tqdm(enumerate(normal_loader)):
        img = img.to(device)
        label = label.to(device)

        rec_img, z_hat, jac = model(img)
        flow_loss, log_z = model.flow_loss()
        anomaly_score = np.array(model.anomaly_score(beta, log_z, img).cpu())
        for score in anomaly_score:
            anomaly_scores.append(score)
            labels.append(0)
        # break
    
    for i, (img, label) in tqdm(enumerate(abnormal_loader)):
        img = img.to(device)
        label = label.to(device)

        rec_img, z_hat, jac = model(img)
        flow_loss, log_z = model.flow_loss()
        anomaly_score = np.array(model.anomaly_score(beta, log_z, img).cpu())
        for score in anomaly_score:
            anomaly_scores.append(score)
            labels.append(1)
        # break
    
    return metrics(labels, anomaly_scores, threshold)
    

    

