import scipy
import numpy as np
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



