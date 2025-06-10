import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score, ndcg_score

THRESHOLD = 3.5
K = 10

def prepare_user_metrics(relevant_items, recommended_items, all_items_scores):
    all_items = list(set(recommended_items + relevant_items))
    y_true = [1 if item in relevant_items else 0 for item in all_items]
    y_pred = [1 if item in recommended_items else 0 for item in all_items]
    y_scores = []
    if all_items_scores is not None:
        y_scores = [score for item, score in all_items_scores if item in all_items]
    return y_true, y_pred, y_scores

def compute_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)

def compute_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)

def compute_ndcg(y_true, y_pred):
    if len(y_true) <= 1:
        return 1.0 if (len(y_true) == 1 and y_true[0] > 0) else 0.0
    return ndcg_score([y_true], [y_pred], k=K)

def compute_map(y_true, y_scores):
    if np.sum(y_true) == 0 or len(y_scores) == 0:
        return 0.0
    return average_precision_score(y_true, y_scores)