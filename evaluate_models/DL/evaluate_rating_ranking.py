import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, average_precision_score, ndcg_score, mean_squared_error
import pickle
import torch

# Cấu hình
THRESHOLD = 3.5
K = 10

def load_common_data():
    test_df = pd.read_csv("./data/test_common.csv")
    if test_df.empty or not all(col in test_df for col in ['userId', 'movieId', 'rating']):
        raise ValueError("Dữ liệu test_common.csv không hợp lệ!")
    with open("./save_model/user_id_map.pkl", "rb") as f:
        user_id_map = pickle.load(f)
    with open("./save_model/movie_id_map.pkl", "rb") as f:
        movie_id_map = pickle.load(f)
    with open("./save_model/movie_content_dict.pkl", "rb") as f:
        movie_content_dict = pickle.load(f)
    df_filtered = pd.read_csv("data/processed_movies.csv")
    if df_filtered.empty or not all(col in df_filtered for col in ['userId', 'movieId', 'rating', 'genres']):
        raise ValueError("Dữ liệu processed_movies.csv không hợp lệ!")
    return test_df, user_id_map, movie_id_map, movie_content_dict, df_filtered

def compute_diversity(recommended_movies, df_filtered):
    genres = df_filtered[df_filtered['movieId'].isin(recommended_movies)]['genres'].str.split().explode().unique()
    return len(genres)

def prepare_user_metrics(relevant_items, recommended_items, test_df, user_id, predicted_scores=None):
    all_candidates = list(set(recommended_items).union(set(relevant_items)))
    user_test_data = test_df[(test_df['userId'] == user_id) & (test_df['movieId'].isin(all_candidates))]
    rating_dict = dict(zip(user_test_data['movieId'], user_test_data['rating']))
    
    y_true = [1 if rating_dict.get(item, 0.0) >= THRESHOLD else 0 for item in all_candidates]
    if predicted_scores is not None:
        pred_dict = dict(zip(recommended_items, predicted_scores))
        y_pred = [1 if pred_dict.get(item, 0.0) >= THRESHOLD else 0 for item in all_candidates]
    else:
        y_pred = [1 if item in recommended_items else 0 for item in all_candidates]
    
    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int)

def compute_ndcg(relevant_items, recommended_items, test_df, user_id, k=K):
    y_true, y_pred = prepare_user_metrics(relevant_items, recommended_items, test_df, user_id)
    if len(y_true) <= 1:
        return 1.0 if (len(y_true) == 1 and y_true[0] > 0) else 0.0
    return ndcg_score([y_true], [y_pred], k=k)

def compute_map(relevant_items, recommended_items, test_df, user_id, k=K):
    y_true, y_pred = prepare_user_metrics(relevant_items, recommended_items, test_df, user_id)
    if np.sum(y_true) == 0:
        return 0.0
    return average_precision_score(y_true, y_pred)

def predict_ratings(model, test_df, model_type='cf', user_id_map=None, movie_id_map=None, movie_content_dict=None, **kwargs):
    predictions = []
    if model_type == 'cf':
        # Logic cho Collaborative Filtering (SVD)
        for _, row in test_df.iterrows():
            user_id, movie_id = row['userId'], row['movieId']
            pred = model.predict(user_id, movie_id).est
            predictions.append(pred)
    elif model_type in ['ae', 'ncf']:  # Gộp chung logic cho cả AE và NCF
        # Xác định device từ model
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for _, row in test_df.iterrows():
                user_id = row['userId']
                movie_id = row['movieId']
                user_idx = user_id_map.get(user_id, -1)
                movie_idx = movie_id_map.get(movie_id, -1)
                if user_idx == -1 or movie_idx == -1:
                    continue
                # Chuyển đổi sang tensor và thêm chiều batch
                user_idx_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
                movie_idx_tensor = torch.tensor([movie_idx], dtype=torch.long, device=device)
                content_features = torch.tensor(
                    movie_content_dict[movie_id], 
                    dtype=torch.float32, 
                    device=device
                ).unsqueeze(0)  # Thêm chiều batch (shape [1, content_dim])
                # Dự đoán
                pred = model(user_idx_tensor, movie_idx_tensor, content_features).squeeze().cpu().numpy()
                predictions.append(pred)
    else:
        raise ValueError("Loại mô hình không được hỗ trợ!")
    return predictions