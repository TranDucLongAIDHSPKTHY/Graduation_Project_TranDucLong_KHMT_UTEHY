import sys
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle
from sklearn.metrics import precision_score, recall_score, mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluate_models/DL')))
from evaluate_rating_ranking import prepare_user_metrics, compute_ndcg, compute_map, compute_diversity, load_common_data, predict_ratings

# Cấu hình
THRESHOLD = 3.5
K = 10
MIN_TEST_RATINGS = 5

def compute_rmse(model, test_df):
    predictions = predict_ratings(model, test_df, model_type='cf')
    actuals = test_df['rating'].values
    return np.sqrt(mean_squared_error(actuals, predictions))

def evaluate_collaborative():
    print(f"\nĐang đánh giá Collaborative Filtering (SVD)...\n")
    os.makedirs("results_model", exist_ok=True)
    
    # Load dữ liệu
    test_df, user_id_map, movie_id_map, movie_content_dict, df_filtered = load_common_data()
    try:
        with open("save_model/cf_svd_model.pkl", "rb") as f:
            svd_model = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Không tìm thấy tệp mô hình: {e}")
    
    # Tính RMSE
    rmse_score = compute_rmse(svd_model, test_df)
    print(f"RMSE trên test_common.csv: {rmse_score:.4f}")
    
    # Đánh giá ranking
    unique_users = test_df['userId'].unique()
    precision_list, recall_list, ndcg_list, map_list, diversity_list = [], [], [], [], []
    
    for user_id in tqdm(unique_users, desc="Evaluating SVD"):
        user_data = test_df[test_df['userId'] == user_id]
        if len(user_data) < MIN_TEST_RATINGS:
            continue
        relevant_movies = user_data[user_data['rating'] >= THRESHOLD]['movieId'].tolist()
        if not relevant_movies:
            continue
        
        # Dự đoán rating và chọn top-K
        test_movies = test_df[test_df['userId'] == user_id]['movieId'].unique()
        predictions = [(movie_id, svd_model.predict(user_id, movie_id).est) for movie_id in test_movies]
        top_k = sorted(predictions, key=lambda x: x[1], reverse=True)[:K]
        recommended_movies = [x[0] for x in top_k]
        predicted_scores = [x[1] for x in top_k]
        
        # Tính metrics
        y_true, y_pred = prepare_user_metrics(relevant_movies, recommended_movies, test_df, user_id, predicted_scores)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        ndcg = compute_ndcg(relevant_movies, recommended_movies, test_df, user_id)
        map_score = compute_map(relevant_movies, recommended_movies, test_df, user_id)
        diversity = compute_diversity(recommended_movies, df_filtered)
        
        precision_list.append(precision)
        recall_list.append(recall)
        ndcg_list.append(ndcg)
        map_list.append(map_score)
        diversity_list.append(diversity)
    
    # Lưu kết quả
    results = {
        'Precision@10': np.mean(precision_list),
        'Recall@10': np.mean(recall_list),
        'NDCG@10': np.mean(ndcg_list),
        'MAP@10': np.mean(map_list),
        'Diversity@10': np.mean(diversity_list),
        'RMSE': rmse_score
    }
    
    with open("results_model/result_collaborative.txt", "w", encoding="utf-8") as f:
        f.write(f"Kết quả Collaborative Filtering:\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print("\nKết quả Collaborative Filtering:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    return results

if __name__ == "__main__":
    evaluate_collaborative()