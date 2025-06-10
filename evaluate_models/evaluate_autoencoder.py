import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, mean_squared_error
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluate_models/DL')))
from autoencoder import Autoencoder, Config as AEConfig
from evaluate_rating_ranking import prepare_user_metrics, compute_ndcg, compute_map, compute_diversity, load_common_data, predict_ratings

# Thiết lập
AEConfig.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 3.5
K = 10

def evaluate_autoencoder():
    print("\nĐang đánh giá Autoencoder...\n")
    os.makedirs("./results_model", exist_ok=True)
    
    # Load dữ liệu và mô hình
    test_df, user_id_map, movie_id_map, movie_content_dict, df_filtered = load_common_data()
    model = Autoencoder(len(user_id_map), len(movie_id_map), AEConfig.content_dim, AEConfig.hidden_dims).to(AEConfig.device)
    model.load_state_dict(torch.load("./save_model/best_autoencoder.pth", map_location=AEConfig.device))
    model.eval()
    
    # Tính RMSE
    test_df['pred_rating'] = predict_ratings(
        model=model,
        test_df=test_df,
        model_type='ae',
        user_id_map=user_id_map,
        movie_id_map=movie_id_map,
        movie_content_dict=movie_content_dict
    )
    rmse_score = np.sqrt(mean_squared_error(test_df['rating'], test_df['pred_rating']))
    print(f"RMSE: {rmse_score:.4f}")
    
    # Đánh giá ranking
    unique_users = test_df['userId'].unique()
    precision_list, recall_list, ndcg_list, map_list, diversity_list = [], [], [], [], []
    
    for user_id in tqdm(unique_users, desc="Evaluating Autoencoder"):
        user_data = test_df[test_df['userId'] == user_id]
        if len(user_data) < 5:
            continue
        relevant_movies = user_data[user_data['rating'] >= THRESHOLD]['movieId'].tolist()
        if not relevant_movies:
            continue
        
        # Dự đoán và chọn top-K
        user_predictions = test_df[test_df['userId'] == user_id][['movieId', 'pred_rating']]
        top_k = user_predictions.sort_values('pred_rating', ascending=False).head(K)
        recommended_movies = top_k['movieId'].tolist()
        predicted_scores = top_k['pred_rating'].tolist()
        
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
    
    # Tổng hợp kết quả
    results = {
        'Precision@10': np.mean(precision_list) if precision_list else 0.0,
        'Recall@10': np.mean(recall_list) if recall_list else 0.0,
        'NDCG@10': np.mean(ndcg_list) if ndcg_list else 0.0,
        'MAP@10': np.mean(map_list) if map_list else 0.0,
        'Diversity@10': np.mean(diversity_list) if diversity_list else 0.0,
        'RMSE': rmse_score
    }
    
    # Lưu kết quả
    result_file = "./results_model/result_autoencoder.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("Kết quả Autoencoder:\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print("\nKết quả Autoencoder:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print(f"Kết quả lưu tại: {result_file}")
    
    return results

if __name__ == "__main__":
    evaluate_autoencoder()