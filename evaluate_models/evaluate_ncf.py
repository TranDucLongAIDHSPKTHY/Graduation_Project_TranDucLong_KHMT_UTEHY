import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, mean_squared_error
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluate_models/DL')))
from ncf import NCF, Config as NCFConfig
from evaluate_rating_ranking import prepare_user_metrics, compute_ndcg, compute_map, compute_diversity, load_common_data

# Thiết lập thiết bị
NCFConfig.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cấu hình đánh giá
THRESHOLD = 3.5
K = 10
MIN_TEST_RATINGS = 5

def predict_ratings(model, test_df, user_id_map, movie_id_map, movie_content_dict):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            user_idx = user_id_map.get(user_id, -1)
            movie_idx = movie_id_map.get(movie_id, -1)
            if user_idx == -1 or movie_idx == -1:
                continue
            
            # Thêm chiều batch cho content_features
            content_features = torch.tensor(
                movie_content_dict[movie_id], 
                dtype=torch.float32,
                device=NCFConfig.device
            ).unsqueeze(0)  # <--- Sửa ở đây
            
            user_idx = torch.tensor([user_idx], dtype=torch.long, device=NCFConfig.device)
            movie_idx = torch.tensor([movie_idx], dtype=torch.long, device=NCFConfig.device)
            
            pred = model(user_idx, movie_idx, content_features).squeeze().cpu().numpy()
            predictions.append(pred)
    return np.array(predictions)

def evaluate_ncf():
    print("\nĐang đánh giá Neural Collaborative Filtering (NCF)...\n")
    os.makedirs("./results_model", exist_ok=True)
    
    # Load dữ liệu và mô hình
    test_df, user_id_map, movie_id_map, movie_content_dict, df_filtered = load_common_data()
    num_users = len(user_id_map)
    num_movies = len(movie_id_map)
    
    # Khởi tạo mô hình
    model = NCF(num_users, num_movies, NCFConfig.content_dim, NCFConfig.embedding_dim, NCFConfig.hidden_dims).to(NCFConfig.device)
    model.load_state_dict(torch.load("./save_model/best_ncf.pth", map_location=NCFConfig.device))
    model.eval()
    
    # Dự đoán ratings
    test_df['pred_rating'] = predict_ratings(model, test_df, user_id_map, movie_id_map, movie_content_dict)
    rmse_score = np.sqrt(mean_squared_error(test_df['rating'], test_df['pred_rating']))
    print(f"RMSE: {rmse_score:.4f}")
    
    # Tính toán các chỉ số ranking
    precision_list, recall_list, ndcg_list, map_list, diversity_list = [], [], [], [], []
    
    for user_id in tqdm(test_df['userId'].unique(), desc="Evaluating NCF"):
        user_data = test_df[test_df['userId'] == user_id]
        if len(user_data) < MIN_TEST_RATINGS:
            continue
        relevant_movies = user_data[user_data['rating'] >= THRESHOLD]['movieId'].tolist()
        if not relevant_movies:
            continue
        
        # Lấy top-K recommendations
        user_predictions = user_data.sort_values('pred_rating', ascending=False).head(K)
        recommended_movies = user_predictions['movieId'].tolist()
        predicted_scores = user_predictions['pred_rating'].tolist()
        
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
    result_file = "./results_model/result_ncf.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("Kết quả đánh giá Neural Collaborative Filtering (NCF):\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print("\nKết quả NCF:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print(f"Kết quả lưu tại: {result_file}")
    
    return results

if __name__ == "__main__":
    evaluate_ncf()