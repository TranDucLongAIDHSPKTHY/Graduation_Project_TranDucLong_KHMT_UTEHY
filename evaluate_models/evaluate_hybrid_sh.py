# evaluate_models/evaluate_hybrid.py
import os
import sys
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluate_models/DL')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluate_models/ML')))
from evaluate_rating_ranking import load_common_data, predict_ratings, compute_ndcg, compute_map, compute_diversity
from evaluate_content_ranking import prepare_user_metrics as prepare_user_metrics_content, compute_precision, compute_recall

# Cấu hình
THRESHOLD = 3.5
K = 10
MIN_RATINGS_LIST = [5, 10, 15]

def prepare_user_metrics_hybrid(relevant_items, recommended_items, test_df, user_id, predicted_scores=None):
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

def evaluate_hybrid(min_ratings):
    print(f"\nĐang đánh giá mô hình Switching Hybrid với MIN_RATINGS={min_ratings}...\n")
    results_model_path = "./results_model/"
    os.makedirs(results_model_path, exist_ok=True)
    
    # Tải dữ liệu
    test_df, user_id_map, movie_id_map, movie_content_dict, df_filtered = load_common_data()
    df_full = pd.read_csv("./data/processed_movies.csv")
    
    # Tải mô hình hybrid
    with open("./save_model/hybrid.pkl", "rb") as f:
        hybrid_model = pickle.load(f)
    svd_model = hybrid_model['svd_model']
    tfidf = hybrid_model['tfidf']
    svd = hybrid_model['svd']
    knn = hybrid_model['knn']
    movie_ids = hybrid_model['movie_ids']
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    
    # Tính số lượng đánh giá của mỗi người dùng
    user_rating_counts = test_df.groupby('userId').size().to_dict()
    
    # Đánh giá
    precision_list, recall_list, ndcg_list, map_list, diversity_list = [], [], [], [], []
    rmse_predictions, rmse_actuals = [], []
    
    for user_id in tqdm(test_df['userId'].unique(), desc=f"Evaluating Hybrid (MIN_RATINGS={min_ratings})"):
        user_data = test_df[test_df['userId'] == user_id]
        if len(user_data) < 1:
            continue
        relevant_movies = user_data[user_data['rating'] >= THRESHOLD]['movieId'].tolist()
        if not relevant_movies:
            continue
        
        num_ratings = user_rating_counts.get(user_id, 0)
        recommended_movies = []
        predicted_scores = []
        
        if num_ratings >= min_ratings:
            # Sử dụng Collaborative Filtering (SVD)
            test_movies = user_data['movieId'].unique()
            predictions = [(movie_id, svd_model.predict(user_id, movie_id).est) for movie_id in test_movies]
            top_k = sorted(predictions, key=lambda x: x[1], reverse=True)[:K]
            recommended_movies = [x[0] for x in top_k]
            predicted_scores = [x[1] for x in top_k]
        else:
            # Sử dụng Content-Based Filtering
            user_watched = user_data['movieId'].tolist()
            if not user_watched or user_watched[0] not in movie_id_to_idx:
                continue
            seed_movie = user_watched[0]
            seed_idx = movie_id_to_idx[seed_movie]
            content = df_full[df_full['movieId'] == seed_movie]['content'].values
            if len(content) == 0:
                continue
            tfidf_vector = tfidf.transform([content[0]])
            reduced_vector = svd.transform(tfidf_vector)
            _, indices = knn.kneighbors(reduced_vector, n_neighbors=K+1)
            similar_indices = indices[0][1:]  # Bỏ phim gốc
            recommended_movies = [movie_ids[i] for i in similar_indices]
        
        # Tính metrics
        y_true, y_pred = prepare_user_metrics_hybrid(relevant_movies, recommended_movies, test_df, user_id, predicted_scores)
        precision_list.append(compute_precision(y_true, y_pred))
        recall_list.append(compute_recall(y_true, y_pred))
        ndcg_list.append(compute_ndcg(relevant_movies, recommended_movies, test_df, user_id))
        map_list.append(compute_map(relevant_movies, recommended_movies, test_df, user_id))
        diversity_list.append(compute_diversity(recommended_movies, df_filtered))
        
        # Tính RMSE (chỉ cho Collaborative Filtering)
        if num_ratings >= min_ratings:
            predictions = predict_ratings(svd_model, user_data, model_type='cf')
            rmse_predictions.extend(predictions)
            rmse_actuals.extend(user_data['rating'].values)
    
    # Tính RMSE tổng
    rmse_score = np.sqrt(mean_squared_error(rmse_actuals, rmse_predictions)) if rmse_predictions else 0.0
    
    # Tính trung bình các chỉ số
    results = {
        'Precision@10': np.mean(precision_list),
        'Recall@10': np.mean(recall_list),
        'NDCG@10': np.mean(ndcg_list),
        'MAP@10': np.mean(map_list),
        'Diversity@10': np.mean(diversity_list),
        'RMSE': rmse_score
    }
    
    # Lưu kết quả
    with open(os.path.join(results_model_path, f"result_hybrid_min_ratings_{min_ratings}.txt"), "w", encoding="utf-8") as f:
        f.write(f"Kết quả Switching Hybrid (MIN_RATINGS={min_ratings}):\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"\nKết quả Switching Hybrid (MIN_RATINGS={min_ratings}):")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    return results

if __name__ == "__main__":
    all_results = {}
    for min_ratings in MIN_RATINGS_LIST:
        results = evaluate_hybrid(min_ratings)
        all_results[min_ratings] = results
    
    # Ghi tóm tắt kết quả
    with open("./results_model/summary_hybrid.txt", "w", encoding="utf-8") as f:
        f.write("Tóm tắt kết quả Switching Hybrid:\n")
        for min_ratings, results in all_results.items():
            f.write(f"\nMIN_RATINGS={min_ratings}:\n")
            for metric, value in results.items():
                f.write(f"{metric}: {value:.4f}\n")

'''Hạn chế: Content-Based Filtering không dự đoán rating trực tiếp, nên RMSE chỉ được tính cho các trường hợp sử dụng Collaborative Filtering. 
Điều này có thể làm giảm giá trị RMSE tổng thể khi số lượng người dùng sử dụng Content-Based tăng lên.'''                