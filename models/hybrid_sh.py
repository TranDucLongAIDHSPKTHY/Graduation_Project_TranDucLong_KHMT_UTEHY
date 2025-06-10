# models/hybrid_sh.py
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
import sys

# Thêm đường dẫn để import hàm đánh giá
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluate_models/DL')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluate_models/ML')))
from evaluate_rating_ranking import prepare_user_metrics, compute_ndcg, compute_map, compute_diversity
from evaluate_content_ranking import prepare_user_metrics as prepare_user_metrics_content, compute_precision, compute_recall

# Cấu hình
THRESHOLD = 3.5
K = 10
MIN_RATINGS = 5  # Ngưỡng để chọn giữa SVD và Content-Based

def evaluate_hybrid_validation(val_df, hybrid_model):
    """Đánh giá mô hình Hybrid trên tập validation."""
    svd_model = hybrid_model['svd_model']
    tfidf = hybrid_model['tfidf']
    svd = hybrid_model['svd']
    knn = hybrid_model['knn']
    movie_ids = hybrid_model['movie_ids']
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    
    user_rating_counts = val_df.groupby('userId').size().to_dict()
    precision_list, recall_list, ndcg_list, map_list = [], [], [], []
    
    for user_id in tqdm(val_df['userId'].unique(), desc="Evaluating Hybrid"):
        user_data = val_df[val_df['userId'] == user_id]
        relevant_movies = user_data[user_data['rating'] >= THRESHOLD]['movieId'].tolist()
        if not relevant_movies:
            continue
        
        num_ratings = user_rating_counts.get(user_id, 0)
        recommended_movies = []
        predicted_scores = []
        
        if num_ratings >= MIN_RATINGS:
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
            content = val_df[val_df['movieId'] == seed_movie]['content'].values
            if len(content) == 0:
                continue
            tfidf_vector = tfidf.transform([content[0]])
            reduced_vector = svd.transform(tfidf_vector)
            _, indices = knn.kneighbors(reduced_vector, n_neighbors=K+1)
            similar_indices = indices[0][1:]
            recommended_movies = [movie_ids[i] for i in similar_indices]
        
        y_true, y_pred = prepare_user_metrics(relevant_movies, recommended_movies, val_df, user_id, predicted_scores)
        precision_list.append(compute_precision(y_true, y_pred))
        recall_list.append(compute_recall(y_true, y_pred))
        ndcg_list.append(compute_ndcg(relevant_movies, recommended_movies, val_df, user_id))
        map_list.append(compute_map(relevant_movies, recommended_movies, val_df, user_id))
    
    return {
        'Precision@10': np.mean(precision_list) if precision_list else 0.0,
        'Recall@10': np.mean(recall_list) if recall_list else 0.0,
        'NDCG@10': np.mean(ndcg_list) if ndcg_list else 0.0,
        'MAP@10': np.mean(map_list) if map_list else 0.0
    }

def train_hybrid_model():
    print("Đang xây dựng mô hình Switching Hybrid...")
    
    # Đường dẫn
    data_path = "./data/"
    save_model_path = "./save_model/"
    results_model_path = "./results_model/"
    os.makedirs(save_model_path, exist_ok=True)
    os.makedirs(results_model_path, exist_ok=True)
    
    # Tải các mô hình đã huấn luyện
    try:
        with open(os.path.join(save_model_path, "cf_svd_model.pkl"), "rb") as f:
            svd_model = pickle.load(f)
        with open(os.path.join(save_model_path, "tfidf_vectorizer_cb.pkl"), "rb") as f:
            tfidf = pickle.load(f)
        with open(os.path.join(save_model_path, "svd_cb.pkl"), "rb") as f:
            svd = pickle.load(f)
        with open(os.path.join(save_model_path, "knn_cb.pkl"), "rb") as f:
            knn = pickle.load(f)
        with open(os.path.join(save_model_path, "movie_ids_cb.pkl"), "rb") as f:
            movie_ids = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Không tìm thấy tệp mô hình: {e}")
    
    # Tạo từ điển chứa các thành phần của mô hình hybrid
    hybrid_model = {
        'svd_model': svd_model,
        'tfidf': tfidf,
        'svd': svd,
        'knn': knn,
        'movie_ids': movie_ids
    }
    
    # Đọc dữ liệu train để chia validation
    df = pd.read_csv(os.path.join(data_path, "train_dl.csv"))
    _, val_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df['userId'])
    print(f"Kích thước tập validation: {len(val_df)}")
    
    # Đánh giá trên tập validation
    print("Đánh giá mô hình Hybrid trên tập validation...")
    results = evaluate_hybrid_validation(val_df, hybrid_model)
    
    # Lưu kết quả
    result_file = os.path.join(results_model_path, "result_train_val_hybrid_sh.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("Kết quả huấn luyện Switching Hybrid trên tập validation:\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print("Kết quả trên tập validation:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print(f"Kết quả lưu tại: {result_file}")
    
    # Lưu mô hình hybrid
    with open(os.path.join(save_model_path, "hybrid.pkl"), "wb") as f:
        pickle.dump(hybrid_model, f)
    
    print(f"Đã lưu mô hình Hybrid tại: {os.path.join(save_model_path, 'hybrid.pkl')}")

if __name__ == "__main__":
    train_hybrid_model()