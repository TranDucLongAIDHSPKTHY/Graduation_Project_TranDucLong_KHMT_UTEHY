# models/content_based.py
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import pickle
import sys

# Thêm đường dẫn để import các hàm đánh giá
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluate_models/ML')))
from evaluate_content_ranking import prepare_user_metrics, compute_precision, compute_recall, compute_ndcg, compute_map

# Cấu hình
THRESHOLD = 3.5
K = 10

def evaluate_content_based_validation(df, tfidf, svd, knn, movie_ids):
    """Đánh giá mô hình Content-Based trên tập validation."""
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    precision_list, recall_list, ndcg_list, map_list = [], [], [], []
    
    for user_id in df['userId'].unique():
        user_data = df[df['userId'] == user_id]
        relevant_movies = user_data[user_data['rating'] >= THRESHOLD]['movieId'].tolist()
        user_watched = user_data['movieId'].tolist()
        
        if len(relevant_movies) == 0 or len(user_watched) == 0:
            continue
            
        # Chọn seed movie có rating cao nhất
        seed_movie = user_data.loc[user_data['rating'].idxmax(), 'movieId']
        if seed_movie not in movie_id_to_idx:
            continue
            
        seed_idx = movie_id_to_idx[seed_movie]
        _, indices = knn.kneighbors([svd.transform(tfidf.transform([df[df['movieId'] == seed_movie]['content'].values[0]]))[0]], n_neighbors=K+1)
        recommended_movies = [movie_ids[i] for i in indices[0][1:]]
        
        y_true, y_pred, _ = prepare_user_metrics(relevant_movies, recommended_movies, None)
        precision_list.append(compute_precision(y_true, y_pred))
        recall_list.append(compute_recall(y_true, y_pred))
        ndcg_list.append(compute_ndcg(y_true, y_pred))
        map_list.append(compute_map(y_true, y_pred))
    
    return {
        'Precision@10': np.mean(precision_list) if precision_list else 0.0,
        'Recall@10': np.mean(recall_list) if recall_list else 0.0,
        'NDCG@10': np.mean(ndcg_list) if ndcg_list else 0.0,
        'MAP@10': np.mean(map_list) if map_list else 0.0
    }

def train_content_based_model():
    print("Đang huấn luyện mô hình Content-Based...")

    # Đường dẫn
    data_path = "./data/"
    save_model_path = "./save_model/"
    results_model_path = "./results_model/"
    os.makedirs(save_model_path, exist_ok=True)
    os.makedirs(results_model_path, exist_ok=True)

    # Đọc dữ liệu
    df = pd.read_csv(os.path.join(data_path, "train_dl.csv"))
    df = df.dropna(subset=['content'])

    # Chia dữ liệu thành train (60%) và validation (20%)
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df['userId'])
    print(f"Kích thước tập train: {len(train_df)}, tập validation: {len(val_df)}")

    # Vector hóa bằng TF-IDF
    print("Vector hóa nội dung phim bằng TF-IDF...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=300)
    tfidf_matrix = tfidf.fit_transform(train_df['content'])

    # Giảm chiều bằng Truncated SVD
    print("Giảm chiều đặc trưng bằng Truncated SVD...")
    svd = TruncatedSVD(n_components=200, random_state=42)
    reduced_features = svd.fit_transform(tfidf_matrix)
    
    # Tìm hàng xóm gần nhất
    print("Xây dựng mô hình kNN...")
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(reduced_features)

    # Đánh giá trên tập validation
    print("Đánh giá mô hình trên tập validation...")
    results = evaluate_content_based_validation(val_df, tfidf, svd, knn, train_df['movieId'].tolist())
    
    # Lưu kết quả vào file
    result_file = os.path.join(results_model_path, "result_train_val_content_based.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("Kết quả huấn luyện Content-Based trên tập validation:\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print("Kết quả trên tập validation:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print(f"Kết quả lưu tại: {result_file}")

    # Lưu mô hình và các thành phần
    with open(os.path.join(save_model_path, "tfidf_vectorizer_cb.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(save_model_path, "svd_cb.pkl"), "wb") as f:
        pickle.dump(svd, f)
    with open(os.path.join(save_model_path, "knn_cb.pkl"), "wb") as f:
        pickle.dump(knn, f)
    with open(os.path.join(save_model_path, "movie_ids_cb.pkl"), "wb") as f:
        pickle.dump(train_df['movieId'].tolist(), f)

    print("Đã huấn luyện và lưu mô hình Content-Based!")

if __name__ == "__main__":
    train_content_based_model()