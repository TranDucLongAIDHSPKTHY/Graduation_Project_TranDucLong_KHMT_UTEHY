# evaluate_models/evaluate_content_based.py
import os
import pickle
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluate_models/ML')))
from evaluate_content_ranking import *

def evaluate_content_based():
    print("\nĐang đánh giá mô hình Content-Based...\n")
    results_model_path = "./results_model/"
    os.makedirs(results_model_path, exist_ok=True)

    # Load dữ liệu
    test_df = pd.read_csv("./data/test_common.csv")
    df_full = pd.read_csv("./data/processed_movies.csv")

    # Load mô hình
    with open("./save_model/tfidf_vectorizer_cb.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("./save_model/svd_cb.pkl", "rb") as f:
        svd = pickle.load(f)
    with open("./save_model/knn_cb.pkl", "rb") as f:
        knn = pickle.load(f)
    with open("./save_model/movie_ids_cb.pkl", "rb") as f:
        movie_ids = pickle.load(f)

    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    precision_list, recall_list, ndcg_list, map_list = [], [], [], []
    user_groups = test_df.groupby('userId')

    for user_id, group in tqdm(user_groups, desc="Evaluating users"):
        relevant_movies = group[group['rating'] >= THRESHOLD]['movieId'].tolist()
        user_watched = group['movieId'].tolist()

        # Bỏ qua người dùng không có phim liên quan hoặc danh sách xem trống
        if len(relevant_movies) == 0 or len(user_watched) == 0:
            continue

        # Logic chọn seed movie:
        # - Ưu tiên phim có rating cao nhất (nếu có rating)
        if 'rating' in group.columns:
            seed_movie = group.loc[group['rating'].idxmax(), 'movieId']
        else:
            # Fallback: Chọn phim đầu tiên nếu không có rating
            seed_movie = user_watched[0]

        # Kiểm tra seed movie có trong danh sách đặc trưng không
        if seed_movie not in movie_id_to_idx:
            continue

        seed_idx = movie_id_to_idx[seed_movie]
        _, indices = knn.kneighbors([svd.transform(tfidf.transform([df_full[df_full['movieId'] == seed_movie]['content'].values[0]]))[0]], n_neighbors=K+1)
        similar_indices = indices[0][1:]  # Bỏ phim gốc
        recommended_movies = [movie_ids[i] for i in similar_indices]

        y_true, y_pred, y_scores = prepare_user_metrics(relevant_movies, recommended_movies, None)
        precision_list.append(compute_precision(y_true, y_pred))
        recall_list.append(compute_recall(y_true, y_pred))
        ndcg_list.append(compute_ndcg(y_true, y_pred))
        map_list.append(compute_map(y_true, y_pred))

    results = {
        'Precision@10': np.mean(precision_list),
        'Recall@10': np.mean(recall_list),
        'NDCG@10': np.mean(ndcg_list),
        'MAP@10': np.mean(map_list)
    }

    # Ghi kết quả
    with open(os.path.join(results_model_path, "result_content_based.txt"), "w", encoding="utf-8") as f:
        f.write("Kết quả đánh giá Content-Based:\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")

    print("\nKết quả Content-Based:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    return results

if __name__ == "__main__":
    evaluate_content_based()  