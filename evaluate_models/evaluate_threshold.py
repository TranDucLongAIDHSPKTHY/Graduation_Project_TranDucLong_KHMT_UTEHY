import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.metrics import precision_score, recall_score

# Cấu hình
THRESHOLD = 3.5
K = 10

def load_data():
    """Đọc dữ liệu train và test để xác định số lượng ratings của user."""
    train_df = pd.read_csv("./data/train_dl.csv")
    user_rating_counts = train_df['userId'].value_counts().to_dict()
    test_df = pd.read_csv("./data/test_common.csv")
    return user_rating_counts, test_df

def evaluate_collaborative(test_subset, model):
    """Đánh giá Collaborative Filtering trên tập user cụ thể."""
    precision_list = []
    recall_list = []
    skipped_users = 0
    for user_id in tqdm(test_subset['userId'].unique(), desc="Collaborative Evaluation", leave=False):
        user_data = test_subset[test_subset['userId'] == user_id]
        relevant = user_data[user_data['rating'] >= THRESHOLD]['movieId'].tolist()
        if not relevant:
            skipped_users += 1
            continue
        
        test_movies = user_data['movieId'].unique()
        predictions = []
        for movie in test_movies:
            try:
                est = model.predict(user_id, movie).est
                predictions.append((movie, est))
            except:
                continue
        if not predictions:
            skipped_users += 1
            continue
        top_k = sorted(predictions, key=lambda x: x[1], reverse=True)[:K]
        recommended = [movie for movie, _ in top_k]
        
        y_true = [1 if movie in relevant else 0 for movie in recommended]
        if not y_true:
            skipped_users += 1
            continue
        precision = precision_score(y_true, [1]*len(y_true), zero_division=0)
        recall = recall_score(y_true, [1]*len(y_true), zero_division=0)
        precision_list.append(precision)
        recall_list.append(recall)
    
    return (np.mean(precision_list) if precision_list else 0.0,
            np.mean(recall_list) if recall_list else 0.0,
            skipped_users)

def evaluate_content_based(test_subset, tfidf, svd, knn, movie_ids, df_full):
    """Đánh giá Content-Based trên tập user cụ thể."""
    precision_list = []
    recall_list = []
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    skipped_users = 0
    
    for user_id in tqdm(test_subset['userId'].unique(), desc="Content-Based Evaluation", leave=False):
        user_data = test_subset[test_subset['userId'] == user_id]
        relevant = user_data[user_data['rating'] >= THRESHOLD]['movieId'].tolist()
        if not relevant or len(user_data) == 0:
            skipped_users += 1
            continue
        
        seed_movie = user_data.iloc[0]['movieId']
        
        if seed_movie not in movie_id_to_idx:
            skipped_users += 1
            continue
        
        try:
            content = df_full[df_full['movieId'] == seed_movie]['content'].values[0]
            tfidf_vec = tfidf.transform([content])
            reduced_vec = svd.transform(tfidf_vec)
            _, indices = knn.kneighbors(reduced_vec, n_neighbors=K+1)
            recommended = [movie_ids[i] for i in indices[0][1:]]  # Bỏ qua seed
        except:
            skipped_users += 1
            continue
        
        y_true = [1 if movie in relevant else 0 for movie in recommended]
        if not y_true:
            skipped_users += 1
            continue
        precision = precision_score(y_true, [1]*len(y_true), zero_division=0)
        recall = recall_score(y_true, [1]*len(y_true), zero_division=0)
        precision_list.append(precision)
        recall_list.append(recall)
    
    return (np.mean(precision_list) if precision_list else 0.0,
            np.mean(recall_list) if recall_list else 0.0,
            skipped_users)

def main():
    user_rating_counts, test_df = load_data()
    
    # Load các mô hình
    with open("./save_model/cf_svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)
    with open("./save_model/tfidf_vectorizer_cb.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("./save_model/svd_cb.pkl", "rb") as f:
        svd = pickle.load(f)
    with open("./save_model/knn_cb.pkl", "rb") as f:
        knn = pickle.load(f)
    with open("./save_model/movie_ids_cb.pkl", "rb") as f:
        movie_ids = pickle.load(f)
    df_full = pd.read_csv("./data/processed_movies.csv")
    
    threshold_groups = [
        {'min': 1, 'max': 5, 'label': '1-5'},
        {'min': 6, 'max': 10, 'label': '6-10'},
        {'min': 11, 'max': 15, 'label': '11-15'},
    ]
    
    results = []
    
    for group in threshold_groups:
        min_r = group['min']
        max_r = group['max']
        
        group_users = [
            user for user, count in user_rating_counts.items()
            if min_r <= count <= max_r
        ]
        
        group_test = test_df[test_df['userId'].isin(group_users)]
        num_users = len(group_test['userId'].unique())
        print(f"\n=== Đang đánh giá nhóm: {group['label']} ===")
        print(f"Số user trong test set: {num_users}")
        
        cb_precision, cb_recall, cb_skipped = evaluate_content_based(group_test, tfidf, svd, knn, movie_ids, df_full)
        cf_precision, cf_recall, cf_skipped = evaluate_collaborative(group_test, svd_model)
        
        print(f"CB - Precision@10: {cb_precision:.4f}, Recall@10: {cb_recall:.4f}, Skipped Users: {cb_skipped}")
        print(f"CF - Precision@10: {cf_precision:.4f}, Recall@10: {cf_recall:.4f}, Skipped Users: {cf_skipped}")
        
        results.append({
            'Group (num_ratings)': group['label'],
            'Precision_CB': round(cb_precision, 4),
            'Recall_CB': round(cb_recall, 4),
            'Precision_CF': round(cf_precision, 4),
            'Recall_CF': round(cf_recall, 4),
            'Users_in_Test': num_users,
            'CB_Skipped_Users': cb_skipped,
            'CF_Skipped_Users': cf_skipped
        })
    
    result_df = pd.DataFrame(results)
    print("\nKết quả đánh giá:")
    print(result_df.to_markdown(index=False, floatfmt=".4f"))
    
    os.makedirs("./results_model", exist_ok=True)
    result_df.to_csv("./results_model/threshold_group_metrics.txt", index=False)
    print("\nKết quả đã được lưu vào: results_model/threshold_group_metrics.txt")

if __name__ == "__main__":
    main()
