# models/collaborative.py
import os
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.accuracy import rmse
from tqdm import tqdm
import pickle
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
import sys

# Thêm đường dẫn để import hàm đánh giá
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluate_models/DL')))
from evaluate_rating_ranking import prepare_user_metrics, compute_ndcg, compute_map, compute_diversity

# Cấu hình
THRESHOLD = 3.5
K = 10

def evaluate_svd_validation(model, valset_df):
    """Đánh giá mô hình SVD trên tập validation."""
    precision_list, recall_list, ndcg_list, map_list = [], [], [], []
    
    # Thêm tqdm để theo dõi tiến độ đánh giá qua các user
    for user_id in tqdm(valset_df['userId'].unique(), desc="Đánh giá user", leave=False):
        user_data = valset_df[valset_df['userId'] == user_id]
        relevant_movies = user_data[user_data['rating'] >= THRESHOLD]['movieId'].tolist()
        if not relevant_movies:
            continue
        
        test_movies = user_data['movieId'].unique()
        predictions = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in test_movies]
        top_k = sorted(predictions, key=lambda x: x[1], reverse=True)[:K]
        recommended_movies = [x[0] for x in top_k]
        predicted_scores = [x[1] for x in top_k]
        
        y_true, y_pred = prepare_user_metrics(relevant_movies, recommended_movies, valset_df, user_id, predicted_scores)
        precision_list.append(precision_score(y_true, y_pred, zero_division=0))
        recall_list.append(recall_score(y_true, y_pred, zero_division=0))
        ndcg_list.append(compute_ndcg(relevant_movies, recommended_movies, valset_df, user_id))
        map_list.append(compute_map(relevant_movies, recommended_movies, valset_df, user_id))
    
    return {
        'Precision@10': np.mean(precision_list) if precision_list else 0.0,
        'Recall@10': np.mean(recall_list) if recall_list else 0.0,
        'NDCG@10': np.mean(ndcg_list) if ndcg_list else 0.0,
        'MAP@10': np.mean(map_list) if map_list else 0.0
    }

def load_and_validate_data():
    df_filtered = pd.read_csv(os.path.join(data_path, "train_dl.csv"))
    if df_filtered.empty or not all(col in df_filtered for col in ['userId', 'movieId', 'rating']):
        raise ValueError("Dữ liệu train_dl.csv không hợp lệ!")
    df_filtered = df_filtered.dropna(subset=['userId', 'movieId', 'rating'])
    return df_filtered

def main():
    global data_path, save_model_path, results_model_path
    data_path = "./data/"
    save_model_path = "./save_model/"
    results_model_path = "./results_model/"
    os.makedirs(save_model_path, exist_ok=True)
    os.makedirs(results_model_path, exist_ok=True)
    
    # Load dữ liệu
    df_filtered = load_and_validate_data()
    
    # Xác định thang đo rating
    min_rating = df_filtered["rating"].min()
    max_rating = df_filtered["rating"].max()
    print(f"Thang đo rating: {min_rating} - {max_rating}")
    
    if min_rating == max_rating:
        raise ValueError("Dữ liệu chỉ có một mức rating duy nhất!")
    if df_filtered["userId"].nunique() < 10 or df_filtered["movieId"].nunique() < 5:
        raise ValueError("Dữ liệu quá nhỏ để huấn luyện!")
    
    # Chia train/validation với stratify
    train_df, val_df = train_test_split(
        df_filtered, test_size=0.25, random_state=42, stratify=df_filtered['userId']
    )
    print(f"Kích thước tập train: {len(train_df)}, tập validation: {len(val_df)}")
    
    # Tạo Reader và Dataset cho Surprise
    reader = Reader(rating_scale=(min_rating, max_rating))
    train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    val_data = Dataset.load_from_df(val_df[['userId', 'movieId', 'rating']], reader)
    
    # Chuyển train_data thành trainset
    trainset = train_data.build_full_trainset()
    valset = [(u, i, r) for u, i, r in val_df[['userId', 'movieId', 'rating']].itertuples(index=False)]
    
    # Tìm tham số tốt nhất
    param_grid = {
        'n_factors': [50, 100, 150],
        'n_epochs': [10, 20],
        'lr_all': [0.001, 0.005],
        'reg_all': [0.01, 0.1],
        'biased': [False]
    }
    print("Tìm tham số tốt nhất...")
    best_score = float('-inf')
    best_params = None
    best_model = None
    
    # Tính tổng số tổ hợp tham số
    total_combinations = len(param_grid['n_factors']) * len(param_grid['n_epochs']) * len(param_grid['lr_all']) * len(param_grid['reg_all'])
    
    # Thêm tqdm để theo dõi tiến độ tìm kiếm tham số
    with tqdm(total=total_combinations, desc="Tìm tham số", unit="combo") as pbar:
        for n_factors in param_grid['n_factors']:
            for n_epochs in param_grid['n_epochs']:
                for lr_all in param_grid['lr_all']:
                    for reg_all in param_grid['reg_all']:
                        model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, biased=False, random_state=42)
                        model.fit(trainset)
                        results = evaluate_svd_validation(model, val_df)
                        score = (results['Precision@10'] + results['Recall@10'] + results['NDCG@10'] + results['MAP@10']) / 4
                        if score > best_score:
                            best_score = score
                            best_params = {'n_factors': n_factors, 'n_epochs': n_epochs, 'lr_all': lr_all, 'reg_all': reg_all}
                            best_model = model
                        print(f"Params: n_factors={n_factors}, n_epochs={n_epochs}, lr_all={lr_all}, reg_all={reg_all} -> "
                              f"Precision@10: {results['Precision@10']:.4f}, Recall@10: {results['Recall@10']:.4f}, "
                              f"NDCG@10: {results['NDCG@10']:.4f}, MAP@10: {results['MAP@10']:.4f}")
                        pbar.update(1)
    
    # Đánh giá mô hình tốt nhất
    final_results = evaluate_svd_validation(best_model, val_df)
    
    # Lưu kết quả
    result_file = os.path.join(results_model_path, "result_train_val_collaborative.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("Kết quả huấn luyện Collaborative Filtering (SVD) trên tập validation:\n")
        f.write(f"Tham số tốt nhất: {best_params}\n")
        for metric, value in final_results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print("\nKết quả trên tập validation:")
    print(f"Tham số tốt nhất: {best_params}")
    for metric, value in final_results.items():
        print(f"{metric}: {value:.4f}")
    print(f"Kết quả lưu tại: {result_file}")
    
    # Lưu mô hình
    with open(os.path.join(save_model_path, "cf_svd_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    
    print("Hoàn tất Collaborative Filtering!")
    print(f"Mô hình lưu tại: {os.path.join(save_model_path, 'cf_svd_model.pkl')}")

if __name__ == "__main__":
    main()