# models/ncf.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from time import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import sys

# Thêm đường dẫn để import hàm đánh giá
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluate_models/DL')))
from evaluate_rating_ranking import prepare_user_metrics, compute_ndcg, compute_map, compute_diversity, predict_ratings

# Đường dẫn
data_path = "./data/"
save_model_path = "./save_model/"
results_model_path = "./results_model/"

# Tham số mô hình
class Config:
    content_dim = 303
    embedding_dim = 64
    hidden_dims = [512, 256, 128, 64]
    batch_size = 64
    epochs = 30
    learning_rate = 0.001
    num_workers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patience = 5
    min_delta = 0.001

# Dataset
class MovieRatingDataset(Dataset):
    def __init__(self, df, movie_content_dict):
        if df.empty or not all(col in df for col in ['user_idx', 'movie_idx', 'rating', 'movieId']):
            raise ValueError("Dữ liệu đầu vào không hợp lệ!")
        self.user_idx = torch.tensor(df['user_idx'].values, dtype=torch.long)
        self.movie_idx = torch.tensor(df['movie_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
        content_features = np.array([movie_content_dict[mid] for mid in df['movieId']], dtype=np.float32)
        self.content_features = torch.tensor(content_features, dtype=torch.float32)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_idx[idx], self.movie_idx[idx], self.content_features[idx], self.ratings[idx]

# Mô hình NCF
class NCF(nn.Module):
    def __init__(self, num_users, num_movies, content_dim, embedding_dim, hidden_dims):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.content_fc = nn.Linear(content_dim, embedding_dim)
        
        layers = []
        input_dim = embedding_dim * 3
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, user_idx, movie_idx, content_features):
        user_emb = self.user_embedding(user_idx)
        movie_emb = self.movie_embedding(movie_idx)
        content_emb = self.content_fc(content_features)
        x = torch.cat([user_emb, movie_emb, content_emb], dim=1)
        rating = self.mlp(x) * 5.0
        return rating

# Hàm huấn luyện
def train_ncf(model, train_loader, criterion, optimizer, device, pbar, epoch, total_epochs):
    model.train()
    total_loss = 0
    for user_idx, movie_idx, content_features, ratings in train_loader:
        user_idx, movie_idx, content_features, ratings = (
            user_idx.to(device), movie_idx.to(device), content_features.to(device), ratings.to(device)
        )
        optimizer.zero_grad()
        outputs = model(user_idx, movie_idx, content_features).squeeze()
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(ratings)
        pbar.set_description(f"Epoch {epoch+1}/{total_epochs} - Train Loss: {loss.item():.4f}")
        pbar.update(1)
    return total_loss / len(train_loader.dataset)

# Hàm tính val loss
def compute_val_loss(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for user_idx, movie_idx, content_features, ratings in val_loader:
            user_idx, movie_idx, content_features, ratings = (
                user_idx.to(device), movie_idx.to(device), content_features.to(device), ratings.to(device)
            )
            outputs = model(user_idx, movie_idx, content_features).squeeze()
            loss = criterion(outputs, ratings)
            total_loss += loss.item() * len(ratings)
            total_samples += len(ratings)
    return total_loss / total_samples if total_samples > 0 else float('inf')

# Hàm đánh giá trên tập validation
def evaluate_ncf_validation(model, val_df, user_id_map, movie_id_map, movie_content_dict, device):
    model.eval()
    predictions = predict_ratings(model, val_df, model_type='ncf', user_id_map=user_id_map, 
                                movie_id_map=movie_id_map, movie_content_dict=movie_content_dict)
    val_df = val_df.copy()
    val_df['pred_rating'] = predictions
    precision_list, recall_list, ndcg_list, map_list = [], [], [], []
    
    for user_id in val_df['userId'].unique():
        user_data = val_df[val_df['userId'] == user_id]
        if len(user_data) < 5:
            continue
        relevant_movies = user_data[user_data['rating'] >= 3.5]['movieId'].tolist()
        if not relevant_movies:
            continue
        
        user_predictions = user_data.sort_values('pred_rating', ascending=False).head(10)
        recommended_movies = user_predictions['movieId'].tolist()
        predicted_scores = user_predictions['pred_rating'].tolist()
        
        y_true, y_pred = prepare_user_metrics(relevant_movies, recommended_movies, val_df, user_id, predicted_scores)
        precision_list.append(precision_score(y_true, y_pred, zero_division=0))
        recall_list.append(recall_score(y_true, y_pred, zero_division=0))
        ndcg_list.append(compute_ndcg(relevant_movies, recommended_movies, val_df, user_id))
        map_list.append(compute_map(relevant_movies, recommended_movies, val_df, user_id))
    
    return {
        'Precision@10': np.mean(precision_list) if precision_list else 0.0,
        'Recall@10': np.mean(recall_list) if recall_list else 0.0,
        'NDCG@10': np.mean(ndcg_list) if ndcg_list else 0.0,
        'MAP@10': np.mean(map_list) if map_list else 0.0
    }

# Main
def main():
    os.makedirs(save_model_path, exist_ok=True)
    os.makedirs(results_model_path, exist_ok=True)
    
    # Load dữ liệu
    train_df = pd.read_csv(os.path.join(data_path, "train_dl.csv"))
    if train_df.empty or not all(col in train_df for col in ['user_idx', 'movie_idx', 'rating', 'movieId']):
        raise ValueError("Dữ liệu train không hợp lệ!")
    
    # Chia train/validation
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42, stratify=train_df['userId'])
    print(f"Kích thước tập train: {len(train_df)}, tập validation: {len(val_df)}")
    
    with open(os.path.join(save_model_path, "user_id_map.pkl"), "rb") as f:
        user_id_map = pickle.load(f)
    with open(os.path.join(save_model_path, "movie_id_map.pkl"), "rb") as f:
        movie_id_map = pickle.load(f)
    with open(os.path.join(save_model_path, "movie_content_dict.pkl"), "rb") as f:
        movie_content_dict = pickle.load(f)
    
    num_users = len(user_id_map)
    num_movies = len(movie_id_map)
    
    # Dataset và DataLoader
    train_dataset = MovieRatingDataset(train_df, movie_content_dict)
    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True,
        num_workers=Config.num_workers, pin_memory=True
    )
    val_dataset = MovieRatingDataset(val_df, movie_content_dict)
    val_loader = DataLoader(
        val_dataset, batch_size=Config.batch_size, shuffle=False,
        num_workers=Config.num_workers, pin_memory=True
    )
    
    # Khởi tạo mô hình
    model = NCF(
        num_users, num_movies, Config.content_dim, Config.embedding_dim, Config.hidden_dims
    ).to(Config.device)
    start_time = time()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    # Huấn luyện
    print("Huấn luyện NCF...")
    best_score = float('-inf')
    patience_counter = 0
    total_batches = Config.epochs * len(train_loader)
    with tqdm(total=total_batches, desc="Huấn luyện", unit="batch") as pbar:
        for epoch in range(Config.epochs):
            epoch_start = time()
            train_loss = train_ncf(model, train_loader, criterion, optimizer, 
                                  Config.device, pbar, epoch, Config.epochs)
            val_loss = compute_val_loss(model, val_loader, criterion, Config.device)
            results = evaluate_ncf_validation(model, val_df, user_id_map, movie_id_map, movie_content_dict, Config.device)
            duration = time() - epoch_start
            score = (results['Precision@10'] + results['Recall@10'] + results['NDCG@10'] + results['MAP@10']) / 4
            pbar.write(f"Epoch {epoch+1}/{Config.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Validation Metrics: {results} - Time: {duration:.2f}s")
            
            if score > best_score + Config.min_delta:
                best_score = score
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(save_model_path, "best_ncf.pth"))
            else:
                patience_counter += 1
                if patience_counter >= Config.patience:
                    print(f"Early stopping tại epoch {epoch+1}")
                    break
    
    # Lưu kết quả cuối cùng
    final_results = evaluate_ncf_validation(model, val_df, user_id_map, movie_id_map, movie_content_dict, Config.device)
    final_val_loss = compute_val_loss(model, val_loader, criterion, Config.device)
    result_file = os.path.join(results_model_path, "result_train_val_ncf.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("Kết quả huấn luyện NCF:\n")
        f.write(f"Train Loss cuối cùng: {train_loss:.4f}\n")
        f.write(f"Val Loss cuối cùng: {final_val_loss:.4f}\n")
        f.write("Validation Metrics:\n")
        for metric, value in final_results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"\nKết quả trên tập validation:")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {final_val_loss:.4f}")
    for metric, value in final_results.items():
        print(f"{metric}: {value:.4f}")
    print(f"Kết quả lưu tại: {result_file}")

    total_time = time() - start_time
    print(f"\nTổng thời gian: {total_time//60:.0f}ph {total_time%60:.2f}s")
    print(f"Score tốt nhất (Precision@10 + Recall@10 + NDCG@10 + MAP@10)/4: {best_score:.4f}")
    print(f"Model lưu tại: {os.path.join(save_model_path, 'best_ncf.pth')}")

if __name__ == "__main__":
    main()