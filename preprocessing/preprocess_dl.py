import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle

# Thiết lập đường dẫn
data_path = "./data/"
save_model_path = "./save_model/"
results_model_path = "./results_model/"
os.makedirs(data_path, exist_ok=True)
os.makedirs(save_model_path, exist_ok=True)
os.makedirs(results_model_path, exist_ok=True)

# Đọc dữ liệu
df = pd.read_csv(os.path.join(data_path, "processed_movies.csv"))

# Kiểm tra dữ liệu
missing_values = df.isna().sum()
print(f"Missing values:\n{missing_values}")
df = df.dropna(subset=['userId', 'movieId', 'rating', 'content'])

# Kiểm tra outlier ratings
df = df[(df['rating'] >= 0) & (df['rating'] <= 5)]
print(f"Số bản ghi sau khi loại outlier ratings: {len(df)}")

# Chuẩn hóa rating
df['rating'] = df['rating'].clip(0, 5)

# Chuyển đổi userId và movieId thành chỉ số liên tục
user_ids = df['userId'].unique()
movie_ids = df['movieId'].unique()
user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
movie_id_map = {mid: idx for idx, mid in enumerate(movie_ids)}
df['user_idx'] = df['userId'].map(user_id_map)
df['movie_idx'] = df['movieId'].map(movie_id_map)

# Tạo đặc trưng nội dung phim bằng TF-IDF
print("Đang tạo đặc trưng nội dung phim...")
tfidf = TfidfVectorizer(stop_words='english', max_features=300)
content_features = tfidf.fit_transform(df['content']).toarray()
scaler = StandardScaler()
content_features = scaler.fit_transform(content_features)

# Thêm đặc trưng thống kê
stat_features = df[['rating_count', 'rating_mean', 'rating_std']].values
content_features = np.hstack([content_features, scaler.fit_transform(stat_features)])

# Lưu TF-IDF và Scaler
with open(os.path.join(save_model_path, "tfidf_vectorizer_dl.pkl"), "wb") as f:
    pickle.dump(tfidf, f)
with open(os.path.join(save_model_path, "scaler_dl.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Gán đặc trưng nội dung cho từng movieId
movie_content_dict = {mid: content_features[df[df['movieId'] == mid]['movie_idx'].iloc[0]] for mid in movie_ids}

# Chia dữ liệu train/test (80-20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['userId'])

# Lưu train/test
train_df.to_csv(os.path.join(data_path, "train_dl.csv"), index=False, encoding="utf-8")
test_df.to_csv(os.path.join(data_path, "test_common.csv"), index=False, encoding="utf-8")

# Lưu mapping
with open(os.path.join(save_model_path, "user_id_map.pkl"), "wb") as f:
    pickle.dump(user_id_map, f)
with open(os.path.join(save_model_path, "movie_id_map.pkl"), "wb") as f:
    pickle.dump(movie_id_map, f)
with open(os.path.join(save_model_path, "movie_content_dict.pkl"), "wb") as f:
    pickle.dump(movie_content_dict, f)

# Thống kê
with open(os.path.join(results_model_path, "preprocess_dl_stats.txt"), "w", encoding="utf-8") as f:
    f.write(f"Number of users: {len(user_ids)}\n")
    f.write(f"Number of movies: {len(movie_ids)}\n")
    f.write(f"Train set size: {len(train_df)}\n")
    f.write(f"Test set size: {len(test_df)}\n")
    f.write(f"Content feature dimension: {content_features.shape[1]}\n")

print("Đã lưu train/test set và các file hỗ trợ.")
