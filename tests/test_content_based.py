import pickle
import numpy as np
import pandas as pd
import faiss  # Thư viện tăng tốc tìm kiếm gần nhất

# Load dữ liệu đã lưu
with open("save_model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("save_model/truncated_svd.pkl", "rb") as f:
    svd = pickle.load(f)

tfidf_reduced = np.load("save_model/tfidf_reduced.npy")

# Load thông tin phim
df_movies = pd.read_csv("data/processed_movies.csv")

# Chuẩn hóa tiêu đề phim
df_movies['title'] = df_movies['title'].str.lower().str.strip()

# Xây dựng chỉ mục FAISS để tìm kiếm nhanh
vector_dim = tfidf_reduced.shape[1]
index = faiss.IndexFlatL2(vector_dim)
index.add(tfidf_reduced.astype(np.float32))

def recommend_movies(movie_title, top_n=10):
    """Gợi ý phim dựa trên nội dung"""
    movie_title = movie_title.lower().strip()  # Chuẩn hóa tiêu đề tìm kiếm
    
    if movie_title not in df_movies['title'].values:
        print(f"❌ Không tìm thấy phim: {movie_title}")
        return []
    
    movie_idx = df_movies[df_movies['title'] == movie_title].index[0]
    movie_vector = tfidf_reduced[movie_idx].reshape(1, -1).astype(np.float32)
    
    # Sử dụng FAISS để tìm phim tương tự
    _, similar_indices = index.search(movie_vector, top_n + 5)  # Lấy thêm để loại trùng lặp
    similar_indices = similar_indices[0][1:]  # Loại bỏ phim gốc
    
    # Lấy danh sách phim gợi ý và loại bỏ trùng lặp
    recommended_movies = df_movies.iloc[similar_indices][['title']].drop_duplicates().head(top_n)
    
    return recommended_movies

# Gợi ý phim tương tự "Man of the House (1995)"
recommended = recommend_movies("major payne 1995")
print("🎥 Gợi ý phim tương tự major payne 1995:")
print(recommended)
