import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# Thiết lập
tqdm.pandas()
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# Đường dẫn dữ liệu
data_path = "./data/"
os.makedirs(data_path, exist_ok=True)
ratings = pd.read_csv(os.path.join(data_path, "ratings.csv"))
movies = pd.read_csv(os.path.join(data_path, "movies.csv"))
tags = pd.read_csv(os.path.join(data_path, "tags.csv"))

# Hàm xử lý văn bản
def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\b(?:{})\b".format("|".join(stop_words)), "", text)
    return re.sub(r"[^a-z0-9\s]", " ", text).strip()

def limit_tag_length(text, max_words=200):
    return " ".join(text.split()[:max_words])

# Thông tin trước xử lý
missing_before = {
    "movies": movies.isna().sum(),
    "ratings": ratings.isna().sum(),
    "tags": tags.isna().sum(),
}
duplicates_before = {
    "movies": movies.duplicated().sum(),
    "ratings": ratings.duplicated().sum(),
    "tags": tags.duplicated().sum(),
}
num_users_before = ratings['userId'].nunique()
num_movies_before = ratings['movieId'].nunique()

# Làm sạch và gộp nội dung
movies["genres"] = movies["genres"].fillna("").apply(clean_text)
tags["tag"] = tags["tag"].apply(clean_text)
ratings.drop_duplicates(inplace=True)
movies.drop_duplicates(inplace=True)
tags.drop_duplicates(inplace=True)

# Tách năm và làm sạch title
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
movies["title"] = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
movies["year"] = movies["year"].fillna("no_year")

# Gộp tag cho mỗi movieId
tag_data = tags.groupby("movieId")["tag"].progress_apply(lambda x: " ".join(x)).reset_index()
movies = movies.merge(tag_data, on="movieId", how="left")
movies["tag"] = movies["tag"].fillna("no_tag").progress_apply(limit_tag_length)

# Kết hợp genres và tags thành content
movies["content"] = (movies["genres"] + " " + movies["tag"]).map(lambda x: " ".join(set(x.split())))

# Merge với ratings
movies = movies.merge(ratings[["userId", "movieId", "rating"]], on="movieId", how="left")
movies.dropna(subset=["userId"], inplace=True)
movies["userId"] = movies["userId"].astype(int)

# Thêm đặc trưng thống kê
movie_stats = movies.groupby("movieId").agg(
    rating_count=("rating", "count"),
    rating_mean=("rating", "mean"),
    rating_std=("rating", "std")
).reset_index()
movies = movies.merge(movie_stats, on="movieId", how="left")
movies["rating_std"] = movies["rating_std"].fillna(0)

# Giới hạn và lọc dữ liệu
max_ratings_per_movie = 250
min_ratings_per_user = 20
max_ratings_per_user = 250

# Lọc phim có ít nhất 10 lượt đánh giá
movie_counts = movies["movieId"].value_counts()
valid_movies = movie_counts[movie_counts >= 10].index
movies = movies[movies["movieId"].isin(valid_movies)]

# Giới hạn tối đa số lượt đánh giá mỗi phim
movie_groups = movies.groupby("movieId")
filtered_movies = []
for movie_id, group in movie_groups:
    sampled = group.sample(n=max_ratings_per_movie, random_state=42) if len(group) > max_ratings_per_movie else group
    filtered_movies.append(sampled)
movies = pd.concat(filtered_movies, ignore_index=True)

# Lọc người dùng có ít nhất min_ratings_per_user lượt đánh giá
user_counts = movies["userId"].value_counts()
valid_users = user_counts[user_counts >= min_ratings_per_user].index
movies = movies[movies["userId"].isin(valid_users)]

# Giới hạn số lượng đánh giá tối đa mỗi người dùng
user_groups = movies.groupby("userId")
filtered_users = []
for user_id, group in user_groups:
    sampled = group.sample(n=max_ratings_per_user, random_state=42) if len(group) > max_ratings_per_user else group
    filtered_users.append(sampled)
movies = pd.concat(filtered_users, ignore_index=True)

# Kiểm tra lại dữ liệu hợp lệ
user_counts = movies["userId"].value_counts()
valid_users = user_counts[user_counts >= min_ratings_per_user].index
movies = movies[movies["userId"].isin(valid_users)]

movie_counts = movies["movieId"].value_counts()
valid_movies = movie_counts[movie_counts >= 10].index
movies = movies[movies["movieId"].isin(valid_movies)]

# Đảm bảo mỗi userId có ít nhất 2 đánh giá để phân tầng
user_counts = movies["userId"].value_counts()
valid_users = user_counts[user_counts >= 2].index
movies = movies[movies["userId"].isin(valid_users)]

# Thống kê sau xử lý
ratings_per_movie = movies.groupby("movieId")["userId"].nunique()
ratings_per_user = movies.groupby("userId")["movieId"].nunique()
user_count_after = movies['userId'].nunique()
movie_count_after = movies['movieId'].nunique()

# Ghi thông tin thống kê
results_model_path = "./results_model/"
os.makedirs(results_model_path, exist_ok=True)
report_path = os.path.join(results_model_path, "result_preprocess_ml.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("### Missing Values Before Processing:\n")
    f.write(str(missing_before) + "\n\n")
    f.write("### Duplicates Before Processing:\n")
    f.write(str(duplicates_before) + "\n\n")
    f.write(f"### Number of Users Before Processing: {num_users_before}\n")
    f.write(f"### Number of Movies Before Processing: {num_movies_before}\n\n")
    f.write(f"### Number of Users After Processing: {user_count_after}\n")
    f.write(f"### Number of Movies After Processing: {movie_count_after}\n\n")
    f.write("### Rating Distribution Per Movie After Processing:\n")
    f.write(str(ratings_per_movie.describe()) + "\n\n")
    f.write("### Rating Distribution Per User After Processing:\n")
    f.write(str(ratings_per_user.describe()) + "\n\n")

# Xuất file CSV
output_cols = ["userId", "movieId", "title", "year", "rating", "genres", "tag", "content", "rating_count", "rating_mean", "rating_std"]
processed_path = os.path.join(data_path, "processed_movies.csv")
movies[output_cols].to_csv(processed_path, index=False, encoding="utf-8")

# Kiểm tra file CSV
processed_movies = pd.read_csv(processed_path)
missing_after = processed_movies.isna().sum()
duplicates_after = processed_movies.duplicated().sum()

# Ghi báo cáo
with open(report_path, "a", encoding="utf-8") as f:
    f.write("\n### Missing Values After Processing (processed_movies.csv):\n")
    f.write(str(missing_after) + "\n\n")
    f.write("### Duplicates After Processing (processed_movies.csv):\n")
    f.write(f"{duplicates_after}\n")

print("File đã lưu tại:", processed_path)
print("Thống kê ghi tại:", report_path)
