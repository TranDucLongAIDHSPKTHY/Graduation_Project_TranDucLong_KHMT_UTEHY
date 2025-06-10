import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import requests
import time
import logging
from pathlib import Path
import stat
import getpass
import platform
from googleapiclient.discovery import build
import html

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s - User: %(user)s')

def get_current_user():
    """Get the current system user for logging."""
    try:
        return getpass.getuser()
    except:
        return "unknown"

logging.getLogger().handlers[0].setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - User: ' + get_current_user())
)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for subdir in ['models', 'app']:
    sys.path.append(os.path.join(base_dir, subdir))

from SQLite import get_db_connection, register_user, validate_login, save_user_rating, save_user_preferences, get_user_history, get_user_rating_count, debug_users_table
from ncf import Config, NCF

data_path = "./data/"
save_model_path = "./save_model/"
cache_path = os.path.join(data_path, "poster_cache.pkl")
TMDB_API_KEY = "3d9519a950e1009b05ec70646db7ea51"
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', 'AIzaSyBsaDP2tBMwyogq5DPdKLD_J7PCIDcxY6w')
PLACEHOLDER_URL = "https://via.placeholder.com/150"

# Global CSS
global_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #FFFFFF;
        color: #000000;
        margin: 0;
        width: 100vw;
        overflow-x: hidden;
    }
    .stApp {
        width: 100%;
        margin: 0;
        padding: 20px;
        box-sizing: border-box;
    }
    .header {
        text-align: center;
        padding: 10px 0;
        margin-bottom: 20px;
        border-bottom: 1px solid #E0E0E0;
        width: 100%;
    }
    .header h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 28px;
        font-weight: 600;
        letter-spacing: 1px;
        margin: 0;
    }
    .footer {
        text-align: center;
        padding: 10px 0;
        color: #666666;
        font-size: 12px;
        margin-top: 30px;
        border-top: 1px solid #E0E0E0;
        width: 100%;
    }
    .sidebar .sidebar-content {
        background-color: #FFFFFF;
        padding: 10px;
        width: 100%;
    }
    .sidebar .stSelectbox, .sidebar .stButton button {
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        padding: 8px;
        transition: all 0.2s ease;
        width: 100%;
    }
    .sidebar .stSelectbox:hover, .sidebar .stButton button:hover {
        border-color: #000000;
        transform: scale(1.02);
    }
    .sidebar .stButton button {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #000000;
        width: 100%;
        font-weight: 500;
    }
    .sidebar .stButton button:hover {
        background-color: #000000;
        color: #FFFFFF;
    }
    .movie-container {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        height: 300px;
        padding: 10px;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        background-color: #FFFFFF;
        transition: all 0.2s ease;
        width: 100%;
        position: relative;
        overflow: hidden;
        pointer-events: auto;
        z-index: 1;
    }
    .movie-container:hover .movie-details {
        display: flex;
    }
    .movie-poster {
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-radius: 4px;
        z-index: 2;
    }
    .movie-title {
        font-size: 14px;
        font-weight: 500;
        margin: 8px 0;
        min-height: 32px;
        overflow: hidden;
        text-overflow: ellipsis;
        text-align: center;
        cursor: pointer;
        z-index: 2;
    }
    .movie-details {
        display: none;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background-color: rgba(255, 255, 255, 0.95);
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        padding: 10px;
        border-radius: 4px;
        transition: all 0.2s ease;
        z-index: 3;
        pointer-events: auto;
    }
    .movie-details p {
        font-size: 12px;
        color: #333333;
        margin: 4px 0;
        text-align: center;
    }
    .action-button {
        width: 100%;
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #000000;
        padding: 8px;
        border-radius: 4px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-top: 4px;
    }
    .action-button:hover {
        background-color: #000000;
        color: #FFFFFF;
        transform: translateY(-2px);
    }
    .refresh-button {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #000000;
        padding: 8px 16px;
        border-radius: 4px;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .refresh-button:hover {
        background-color: #000000;
        color: #FFFFFF;
        transform: translateY(-2px);
    }
    .section-title {
        font-size: 18px;
        font-weight: 500;
        color: #000000;
        margin-bottom: 15px;
    }
    .form-container {
        background-color: #FFFFFF;
        padding: 15px;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        max-width: 600px;
        margin: 0 auto;
    }
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        padding: 8px;
        transition: border-color 0.2s ease;
        width: 100%;
    }
    .stTextInput input:hover, .stTextArea textarea:hover, .stSelectbox select:hover {
        border-color: #000000;
    }
    .stSlider > div > div > div > div {
        background: #000000;
    }
    .stButton button {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #000000;
        border-radius: 4px;
        padding: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #000000;
        color: #FFFFFF;
        transform: translateY(-2px);
    }
    .movie-container .stButton > button {
        width: 100%;
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #000000;
        padding: 8px;
        border-radius: 4px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .movie-container .stButton > button:hover {
        background-color: #000000;
        color: #FFFFFF;
        transform: translateY(-2px);
    }
    .st-emotion-cache-1r4snee {
        flex: 1 1 0;
        width: 100%;
    }
    .st-emotion-cache-1wmy9hl {
        width: 100%;
    }
    /* Hide Streamlit's loading overlay */
    div[data-testid="stAppViewBlockContainer"] + div {
        display: none !important;
    }
    @media (max-width: 768px) {
        .stApp {
            padding: 10px;
        }
        .movie-container {
            height: 300px;
            margin: 8px;
        }
        .movie-poster {
            height: 160px;
        }
        .header h1 {
            font-size: 20px;
        }
        .section-title {
            font-size: 16px;
        }
        .form-container {
            max-width: 100%;
            padding: 10px;
        }
        .st-emotion-cache-1r4snee {
            flex: 1 1 100%;
            width: 100%;
        }
        .movie-details p {
            font-size: 10px;
        }
        .action-button {
            font-size: 10px;
            padding: 6px;
        }
    }
</style>
"""

def log_file_status(path):
    """Log detailed file/directory status."""
    try:
        stat_info = os.stat(path)
        permissions = oct(stat_info.st_mode & 0o777)[2:]
        owner = stat_info.st_uid
        group = stat_info.st_gid
        writable = os.access(path, os.W_OK)
        logging.debug(f"Status for {path}: Permissions={permissions}, Owner={owner}, Group={group}, Writable={writable}")
        return writable
    except Exception as e:
        logging.error(f"Cannot get status for {path}: {str(e)}")
        return False

def ensure_directory_permissions(directory):
    """Ensure directory exists and has write permissions."""
    try:
        Path(directory).mkdir(exist_ok=True, parents=True)
        if platform.system() != "Windows":
            try:
                os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
            except PermissionError as e:
                logging.warning(f"Cannot set permissions for {directory}: {str(e)}")
        if not os.access(directory, os.W_OK):
            logging.error(f"No write permission for directory: {directory}")
            return False
        log_file_status(directory)
        return True
    except Exception as e:
        logging.error(f"Failed to set up directory {directory}: {str(e)}")
        return False

def ensure_file_writable(filepath):
    """Ensure a file is writable, creating it if it doesn't exist."""
    try:
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        if not os.path.exists(filepath):
            with open(filepath, 'a'):
                os.utime(filepath, None)
        if platform.system() != "Windows":
            try:
                os.chmod(filepath, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
            except PermissionError as e:
                logging.warning(f"Cannot set permissions for {filepath}: {str(e)}")
        if not os.access(filepath, os.W_OK):
            logging.error(f"No write permission for file: {filepath}")
            return False
        log_file_status(filepath)
        return True
    except Exception as e:
        logging.error(f"Failed to ensure file {filepath} is writable: {str(e)}")
        return False

for d in [data_path, save_model_path]:
    if not ensure_directory_permissions(d):
        st.error(f"Không thể tạo hoặc truy cập thư mục {d}. Vui lòng kiểm tra quyền truy cập.")
        st.markdown("Hướng dẫn sửa lỗi:")
        st.markdown(f"- Chạy lệnh: `chmod -R 775 {d}`")
        st.markdown(f"- Đảm bảo quyền ghi: `ls -ld {d}`")
        st.markdown(f"- Đổi quyền sở hữu nếu cần: `chown -R $(whoami) {d}`")
        st.markdown("- Kiểm tra dung lượng đĩa: `df -h`")
        st.stop()

if not ensure_file_writable(cache_path):
    st.warning(f"Không thể ghi vào {cache_path}. Poster cache có thể không hoạt động.")

def load_poster_cache():
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Failed to load poster_cache.pkl, recreating: {str(e)}")
            return {}
    return {}

def save_poster_cache(cache):
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        if platform.system() != "Windows":
            try:
                os.chmod(cache_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
            except PermissionError as e:
                logging.warning(f"Cannot set permissions for {cache_path}: {str(e)}")
    except Exception as e:
        logging.error(f"Failed to save poster cache: {str(e)}")

@st.cache_data
def get_movie_poster(title, year, retries=3):
    cache = load_poster_cache()
    cache_key = f"{title}_{year}"
    
    if cache_key in cache:
        return cache[cache_key]
    
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}&year={year}"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data['results']:
                poster_path = data['results'][0].get('poster_path')
                if poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                    cache[cache_key] = poster_url
                    save_poster_cache(cache)
                    return poster_url
            cache[cache_key] = None
            save_poster_cache(cache)
            return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                logging.warning(f"Failed to fetch poster for {title} ({year}): {str(e)}")
                st.warning(f"Không thể lấy poster cho {title} ({year}): {str(e)}")
                cache[cache_key] = None
                save_poster_cache(cache)
                return None

def get_movie_trailer(title, year, retries=3):
    """Lấy URL trailer phim từ YouTube bằng YouTube Data API với cache."""
    cache = load_poster_cache()
    cache_key = f"trailer_{title}_{year}"
    
    if cache_key in cache:
        return cache[cache_key]
    
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    query = f"{title} {year} official trailer"
    
    for attempt in range(retries):
        try:
            request = youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=1,
                videoDuration="short"
            )
            response = request.execute()
            
            if response['items']:
                video_id = response['items'][0]['id']['videoId']
                trailer_url = f"https://www.youtube.com/watch?v={video_id}"
                cache[cache_key] = trailer_url
                save_poster_cache(cache)
                logging.debug(f"Fetched trailer for {title} ({year}): {trailer_url}")
                return trailer_url
            else:
                cache[cache_key] = f"https://www.youtube.com/results?search_query={title}+{year}+trailer"
                save_poster_cache(cache)
                logging.warning(f"No trailer found for {title} ({year})")
                return cache[cache_key]
                
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                logging.error(f"Failed to fetch trailer for {title} ({year}): {str(e)}")
                st.warning(f"Không thể lấy trailer cho {title} ({year}): {str(e)}")
                cache[cache_key] = f"https://www.youtube.com/results?search_query={title}+{year}+trailer"
                save_poster_cache(cache)
                return cache[cache_key]

try:
    df_movies = pd.read_csv(os.path.join(data_path, "processed_movies.csv"))
    df_movies = df_movies[['movieId', 'title', 'year', 'rating', 'genres', 'content']].dropna(subset=['content'])
except FileNotFoundError:
    st.error("Không tìm thấy file processed_movies.csv trong thư mục data.")
    logging.error("processed_movies.csv not found")
    st.stop()

popular_movies = df_movies.sort_values(by='rating', ascending=False).head(50)[['movieId', 'title', 'year', 'rating', 'genres']].to_dict('records')

@st.cache_resource
def load_cbf_components():
    try:
        with open(os.path.join(save_model_path, "tfidf_vectorizer_cb.pkl"), "rb") as f:
            tfidf = pickle.load(f)
        with open(os.path.join(save_model_path, "svd_cb.pkl"), "rb") as f:
            svd = pickle.load(f)
        with open(os.path.join(save_model_path, "knn_cb.pkl"), "rb") as f:
            knn = pickle.load(f)
        with open(os.path.join(save_model_path, "movie_ids_cb.pkl"), "rb") as f:
            movie_ids = pickle.load(f)
        return tfidf, svd, knn, movie_ids
    except FileNotFoundError as e:
        logging.error(f"Content-Based model files not found: {str(e)}")
        st.error(f"Không tìm thấy file mô hình Content-Based: {str(e)}")
        return None, None, None, None
    except Exception as e:
        logging.error(f"Error loading Content-Based model: {str(e)}")
        st.error(f"Lỗi khi tải mô hình Content-Based: {str(e)}")
        return None, None, None, None

@st.cache_resource
def load_ncf_model():
    try:
        with open(os.path.join(save_model_path, "user_id_map.pkl"), "rb") as f:
            user_id_map = pickle.load(f)
        with open(os.path.join(save_model_path, "movie_id_map.pkl"), "rb") as f:
            movie_id_map = pickle.load(f)
        with open(os.path.join(save_model_path, "movie_content_dict.pkl"), "rb") as f:
            movie_content_dict = pickle.load(f)
        
        if not all([user_id_map, movie_id_map, movie_content_dict]):
            raise ValueError("Dữ liệu ánh xạ hoặc đặc trưng nội dung không hợp lệ.")
        
        num_users = len(user_id_map)
        num_movies = len(movie_id_map)
        
        model = NCF(
            num_users=num_users,
            num_movies=num_movies,
            content_dim=Config.content_dim,
            embedding_dim=Config.embedding_dim,
            hidden_dims=Config.hidden_dims
        ).to(Config.device)
        
        model.load_state_dict(torch.load(os.path.join(save_model_path, "best_ncf.pth"), map_location=Config.device))
        model.eval()
        return model, user_id_map, movie_id_map, movie_content_dict
    except FileNotFoundError as e:
        logging.error(f"NCF model files not found: {str(e)}")
        st.error(f"Không tìm thấy file mô hình NCF: {str(e)}")
        return None, None, None, None
    except Exception as e:
        logging.error(f"Error loading NCF model: {str(e)}")
        st.error(f"Lỗi khi tải mô hình NCF: {str(e)}")
        return None, None, None, None

def recommend_cbf(movie_title):
    try:
        tfidf, svd, knn, movie_ids = load_cbf_components()
        if any(x is None for x in [tfidf, svd, knn, movie_ids]):
            return []
        
        movie_idx = df_movies[df_movies['title'] == movie_title].index
        if len(movie_idx) == 0:
            st.warning(f"Không tìm thấy phim '{movie_title}' trong dữ liệu.")
            return []
        
        movie_idx = movie_idx[0]
        content = df_movies.iloc[movie_idx]['content']
        tfidf_matrix = tfidf.transform([content])
        reduced_features = svd.transform(tfidf_matrix)
        
        distances, indices = knn.kneighbors(reduced_features, n_neighbors=101)
        sim_indices = indices[0][1:]
        recommendations = df_movies.iloc[sim_indices][['movieId', 'title', 'year', 'rating', 'genres', 'content']].to_dict('records')
        
        seen_titles = set()
        unique_recommendations = []
        for rec in recommendations:
            title = rec['title']
            if title != movie_title and title not in seen_titles:
                seen_titles.add(title)
                rec['poster_url'] = get_movie_poster(title, rec['year'])
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]
    except Exception as e:
        logging.error(f"Error in CBF recommendation: {str(e)}")
        st.error(f"Lỗi khi tạo gợi ý CBF: {str(e)}")
        return []

def recommend_movies(user_id):
    try:
        model, user_id_map, movie_id_map, movie_content_dict = load_ncf_model()
        if model is None:
            return None
        if user_id not in user_id_map:
            st.warning(f"UserID {user_id} không tồn tại trong hệ thống NCF. Vui lòng nhập UserID khác.")
            return None
        
        user_idx = user_id_map[user_id]
        movie_ids = list(movie_id_map.keys())
        movie_indices = [movie_id_map[mid] for mid in movie_ids]
        content_features = np.array([movie_content_dict[mid] for mid in movie_ids], dtype=np.float32)
        
        user_tensor = torch.tensor([user_idx] * len(movie_ids), dtype=torch.long).to(Config.device)
        movie_tensor = torch.tensor(movie_indices, dtype=torch.long).to(Config.device)
        content_tensor = torch.tensor(content_features, dtype=torch.float32).to(Config.device)
        
        with torch.no_grad():
            predictions = model(user_tensor, movie_tensor, content_tensor).squeeze().cpu().numpy()
        
        top_indices = np.argsort(predictions)[::-1][:30]
        top_movie_ids = [movie_ids[i] for i in top_indices]
        
        recommendations_df = df_movies[df_movies['movieId'].isin(top_movie_ids)][['movieId', 'title', 'year', 'rating', 'genres', 'content']]
        
        recommendations_df = recommendations_df.sort_values(by=['title', 'rating'], ascending=[True, False])
        recommendations_df = recommendations_df.drop_duplicates(subset=['title'], keep='first')
        
        recommendations = recommendations_df[['movieId', 'title', 'year', 'rating', 'genres', 'content']].to_dict('records')
        
        filtered_recommendations = []
        for rec in recommendations:
            poster_url = get_movie_poster(rec['title'], rec['year'])
            if poster_url:
                rec['poster_url'] = poster_url
                filtered_recommendations.append(rec)
        
        filtered_recommendations = filtered_recommendations[:10]
        
        return filtered_recommendations if filtered_recommendations else None
    except Exception as e:
        logging.error(f"Error in NCF recommendation: {str(e)}")
        st.error(f"Lỗi khi tạo gợi ý NCF: {str(e)}")
        return None

def recommend_by_genres(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.execute('SELECT genres FROM preferences WHERE user_id = ?', (user_id,))
        prefs = cursor.fetchone()
        conn.close()
        
        if not prefs or not prefs[0]:
            filtered_popular = []
            for rec in popular_movies[:15]:
                rec = rec.copy()
                poster_url = get_movie_poster(rec['title'], rec['year'])
                if poster_url:
                    rec['poster_url'] = poster_url
                    filtered_popular.append(rec)
            return filtered_popular[:10]
        
        genres = prefs[0].split('|')
        recommended_movies = df_movies[df_movies['genres'].apply(lambda x: any(genre in x for genre in genres))]
        recommended_movies = recommended_movies.sort_values(by='rating', ascending=False).head(15)
        recommendations = recommended_movies[['movieId', 'title', 'year', 'rating', 'genres', 'content']].to_dict('records')
        
        filtered_recommendations = []
        for rec in recommendations:
            poster_url = get_movie_poster(rec['title'], rec['year'])
            if poster_url:
                rec['poster_url'] = poster_url
                filtered_recommendations.append(rec)
        
        return filtered_recommendations[:10]
    except Exception as e:
        logging.error(f"Error in genre-based recommendation: {str(e)}")
        st.error(f"Lỗi khi tạo gợi ý theo thể loại: {str(e)}")
        return []

def recommend_by_user_ratings(user_id):
    try:
        # Check if user has any rating history
        history = get_user_history(user_id)
        if not history:
            logging.info(f"No rating history for user {user_id}")
            return []
        
        # Check if user has liked movies (rating >= 3.5)
        liked_movies = [item['movie_id'] for item in history if item['rating'] >= 3.5]
        if not liked_movies:
            logging.info(f"No liked movies (rating >= 3.5) for user {user_id}")
            return []
        
        # Generate recommendations using NCF model
        recommendations = recommend_movies(user_id)
        if not recommendations:
            logging.warning(f"No recommendations generated for user {user_id}")
            return []
        
        # Filter unique recommendations with posters
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec['title'] not in seen and 'poster_url' in rec and rec['poster_url']:
                seen.add(rec['title'])
                unique_recs.append(rec)
        
        return unique_recs[:10]
    except Exception as e:
        logging.error(f"Error in user ratings recommendation: {str(e)}")
        st.error(f"Lỗi khi tạo gợi ý dựa trên đánh giá: {str(e)}")
        return []

def check_movie_id_alignment():
    try:
        df_processed = pd.read_csv(os.path.join(data_path, "processed_movies.csv"))
        preprocess_ids = set(df_processed['movieId'])
        
        train_missing = False
        try:
            df_train = pd.read_csv(os.path.join(data_path, "train_dl.csv"))
            train_ids = set(df_train['movieId'])
            if not train_ids.issubset(preprocess_ids):
                st.warning("Some movieIds in train_dl.csv are not in processed_movies.csv")
                train_missing = True
        except FileNotFoundError:
            st.warning("train_dl.csv not found in data directory")
            train_missing = True
        
        test_missing = False
        try:
            df_test = pd.read_csv(os.path.join(data_path, "test_common.csv"))
            test_ids = set(df_test['movieId'])
            if not test_ids.issubset(preprocess_ids):
                st.warning("Some movieIds in test_common.csv are not in processed_movies.csv")
                test_missing = True
        except FileNotFoundError:
            st.warning("test_common.csv not found in data directory")
            test_missing = True
        
        if not train_missing and not test_missing:
            st.info("Hệ thống đã sẵn sàng.")
    except FileNotFoundError:
        logging.error("processed_movies.csv not found")
        st.error("Không tìm thấy file processed_movies.csv trong thư mục data.")
    except Exception as e:
        logging.error(f"Error checking movieId alignment: {str(e)}")
        st.error(f"Lỗi khi kiểm tra movieId: {str(e)}")

def handle_login():
    st.markdown('<div class="section-title">Đăng nhập</div>', unsafe_allow_html=True)
    with st.container():
        username = st.text_input("Tên đăng nhập", placeholder="Nhập tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password", placeholder="Nhập mật khẩu")
        
        if st.button("Đăng nhập", type="primary", key="login_button"):
            user_id = validate_login(username, password)
            if user_id:
                st.session_state.auth.update({'user_id': user_id, 'username': username, 'new_user': get_user_rating_count(user_id) < 5})
                st.success("Đăng nhập thành công.")
                st.rerun()
            else:
                st.error("Tên đăng nhập hoặc mật khẩu không đúng.")
        st.markdown('<a href="#" onclick="st.session_state.choice=\'📝 Đăng ký\';st.rerun()">Chưa có tài khoản? Đăng ký</a>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def handle_register():
    st.markdown('<div class="section-title">Đăng ký</div>', unsafe_allow_html=True)
    with st.container():
        username = st.text_input("Tên đăng nhập", placeholder="3-20 ký tự, chỉ chữ cái, số, dấu gạch dưới")
        password = st.text_input("Mật khẩu", type="password", placeholder="Tối thiểu 6 ký tự")
        confirm_password = st.text_input("Xác nhận mật khẩu", type="password", placeholder="Nhập lại mật khẩu")
        
        if st.button("Đăng ký", type="primary", key="register_button"):
            if not username:
                st.error("Tên đăng nhập không được để trống.")
                return
            if len(password) < 6:
                st.error("Mật khẩu phải có ít nhất 6 ký tự.")
                return
            if password != confirm_password:
                st.error("Mật khẩu và xác nhận mật khẩu không khớp.")
                return
            existing_usernames = debug_users_table()
            if username.lower() in [u.lower() for u in existing_usernames]:
                st.error(f"Tên đăng nhập {username} đã tồn tại.")
                return
            user_id = register_user(username, password, confirm_password)
            if user_id:
                st.session_state.auth.update({'user_id': user_id, 'username': username, 'new_user': True})
                st.success(f"Đăng ký thành công. Tên đăng nhập: {username}, User ID: {user_id}. Vui lòng đánh giá ít nhất 5 phim để nhận đề xuất cá nhân.")
                st.rerun()
            else:
                st.error("Đăng ký thất bại. Vui lòng kiểm tra quyền truy cập thư mục `data` và `save_model`, hoặc xóa users.db.")
                st.markdown("Hướng dẫn sửa lỗi:")
                st.markdown("- Chạy lệnh: `chmod -R 775 data save_model`")
                st.markdown("- Đảm bảo quyền ghi: `ls -ld data save_model`")
                st.markdown("- Xóa file database: `rm data/users.db`")
                st.markdown("- Kiểm tra user_id_map.pkl: `rm save_model/user_id_map.pkl` nếu bị hỏng")
                st.markdown("- Đổi quyền sở hữu: `chown -R $(whoami) data save_model`")
                st.markdown("- Kiểm tra dung lượng đĩa: `df -h`")
                if platform.system() == "Windows":
                    st.markdown("- Windows: Đảm bảo chạy với quyền admin và cấp quyền 'Full control' cho thư mục `data`, `save_model`.")
                logging.error(f"Registration failed for username: {username}")
        st.markdown('<a href="#" onclick="st.session_state.choice=\'🔑 Đăng nhập\';st.rerun()">Đã có tài khoản? Đăng nhập</a>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def handle_home():
    st.markdown('<div class="section-title">Chào mừng ' + st.session_state.auth['username'] + '</div>', unsafe_allow_html=True)
    
    recs = recommend_movies(st.session_state.auth['user_id'])
    
    if recs is None or st.session_state.auth['new_user']:
        st.info("Vui lòng đánh giá ít nhất 5 phim để nhận đề xuất cá nhân.")
        
        st.markdown('<div class="section-title">Phim được đánh giá cao</div>', unsafe_allow_html=True)
        filtered_popular = []
        for movie in popular_movies:
            movie = movie.copy()
            poster_url = get_movie_poster(movie['title'], movie['year'])
            if poster_url:
                movie['poster_url'] = poster_url
                filtered_popular.append(movie)
        
        if not filtered_popular:
            st.warning("Không có phim nào có poster để hiển thị.")
            return
        
        for i in range(0, len(filtered_popular), 5):
            cols = st.columns(5)
            for j, movie in enumerate(filtered_popular[i:i+5]):
                with cols[j]:
                    with st.container():
                        title_escaped = movie['title'].replace("'", "\\'")
                        st.markdown(
                            f"""
                            <div class="movie-container" data-movie-id="{movie['movieId']}">
                                <img src="{movie['poster_url']}" class="movie-poster" alt="{title_escaped}"/>
                                <div class="movie-title">{title_escaped} ({movie['year']})</div>
                                <div class="movie-details">
                                    <p>Tên phim: {movie['title']}</p>
                                    <p>Năm: {movie['year']}</p>
                                    <p>Điểm: {movie['rating']}⭐</p>
                                    <p>Thể loại: {movie['genres']}</p>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        # Streamlit buttons for actions
                        if st.button("Đánh giá", key=f"rate_popular_{movie['movieId']}_{i}_{j}_{st.session_state.recommendation_key}"):
                            st.session_state.selected_movie = movie['movieId']
                            st.session_state.rating_mode = True
                            st.rerun()
                        if st.button("Xem trailer", key=f"trailer_popular_{movie['movieId']}_{i}_{j}_{st.session_state.recommendation_key}"):
                            trailer_url = get_movie_trailer(movie['title'], movie['year'])
                            # LongLong
                            st.markdown(f'<a href="{trailer_url}" target="_blank">Mở trailer</a>', unsafe_allow_html=True)
                            #st.markdown(f'<meta http-equiv="refresh" content="0;url={trailer_url}">', unsafe_allow_html=True)
            st.divider()
        
        if 'selected_movie' in st.session_state and st.session_state.rating_mode:
            movie = next(m for m in filtered_popular if m['movieId'] == st.session_state.selected_movie)
            st.markdown(f'<div class="section-title">Đánh giá bộ phim: {movie["title"]}</div>', unsafe_allow_html=True)
            with st.container():
                # st.image(movie['poster_url'], width=50 , use_container_width=True)
                rating = st.slider(
                    "Đánh giá (1-5)",
                    min_value=1.0,
                    max_value=5.0,
                    step=0.5,
                    value=3.0,
                    key=f"rating_{movie['movieId']}_{st.session_state.recommendation_key}"
                )
                comment = st.text_area(
                    "Bình luận (tối đa 200 từ)",
                    key=f"comment_{movie['movieId']}_{st.session_state.recommendation_key}",
                    height=80
                )
                if st.button(
                    "Lưu",
                    key=f"save_{movie['movieId']}_{st.session_state.recommendation_key}",
                    type="primary"
                ):
                    word_count = len(comment.split()) if comment else 0
                    if word_count > 200:
                        st.error("Bình luận không được vượt quá 200 từ.")
                    else:
                        success = save_user_rating(st.session_state.auth['user_id'], movie['movieId'], rating, comment)
                        if success:
                            st.session_state.auth['new_user'] = get_user_rating_count(st.session_state.auth['user_id']) < 5
                            del st.session_state.selected_movie
                            del st.session_state.rating_mode
                            st.session_state.recommendation_key += 1
                            st.success("Đã lưu đánh giá.")
                            st.rerun()
                        else:
                            st.error("Lỗi khi lưu đánh giá. Vui lòng thử lại.")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-title">Phim đề xuất</div>', unsafe_allow_html=True)
        if not recs:
            st.warning("Không có phim đề xuất nào có poster để hiển thị.")
            return
        
        if st.button("Làm mới", key="reset_recommendations", type="primary"):
            st.session_state.recommendation_key += 1
            st.rerun()
        
        for i in range(0, len(recs), 5):
            cols = st.columns(5)
            for j, movie in enumerate(recs[i:i+5]):
                with cols[j]:
                    with st.container():
                        title_escaped = movie['title'].replace("'", "\\'")
                        st.markdown(
                            f"""
                            <div class="movie-container" data-movie-id="{movie['movieId']}">
                                <img src="{movie['poster_url']}" class="movie-poster" alt="{title_escaped}"/>
                                <div class="movie-title">{title_escaped} ({movie['year']})</div>
                                <div class="movie-details">
                                    <p>Tên phim: {movie['title']}</p>
                                    <p>Năm: {movie['year']}</p>
                                    <p>Điểm: {movie['rating']}⭐</p>
                                    <p>Thể loại: {movie['genres']}</p>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.markdown("<div style='height: 9px;'></div>", unsafe_allow_html=True)
                        # Streamlit buttons for actions
                        if st.button("Đánh giá", key=f"rate_rec_{movie['movieId']}_{i}_{j}_{st.session_state.recommendation_key}"):
                            st.session_state.selected_movie = movie['movieId']
                            st.session_state.rating_mode = True
                            st.rerun()
                        if st.button("Xem trailer", key=f"trailer_rec_{movie['movieId']}_{i}_{j}_{st.session_state.recommendation_key}"):
                            trailer_url = get_movie_trailer(movie['title'], movie['year'])
                            # Long 
                            st.markdown(f'<a href="{trailer_url}" target="_blank">Mở trailer</a>', unsafe_allow_html=True)
                            #st.markdown(f'<meta http-equiv="refresh" content="0;url={trailer_url}">', unsafe_allow_html=True)
            st.divider()
        
        if 'selected_movie' in st.session_state and st.session_state.rating_mode:
            movie = next(m for m in recs if m['movieId'] == st.session_state.selected_movie)
            st.markdown(f'<div class="section-title">Đánh giá: {movie["title"]}</div>', unsafe_allow_html=True)
            with st.container():
                st.image(movie['poster_url'], width=50, use_container_width=True)
                rating = st.slider(
                    "Đánh giá (1-5)",
                    min_value=1.0,
                    max_value=5.0,
                    step=0.5,
                    value=3.0,
                    key=f"rating_rec_{movie['movieId']}_{st.session_state.recommendation_key}"
                )
                comment = st.text_area(
                    "Bình luận (tối đa 200 từ)",
                    key=f"comment_rec_{movie['movieId']}_{st.session_state.recommendation_key}",
                    height=80
                )
                if st.button(
                    "Lưu",
                    key=f"save_rec_{movie['movieId']}_{st.session_state.recommendation_key}",
                    type="primary"
                ):
                    word_count = len(comment.split()) if comment else 0
                    if word_count > 200:
                        st.error("Bình luận không được vượt quá 200 từ.")
                    else:
                        success = save_user_rating(st.session_state.auth['user_id'], movie['movieId'], rating, comment)
                        if success:
                            st.session_state.auth['new_user'] = get_user_rating_count(st.session_state.auth['user_id']) < 5
                            del st.session_state.selected_movie
                            del st.session_state.rating_mode
                            st.session_state.recommendation_key += 1
                            st.success("Đã lưu đánh giá.")
                            st.rerun()
                        else:
                            st.error("Lỗi khi lưu đánh giá. Vui lòng thử lại.")
                # st.markdown('</div>', unsafe_allow_html=True)

def handle_profile():
    st.markdown('<div class="section-title">Hồ sơ</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown(f"Tên người dùng: {st.session_state.auth['username']}")
        st.markdown(f"User ID: {st.session_state.auth['user_id']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    st.markdown('<div class="section-title">Thể loại yêu thích</div>', unsafe_allow_html=True)
    all_genres = sorted(set([genre for sublist in df_movies['genres'].str.split('|') for genre in sublist]))
    try:
        conn = get_db_connection()
        cursor = conn.execute('SELECT genres FROM preferences WHERE user_id = ?', (st.session_state.auth['user_id'],))
        current_prefs = cursor.fetchone()
        conn.close()
    except Exception as e:
        logging.error(f"Error fetching user preferences: {str(e)}")
        st.error(f"Lỗi khi lấy sở thích: {str(e)}")
        current_prefs = None
    
    with st.container():
        selected_genres = st.multiselect(
            "Chọn thể loại",
            options=all_genres,
            default=current_prefs[0].split('|') if current_prefs and current_prefs[0] else [],
            placeholder="Chọn thể loại yêu thích"
        )
        if st.button("Lưu", type="primary"):
            try:
                save_user_preferences(st.session_state.auth['user_id'], "|".join(selected_genres))
                st.success("Đã cập nhật sở thích.")
            except Exception as e:
                logging.error(f"Error saving preferences: {str(e)}")
                st.error(f"Lỗi khi lưu sở thích: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    st.markdown('<div class="section-title">Lịch sử đánh giá</div>', unsafe_allow_html=True)
    history = get_user_history(st.session_state.auth['user_id'])
    
    if history:
        filtered_history = []
        for item in history:
            movie_info = df_movies[df_movies['movieId'] == item['movie_id']].iloc[0]
            poster_url = get_movie_poster(movie_info['title'], movie_info['year'])
            if poster_url:
                filtered_history.append({
                    'movie_id': item['movie_id'],
                    'rating': item['rating'],
                    'title': movie_info['title'],
                    'year': movie_info['year'],
                    'genres': movie_info['genres'],
                    'poster_url': poster_url,
                    'content': movie_info['content']
                })
        
        if not filtered_history:
            st.info("Không có phim nào trong lịch sử đánh giá có poster để hiển thị.")
            return
        
        for i in range(0, len(filtered_history), 5):
            cols = st.columns(5)
            for j, item in enumerate(filtered_history[i:i+5]):
                with cols[j]:
                    with st.container():
                        title_escaped = item['title'].replace("'", "\\'")
                        st.markdown(
                            f"""
                            <div class="movie-container" data-movie-id="{item['movie_id']}">
                                <img src="{item['poster_url']}" class="movie-poster" alt="{title_escaped}"/>
                                <div class="movie-title">{title_escaped} ({item['year']})</div>
                                <div class="movie-details">
                                    <p>Tên phim: {item['title']}</p>
                                    <p>Năm: {item['year']}</p>
                                    <p>Điểm: {item['rating']}⭐</p>
                                    <p>Thể loại: {item['genres']}</p>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        # Streamlit button for action
                        if st.button("Xem trailer", key=f"trailer_history_{item['movie_id']}_{i}_{j}_{st.session_state.recommendation_key}"):
                            trailer_url = get_movie_trailer(item['title'], item['year'])
                            # Long
                            st.markdown(f'<a href="{trailer_url}" target="_blank">Mở trailer</a>', unsafe_allow_html=True)
                            #st.markdown(f'<meta http-equiv="refresh" content="0;url={trailer_url}">', unsafe_allow_html=True)
            st.divider()
    else:
        st.info("Chưa có lịch sử đánh giá.")

def handle_search():
    st.markdown('<div class="section-title">Tìm kiếm</div>', unsafe_allow_html=True)
    with st.container():
        search_term = st.text_input("Tìm kiếm", key="search_input", placeholder="Nhập tên phim hoặc thể loại")
        
        if search_term:
            try:
                search_results = df_movies[df_movies['title'].str.contains(search_term, case=False, na=False) | df_movies['genres'].str.contains(search_term, case=False, na=False)]
                if not search_results.empty:
                    search_results = search_results.sort_values(by='rating', ascending=False)
                    search_results = search_results.drop_duplicates(subset=['title'], keep='first')
                    
                    st.markdown(f'<div class="section-title">Kết quả: {len(search_results)} phim</div>', unsafe_allow_html=True)
                    filtered_results = []
                    for movie in search_results.itertuples():
                        poster_url = get_movie_poster(movie.title, movie.year)
                        if poster_url:
                            filtered_results.append({
                                'movieId': movie.movieId,
                                'title': movie.title,
                                'year': movie.year,
                                'rating': movie.rating,
                                'genres': movie.genres,
                                'content': movie.content if hasattr(movie, 'content') else 'Chưa có mô tả',
                                'poster_url': poster_url
                            })
                    
                    if not filtered_results:
                        st.warning("Không tìm thấy phim nào có poster phù hợp.")
                        return
                    
                    for i in range(0, len(filtered_results), 5):
                        cols = st.columns(5)
                        for j, movie in enumerate(filtered_results[i:i+5]):
                            with cols[j]:
                                with st.container():
                                    title_escaped = movie['title'].replace("'", "\\'")
                                    st.markdown(
                                        f"""
                                        <div class="movie-container" data-movie-id="{movie['movieId']}">
                                            <img src="{movie['poster_url']}" class="movie-poster" alt="{title_escaped}"/>
                                            <div class="movie-title">{title_escaped} ({movie['year']})</div>
                                            <div class="movie-details">
                                                <p>Tên phim: {movie['title']}</p>
                                                <p>Năm: {movie['year']}</p>
                                                <p>Điểm: {movie['rating']}⭐</p>
                                                <p>Thể loại: {movie['genres']}</p>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                    # Streamlit buttons for actions
                                    if st.button("Đánh giá", key=f"rate_search_{movie['movieId']}_{i}_{j}_{st.session_state.recommendation_key}"):
                                        if not st.session_state.auth['user_id']:
                                            st.error("Vui lòng đăng nhập để đánh giá phim.")
                                        else:
                                            st.session_state.selected_movie = movie['movieId']
                                            st.session_state.rating_mode = True
                                            st.rerun()
                                    if st.button("Xem trailer", key=f"trailer_search_{movie['movieId']}_{i}_{j}_{st.session_state.recommendation_key}"):
                                        trailer_url = get_movie_trailer(movie['title'], movie['year'])
                                        st.markdown(f'<a href="{trailer_url}" target="_blank">Mở trailer</a>', unsafe_allow_html=True)
                        st.divider()
                    
                    # Display rating interface if a movie is selected
                    if 'selected_movie' in st.session_state and st.session_state.rating_mode and st.session_state.auth['user_id']:
                        movie = next(m for m in filtered_results if m['movieId'] == st.session_state.selected_movie)
                        st.markdown(f'<div class="section-title">Đánh giá: {movie["title"]}</div>', unsafe_allow_html=True)
                        with st.container():
                            st.image(movie['poster_url'], width=50, use_container_width=True)
                            rating = st.slider(
                                "Đánh giá (1-5)",
                                min_value=1.0,
                                max_value=5.0,
                                step=0.5,
                                value=3.0,
                                key=f"rating_search_{movie['movieId']}_{st.session_state.recommendation_key}"
                            )
                            comment = st.text_area(
                                "Bình luận (tối đa 200 từ)",
                                key=f"comment_search_{movie['movieId']}_{st.session_state.recommendation_key}",
                                height=80
                            )
                            if st.button(
                                "Lưu",
                                key=f"save_search_{movie['movieId']}_{st.session_state.recommendation_key}",
                                type="primary"
                            ):
                                word_count = len(comment.split()) if comment else 0
                                if word_count > 200:
                                    st.error("Bình luận không được vượt quá 200 từ.")
                                else:
                                    success = save_user_rating(st.session_state.auth['user_id'], movie['movieId'], rating, comment)
                                    if success:
                                        st.session_state.auth['new_user'] = get_user_rating_count(st.session_state.auth['user_id']) < 5
                                        del st.session_state.selected_movie
                                        del st.session_state.rating_mode
                                        st.session_state.recommendation_key += 1
                                        st.success("Đã lưu đánh giá.")
                                        st.rerun()
                                    else:
                                        st.error("Lỗi khi lưu đánh giá. Vui lòng thử lại.")
                else:
                    st.warning("Không tìm thấy phim phù hợp.")
            except Exception as e:
                logging.error(f"Error in search: {str(e)}")
                st.error(f"Lỗi khi tìm kiếm: {str(e)}")

def handle_guest():
    st.markdown('<div class="section-title">Chế độ khách</div>', unsafe_allow_html=True)
    st.info("Bạn đang ở chế độ khách. Đăng nhập để sử dụng đầy đủ tính năng.")
    
    st.divider()
    st.markdown('<div class="section-title">Phim thịnh hành</div>', unsafe_allow_html=True)
    
    filtered_popular = []
    for movie in popular_movies[:30]:
        movie = movie.copy()
        poster_url = get_movie_poster(movie['title'], movie['year'])
        if poster_url:
            movie['poster_url'] = poster_url
            filtered_popular.append(movie)
    
    if not filtered_popular:
        st.warning("Không có phim thịnh hành nào có poster để hiển thị.")
    else:
        for i in range(0, len(filtered_popular[:10]), 5):
            cols = st.columns(5)
            for j, movie in enumerate(filtered_popular[i:i+5]):
                with cols[j]:
                    with st.container():
                        title = html.escape(movie['title'])
                        genres = html.escape(movie['genres'])
                        st.markdown(
                            f"""
                            <div class="movie-container" data-movie-id="{movie['movieId']}">
                                <img src="{movie['poster_url']}" class="movie-poster" alt="{title}"/>
                                <div class="movie-title">{title} ({movie['year']})</div>
                                <div class="movie-details">
                                    <p><strong>Tên phim:</strong> {title}</p>
                                    <p><strong>Năm:</strong> {movie['year']}</p>
                                    <p><strong>Điểm:</strong> {movie['rating']}⭐</p>
                                    <p><strong>Thể loại:</strong> {genres}</p>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if st.button("Xem trailer", key=f"trailer_guest_popular_{movie['movieId']}_{i}_{j}_{st.session_state.recommendation_key}"):
                            trailer_url = get_movie_trailer(movie['title'], movie['year'])
                            st.markdown(f'<a href="{trailer_url}" target="_blank">Mở trailer</a>', unsafe_allow_html=True)
            st.divider()

    # Gợi ý tương tự
    st.divider()
    st.markdown('<div class="section-title">Khuyến nghị phim tương tự</div>', unsafe_allow_html=True)
    
    with st.container():
        random_titles = df_movies['title'].head(500).values
        selected_movie = st.selectbox("Chọn phim", random_titles, placeholder="Chọn phim để tìm tương tự")
        
        if st.button("Tìm", type="primary"):
            recommendations = recommend_cbf(selected_movie)
            if recommendations:
                st.markdown('<div class="section-title">Phim tương tự</div>', unsafe_allow_html=True)
                for i in range(0, len(recommendations), 5):
                    cols = st.columns(5)
                    for j, movie in enumerate(recommendations[i:i+5]):
                        with cols[j]:
                            with st.container():
                                title = html.escape(movie['title'])
                                genres = html.escape(movie['genres'])
                                st.markdown(
                                    f"""
                                    <div class="movie-container" data-movie-id="{movie['movieId']}">
                                        <img src="{movie['poster_url']}" class="movie-poster" alt="{title}"/>
                                        <div class="movie-title">{title} ({movie['year']})</div>
                                        <div class="movie-details">
                                            <p><strong>Tên phim:</strong> {title}</p>
                                            <p><strong>Năm:</strong> {movie['year']}</p>
                                            <p><strong>Điểm:</strong> {movie['rating']}⭐</p>
                                            <p><strong>Thể loại:</strong> {genres}</p>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                if st.button("Xem trailer", key=f"trailer_guest_similar_{movie['movieId']}_{i}_{j}_{st.session_state.recommendation_key}"):
                                    trailer_url = get_movie_trailer(movie['title'], movie['year'])
                                    st.markdown(f'<a href="{trailer_url}" target="_blank">Mở trailer</a>', unsafe_allow_html=True)
                    st.divider()
            else:
                st.warning("Không tìm thấy phim tương tự có poster.")

def handle_userid_recommendation():
    st.markdown('<div class="section-title">Gợi ý theo UserID</div>', unsafe_allow_html=True)
    st.info("Nhập UserID để nhận gợi ý phim cá nhân hóa.")
    
    with st.container():
        user_id_input = st.text_input("UserID", key="userid_input", placeholder="Nhập UserID")
        
        if st.button("Tìm", type="primary"):
            if not user_id_input:
                st.error("Vui lòng nhập UserID.")
                return
            
            try:
                user_id = int(user_id_input)
                recs = recommend_movies(user_id)
                
                if recs is None:
                    st.warning(f"Không thể tạo gợi ý cho UserID {user_id}. Vui lòng kiểm tra UserID.")
                    return
                
                st.markdown(f'<div class="section-title">Gợi ý cho UserID {user_id}</div>', unsafe_allow_html=True)
                for i in range(0, len(recs), 5):
                    cols = st.columns(5)
                    for j, movie in enumerate(recs[i:i+5]):
                        with cols[j]:
                            with st.container():
                                title_escaped = movie['title'].replace("'", "\\'")
                                st.markdown(
                                    f"""
                                    <div class="movie-container" data-movie-id="{movie['movieId']}">
                                        <img src="{movie['poster_url']}" class="movie-poster" alt="{title_escaped}"/>
                                        <div class="movie-title">{title_escaped} ({movie['year']})</div>
                                        <div class="movie-details">
                                            <p>Tên phim:{movie['title']}</p>
                                            <p>Năm: {movie['year']}</p>
                                            <p>Điểm: {movie['rating']}⭐</p>
                                            <p>Thể loại: {movie['genres']}</p>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                # Streamlit button for action
                                if st.button("Đánh giá", key=f"rate_search_{movie['movieId']}_{i}_{j}_{st.session_state.recommendation_key}"):
                                    st.session_state.selected_movie = movie['movieId']
                                    st.session_state.rating_mode = True
                                    st.rerun()
                                if st.button("Xem trailer", key=f"trailer_search_{movie['movieId']}_{i}_{j}_{st.session_state.recommendation_key}"):
                                    trailer_url = get_movie_trailer(movie['title'], movie['year'])
                                    # Long 
                                    st.markdown(f'<a href="{trailer_url}" target="_blank">Mở trailer</a>', unsafe_allow_html=True)
                                    #st.markdown(f'<meta http-equiv="refresh" content="0;url={trailer_url}">', unsafe_allow_html=True)
                    st.divider()
            except ValueError:
                st.error("UserID phải là số nguyên.")
            except Exception as e:
                logging.error(f"Error in UserID recommendation: {str(e)}")
                st.error(f"Lỗi khi tạo gợi ý: {str(e)}")
        # st.markdown('</div>', unsafe_allow_html=True)

def handle_logout():
    try:
        st.session_state.auth = {'user_id': None, 'username': None, 'new_user': False}
        if 'selected_movie' in st.session_state:
            del st.session_state.selected_movie
        if 'rating_mode' in st.session_state:
            del st.session_state.rating_mode
        if 'recommendation_key' in st.session_state:
            del st.session_state.recommendation_key
        logging.info(f"User {st.session_state.auth.get('username', 'unknown')} logged out successfully")
        st.success("Đăng xuất thành công.")
        st.rerun()
    except Exception as e:
        logging.error(f"Error during logout: {str(e)}")
        st.error(f"Lỗi khi đăng xuất: {str(e)}")

def main():
    # Inject global CSS
    st.markdown(global_css, unsafe_allow_html=True)
    
    # Header
    st.markdown(
    '''
    <div class="header">
        <h1>Recommendation System Movies</h1>
        <h2 style="font-size: 18px; font-weight: 400; margin-top: 5px;">Duc-LongTran 124211</h2>
    </div>
    ''',
    unsafe_allow_html=True
)

    
    check_movie_id_alignment()
    
    # Initialize session state
    if 'auth' not in st.session_state:
        st.session_state.auth = {'user_id': None, 'username': None, 'new_user': False}
    if 'recommendation_key' not in st.session_state:
        st.session_state.recommendation_key = 0
    
    try:
        conn = get_db_connection()
        conn.close()
    except Exception as e:
        logging.error(f"Database connection failed: {str(e)}")
        st.error(f"Kết nối database thất bại: {str(e)}")
        st.markdown("Hướng dẫn sửa lỗi:")
        st.markdown("- Chạy lệnh: `chmod 660 data/users.db`")
        st.markdown("- Đảm bảo quyền thư mục: `chmod 775 data`")
        st.markdown("- Đổi quyền sở hữu: `chown -R $(whoami) data`")
        st.stop()
    
    # Sidebar
    menu_options = ["🏠 Trang chủ", "🔍 Tìm kiếm", "👤 Hồ sơ", "🎬 Gợi ý theo UserID"] if st.session_state.auth['user_id'] else ["🔑 Đăng nhập", "📝 Đăng ký", "👤 Khách"]
    choice = st.sidebar.selectbox("Menu", menu_options, format_func=lambda x: x[2:])
    
    if st.session_state.auth['user_id']:
        if st.sidebar.button("Đăng xuất", key="logout_button"):
            handle_logout()
    
    # Route to appropriate handler
    if choice == "🔑 Đăng nhập":
        handle_login()
    elif choice == "📝 Đăng ký":
        handle_register()
    elif choice == "🏠 Trang chủ":
        handle_home()
    elif choice == "👤 Hồ sơ":
        handle_profile()
    elif choice == "🔍 Tìm kiếm":
        handle_search()
    elif choice == "👤 Khách":
        handle_guest()
    elif choice == "🎬 Gợi ý theo UserID":
        handle_userid_recommendation()
    
    # Footer
    st.markdown(
        '<div class="footer">© Recommemdation System Movies Duc-LongTran 124211</div>',
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()