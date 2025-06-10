import sqlite3
import pickle
import os
import logging
from pathlib import Path
import stat
import time
import re
import getpass
import platform
import numpy as np
import pandas as pd
import random
import torch

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s - User: %(user)s')

def get_current_user():
    """Get the current system user for logging."""
    try:
        return getpass.getuser()
    except:
        return "unknown"

# Inject user into logging context
logging.getLogger().handlers[0].setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - User: ' + get_current_user())
)

DATABASE_PATH = "data/users.db"
MODEL_DIR = "save_model"
MOVIES_CSV_PATH = "data/processed_movies.csv"

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

def is_file_locked(filepath):
    """Check if a file is locked by attempting to open it exclusively."""
    try:
        with open(filepath, 'a') as f:
            if platform.system() == "Windows":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return False
    except (IOError, OSError) as e:
        logging.warning(f"File {filepath} may be locked: {str(e)}")
        return True
    except ImportError:
        logging.debug(f"Lock check not supported on {platform.system()}")
        return False

def ensure_directory_permissions(directory):
    """Ensure directory exists and has write permissions."""
    try:
        Path(directory).mkdir(exist_ok=True, parents=True)
        if platform.system() != "Windows":
            try:
                os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)  # 775
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
                os.chmod(filepath, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)  # 660
            except PermissionError as e:
                logging.warning(f"Cannot set permissions for {filepath}: {str(e)}")
        if is_file_locked(filepath):
            logging.error(f"File {filepath} is locked by another process")
            return False
        if not os.access(filepath, os.W_OK):
            logging.error(f"No write permission for file: {filepath}")
            return False
        log_file_status(filepath)
        return True
    except Exception as e:
        logging.error(f"Failed to ensure file {filepath} is writable: {str(e)}")
        return False

def get_db_connection():
    """Establish a database connection."""
    try:
        if not ensure_directory_permissions("data"):
            raise Exception("Cannot create or access data directory")
        if not ensure_file_writable(DATABASE_PATH):
            raise Exception(f"Cannot write to database file: {DATABASE_PATH}")
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key support
        logging.debug(f"Connected to database: {DATABASE_PATH}")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Failed to connect to database: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error setting up database file: {str(e)}")
        raise

def init_db():
    """Initialize the database with required tables and indices."""
    try:
        conn = get_db_connection()
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                        password TEXT NOT NULL)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS ratings (
                        user_id INTEGER NOT NULL,
                        movie_id INTEGER NOT NULL,
                        rating REAL NOT NULL,
                        comment TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(user_id) REFERENCES users(id))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS preferences (
                        user_id INTEGER PRIMARY KEY,
                        genres TEXT,
                        FOREIGN KEY(user_id) REFERENCES users(id))''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ratings_user_id ON ratings(user_id)')
        conn.commit()
        logging.info("Database initialized successfully")
    except sqlite3.Error as e:
        logging.error(f"Database initialization failed: {str(e)}")
        raise
    finally:
        conn.close()

def debug_users_table():
    """Debug function to list all usernames in the users table."""
    try:
        conn = get_db_connection()
        cursor = conn.execute('SELECT username FROM users')
        usernames = [row[0] for row in cursor.fetchall()]
        conn.close()
        logging.debug(f"Current usernames in users table: {usernames}")
        return usernames
    except sqlite3.Error as e:
        logging.error(f"Error debugging users table: {str(e)}")
        return []

def add_user_to_ncf_system(max_retries=3, retry_delay=1):
    """Select a random unused userID from user_id_map.pkl for a new user."""
    user_map_path = os.path.join(MODEL_DIR, "user_id_map.pkl")
    
    for attempt in range(max_retries):
        try:
            if not ensure_directory_permissions(MODEL_DIR):
                logging.error("Cannot create or access save_model directory")
                try:
                    import streamlit as st
                    st.error("Không thể truy cập thư mục save_model. Kiểm tra quyền truy cập!")
                except ImportError:
                    pass
                return None

            if not os.path.exists(user_map_path):
                logging.error(f"File {user_map_path} does not exist")
                try:
                    import streamlit as st
                    st.error(f"File user_id_map.pkl không tồn tại!")
                except ImportError:
                    pass
                return None

            if is_file_locked(user_map_path):
                logging.error(f"File {user_map_path} is locked, cannot proceed")
                try:
                    import streamlit as st
                    st.error(f"File user_id_map.pkl đang bị khóa. Đóng các tiến trình khác hoặc xóa file.")
                except ImportError:
                    pass
                return None

            # Load user_id_map
            try:
                with open(user_map_path, "rb") as f:
                    user_id_map = pickle.load(f)
                if not isinstance(user_id_map, dict):
                    logging.error(f"Corrupted user_id_map.pkl, not a dictionary: {user_map_path}")
                    try:
                        import streamlit as st
                        st.error(f"File user_id_map.pkl bị hỏng!")
                    except ImportError:
                        pass
                    return None
            except (pickle.UnpicklingError, EOFError, ValueError) as e:
                logging.error(f"Corrupted user_id_map.pkl: {str(e)}")
                try:
                    import streamlit as st
                    st.error(f"File user_id_map.pkl bị hỏng: {str(e)}")
                except ImportError:
                    pass
                return None
            except Exception as e:
                logging.error(f"Unexpected error loading user_id_map.pkl: {str(e)}")
                return None

            # Chuyển đổi khóa numpy.int64 thành int
            valid_user_id_map = {int(k) if isinstance(k, np.int64) else k: v for k, v in user_id_map.items()}
            if not all(isinstance(k, int) for k in valid_user_id_map.keys()):
                logging.error(f"user_id_map.pkl contains non-integer keys: {list(valid_user_id_map.keys())[:10]}")
                try:
                    import streamlit as st
                    st.error("Dữ liệu trong user_id_map.pkl không hợp lệ: Có ID không phải số nguyên!")
                except ImportError:
                    pass
                return None

            # Lưu lại file với khóa đã chuyển đổi
            try:
                with open(user_map_path, "wb") as f:
                    pickle.dump(valid_user_id_map, f)
                logging.info(f"Updated user_id_map.pkl with integer keys")
            except Exception as e:
                logging.error(f"Failed to update user_id_map.pkl: {str(e)}")
                return None

            # Get available user IDs
            available_ids = list(valid_user_id_map.keys())

            # Get used IDs from users.db
            used_ids = set()
            try:
                conn = get_db_connection()
                cursor = conn.execute('SELECT id FROM users')
                used_ids.update(row[0] for row in cursor.fetchall())
                conn.close()
            except Exception as e:
                logging.error(f"Error checking used IDs in users.db: {str(e)}")
                try:
                    import streamlit as st
                    st.error(f"Lỗi khi kiểm tra ID người dùng trong cơ sở dữ liệu: {str(e)}")
                except ImportError:
                    pass
                return None

            # Select random unused ID
            available_ids = [id_ for id_ in available_ids if id_ not in used_ids]
            if not available_ids:
                logging.error("No unused user IDs available in user_id_map.pkl")
                try:
                    import streamlit as st
                    st.error("Hết ID người dùng khả dụng trong user_id_map.pkl!")
                except ImportError:
                    pass
                return None

            new_id = random.choice(available_ids)
            logging.debug(f"Selected random userID: {new_id}")
            logging.info(f"Assigned user ID {new_id} for new user")
            return new_id

        except Exception as e:
            logging.error(f"Error in add_user_to_ncf_system (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying add_user_to_ncf_system (attempt {attempt + 2}/{max_retries})")
                time.sleep(retry_delay)
                continue
            try:
                import streamlit as st
                st.error(f"Lỗi khi cấp ID người dùng NCF: {str(e)}")
            except ImportError:
                pass
            return None
    return None

def get_user_rating_count(user_id):
    """Get the number of ratings for a user."""
    try:
        conn = get_db_connection()
        cursor = conn.execute('SELECT COUNT(*) FROM ratings WHERE user_id = ?', (user_id,))
        count = cursor.fetchone()[0]
        return count
    except sqlite3.Error as e:
        logging.error(f"Error getting user rating count: {str(e)}")
        return 0
    finally:
        conn.close()

def is_new_user(user_id):
    """Check if user has fewer than 3 ratings."""
    return get_user_rating_count(user_id) < 3

def register_user(username, password, confirm_password):
    """Register a new user with a random userID from user_id_map.pkl."""
    # Hỗ trợ ký tự đặc biệt trong tên người dùng
    if not re.match(r'^[\w@#$%^&*+\-!]{3,20}$', username):
        logging.error(f"Invalid username format: {username}")
        try:
            import streamlit as st
            st.error("Tên đăng nhập chỉ được chứa chữ cái, số, ký tự đặc biệt (@#$%^&*+-!) và từ 3-20 ký tự!")
        except ImportError:
            pass
        return None

    if password != confirm_password:
        logging.error(f"Password and confirmation do not match for username: {username}")
        try:
            import streamlit as st
            st.error("Mật khẩu và xác nhận mật khẩu không khớp!")
        except ImportError:
            pass
        return None

    try:
        init_db()
        conn = get_db_connection()
        cursor = conn.execute('SELECT id FROM users WHERE UPPER(username) = UPPER(?)', (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            logging.error(f"Username already exists (case-insensitive): {username}")
            try:
                import streamlit as st
                st.error(f"Tên đăng nhập {username} đã tồn tại!")
            except ImportError:
                pass
            conn.close()
            return None

        user_id = add_user_to_ncf_system()
        if not isinstance(user_id, int):
            logging.error(f"Invalid user_id type: {type(user_id)}, value: {user_id}")
            try:
                import streamlit as st
                st.error("Lỗi hệ thống: ID người dùng không hợp lệ. Kiểm tra user_id_map.pkl!")
            except ImportError:
                pass
            conn.close()
            return None

        logging.debug(f"Attempting to insert user: id={user_id}, username={username}")
        conn.execute('INSERT INTO users (id, username, password) VALUES (?, ?, ?)',
                     (user_id, username, password))
        conn.commit()
        logging.info(f"Successfully registered user: {username} with id: {user_id}")
        debug_users_table()
        return user_id
    except sqlite3.IntegrityError as e:
        logging.error(f"Database IntegrityError during registration: {str(e)}")
        try:
            import streamlit as st
            st.error(f"Lỗi cơ sở dữ liệu khi đăng ký: {str(e)}. Kiểm tra user_id_map.pkl hoặc xóa users.db!")
        except ImportError:
            pass
        return None
    except Exception as e:
        logging.error(f"Unexpected error during registration: {str(e)}")
        try:
            import streamlit as st
            st.error(f"Lỗi đăng ký: {str(e)}")
        except ImportError:
            pass
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def validate_login(username, password):
    """Validate user login credentials."""
    try:
        conn = get_db_connection()
        cursor = conn.execute('SELECT id FROM users WHERE username = ? AND password = ?',
                             (username, password))
        user = cursor.fetchone()
        return user[0] if user else None
    except sqlite3.Error as e:
        logging.error(f"Error validating login: {str(e)}")
        return None
    finally:
        conn.close()

def save_user_rating(user_id, movie_id, rating, comment=None):
    """Save a user rating for a movie, ensuring comment is max 200 words."""
    try:
        if comment:
            word_count = len(comment.split())
            if word_count > 200:
                logging.error(f"Comment exceeds 200 words: {word_count} words")
                try:
                    import streamlit as st
                    st.error("Bình luận không được vượt quá 200 từ!")
                except ImportError:
                    pass
                return False
        conn = get_db_connection()
        conn.execute('INSERT INTO ratings (user_id, movie_id, rating, comment) VALUES (?, ?, ?, ?)',
                    (user_id, movie_id, rating, comment))
        conn.commit()
        logging.info(f"Saved rating for user_id={user_id}, movie_id={movie_id}")
        return True
    except sqlite3.Error as e:
        logging.error(f"Error saving rating: {str(e)}")
        return False
    finally:
        conn.close()

def save_user_preferences(user_id, genres):
    """Save user genre preferences."""
    try:
        conn = get_db_connection()
        conn.execute('INSERT OR REPLACE INTO preferences (user_id, genres) VALUES (?, ?)',
                    (user_id, genres))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Error saving preferences: {str(e)}")
    finally:
        conn.close()

def get_user_history(user_id):
    """Get user rating history, limited to the 20 most recent ratings."""
    try:
        conn = get_db_connection()
        cursor = conn.execute('''SELECT movie_id, rating, comment, timestamp 
                              FROM ratings WHERE user_id = ? 
                              ORDER BY timestamp DESC LIMIT 20''', (user_id,))
        history = [{
            'movie_id': row[0],
            'rating': row[1],
            'comment': row[2],
            'timestamp': row[3]
        } for row in cursor.fetchall()]
        return history
    except sqlite3.Error as e:
        logging.error(f"Error getting user history: {str(e)}")
        return []
    finally:
        conn.close()

try:
    init_db()
except Exception as e:
    logging.error(f"Failed to initialize database on import: {str(e)}")
