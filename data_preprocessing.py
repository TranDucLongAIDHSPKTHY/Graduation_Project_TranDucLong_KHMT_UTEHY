import os

print(" Data Preprocessing...")

os.system("python preprocessing/preprocess_ml.py")
os.system("python preprocessing/preprocess_dl.py")


print("Đã xử lý dữ liệu xong!")
