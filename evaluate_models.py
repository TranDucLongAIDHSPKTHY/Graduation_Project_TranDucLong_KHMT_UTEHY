import os

print("Evaluate models...")

os.system("python evaluate_models/evaluate_content_based.py")
os.system("python evaluate_models/evaluate_collaborative.py")
os.system("python evaluate_models/evaluate_hybrid_sh.py")
os.system("python evaluate_models/evaluate_autoencoder.py")
os.system("python evaluate_models/evaluate_ncf")

print("Đã đánh giá xong 3 mô hình ML!")
