import os

print("Evaluate models...")

os.system("python evaluate_models/evaluate_collaborative.py")
os.system("python evaluate_models/evaluate_hybrid_sh.py")
os.system("python evaluate_models/evaluate_autoencoder.py")
os.system("python evaluate_models/evaluate_ncf")
os.system("python evaluate_models/evaluate_content_based.py")

print("Đã đánh giá xong 5 mô hình!")
