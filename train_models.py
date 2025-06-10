import os

print("Training models...")

os.system("python models/content_based.py")
os.system("python models/collaborative.py")
os.system("python models/hybrid_sh.py")
os.system("python models/autoencoder.py")
os.system("python models/ncf.py")

print("Đã huấn luyện xong tất cả mô hình!")
