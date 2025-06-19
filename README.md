# Hướng Dẫn Triển Khai Dự Án

Dự án này là một hệ thống gợi ý phim sử dụng bộ dữ liệu MovieLens 25M, tích hợp các mô hình học máy và học sâu để đưa ra các gợi ý phim chính xác dựa trên đánh giá và nội dung. Hướng dẫn dưới đây cung cấp các bước triển khai dự án từ việc chuẩn bị dữ liệu đến chạy ứng dụng demo.

## 1. Dữ Liệu
- **Bộ dữ liệu**: MovieLens 25M (Kaggle Dataset)  
- **Link**: [MovieLens 25M Dataset](https://www.kaggle.com/datasets/garymk/movielens-25m-dataset)  
- **Tệp sử dụng**:  
  - `ratings.csv`: Điểm đánh giá phim  
  - `movies.csv`: Thông tin phim  
  - `tags.csv`: Thẻ gắn với phim  

## 2. Quy Trình Thực Hiện
Thực hiện lần lượt các bước sau (hoàn thành bước trước mới chuyển sang bước tiếp theo):

1. **Tiền xử lý dữ liệu**: Chạy file `data_preprocessing.py`.  
2. **Huấn luyện mô hình**: Chạy file `train_models.py`.  
3. **Đánh giá mô hình**: Chạy file `evaluate_models.py`.  
4. **Tạo cơ sở dữ liệu demo**: Chạy file `SQLite.py`.  
5. **Chạy ứng dụng demo**: Sử dụng lệnh sau:  
   ```bash
   streamlit run app/app.py --server.fileWatcherType none

## 3. Chức Năng Các Thư Mục/File

1. **app**: Chứa cơ sở dữ liệu và mã nguồn chương trình demo.  
2. **data**: Lưu trữ bộ dữ liệu MovieLens và cơ sở dữ liệu chương trình demo.    
3. **evaluate_models**: Chứa hàm đánh giá các mô hình.
4. **models**: Chứa các file huấn luyện mô hình.  
5. **notebook**: Dùng để khám phá và trực quan hóa dữ liệu. 
6. **preprocessing**: Chứa các file tiền xử lý dữ liệu.  
7. **results_model**: Lưu kết quả huấn luyện và đánh giá (định dạng .txt).
8. **save_model**: Lưu mô hình đã huấn luyện (định dạng .pkl, .pth).
9. **test**: Chứa file kiểm tra gợi ý phim của các mô hình (đang phát triển).
10. **venv**: Chứa các thư viện Python được sử dụng trong dự án.

## 4. Lưu ý
Khi clone về thiết bị khác nên import lại các thư viện.

## 5. Link báo cáo Đồ án tốt nghiệp
Truy cập link: https://drive.google.com/drive/folders/1RS00TQ_cXY7qJt0zOUabaruMSvaLfXK4
