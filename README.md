# 🇻🇳 Trợ lý Phân loại Cảm xúc Tiếng Việt (Transformer)

## 🧠 Giới thiệu
Ứng dụng giúp **phân loại cảm xúc tiếng Việt** (tích cực, tiêu cực, trung lập) dựa trên mô hình **Transformer** (PhoBERT) .  
Dự án được xây dựng trong khuôn khổ **đồ án môn học** với yêu cầu đạt tối thiểu 65% độ chính xác.

---

## 🧩 Cấu trúc dự án
```bash
sentiment_check/
│
├── app.py                 
├── database.py            
├── sentiment_engine.py      
└── README.md              
```

---

## ⚙️ Cài đặt môi trường
### 1️⃣ Tạo môi trường ảo
```bash
python -m venv venv
source venv/bin/activate    # Trên Linux/Mac
venv\Scripts\activate       # Trên Windows
```

### 2️⃣ Cài thư viện cần thiết
```bash
pip install streamlit transformers underthesea torch sqlite3
```

---

## 🚀 Chạy ứng dụng chính
```bash
streamlit run app.py
```
Sau đó mở trình duyệt tại địa chỉ:
```
http://localhost:8501
```

Nhập câu tiếng Việt → Nhấn **Phân loại cảm xúc** để xem kết quả.

---

## 🗄️ Cơ sở dữ liệu
- SQLite file: `sentiments.db`
- Bảng: `sentiments(id, text, sentiment, timestamp)`

Lịch sử được lưu tự động mỗi khi bạn phân loại cảm xúc.

---

## 🧾 Báo cáo ngắn
**Mục tiêu:** Xây dựng ứng dụng AI nhận biết cảm xúc trong tiếng Việt bằng Transformer.  
**Phương pháp:** Sử dụng PhoBERT được tích hợp sẵn để phân loại.  
**Chức năng:**
- Giao diện web bằng Streamlit.
- Xử lý tiếng Việt bằng underthesea.
- Lưu lịch sử bằng SQLite.
- Đánh giá độ chính xác qua 10 test case.

**Kết quả mong đợi:**
- Độ chính xác ≥ 65%.
- Thời gian phản hồi < 2s.
- Ứng dụng chạy ổn định, không crash.

---


