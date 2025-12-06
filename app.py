import streamlit as st
from transformers import pipeline
import underthesea
import sqlite3
from datetime import datetime

from sentiment_engine import classify_sentiment


# Thiết lập cơ sở dữ liệu SQLite
def init_db():
    conn = sqlite3.connect('sentiments.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sentiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    timestamp TEXT NOT NULL)''')
    conn.commit()
    conn.close()

# Lưu kết quả phân loại vào DB
def save_to_db(text, sentiment):
    conn = sqlite3.connect('sentiments.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)', (text, sentiment, timestamp))
    conn.commit()
    conn.close()

# Lấy lịch sử phân loại 
def get_history(limit=None):
    conn = sqlite3.connect('sentiments.db')
    c = conn.cursor()
    if limit:
        c.execute('SELECT text, sentiment, timestamp FROM sentiments ORDER BY timestamp DESC LIMIT ?', (limit,))
    else:
        c.execute('SELECT text, sentiment, timestamp FROM sentiments ORDER BY timestamp DESC')
    data = c.fetchall()
    conn.close()
    return data


# Hàm xử lý tiếng Việt (chuẩn hóa đầu vào)
def preprocess_text(text):
    text = text.strip().lower()
    tokens = underthesea.word_tokenize(text)
    return ' '.join(tokens)

# Pipeline Transformer để phân loại cảm xúc
@st.cache_resource
def load_pipeline():
    try:
        model = pipeline("sentiment-analysis", model="vinai/phobert-base-v2")
    except:
        model = pipeline("sentiment-analysis", model="distilbert-base-multilingual-cased")
    return model

# Giao diện Streamlit
def main():
    st.set_page_config(page_title="Trợ lý phân loại cảm xúc Tiếng Việt", layout="centered")
    st.title("Trợ lý phân loại cảm xúc Tiếng Việt")
    st.write("Ứng dụng sử dụng mô hình Transformer PhoBERT. 🧠")

    init_db()
    user_input = st.text_area("Nhập câu tiếng Việt:", placeholder="Ví dụ: Hôm nay tôi rất vui") or ""
    text = user_input.strip()
    if st.button("Phân loại cảm xúc"):
    # Kiểm tra lỗi nhập liệu
        if len(text) == 0:
            st.warning("⚠️ Bạn chưa nhập nội dung. Vui lòng nhập câu tiếng Việt.")
        elif len(text) < 4:
            st.error("❌ Câu quá ngắn. Vui lòng nhập câu dài hơn để phân tích cảm xúc.")
        elif not any(char.isalpha() for char in text):
            st.error("⚠️ Câu nhập không hợp lệ. Vui lòng nhập văn bản tiếng Việt hợp lệ.")
        else:
            with st.spinner("Đang phân tích cảm xúc..."):
                try:
                    result = classify_sentiment(text)
                    sentiment = result["sentiment"]
                    scores = result["scores"]

                    max_prob = max(scores.values())
                    if max_prob < 0.5:
                        st.warning("🤔 Mô hình không chắc chắn về cảm xúc của câu này (độ tin cậy thấp).")

                    st.success(f"**Kết quả cảm xúc:** {sentiment}")
                    st.subheader("📊 Xác suất dự đoán")
                    st.write(f"🟥 Tiêu cực (NEG): {scores['NEGATIVE']}")
                    st.write(f"🟩 Tích cực (POS): {scores['POSITIVE']}")
                    st.write(f"🟦 Trung lập (NEU): {scores['NEUTRAL']}")

                    save_to_db(text, sentiment)

                except Exception as e:
                    st.error(f"❌ Lỗi khi xử lý mô hình: {e}")


    # Hiển thị lịch sử phân loại
    st.subheader("📜 Lịch sử phân loại cảm xúc")

    # Trạng thái xem thêm (giữ nguyên giữa các lần refresh)
    if "show_all" not in st.session_state:
        st.session_state.show_all = False

    # Lấy dữ liệu
    if st.session_state.show_all:
        history = get_history()  # toàn bộ
    else:
        history = get_history(limit=10)  # 10 gần nhất

    # Hiển thị danh sách
    if history:
        for row in history:
            st.write(f"🕒 {row[2]} — '{row[0]}' → **{row[1]}**")

        if st.session_state.show_all:
            if st.button("⬆️ Thu gọn"):
                st.session_state.show_all = False
        else:
            if st.button("📂 Xem thêm"):
                st.session_state.show_all = True
    else:
        st.info("Chưa có lịch sử phân loại.")


    # Hướng dẫn sử dụng
    st.markdown("---")
    st.markdown("### 🧾 Hướng dẫn nhanh")
    st.markdown("1️⃣ Nhập câu tiếng Việt tự nhiên.  ")
    st.markdown("2️⃣ Nhấn **Phân loại cảm xúc**.  ")
    st.markdown("3️⃣ Xem kết quả và lịch sử phân loại ở bên dưới.  ")

if __name__ == "__main__":
    main()
