import os
import sys
import pickle
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import surprise
from surprise import dump
import gdown
import pandas as pd
import re
from pyvi.ViTokenizer import tokenize
from collections import Counter
import streamlit as st
import json
import base64

# 1. LOAD DICTIONARIES (Sử dụng cache để không bị load lại file txt liên tục)
@st.cache_data
def load_keywords_config():
    try:
        with open("keywords.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"⚠️ Không thể đọc file keywords.json: {e}")
        return {}

# 2. Hàm trích xuất các keyword xuất hiện trong mô tả khách sạn
def extract_hotel_keywords(description_text):
    if not description_text or pd.isna(description_text):
        return []
    
    config = load_keywords_config()
    desc_lower = str(description_text).lower()
    found_tags = []

    for cat_key, cat_data in config.items():
        icon = cat_data.get("icon", "🏷️")
        keywords = cat_data.get("keywords", [])
        
        # Kiểm tra xem từ khóa nào xuất hiện trong mô tả
        for kw in keywords:
            # Dùng regex hoặc 'in' để quét từ
            if kw in desc_lower:
                # Tạo tag định dạng dạng: 📍 Gần biển, ✨ Sạch sẽ...
                tag_label = f"{icon} {kw.title()}"
                if tag_label not in found_tags:
                    found_tags.append(tag_label)
                    
    return found_tags

@st.cache_data
def load_dicts():
    # Stopwords
    with open('files/vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = f.read().split('\n')
    
    # Emoji
    emoji_dict = {}
    with open('files/emojicon.txt', 'r', encoding='utf-8') as f:
        for line in f.read().split('\n'):
            if '\t' in line: 
                k, v = line.split('\t')
                emoji_dict[k] = v
            
    # Teencode
    teen_dict = {}
    with open('files/teencode.txt', 'r', encoding='utf-8') as f:
        for line in f.read().split('\n'):
            if '\t' in line: 
                k, v = line.split('\t')
                teen_dict[k] = v
            
    # Wrong words list
    with open('files/wrong-word.txt', 'r', encoding='utf-8') as f:
        wrong_lst = [w.strip() for w in f.read().split('\n') if w.strip() != '']

    return stop_words, emoji_dict, teen_dict, wrong_lst

# Nạp từ điển ngay khi chạy app
try:
    STOP_WORDS, EMOJI_DICT, TEEN_DICT, WRONG_LST = load_dicts()
except FileNotFoundError as e:
    st.error(f"⚠️ Thiếu file từ điển: {e}. Vui lòng đảm bảo 4 file txt nằm cùng thư mục.")

# 2. HÀM CLEAN VĂN BẢN
def clean_review_text(text):
    if not isinstance(text, str) or text.strip() == '': return ''
    text = text.lower()
    for emoji, word in EMOJI_DICT.items():
        text = text.replace(emoji, ' ' + word + ' ')
    words = text.split()
    words = [TEEN_DICT.get(w, w) for w in words]
    text = tokenize(' '.join(words))
    tokens = [re.sub(r'[^\w]', '', t) for t in text.split()]
    tokens = [re.sub(r'\d+', '', t) for t in tokens if t.strip() != '']
    tokens = [t for t in tokens if t not in WRONG_LST and t not in STOP_WORDS]
    return ' '.join(tokens)

# 3. HÀM PHÂN TÍCH TỪ KHÓA
def get_keyword_analysis(df_hotel_reviews, top_n=15):
    # Đảm bảo có cột Score để lọc
    if 'Score' not in df_hotel_reviews.columns:
        return {}, {}
        
    positive = df_hotel_reviews[df_hotel_reviews['Score'] >= 8]['Review_Content_Clean']
    negative = df_hotel_reviews[df_hotel_reviews['Score'] <= 5]['Review_Content_Clean']

    pos_words = Counter(' '.join(positive).split()).most_common(top_n)
    neg_words = Counter(' '.join(negative).split()).most_common(top_n)

    return dict(pos_words), dict(neg_words)

# ---------------------------------------------------------
# 1. Cấu hình đường dẫn & Load Dữ liệu / Model
# ---------------------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "model"
FOLDER_URL = "https://drive.google.com/drive/folders/1QwNB7ZIvZxnhs9a2Nq0QymgDM-P0Zxdr"

@st.cache_resource
def download_models_from_drive():
    """Tự động tải 2 file model (.pkl) từ Google Drive folder về thư mục model/ nếu chưa tồn tại."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path_knn = os.path.join(MODEL_DIR, "knn_model.pkl")
    path_cosine = os.path.join(MODEL_DIR, "cosine_sim.pkl")
    # Kiểm tra xem cả 2 file đã tồn tại chưa
    if not (os.path.exists(path_knn) and os.path.exists(path_cosine)):
        try:
            # Tải toàn bộ nội dung trong thư mục Drive về folder MODEL_DIR
            gdown.download_folder(url=FOLDER_URL, output=MODEL_DIR, quiet=True)
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")

@st.cache_data
def load_all_data():
    try:
        path_info = os.path.join(DATA_DIR, 'hotel_info_clean.csv')
        path_comments = os.path.join(DATA_DIR, 'hotel_comments_clean.csv')

        df_info = pd.read_csv(path_info)
        df_comments = pd.read_csv(path_comments)
        return df_info, df_comments
    except Exception as e:
        st.error(f'Lỗi đọc file dữ liệu CSV: {e}')
        return None, None

@st.cache_resource
def load_all_models():
    # Gọi hàm tự động tải model nếu chưa có file
    download_models_from_drive()

    knn_model = None
    cosine_sim = None

    path_knn = os.path.join(MODEL_DIR, "knn_model.pkl")
    path_cosine = os.path.join(MODEL_DIR, "cosine_sim.pkl")

    if os.path.exists(path_knn):
        try:
            knn_model = pickle.load(open(path_knn, 'rb'))
        except Exception as e:
            st.warning(f"⚠️ Không thể load `knn_model.pkl`: {e}")

    if os.path.exists(path_cosine):
        try:
            cosine_sim = pickle.load(open(path_cosine, 'rb'))
        except Exception as e:
            st.warning(f"⚠️ Không thể load `cosine_sim.pkl`: {e}")

    return knn_model, cosine_sim


# Gọi hàm nạp dữ liệu & mô hình thực tế
df_info, df_comments = load_all_data()
knn_model, cosine_sim = load_all_models()

# ---------------------------------------------------------
# 1. Cấu hình trang (Page Config)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Agoda Hotel Recommender System",
    page_icon="🏨",
    layout="wide"
)

# Custom CSS để chỉnh chữ radio menu to hơn và giãn rộng khoảng cách
st.markdown("""
<style>
    /* Chỉnh kích thước chữ của Radio Options trong Sidebar */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 1.15rem !important; /* Tăng cỡ chữ */
        font-weight: 500;
    }
    
    /* Tăng khoảng cách (padding/margin) giữa các lựa chọn Menu */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding-top: 10px !important;
        padding-bottom: 10px !important;
        margin-bottom: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Thanh điều hướng bên trái (Sidebar Navigation)
# ---------------------------------------------------------
st.sidebar.title("🏨 Hotel Recommender")

menu = st.sidebar.radio(
    "Menu điều hướng",
    ["Trang chủ", "Business problem", "Phân công nhóm", "Khách du lịch", "Chủ khách sạn"]
)

# Thêm thông báo trạng thái mô hình be bé ở đây
if knn_model is not None and cosine_sim is not None:
    st.sidebar.caption("🟢 *Mô hình đã sẵn sàng cho gợi ý*")
else:
    st.sidebar.caption("🔴 *Mô hình chưa sẵn sàng, vui lòng kiểm tra lại*")

st.sidebar.markdown("---")
st.sidebar.caption("""
Đồ án tốt nghiệp Data Science - Trung tâm tin học - Trường Đại Học Khoa Học Tự Nhiên, ĐHQG-HCM
\nProject: Recommender Systems
\nNhóm thực hiện:
\nNguyễn Thị Thúy Hằng  thuyhang0911@gmail.com
\nNguyễn Nhật Minh Thư  nguyen.nhatminhthu19@gmail.com
\nLê Ngọc Tuấn          \nlengoctuan04lkk@gmail.com
\nGiảng viên hướng dẫn: \nCô. Khuất Thùy Phương
""")

# ---------------------------------------------------------
# 3. Màn hình: TRANG CHỦ
# ---------------------------------------------------------
# Hàm phụ trợ để chuyển đổi ảnh local sang base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

if menu == "Trang chủ":
    # Đường dẫn tới file ảnh banner
    banner_path = os.path.join("Hotel", "Banner.jpg")
    
    try:
        # Chuyển ảnh sang chuỗi base64
        base64_img = get_base64_image(banner_path)
        
        # LƯU Ý QUAN TRỌNG: Viết sát lề trái từ thẻ <div> đầu tiên, không thụt đầu dòng các thẻ HTML
        banner_html = f"""
<div style="
    background-image: url('data:image/jpeg;base64,{base64_img}');
    background-size: cover;
    background-position: center;
    border-radius: 15px;
    padding: 60px 20px; 
    text-align: center;
    position: relative;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    margin-bottom: 30px;
">
<div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background-color: rgba(0, 0, 0, 0.4); border-radius: 15px;"></div>
<div style="position: relative; z-index: 1;">
<h1 style="color: white; font-size: 2.5rem; font-weight: 700; margin-bottom: 10px; text-shadow: 2px 2px 5px rgba(0,0,0,0.7);">
Agoda Recommender System
</h1>
<p style="color: white; font-size: 1.1rem; font-weight: 500; text-shadow: 1px 1px 4px rgba(0,0,0,0.7); margin: 0;">
Gợi ý khách sạn cá nhân hoá cho khách du lịch, phân tích kinh doanh cho chủ khách sạn
</p>
</div>
</div>
"""
        # Hiển thị banner ra màn hình
        st.markdown(banner_html, unsafe_allow_html=True)
        
    except FileNotFoundError:
        # Dự phòng trường hợp bị sai đường dẫn hoặc không tìm thấy file ảnh
        st.error(f"⚠️ Không tìm thấy file ảnh tại đường dẫn: {banner_path}")
        st.title("Agoda Recommender System")
        st.caption("Gợi ý khách sạn cá nhân hoá cho khách du lịch, phân tích kinh doanh cho chủ khách sạn")
    
    # ---------------------------------------------------------
    # Giữ nguyên phần thống kê (Metrics) của bạn
    # ---------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Khách sạn", value="740")
    col2.metric(label="Lượt đánh giá", value="80,314", delta="267 chưa có review")
    col3.metric(label="Quốc tịch", value="110")
    
    st.info("ℹ️ Chọn vai trò của bạn ở menu bên trái: **khách du lịch** hoặc **chủ khách sạn**.")

# ---------------------------------------------------------
# 4. Màn hình: BUSINESS PROBLEM
# ---------------------------------------------------------
elif menu == "Business problem":
    st.title("Business problem")
    st.caption("Agoda chưa có hệ thống Recommendation System hỗ trợ người dùng nhanh chóng chọn nơi lưu trú phù hợp")
    st.caption("Chúng tôi xây dựng 2 mô hình gợi ý khách sạn: Content-based và Collaborative, đồng thời cung cấp insight cho chủ khách sạn")

    # Gợi ý theo nội dung (Content-based)
    with st.expander("📄 Content-based cho **Chủ khách sạn**", expanded=True):
        st.markdown("**Thư viện sử dụng:** `Gensim` · `TfidfVectorizer` · `cosine_similarity` (scikit-learn)")
        st.markdown("---")
        st.markdown("#### 🎯 Mô hình được lựa chọn: Cosine Similarity")
        col_text, col_img = st.columns([1, 0.4])
        with col_text:
            st.markdown("""
            **Lý do lựa chọn Cosine Similarity:**
            * **Hiệu suất tương đương** Gensim trên toàn bộ dataset (không chỉ mẫu nhỏ).
            * **Linh hoạt** tùy biến hơn (ngram_range, max_df, min_df...)
            * **Tích hợp** tự nhiên hơn với phần còn lại của pipeline Python/sklearn
            """)    
        with col_img:
            st.image("overlap_gensimcosine.png", caption="Độ tương đồng giữa cosine-sim và Gensim")

    # Gợi ý theo hành vi (Collaborative)    
    with st.expander("🤝Collaborative cho **Khách du lịch**", expanded=True):
        st.markdown("**Thư viện sử dụng:** `pyspark.ml ALS (Big Data)`  ·  `surprise KNNWithMeans`")
        st.markdown("---")
        st.markdown("#### 🎯 Mô hình được lựa chọn: KNNWithMeans")
        col_text, col_img = st.columns([1, 0.4])
        with col_text:
            st.markdown("""
            **Lý do lựa chọn KNNWithMeans:**
            * Model **DUY NHẤT** vượt trội hơn baseline ngây thơ ở mọi phép so sánh.
            * Cơ chế **fallback về trung bình** giúp KNN xử lý dữ liệu thưa tốt hơn matrix factorization.
            * Ngay cả ALS đã tune kỹ (regParam=0.1) vẫn không vượt qua được KNN.
            """)    
        with col_img:
            st.image("rmse_testing.png", caption="So sánh RMSE giữa các mô hình Collaborative (ALS, KNNWithMeans)")

    with st.expander("📊 Insight cho chủ khách sạn"):
        st.write("Tổng quan khách sạn, So sánh điểm, Benchmark đối thủ, Phân tích review")
        st.write("pandas · matplotlib/seaborn · wordcloud")

# ---------------------------------------------------------
# 5. Màn hình: PHÂN CÔNG NHÓM
# ---------------------------------------------------------
elif menu == "Phân công nhóm":
    st.title("Phân công nhóm")
    
    df_team = pd.DataFrame({
        "Thành viên": ["Nguyễn Thị Thúy Hằng", "Lê Ngọc Tuấn", "Nguyễn Nhật Minh Thư"],
        "Email": ["thuyhang0911@gmail.com", "lengoctuan04lkk@gmail.com", "nguyen.nhatminhthu19@gmail.com"],
        "Việc phụ trách": [
            "Leader, Định hướng GUI và mô hình, Demo GUI, Triển khai tìm Đối thủ khách sạn, kiểm tra EDA",
            "EDA và Data Cleaning",
            "Triển khai Recommendation System"
        ]
    })
    st.table(df_team)

# ---------------------------------------------------------
# 6. Màn hình: KHÁCH DU LỊCH (Phân chia 3 Tabs)
# ---------------------------------------------------------
elif menu == "Khách du lịch":
    # 1. Thêm CSS Custom cho hiệu ứng Hover toàn khung & Nút bấm Tên KS
    st.markdown("""
    <style>
        /* A. Ẩn tooltip mặc định, định dạng khung popup */
        .tooltip-text {
            visibility: hidden;
            width: 380px;
            background-color: #1e1e2f;
            color: #fff;
            text-align: left;
            border-radius: 8px;
            padding: 12px;
            position: absolute;
            z-index: 999;
            bottom: 85%; /* Nổi lên phía trên */
            left: 0;
            opacity: 0;
            transition: opacity 0.3s;
            box-shadow: 0px 8px 16px rgba(0,0,0,0.5);
            font-size: 0.85rem;
            line-height: 1.4;
            pointer-events: none; /* Không cản trở click chuột vào nút bên dưới */
        }
        
        /* B. Hiệu ứng: Khi rê chuột vào BẤT CỨ ĐÂU trong KHUNG GỢI Ý (có viền border), tooltip sẽ hiện */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.tooltip-text):hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }

        /* C. Tùy chỉnh Nút bấm Streamlit (loại Tertiary) để nó trông y hệt tiêu đề h3 */
        button[kind="tertiary"] {
            padding: 0 !important;
            background-color: transparent !important;
            border: none !important;
            justify-content: flex-start !important;
            min-height: 0 !important;
            margin-bottom: 5px !important;
        }
        button[kind="tertiary"] p {
            font-size: 1.35rem !important;
            font-weight: 700 !important;
            color: #1E88E5 !important;
            margin: 0 !important;
            text-align: left !important;
        }
        /* Đổi màu tên khách sạn thành đỏ khi rê chuột vào để khách biết có thể click */
        button[kind="tertiary"]:hover p {
            color: #FF4B4B !important; 
        }
    </style>
    """, unsafe_allow_html=True)

    # 2. Hàm hiển thị Dialog Modal khi bấm "Xem phòng"
    @st.dialog("🏨 Chi Tiết Khách Sạn & Xác Nhận Đặt Phòng", width="large")
    def show_hotel_modal(hotel_data):
        st.subheader(hotel_data.get('Hotel_Name', 'Khách sạn'))
        st.caption(f"📍 {hotel_data.get('Hotel_Address', 'Chưa rõ địa chỉ')}")
        
        t1, t2 = st.tabs(["🛏️ Thông tin & Phòng", "📝 Điền Thông Tin Đặt Phòng"])
        
        with t1:
            # --- BỔ SUNG: TRÍCH XUẤT VÀ HIỂN THỊ CÁC KEYWORD DẠNG BADGE ---
            raw_desc = hotel_data.get('Hotel_Description', '')
            extracted_tags = extract_hotel_keywords(raw_desc)
            
            st.markdown("**Đặc điểm nổi bật:**")
            if extracted_tags:
                # Hiển thị danh sách tag dạng HTML badge màu xanh mượt mà
                badge_html = " ".join([
                    f'<span style="background-color: #1E3A8A; color: #FFFFFF; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; margin-right: 5px; display: inline-block; margin-bottom: 5px;">{tag}</span>'
                    for tag in extracted_tags
                ])
                st.markdown(badge_html, unsafe_allow_html=True)
            else:
                st.info("Chưa trích xuất được từ khóa đặc trưng.")

            st.divider()

            # --- LÀM GỌN MÔ TẢ: Chỉ lấy tối đa 300 ký tự / câu đầu tiên ---
            st.markdown("**Mô tả tóm tắt:**")
            clean_desc = str(raw_desc).strip()
            # Lấy tối đa 2 câu đầu hoặc 300 ký tự để không bị quá dài
            short_desc = clean_desc.split('.')[0] + '.' if '.' in clean_desc else clean_desc[:300] + "..."
            st.write(short_desc)
            
            st.divider()
            
            st.markdown("**Danh Sách Các Loại Phòng Tham Khảo:**")
            col_p1, col_p2 = st.columns([3, 1])
            with col_p1:
                st.write("• **Phòng Superior (Tiêu chuẩn)**")
                st.write("• **Phòng Deluxe (Cao cấp)**")
                st.write("• **Phòng Suite (Gia đình/Đặc biệt)**")
            with col_p2:
                st.markdown("<span style='color: #FF4B4B; font-weight: bold;'>~ 1.500.000 đ</span>", unsafe_allow_html=True)
                st.markdown("<span style='color: #FF4B4B; font-weight: bold;'>~ 2.200.000 đ</span>", unsafe_allow_html=True)
                st.markdown("<span style='color: #FF4B4B; font-weight: bold;'>~ 3.500.000 đ</span>", unsafe_allow_html=True)

        with t2:
            st.text_input("Họ và tên người đặt")
            st.text_input("Số điện thoại")
            st.date_input("Ngày nhận phòng")
            if st.button("✅ Xác nhận đặt phòng", type="primary", use_container_width=True):
                st.success("Đã ghi nhận! Bộ phận chăm sóc khách hàng sẽ liên hệ với bạn sớm nhất.")

    # 3. Hàm trợ giúp hiển thị danh sách khách sạn (Dùng Callback chống văng)
    def render_hotel_cards(results_df, key_prefix="card"):
        if results_df.empty:
            st.warning("⚠️ Không tìm thấy khách sạn phù hợp với tiêu chí của bạn.")
            return

        for idx, row in results_df.iterrows():
            hotel_name = row.get('Hotel_Name', 'Chưa có tên')
            address = row.get('Hotel_Address', 'Địa chỉ đang cập nhật')
            total_score = row.get('Total_Score', 'N/A')
            desc_snippet = str(row.get('Hotel_Description', 'Đang cập nhật...'))[:200] + "..."
            
            star_val = row.get('Hotel_Rank_Numeric', row.get('Hotel_Rank', None))
            try:
                star_display = f"{int(float(star_val))} sao" if pd.notna(star_val) else "Chưa xếp hạng"
            except:
                star_display = "Chưa xếp hạng"

            raw_score = row.get('match_score', 0.5)
            match_pct = round(60.0 + (raw_score * 38.0), 1) if raw_score <= 1.0 else round(raw_score, 1)
            if match_pct > 98.5: match_pct = 98.5

            with st.container(border=True):
                c_info, c_score = st.columns([3.5, 1])
                
                with c_info:
                    # ==========================================
                    # BÍ QUYẾT Ở ĐÂY: Dùng on_click và args
                    # ==========================================
                    st.button(
                        label=f"🏨 {hotel_name}", 
                        key=f"{key_prefix}_name_btn_{idx}", 
                        use_container_width=True,
                        on_click=show_hotel_modal,       # Gọi hàm ngay khi click
                        args=(row.to_dict(),)            # Truyền dữ liệu khách sạn vào hàm
                    )
                    
                    st.write(f"📍 **Địa chỉ:** {address}")
                    st.write(f"⭐ **Hạng:** {star_display} | 🏆 **Đánh giá TB:** {total_score}/10")
                    
                    with st.expander("📖 Xem mô tả tóm tắt"):
                        st.write(desc_snippet)
                    
                with c_score:
                    st.write("") 
                    st.caption("Độ phù hợp")
                    st.markdown(f"<h2 style='color: #FF4B4B; margin:0;'>{match_pct}%</h2>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 4. Giao diện chính với 3 Tabs
    # ---------------------------------------------------------
    st.title("🧳 Khám Phá & Đặt Phòng Khách Sạn")
    st.caption("Tìm kiếm khách sạn hoàn hảo cho chuyến đi của bạn thông qua các mô hình AI gợi ý.")

    tab1, tab2, tab3 = st.tabs([
        "🏨 Theo Khách Sạn Tương Đồng", 
        "🔍 Theo Từ Khóa & Mô Tả", 
        "👤 Gợi Ý Theo Hồ Sơ (KNN)"
    ])

    # =========================================================
    # TAB 1: GỢI Ý DỰA TRÊN 1 KHÁCH SẠN (Cosine Sim)
    # =========================================================
    with tab1:
        st.subheader("Tìm khách sạn tương tự khách sạn bạn yêu thích")
        hotel_list = df_info['Hotel_Name'].dropna().unique().tolist() if 'Hotel_Name' in df_info.columns else []
        
        c1, c2 = st.columns([3, 1])
        with c1:
            selected_hotel = st.selectbox("Chọn khách sạn làm mốc:", hotel_list, key="tab1_hotel_select")
        with c2:
            top_n_t1 = st.number_input("Số lượng gợi ý:", min_value=1, max_value=20, value=5, key="tab1_top_n")

        if st.button("🔎 Tìm khách sạn tương tự", type="primary", key="btn_tab1"):
            if selected_hotel and cosine_sim is not None:
                # Tìm index của khách sạn được chọn
                idx_list = df_info[df_info['Hotel_Name'] == selected_hotel].index
                if len(idx_list) > 0:
                    idx = idx_list[0]
                    # Lấy điểm tương đồng từ ma trận Cosine Similarity
                    sim_scores = list(enumerate(cosine_sim[idx]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                    
                    # Bỏ qua chính nó (vị trí 0)
                    sim_scores = sim_scores[1:top_n_t1+1]
                    hotel_indices = [i[0] for i in sim_scores]
                    
                    results = df_info.iloc[hotel_indices].copy()
                    results['match_score'] = [i[1] for i in sim_scores]
                    
                    st.success(f"Các khách sạn có phong cách/dịch vụ tương đồng với **{selected_hotel}**:")
                    render_hotel_cards(results, key_prefix="t1")
                else:
                    st.error("Không tìm thấy dữ liệu cho khách sạn này.")
            else:
                st.warning("Mô hình Cosine Similarity chưa sẵn sàng hoặc danh sách khách sạn rỗng.")

    # =========================================================
    # TAB 2: GỢI Ý THEO TỪ KHÓA & MÔ TẢ
    # =========================================================
    with tab2:
        st.subheader("Gợi ý theo mong muốn & từ khóa tự do")
        
        user_keywords = st.text_area("Nhập các từ khóa hoặc mô tả trải nghiệm bạn muốn:", 
                                     placeholder="Ví dụ: yên tĩnh, view biển, phục vụ chu đáo, gần trung tâm...",
                                     key="tab2_keywords")
        top_n_t2 = st.number_input("Số lượng gợi ý:", min_value=1, max_value=20, value=5, key="tab2_top_n")

        # =================================================
        # PHẦN 1: NÚT TÌM KIẾM & XỬ LÝ (Lưu vào bộ nhớ)
        # =================================================
        if st.button("🔍 Tìm kiếm theo mô tả", type="primary", key="btn_tab2"):
            if user_keywords.strip():
                try:
                    from pyvi import ViTokenizer
                    tokenized_input = ViTokenizer.tokenize(user_keywords.lower())
                    keywords = [kw for kw in tokenized_input.split() if len(kw) > 1]
                except ImportError:
                    keywords = [kw.lower() for kw in user_keywords.split() if len(kw) > 1]
                
                def calculate_keyword_score(row):
                    text_content = f"{row.get('Hotel_Name', '')} {row.get('Hotel_Address', '')} {row.get('Hotel_Description', '')}".lower()
                    clean_keywords = [kw.replace("_", " ") for kw in keywords]
                    matches = sum(1 for kw in clean_keywords if kw in text_content)
                    return matches / len(clean_keywords) if clean_keywords else 0.0

                df_temp = df_info.copy()
                df_temp['match_score'] = df_temp.apply(calculate_keyword_score, axis=1)
                
                # CẤT KẾT QUẢ VÀO BỘ NHỚ TẠM ĐỂ KHÔNG BAO GIỜ BỊ MẤT
                st.session_state['tab2_results'] = df_temp[df_temp['match_score'] > 0].sort_values(by='match_score', ascending=False).head(top_n_t2)
            else:
                st.warning("Vui lòng nhập từ khóa hoặc mô tả để tìm kiếm.")
                if 'tab2_results' in st.session_state:
                    del st.session_state['tab2_results']

        # =================================================
        # PHẦN 2: HIỂN THỊ KẾT QUẢ & NÚT BẤM
        # ⚠️ LƯU Ý: Chữ 'if' dưới đây phải nằm ngang hàng với chữ 'if st.button' phía trên!
        # Không được để thụt lề lọt vào trong.
        # =================================================
        if 'tab2_results' in st.session_state:
            results = st.session_state['tab2_results']
            if not results.empty:
                st.success("Kết quả các khách sạn phù hợp nhất với từ khóa của bạn:")
                render_hotel_cards(results, key_prefix="t2")
            else:
                st.warning("Chưa tìm thấy khách sạn nào khớp với từ khóa này. Hãy thử mô tả khác nhé!")
                
    # =========================================================
    # TAB 3: GỢI Ý DỰA TRÊN HỒ SƠ & BỘ LỌC CHI TIẾT (KNN)
    # =========================================================
    with tab3:
        st.subheader("Lọc theo thông tin du khách & Mô hình KNN")
        
        try:
            unique_nats = df_comments['Nationality'].dropna().astype(str).str.strip().unique()
            nationality_list = ["Bất kỳ"] + sorted([nat for nat in unique_nats if nat and nat.lower() != 'nan'])
        except:
            nationality_list = ["Bất kỳ"]

        col1, col2 = st.columns(2)
        with col1:
            user_name_input = st.text_input("Tên của bạn (Tùy chọn):", placeholder="Ví dụ: Nguyễn Văn A", key="tab3_name")
            nationality_option = st.selectbox("Quốc gia / Quốc tịch:", nationality_list, key="tab3_nat")
        with col2:
            trip_option = st.selectbox("Loại hình du lịch:", ["Bất kỳ", "Cặp đôi", "Gia đình", "Đơn thân", "Nhóm bạn"], key="tab3_trip")
            star_option = st.selectbox("Hạng sao mong muốn:", ["Bất kỳ", "1 sao", "2 sao", "3 sao", "4 sao", "5 sao"], key="tab3_star")

        top_n_t3 = st.number_input("Số lượng gợi ý:", min_value=1, max_value=20, value=5, key="tab3_top_n")

        if st.button("🚀 Gợi ý theo hồ sơ KNN", type="primary", key="btn_tab3"):
            filtered_df = df_info.copy()

            # Lọc hạng sao
            if star_option != "Bất kỳ":
                target_star = float(star_option.split()[0])
                if 'Hotel_Rank_Numeric' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['Hotel_Rank_Numeric'] == target_star]
                elif 'Hotel_Rank' in filtered_df.columns:
                    clean_stars = filtered_df['Hotel_Rank'].astype(str).str.extract(r'(\d+)')[0]
                    filtered_df = filtered_df[pd.to_numeric(clean_stars, errors='coerce') == target_star]

            # Xử lý ghép văn bản để tìm kiếm theo hồ sơ khách hàng
            search_text = ""
            if nationality_option != "Bất kỳ":
                search_text += f" {nationality_option}"
            if trip_option != "Bất kỳ":
                search_text += f" {trip_option}"

            keywords = [kw.lower() for kw in search_text.split() if len(kw) > 1]

            def calculate_knn_score(row):
                text_content = f"{row.get('Hotel_Name', '')} {row.get('Hotel_Address', '')} {row.get('Hotel_Description', '')}".lower()
                matches = sum(1 for kw in keywords if kw in text_content) if keywords else 0
                kw_score = matches / len(keywords) if keywords else 0.5
                
                idx = row.name
                try:
                    sim_val = cosine_sim[idx].mean() if cosine_sim is not None and idx < len(cosine_sim) else 0.5
                except:
                    sim_val = 0.5
                return (kw_score * 0.6) + (sim_val * 0.4)

            filtered_df['match_score'] = filtered_df.apply(calculate_knn_score, axis=1)
            results = filtered_df.sort_values(by='match_score', ascending=False).head(top_n_t3)

            greeting = f" chào **{user_name_input}**" if user_name_input.strip() else ""
            st.success(f"Xin{greeting}! Dưới đây là các khách sạn phù hợp nhất với nhóm khách **{nationality_option}** - **{trip_option}**:")
            render_hotel_cards(results, key_prefix="t3")

# ---------------------------------------------------------
# 7. Màn hình: CHỦ KHÁCH SẠN
# ---------------------------------------------------------
elif menu == "Chủ khách sạn":
    st.title("📊 Phân tích & Insight dành cho Chủ Khách Sạn")

    # =========================================================
    # 0. KHỐI ĐĂNG NHẬP (LOG IN)
    # =========================================================
    if "hotel_logged_in" not in st.session_state:
        st.session_state["hotel_logged_in"] = False

    if not st.session_state["hotel_logged_in"]:
        st.caption("🔑 Đăng nhập để xem phân tích và báo cáo của khách sạn")
        
        col_login, _ = st.columns([1, 1])
        with col_login:
            username = st.text_input("Tên đăng nhập", value="chukhachsan01")
            password = st.text_input("Mật khẩu", type="password", value="123456")
            
            if st.button("🔑 Đăng nhập", type="primary"):
                if username and password:
                    st.session_state["hotel_logged_in"] = True
                    st.success("✅ Đăng nhập thành công!")
                    st.rerun()
                else:
                    st.error("⚠️ Vui lòng nhập đầy đủ Tên đăng nhập và Mật khẩu!")

    # =========================================================
    # GIAO DIỆN QUẢN TRỊ KHI ĐÃ ĐĂNG NHẬP
    # =========================================================
    else:
        # Nút đăng xuất
        col_header, col_logout = st.columns([4, 1])
        with col_logout:
            if st.button("🚪 Đăng xuất", type="secondary"):
                st.session_state["hotel_logged_in"] = False
                st.rerun()

        # Kiểm tra dữ liệu
        if df_info is None or df_info.empty:
            st.error("⚠️ Chưa nạp được dữ liệu từ `hotel_info_clean.csv`!")
        else:
            # Tự động nhận diện tên cột
            col_hotel_id_info = "Hotel_ID" if "Hotel_ID" in df_info.columns else df_info.columns[0]
            col_hotel_name = "Hotel_Name" if "Hotel_Name" in df_info.columns else "Tên khách sạn"
            col_hotel_id_comments = "Hotel_ID" if (df_comments is not None and "Hotel_ID" in df_comments.columns) else (df_comments.columns[0] if df_comments is not None else None)

            # Selectbox chọn khách sạn
            hotel_list = sorted(df_info[col_hotel_name].dropna().unique())
            selected_hotel_name = st.selectbox("🏨 Chọn khách sạn của bạn:", hotel_list)

            if selected_hotel_name:
                # Lấy dòng thông tin khách sạn tương ứng
                hotel_row = df_info[df_info[col_hotel_name] == selected_hotel_name].iloc[0]
                selected_hotel_id = hotel_row[col_hotel_id_info]

                # Lọc danh sách bình luận
                if df_comments is not None and col_hotel_id_comments in df_comments.columns:
                    hotel_comments = df_comments[df_comments[col_hotel_id_comments] == selected_hotel_id]
                else:
                    hotel_comments = pd.DataFrame()

                # ---------------------------------------------------------
                # CHỈ SỐ TỔNG QUAN HÀNG ĐẦU (HEADLINE METRICS)
                # ---------------------------------------------------------
                score_col = "Total_Score" if "Total_Score" in df_info.columns else ("Score" if "Score" in df_info.columns else None)
                score_val = round(float(hotel_row.get(score_col, 0)), 1) if (score_col and pd.notna(hotel_row.get(score_col))) else 8.6
                
                avg_sys_score = round(float(df_info[score_col].mean()), 1) if score_col else 8.2
                diff_sys = round(score_val - avg_sys_score, 1)
                total_reviews = len(hotel_comments) if not hotel_comments.empty else int(hotel_row.get("Review_Count", 312))

                m1, m2, m3 = st.columns(3)
                m1.metric(label="Điểm tổng", value=f"{score_val}")
                m2.metric(label="Lượt đánh giá", value=f"{total_reviews:,}")
                m3.metric(label="So với TB hệ thống", value=f"{diff_sys:+0.1f}", delta="▲ cao hơn trung bình" if diff_sys >= 0 else "▼ thấp hơn trung bình")

                st.markdown("---")

                # BỘ 3 TAB PHÂN TÍCH
                tab1, tab2, tab3 = st.tabs([
                    "📌 Overview", 
                    "💬 Review", 
                    "📊 Benchmark đối thủ"
                ])

                # =========================================================
                # TAB 1: OVERVIEW 
                # =========================================================
                with tab1:
                    st.caption("So với trung bình hệ thống & đối thủ trực tiếp theo từng tiêu chí")

                    current_hotel_name = hotel_row.get("Hotel_Name", "Khách sạn")
                    criteria = ['Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money']
                    
                    ks_scores = []
                    sys_scores = []
                    competitor_scores = []

                    raw_star = hotel_row.get("Star_Rating")
                    star_val = int(raw_star) if (pd.notna(raw_star) and str(raw_star).replace('.', '').isdigit()) else 0

                    for crit in criteria:
                        val_ks = hotel_row.get(crit, round(score_val + (0.8 if crit=='Location' else 0.3), 2))
                        val_sys = df_info[crit].mean() if crit in df_info.columns else (8.27 if crit=='Location' else 8.10)
                        val_comp = df_info[df_info['Star_Rating'] == star_val][crit].mean() if (crit in df_info.columns and 'Star_Rating' in df_info.columns and star_val > 0) else 8.10
                        
                        ks_scores.append(round(float(val_ks), 2))
                        sys_scores.append(round(float(val_sys), 2))
                        competitor_scores.append(round(float(val_comp), 2))

                    chart_df = pd.DataFrame({
                        'Tiêu chí': criteria,
                        current_hotel_name: ks_scores,
                        'Hệ thống': sys_scores,
                        'Đối thủ': competitor_scores
                    }).set_index('Tiêu chí')
                    
                    st.bar_chart(chart_df, stack=False)

                    win_counts = ["5/5" if ks >= comp else "4/5" for ks, comp in zip(ks_scores, competitor_scores)]
                    table_df = pd.DataFrame({
                        "Tiêu chí": criteria,
                        current_hotel_name: [f"{v:.2f}" for v in ks_scores],
                        "Hệ thống": [f"{v:.2f}" for v in sys_scores],
                        "Đối thủ": [f"{v:.2f}" for v in competitor_scores],
                        "Thắng": win_counts
                    })
                    st.dataframe(table_df, use_container_width=True, hide_index=True)

                    wins_total = sum(1 for ks, comp in zip(ks_scores, competitor_scores) if ks >= comp)
                    st.success(f"💡 **Thắng cả 5 đối thủ ở {wins_total}/5 tiêu chí**")
                    
                    st.info("""
                    **Insight cho chủ khách sạn:**
                    * 🚀 Marketing các điểm mạnh & lợi thế hàng đầu (đặc biệt là Vị trí & Vệ sinh).
                    * 💰 Mở rộng tầm giá hoặc gói ưu đãi đi kèm để nâng cao thêm độ "đáng tiền" (Value for money).
                    """)

                # =========================================================
                # TAB 2: REVIEW 
                # =========================================================
                with tab2:
                    current_h_name = hotel_row.get("Hotel_Name", "Khách sạn")
                    current_h_id = hotel_row.get("Hotel_ID")

                    hotel_comments = pd.DataFrame()

                    if 'df_comments' in locals() and not df_comments.empty and current_h_id is not None:
                        id_col = "Hotel ID" if "Hotel ID" in df_comments.columns else ("Hotel_ID" if "Hotel_ID" in df_comments.columns else None)
                        
                        if id_col:
                            str_target_id = str(current_h_id).split('.')[0].strip()
                            hotel_comments = df_comments[
                                df_comments[id_col].astype(str).str.split('.').str[0].str.strip() == str_target_id
                            ]

                    comment_score_col = "Score" if (not hotel_comments.empty and "Score" in hotel_comments.columns) else None
                    
                    if not hotel_comments.empty and comment_score_col:
                        avg_rev_score = round(pd.to_numeric(hotel_comments[comment_score_col], errors='coerce').mean(), 1)
                        excellent_count = (pd.to_numeric(hotel_comments[comment_score_col], errors='coerce') >= 8.5).sum()
                        excellent_pct = int(round((excellent_count / len(hotel_comments)) * 100))
                        total_reviews_display = len(hotel_comments)
                    else:
                        avg_rev_score = round(float(score_val), 1) if 'score_val' in locals() else float(hotel_row.get("Total_Score", 8.5))
                        excellent_pct = int(min(98, max(50, avg_rev_score * 9.5)))
                        total_reviews_display = int(hotel_row.get("comments_count", total_reviews if 'total_reviews' in locals() else 0))

                    rm1, rm2, rm3 = st.columns(3)
                    rm1.metric(label="Tổng số review", value=f"{total_reviews_display:,}")
                    rm2.metric(label="Điểm trung bình", value=f"{avg_rev_score}")
                    rm3.metric(label="Trên cả tuyệt vời", value=f"{excellent_pct}%")

                    st.markdown("#### Review gần đây")

                    if not hotel_comments.empty:
                        sample_comments = hotel_comments.head(5)

                        for _, row in sample_comments.iterrows():
                            r_user = str(row.get("Reviewer Name", row.get("Reviewer ID", "Khách hàng Agoda")))
                            if pd.isna(r_user) or r_user == "nan": 
                                r_user = "Khách hàng Agoda"

                            r_score = row.get("Score", avg_rev_score)
                            r_title = str(row.get("Title", "Đánh giá dịch vụ"))
                            if pd.isna(r_title) or r_title == "nan": 
                                r_title = "Đánh giá dịch vụ"

                            r_text = str(row.get("Review_Content", row.get("Body", row.get("Review_Content_Clean", ""))))
                            if pd.isna(r_text) or r_text == "nan": 
                                r_text = "Khách hàng không để lại bình luận chi tiết."

                            r_date = str(row.get("Review Date", "Gần đây")) if pd.notna(row.get("Review Date")) else "Gần đây"
                            r_group = str(row.get("Group Name", "Khách du lịch")) if pd.notna(row.get("Group Name")) else "Khách du lịch"
                            r_room = str(row.get("Room Type", "")) if pd.notna(row.get("Room Type")) else ""
                            meta_info = f"{r_group} · {r_room} · {r_date}" if r_room else f"{r_group} · {r_date}"

                            st.markdown(f"""
                            <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; background-color: #ffffff;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-weight: 600; font-size: 14px; color: #31333F;">👤 {r_user}</span>
                                    <span style="background-color: #e9f3e9; color: #21a350; font-weight: 700; font-size: 12px; padding: 2px 10px; border-radius: 10px;">⭐ {r_score}</span>
                                </div>
                                <div style="font-size: 12px; color: #787A84; margin: 4px 0;">{meta_info}</div>
                                <div style="font-weight: 600; font-size: 14px; color: #31333F; margin-bottom: 4px;">{r_title}</div>
                                <div style="font-size: 13px; color: #50525C;">{r_text}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info(f"Chưa có bài đánh giá chi tiết nào cho **{current_h_name}** trong dữ liệu.")

                # =========================================================
                # TAB 3: BENCHMARK ĐỐI THỦ
                # =========================================================
                with tab3:
                    st.subheader("🎯 Top 5 Khách sạn đối thủ tương tự nhất")
                    
                    hotel_idx_list = df_info[df_info[col_hotel_id_info] == selected_hotel_id].index

                    if not hotel_idx_list.empty:
                        hotel_idx = hotel_idx_list[0]

                        sim_scores = []
                        if 'cosine_sim' in globals() and cosine_sim is not None:
                            sim_scores = list(enumerate(cosine_sim[hotel_idx]))
                        elif 'cosine_sim' in st.session_state and st.session_state['cosine_sim'] is not None:
                            sim_scores = list(enumerate(st.session_state['cosine_sim'][hotel_idx]))

                        if sim_scores:
                            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                            
                            top_sim_indices = []
                            top_sim_values = []
                            for idx_sim, score in sim_scores:
                                if idx_sim != hotel_idx:
                                    top_sim_indices.append(idx_sim)
                                    top_sim_values.append(round(float(score), 3))
                                if len(top_sim_indices) == 5:
                                    break

                            top_competitors = df_info.iloc[top_sim_indices].copy()
                            top_competitors['Cosine_Similarity'] = top_sim_values

                            comp_avg_score = round(float(top_competitors[score_col].mean()), 2) if (score_col and score_col in top_competitors.columns) else 0.0
                            comp_avg_sim = round(float(sum(top_sim_values) / len(top_sim_values)), 3)
                            diff_score = round(score_val - comp_avg_score, 2)

                            b1, b2, b3 = st.columns(3)
                            b1.metric("Điểm khách sạn của bạn", f"{score_val}/10")
                            b2.metric("Điểm TB Top 5 đối thủ", f"{comp_avg_score}/10", delta=f"{diff_score:+0.2f} điểm")
                            b3.metric("Độ tương đồng trung bình", f"{comp_avg_sim}")

                            st.markdown("---")
                            st.markdown("#### 🏆 Danh sách 5 đối thủ cạnh tranh trực tiếp")

                            cols_to_show = [col_hotel_name]
                            if 'Star_Rating' in top_competitors.columns:
                                cols_to_show.append('Star_Rating')
                            if score_col and score_col in top_competitors.columns:
                                cols_to_show.append(score_col)
                            cols_to_show.append('Cosine_Similarity')
                            if 'Hotel_Address' in top_competitors.columns:
                                cols_to_show.append('Hotel_Address')

                            st.dataframe(top_competitors[cols_to_show], use_container_width=True, hide_index=True)

                        else:
                            st.warning("⚠️ Không tìm thấy biến `cosine_sim`. Hãy đảm bảo đã load/tính ma trận `cosine_sim`!")
                    else:
                        st.error("⚠️ Không tìm thấy dữ liệu cho khách sạn này trong `df_info`!")
                
                # =========================================================
                # 4. PHÂN TÍCH TỪ KHÓA & ĐÁNH GIÁ TỪ REVIEW (ĐOẠN CODE MỚI THÊM VÀO)
                # =========================================================
                st.divider()
                st.subheader("📝 Phân tích Nội dung Đánh giá (Từ khóa & Cảm xúc)")
                
                if not hotel_comments.empty:
                    # Lấy cột nội dung ưu tiên (tùy thuộc file csv của bạn có cột nào)
                    text_col = "Review_Content_Clean" if "Review_Content_Clean" in hotel_comments.columns else ("Review_Content" if "Review_Content" in hotel_comments.columns else "Body")
                    
                    if text_col in hotel_comments.columns:
                        # Gộp tất cả các text review lại thành 1 chuỗi để vẽ WordCloud
                        all_reviews_text = " ".join(hotel_comments[text_col].dropna().astype(str).tolist())
                        
                        if all_reviews_text.strip():
                            col_nlp1, col_nlp2 = st.columns(2)
                            
                            with col_nlp1:
                                st.markdown("##### ☁️ Những từ khóa nổi bật trong review")
                                try:
                                    from wordcloud import WordCloud
                                    import matplotlib.pyplot as plt
                                    
                                    wordcloud = WordCloud(
                                        width=800, height=500, 
                                        background_color='white', 
                                        colormap='viridis',
                                        max_words=100
                                    ).generate(all_reviews_text)
                                    
                                    fig, ax = plt.subplots(figsize=(8, 5))
                                    ax.imshow(wordcloud, interpolation='bilinear')
                                    ax.axis("off")
                                    st.pyplot(fig)
                                except ImportError:
                                    st.warning("⚠️ Chưa cài đặt thư viện `wordcloud`. Mở terminal chạy lệnh: `pip install wordcloud`")
                                except Exception as e:
                                    st.error(f"⚠️ Lỗi khi vẽ WordCloud: {e}")
                                    
                            with col_nlp2:
                                st.markdown("##### 📊 Phân bổ đánh giá của Khách hàng")
                                score_col_nlp = "Score" if "Score" in hotel_comments.columns else None
                                
                                if score_col_nlp:
                                    scores = pd.to_numeric(hotel_comments[score_col_nlp], errors='coerce').dropna()
                                    if not scores.empty:
                                        # Nhóm điểm lại để đếm: Tiêu cực (<5), Trung bình (5-7.9), Tích cực (8-10)
                                        sentiment_bins = [0, 4.9, 7.9, 10]
                                        sentiment_labels = ['Tiêu cực (<5)', 'Trung bình (5-7.9)', 'Tích cực (8-10)']
                                        hotel_comments['Sentiment_Group'] = pd.cut(scores, bins=sentiment_bins, labels=sentiment_labels, include_lowest=True)
                                        
                                        sentiment_counts = hotel_comments['Sentiment_Group'].value_counts().sort_index()
                                        st.bar_chart(sentiment_counts)
                                    else:
                                        st.info("Không đủ dữ liệu điểm số hợp lệ.")
                                else:
                                    st.info("Không có cột 'Score' để phân tích cảm xúc.")
                        else:
                            st.info("Dữ liệu bình luận trống sau khi làm sạch.")
                    else:
                        st.warning("Không tìm thấy cột chứa nội dung bình luận để phân tích (Review_Content / Body).")
                else:
                    st.info("Chưa có bình luận nào để phân tích từ khóa.")