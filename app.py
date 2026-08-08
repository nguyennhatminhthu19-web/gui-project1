import base64
import json
import os
import random
import re
import sys
from collections import Counter

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pyvi.ViTokenizer import tokenize

# ---------------------------------------------------------
# 1. Cấu hình trang & CSS Custom (Page Config)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Agoda Hotel Recommender System",
    page_icon="🏨",
    layout="wide"
)

st.markdown("""
<style>
    /* Chỉnh kích thước chữ của Radio Options trong Sidebar */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 1rem !important;
        font-weight: 400;
    }
    
    /* Tăng khoảng cách giữa các lựa chọn Menu */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        margin-bottom: 2px !important;
    }
    /* ÉP KHOẢNG CÁCH CHO ĐƯỜNG KẺ NGANG */
    [data-testid="stSidebar"] hr {
        margin-top: 10px !important;
        margin-bottom: 10px !important;
    }
    /* FOOTER SIDEBAR */
    .sidebar-footer {
        font-size: 0.85rem !important;
        line-height: 1.4 !important;
        color: #7A7A7A;
    }
    /* Tooltip Hover Khung Gợi Ý */
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
        bottom: 85%;
        left: 0;
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0px 8px 16px rgba(0,0,0,0.5);
        font-size: 0.85rem;
        line-height: 1.4;
        pointer-events: none;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.tooltip-text):hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
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
    button[kind="tertiary"]:hover p {
        color: #FF4B4B !important; 
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Cấu hình Đường dẫn & Load Dữ liệu / Model / Dictionaries
# ---------------------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "model"
FILES_DIR = "files"
FOLDER_URL = "https://drive.google.com/drive/folders/1QwNB7ZIvZxnhs9a2Nq0QymgDM-P0Zxdr"

@st.cache_data
def load_keywords_config():
    try:
        with open("keywords.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data
def load_dicts():
    try:
        with open(os.path.join(FILES_DIR, 'vietnamese-stopwords.txt'), 'r', encoding='utf-8') as f:
            stop_words = f.read().split('\n')
        
        emoji_dict = {}
        with open(os.path.join(FILES_DIR, 'emojicon.txt'), 'r', encoding='utf-8') as f:
            for line in f.read().split('\n'):
                if '\t' in line: 
                    k, v = line.split('\t')
                    emoji_dict[k] = v
                
        teen_dict = {}
        with open(os.path.join(FILES_DIR, 'teencode.txt'), 'r', encoding='utf-8') as f:
            for line in f.read().split('\n'):
                if '\t' in line: 
                    k, v = line.split('\t')
                    teen_dict[k] = v
                
        with open(os.path.join(FILES_DIR, 'wrong-word.txt'), 'r', encoding='utf-8') as f:
            wrong_lst = [w.strip() for w in f.read().split('\n') if w.strip() != '']
        return stop_words, emoji_dict, teen_dict, wrong_lst
    except FileNotFoundError:
        return [], {}, {}, []

STOP_WORDS, EMOJI_DICT, TEEN_DICT, WRONG_LST = load_dicts()

@st.cache_resource
def download_models_from_drive():
    os.makedirs(MODEL_DIR, exist_ok=True)
    path_knn = os.path.join(MODEL_DIR, "knn_model.pkl")
    path_cosine = os.path.join(MODEL_DIR, "cosine_sim.pkl")
    if not (os.path.exists(path_knn) and os.path.exists(path_cosine)):
        try:
            import gdown
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
    download_models_from_drive()
    knn_model = None
    cosine_sim = None
    path_knn = os.path.join(MODEL_DIR, "knn_model.pkl")
    path_cosine = os.path.join(MODEL_DIR, "cosine_sim.pkl")
    
    if os.path.exists(path_knn):
        try:
            knn_model = joblib.load(path_knn)
        except Exception:
            try:
                import pickle
                knn_model = pickle.load(open(path_knn, 'rb'))
            except Exception as e:
                st.warning(f"⚠️ Không thể load `knn_model.pkl`: {e}")
                
    if os.path.exists(path_cosine):
        try:
            import pickle
            cosine_sim = pickle.load(open(path_cosine, 'rb'))
        except Exception as e:
            st.warning(f"⚠️ Không thể load `cosine_sim.pkl`: {e}")
            
    return knn_model, cosine_sim

df_info, df_comments = load_all_data()
knn_model, cosine_sim = load_all_models()

# ---------------------------------------------------------
# 3. Các hàm Xử lý Chuỗi & Nghiệp vụ (Helper Functions)
# ---------------------------------------------------------
def extract_hotel_keywords(description_text):
    if not description_text or pd.isna(description_text):
        return []
    
    config = load_keywords_config()
    desc_lower = str(description_text).lower()
    found_tags = []
    for cat_key, cat_data in config.items():
        icon = cat_data.get("icon", "🏷️")
        keywords = cat_data.get("keywords", [])
        for kw in keywords:
            if kw in desc_lower:
                tag_label = f"{icon} {kw.title()}"
                if tag_label not in found_tags:
                    found_tags.append(tag_label)
    return found_tags

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

def get_keyword_analysis(df_hotel_reviews, top_n=15):
    if 'Score' not in df_hotel_reviews.columns:
        return {}, {}
    positive = df_hotel_reviews[df_hotel_reviews['Score'] >= 8]['Review_Content_Clean']
    negative = df_hotel_reviews[df_hotel_reviews['Score'] <= 5]['Review_Content_Clean']
    pos_words = Counter(' '.join(positive.dropna()).split()).most_common(top_n)
    neg_words = Counter(' '.join(negative.dropna()).split()).most_common(top_n)
    return dict(pos_words), dict(neg_words)

@st.cache_data
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

def render_banner(title, subtitle):
    banner_path = os.path.join("Hotel", "Banner.jpg")
    base64_img = get_base64_image(banner_path)
    if base64_img:
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
        {title} 
        </h1>
        <p style="color: white; font-size: 1.1rem; font-weight: 500; text-shadow: 1px 1px 4px rgba(0,0,0,0.7); margin: 0;">
        {subtitle}
        </p>
        </div>
        </div>
        """
        st.markdown(banner_html, unsafe_allow_html=True)
    else:
        st.title(title)
        st.caption(subtitle)

@st.dialog("🏨 Chi Tiết Khách Sạn & Xác Nhận Đặt Phòng", width="large")
def show_hotel_modal(hotel_data):
    st.subheader(hotel_data.get('Hotel_Name', 'Khách sạn'))
    st.caption(f"📍 {hotel_data.get('Hotel_Address', 'Chưa rõ địa chỉ')}")
    
    t1, t2 = st.tabs(["🛏️ Thông tin & Phòng", "📝 Điền Thông Tin Đặt Phòng"])
    
    with t1:
        raw_desc = hotel_data.get('Hotel_Description', '')
        extracted_tags = extract_hotel_keywords(raw_desc)
        
        st.markdown("**Đặc điểm nổi bật:**")
        if extracted_tags:
            badge_html = " ".join([
                f'<span style="background-color: #1E3A8A; color: #FFFFFF; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; margin-right: 5px; display: inline-block; margin-bottom: 5px;">{tag}</span>'
                for tag in extracted_tags
            ])
            st.markdown(badge_html, unsafe_allow_html=True)
        else:
            st.info("Chưa trích xuất được từ khóa đặc trưng.")
        st.divider()
        st.markdown("**Mô tả tóm tắt:**")
        clean_desc = str(raw_desc).strip()
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
        except Exception:
            star_display = "Chưa xếp hạng"
            
        raw_score = row.get('match_score', 0.5)
        match_pct = round(60.0 + (raw_score * 38.0), 1) if raw_score <= 1.0 else round(raw_score, 1)
        if match_pct > 98.5: match_pct = 98.5
        
        img_id = random.Random(hotel_name).randint(1, 20)
        img_path = os.path.join("Hotel", f"H{img_id}.jpg")
        
        with st.container(border=True):
            c_img, c_info, c_score = st.columns([1.2, 3, 1])
            
            with c_img:
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.caption("🏨 Agoda Hotel")
            
            with c_info:
                st.button(
                    label=f"🏨 {hotel_name}",
                    key=f"{key_prefix}_name_btn_{idx}",
                    use_container_width=True,
                    on_click=show_hotel_modal,
                    args=(row.to_dict(),)
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
# 4. Thanh điều hướng bên trái (Sidebar Navigation)
# ---------------------------------------------------------
st.sidebar.title("🏨 Hotel Recommender")
menu = st.sidebar.radio(
    "Menu điều hướng",
    ["Trang chủ", "Business problem", "Phân công nhóm", "Khách du lịch", "Chủ khách sạn"]
)

if knn_model is not None and cosine_sim is not None:
    st.sidebar.caption("🟢 *Mô hình đã sẵn sàng cho gợi ý*")
else:
    st.sidebar.caption("🔴 *Mô hình chưa sẵn sàng, vui lòng kiểm tra lại*")
st.sidebar.markdown("---")

st.sidebar.markdown("""
<div class="sidebar-footer">
    ĐATN Data Science - Trung tâm tin học - Trường ĐHKHTN, ĐHQG-HCM<br><br>
    <b>Project:</b> Recommender Systems<br><br>
    <b>Nhóm thực hiện:</b><br>
    Nguyễn Thị Thúy Hằng<br>thuyhang0911@gmail.com<br>
    Nguyễn Nhật Minh Thư<br>nguyen.nhatminhthu19@gmail.com<br>
    Lê Ngọc Tuấn<br>lengoctuan04lkk@gmail.com<br><br>
    <b>Giảng viên hướng dẫn:</b><br>
    Cô Khuất Thùy Phương
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. Router Các Trang
# ---------------------------------------------------------
if menu == "Trang chủ":
    render_banner(
        title="Agoda Recommender System", 
        subtitle="Gợi ý khách sạn cá nhân hoá cho khách du lịch, phân tích kinh doanh cho chủ khách sạn"
    )
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Khách sạn", value="740")
    col2.metric(label="Lượt đánh giá", value="80,314", delta="267 chưa có review")
    col3.metric(label="Quốc tịch", value="110")
    st.markdown("#### Về **Agoda**")
    st.markdown("---")
    col_text, col_img = st.columns([1, 0.4])
    with col_text:
        st.markdown("Nền tảng đặt phòng trực tuyến có trụ sở tại Singapore (2005), thuộc Booking Holdings Inc. \nCung cấp dịch vụ đặt khách sạn, căn hộ, resort trên toàn cầu, cho phép người dùng tìm kiếm – so sánh – đặt chỗ với giá ưu đãi.")
    with col_img:
        if os.path.exists("Agoda_transparent_logo.png"):
            st.image("Agoda_transparent_logo.png", width=250)
    st.info("ℹ️ Chọn vai trò của bạn ở menu bên trái: **khách du lịch** hoặc **chủ khách sạn**.")

elif menu == "Business problem":
    render_banner(
        title="Business problem", 
        subtitle="Agoda chưa có hệ thống Recommendation System hỗ trợ người dùng nhanh chóng chọn nơi lưu trú phù hợp"
    )
    st.markdown("Chúng tôi xây dựng 2 mô hình gợi ý khách sạn: Content-based và Collaborative, đồng thời cung cấp insight cho chủ khách sạn")
    
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
            if os.path.exists("overlap_gensimcosine.png"):
                st.image("overlap_gensimcosine.png", caption="Độ tương đồng giữa cosine-sim và Gensim")
            
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
            if os.path.exists("rmse_testing.png"):
                st.image("rmse_testing.png", caption="So sánh RMSE giữa các mô hình Collaborative (ALS, KNNWithMeans)")
                
    with st.expander("📊 Insight cho chủ khách sạn"):
        if os.path.exists("business_insights.png"):
            st.image("business_insights.png", caption="Nội dung insight cho chủ khách sạn")

elif menu == "Phân công nhóm":
    render_banner(
        title="Phân công nhóm", 
        subtitle="Thông tin các thành viên thực hiện dự án và phân công nhiệm vụ"
    )
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

elif menu == "Khách du lịch":
    render_banner(
        title="🧳 Khám Phá & Đặt Phòng Khách Sạn", 
        subtitle="Tìm kiếm khách sạn hoàn hảo cho chuyến đi của bạn thông qua các mô hình AI gợi ý."
    )
    tab1, tab2, tab3 = st.tabs([
        "🏨 Theo Khách Sạn Tương Đồng", 
        "🔍 Theo Từ Khóa & Mô Tả", 
        "👤 Gợi Ý Theo Hồ Sơ"
    ])
    
    # TAB 1
    with tab1:
        st.subheader("Tìm khách sạn tương tự khách sạn bạn yêu thích")
        hotel_list = df_info['Hotel_Name'].dropna().unique().tolist() if df_info is not None and 'Hotel_Name' in df_info.columns else []
        
        c1, c2 = st.columns([3, 1])
        with c1:
            selected_hotel = st.selectbox("Chọn khách sạn làm mốc:", hotel_list, key="tab1_hotel_select")
        with c2:
            top_n_t1 = st.number_input("Số lượng gợi ý:", min_value=1, max_value=20, value=5, key="tab1_top_n")
            
        if st.button("🔎 Tìm khách sạn tương tự", type="primary", key="btn_tab1"):
            if selected_hotel and cosine_sim is not None and df_info is not None:
                idx_list = df_info[df_info['Hotel_Name'] == selected_hotel].index
                if len(idx_list) > 0:
                    idx = idx_list[0]
                    sim_scores = list(enumerate(cosine_sim[idx]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                    sim_scores = sim_scores[1:top_n_t1+1]
                    hotel_indices = [i[0] for i in sim_scores]
                    
                    results = df_info.iloc[hotel_indices].copy()
                    results['match_score'] = [i[1] for i in sim_scores]
                    
                    st.session_state['tab1_results'] = results
                    st.session_state['tab1_selected_hotel'] = selected_hotel
                else:
                    st.error("Không tìm thấy dữ liệu cho khách sạn này.")
            else:
                st.warning("Mô hình Cosine Similarity chưa sẵn sàng hoặc danh sách khách sạn rỗng.")
                
        if 'tab1_results' in st.session_state:
            saved_selected_hotel = st.session_state.get('tab1_selected_hotel', '')
            st.success(f"Các khách sạn có phong cách/dịch vụ tương đồng với **{saved_selected_hotel}**:")
            render_hotel_cards(st.session_state['tab1_results'], key_prefix="t1")

    # TAB 2
    with tab2:
        st.subheader("Gợi ý theo mong muốn & từ khóa tự do")
        user_keywords = st.text_area("Nhập các từ khóa hoặc mô tả trải nghiệm bạn muốn:", 
                                     placeholder="Ví dụ: yên tĩnh, view biển, phục vụ chu đáo, gần trung tâm...",
                                     key="tab2_keywords")
        top_n_t2 = st.number_input("Số lượng gợi ý:", min_value=1, max_value=20, value=5, key="tab2_top_n")
        
        if st.button("🔍 Tìm kiếm theo mô tả", type="primary", key="btn_tab2"):
            if user_keywords.strip() and df_info is not None:
                try:
                    tokenized_input = tokenize(user_keywords.lower())
                    keywords = [kw for kw in tokenized_input.split() if len(kw) > 1]
                except Exception:
                    keywords = [kw.lower() for kw in user_keywords.split() if len(kw) > 1]
                
                def calculate_keyword_score(row):
                    text_content = f"{row.get('Hotel_Name', '')} {row.get('Hotel_Address', '')} {row.get('Hotel_Description', '')}".lower()
                    clean_keywords = [kw.replace("_", " ") for kw in keywords]
                    matches = sum(1 for kw in clean_keywords if kw in text_content)
                    return matches / len(clean_keywords) if clean_keywords else 0.0
                
                df_temp = df_info.copy()
                df_temp['match_score'] = df_temp.apply(calculate_keyword_score, axis=1)
                st.session_state['tab2_results'] = df_temp[df_temp['match_score'] > 0].sort_values(by='match_score', ascending=False).head(top_n_t2)
            else:
                st.warning("Vui lòng nhập từ khóa hoặc mô tả để tìm kiếm.")
                if 'tab2_results' in st.session_state:
                    del st.session_state['tab2_results']
                    
        if 'tab2_results' in st.session_state:
            results = st.session_state['tab2_results']
            if not results.empty:
                st.success("Kết quả các khách sạn phù hợp nhất với từ khóa của bạn:")
                render_hotel_cards(results, key_prefix="t2")
            else:
                st.warning("Chưa tìm thấy khách sạn nào khớp với từ khóa này. Hãy thử mô tả khác nhé!")

    # TAB 3
    with tab3:
        st.subheader("Lọc theo thông tin du khách")
        try:
            unique_nats = df_comments['Nationality'].dropna().astype(str).str.strip().unique() if df_comments is not None else []
            nationality_list = ["Bất kỳ"] + sorted([nat for nat in unique_nats if nat and nat.lower() != 'nan'])
        except Exception:
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
            if df_info is not None:
                filtered_df = df_info.copy()
                if star_option != "Bất kỳ":
                    target_star = float(star_option.split()[0])
                    if 'Hotel_Rank_Numeric' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['Hotel_Rank_Numeric'] == target_star]
                    elif 'Hotel_Rank' in filtered_df.columns:
                        clean_stars = filtered_df['Hotel_Rank'].astype(str).str.extract(r'(\d+)')[0]
                        filtered_df = filtered_df[pd.to_numeric(clean_stars, errors='coerce') == target_star]
                
                search_text = ""
                if nationality_option != "Bất kỳ": search_text += f" {nationality_option}"
                if trip_option != "Bất kỳ": search_text += f" {trip_option}"
                keywords = [kw.lower() for kw in search_text.split() if len(kw) > 1]
                
                def calculate_knn_score(row):
                    text_content = f"{row.get('Hotel_Name', '')} {row.get('Hotel_Address', '')} {row.get('Hotel_Description', '')}".lower()
                    matches = sum(1 for kw in keywords if kw in text_content) if keywords else 0
                    kw_score = matches / len(keywords) if keywords else 0.5
                    idx = row.name
                    try:
                        sim_val = cosine_sim[idx].mean() if cosine_sim is not None and idx < len(cosine_sim) else 0.5
                    except Exception:
                        sim_val = 0.5
                    return (kw_score * 0.6) + (sim_val * 0.4)
                
                filtered_df['match_score'] = filtered_df.apply(calculate_knn_score, axis=1)
                results = filtered_df.sort_values(by='match_score', ascending=False).head(top_n_t3)
                greeting = f"**{user_name_input}**" if user_name_input.strip() else ""
                
                st.session_state['tab3_results'] = results
                st.session_state['tab3_greeting'] = greeting
                st.session_state['tab3_nat_option'] = nationality_option
                st.session_state['tab3_trip_option'] = trip_option
            
        if 'tab3_results' in st.session_state:
            res = st.session_state['tab3_results']
            greet = st.session_state['tab3_greeting']
            nat = st.session_state['tab3_nat_option']
            trip = st.session_state['tab3_trip_option']
            
            if not res.empty:
                st.success(f"Xin chào {greet}! Dưới đây là các khách sạn phù hợp nhất với nhóm khách **{nat}** - **{trip}**:")
                render_hotel_cards(res, key_prefix="t3")
            else:
                st.warning("Không tìm thấy khách sạn nào phù hợp với bộ lọc hồ sơ của bạn.")

elif menu == "Chủ khách sạn":
    render_banner(
        title="📊 Phân tích & Insight dành cho Chủ Khách Sạn", 
        subtitle="Xem tổng quan, benchmark đối thủ, phân tích review và các chỉ số kinh doanh của khách sạn bạn quản lý."
    )
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
    else:
        col_header, col_logout = st.columns([4, 1])
        with col_logout:
            if st.button("🚪 Đăng xuất", type="secondary"):
                st.session_state["hotel_logged_in"] = False
                st.rerun()
                
        if df_info is None or df_info.empty:
            st.error("⚠️ Chưa nạp được dữ liệu từ `hotel_info_clean.csv`!")
        else:
            col_hotel_id_info = "Hotel_ID" if "Hotel_ID" in df_info.columns else df_info.columns[0]
            col_hotel_name = "Hotel_Name" if "Hotel_Name" in df_info.columns else "Tên khách sạn"
            
            if df_comments is not None:
                if "Hotel ID" in df_comments.columns:
                    col_hotel_id_comments = "Hotel ID"
                elif "Hotel_ID" in df_comments.columns:
                    col_hotel_id_comments = "Hotel_ID"
                else:
                    col_hotel_id_comments = df_comments.columns[0]
            else:
                col_hotel_id_comments = None
                
            hotel_list = sorted(df_info[col_hotel_name].dropna().unique())
            selected_hotel_name = st.selectbox("🏨 Chọn khách sạn của bạn:", hotel_list)
            
            if selected_hotel_name:
                hotel_row = df_info[df_info[col_hotel_name] == selected_hotel_name].iloc[0]
                selected_hotel_id = hotel_row[col_hotel_id_info]
                
                if df_comments is not None and col_hotel_id_comments in df_comments.columns:
                    str_target_id = str(selected_hotel_id).split('.')[0].strip()
                    hotel_comments = df_comments[
                        df_comments[col_hotel_id_comments].astype(str).str.split('.').str[0].str.strip() == str_target_id
                    ].copy()
                else:
                    hotel_comments = pd.DataFrame()
                    
                score_col = "Total_Score" if "Total_Score" in df_info.columns else ("Score" if "Score" in df_info.columns else None)
                score_val = round(float(hotel_row.get(score_col, 0)), 1) if (score_col and pd.notna(hotel_row.get(score_col))) else 8.6
                
                avg_sys_score = round(float(df_info[score_col].mean()), 1) if score_col else 8.2
                diff_sys = round(score_val - avg_sys_score, 1)
                total_reviews = len(hotel_comments) if not hotel_comments.empty else int(hotel_row.get("Review_Count", 0))
                
                m1, m2, m3 = st.columns(3)
                m1.metric(label="Điểm tổng", value=f"{score_val}/10")
                m2.metric(label="Lượt đánh giá", value=f"{total_reviews:,}")
                m3.metric(label="So với TB hệ thống", value=f"{diff_sys:+0.1f}", delta="▲ cao hơn trung bình" if diff_sys >= 0 else "▼ thấp hơn trung bình")
                st.markdown("---")
                
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📌 Overview",  
                    "📊 Benchmark đối thủ",
                    "👥 Thống kê khách hàng",
                    "💬 Review"
                ])
                
                # OVERVIEW TAB
                with tab1:
                    st.caption("So với trung bình hệ thống & đối thủ trực tiếp theo từng tiêu chí")
                    current_hotel_name = hotel_row.get("Hotel_Name", "Khách sạn")
                    criteria = ['Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money']
                    
                    ks_scores, sys_scores, competitor_scores = [], [], []
                    raw_star = hotel_row.get("Star_Rating")
                    star_val = int(raw_star) if (pd.notna(raw_star) and str(raw_star).replace('.', '').isdigit()) else 0
                    
                    for crit in criteria:
                        val_ks = hotel_row.get(crit, round(score_val + (0.8 if crit=='Location' else 0.3), 2))
                        val_sys = df_info[crit].mean() if crit in df_info.columns else (8.27 if crit=='Location' else 8.10)
                        val_comp = df_info[df_info['Star_Rating'] == star_val][crit].mean() if (crit in df_info.columns and 'Star_Rating' in df_info.columns and star_val > 0) else 8.10
                        
                        ks_scores.append(round(float(val_ks), 2))
                        sys_scores.append(round(float(val_sys), 2))
                        competitor_scores.append(round(float(val_comp), 2))
                    
                    categories = criteria + [criteria[0]]
                    r_ks = ks_scores + [ks_scores[0]]
                    r_sys = sys_scores + [sys_scores[0]]
                    r_comp = competitor_scores + [competitor_scores[0]]
                    
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(r=r_ks, theta=categories, fill='toself', name=current_hotel_name, line_color='#eb891a'))
                    fig_radar.add_trace(go.Scatterpolar(r=r_comp, theta=categories, fill=None, name='Đối thủ TB', line_color='#508b6f'))
                    fig_radar.add_trace(go.Scatterpolar(r=r_sys, theta=categories, fill=None, name='Hệ thống TB', line_color='#4da9ff', line_dash='dot'))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            bgcolor='rgba(0,0,0,0)',
                            radialaxis=dict(visible=True, range=[0, 10], gridcolor='#333333', linecolor='#333333'),
                            angularaxis=dict(gridcolor='#333333', linecolor='#333333')
                        ),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
                        margin=dict(l=30, r=30, t=30, b=30),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white") 
                    )
                    
                    win_counts = ["5/5" if ks >= comp else "4/5" for ks, comp in zip(ks_scores, competitor_scores)]
                    table_df = pd.DataFrame({
                        "Tiêu chí": criteria,
                        current_hotel_name: [f"{v:.2f}" for v in ks_scores],
                        "Hệ thống": [f"{v:.2f}" for v in sys_scores],
                        "Đối thủ": [f"{v:.2f}" for v in competitor_scores],
                        "Thắng": win_counts
                    })
                    
                    col_chart, col_table = st.columns(2)
                    with col_chart:
                        st.markdown("**Biểu đồ Radar so sánh các tiêu chí**")
                        st.plotly_chart(fig_radar, use_container_width=True)
                    with col_table:
                        st.markdown("**Bảng điểm chi tiết**")
                        st.dataframe(table_df, use_container_width=True, hide_index=True)
                        wins_total = sum(1 for ks, comp in zip(ks_scores, competitor_scores) if ks >= comp)
                        st.success(f"💡 **Thắng cả 5 đối thủ ở {wins_total}/5 tiêu chí**")
                        
                    st.divider()
                    st.subheader("📝 Phân tích Nội dung Đánh giá (Từ khóa & Cảm xúc)")
                    
                    if not hotel_comments.empty:
                        text_col = "Review_Content_Clean" if "Review_Content_Clean" in hotel_comments.columns else ("Review_Content" if "Review_Content" in hotel_comments.columns else "Body")
                        if text_col in hotel_comments.columns:
                            all_reviews_text = " ".join(hotel_comments[text_col].dropna().astype(str).tolist())
                            if all_reviews_text.strip():
                                col_nlp1, col_nlp2 = st.columns(2)
                                with col_nlp1:
                                    st.markdown("##### ☁️ Từ khóa Tích cực & Tiêu cực")
                                    score_col_nlp = "Score" if "Score" in hotel_comments.columns else ("Reviewer_Score" if "Reviewer_Score" in hotel_comments.columns else None)
                                    if score_col_nlp:
                                        try:
                                            from wordcloud import WordCloud
                                            import matplotlib.pyplot as plt
                                            
                                            df_temp = hotel_comments.copy()
                                            df_temp['Score'] = pd.to_numeric(df_temp[score_col_nlp], errors='coerce')
                                            if 'Review_Content_Clean' not in df_temp.columns:
                                                df_temp['Review_Content_Clean'] = df_temp[text_col].fillna('').astype(str).apply(clean_review_text)
                                            
                                            pos_dict, neg_dict = get_keyword_analysis(df_temp, top_n=50)
                                            
                                            def draw_freq_wordcloud(freq_dict, colormap, title):
                                                if freq_dict:
                                                    st.markdown(f"**{title}**")
                                                    display_dict = {str(k).replace('_', ' '): v for k, v in freq_dict.items() if pd.notna(k)}
                                                    wordcloud = WordCloud(
                                                        width=600, height=300,
                                                        background_color='white',
                                                        colormap=colormap,
                                                        max_words=50
                                                    ).generate_from_frequencies(display_dict)
                                                    
                                                    fig, ax = plt.subplots(figsize=(6, 3))
                                                    ax.imshow(wordcloud, interpolation='bilinear')
                                                    ax.axis("off")
                                                    st.pyplot(fig)
                                                else:
                                                    st.info(f"Chưa đủ dữ liệu từ khóa cho {title}.")
                                                    
                                            draw_freq_wordcloud(pos_dict, 'Greens', '🟢 Điểm khen ngợi (Điểm 8-10)')
                                            st.write("")
                                            draw_freq_wordcloud(neg_dict, 'Reds', '🔴 Điểm cần cải thiện (Điểm <= 5)')
                                        except ImportError:
                                            st.warning("⚠️ Vui lòng cài đặt thư viện `wordcloud`.")
                                        except Exception as e:
                                            st.error(f"⚠️ Lỗi khi vẽ WordCloud: {e}")
                                            
                                with col_nlp2:
                                    st.markdown("##### 📊 Phân bổ đánh giá của Khách hàng")
                                    score_col_nlp = "Score" if "Score" in hotel_comments.columns else ("Reviewer_Score" if "Reviewer_Score" in hotel_comments.columns else None)
                                    if score_col_nlp:
                                        scores = pd.to_numeric(hotel_comments[score_col_nlp], errors='coerce').dropna()
                                        if not scores.empty:
                                            sentiment_bins = [0, 4.9, 7.9, 10]
                                            sentiment_labels = ['Tiêu cực (<5)', 'Trung bình (5-7.9)', 'Tích cực (8-10)']
                                            sentiment_series = pd.cut(scores, bins=sentiment_bins, labels=sentiment_labels, include_lowest=True)
                                            sentiment_counts = sentiment_series.value_counts().reindex(sentiment_labels, fill_value=0)
                                            st.bar_chart(sentiment_counts)
                                        else:
                                            st.info("Không đủ dữ liệu điểm số hợp lệ.")
                    else:
                        st.info("ℹ️ Khách sạn này hiện chưa có dữ liệu bình luận chi tiết trong hệ thống.")

                # BENCHMARK TAB
                with tab2:
                    st.subheader("🎯 Top 5 Khách sạn đối thủ tương tự nhất")
                    hotel_idx_list = df_info[df_info[col_hotel_id_info] == selected_hotel_id].index
                    if not hotel_idx_list.empty:
                        hotel_idx = hotel_idx_list[0]
                        sim_scores = []
                        if cosine_sim is not None:
                            sim_scores = list(enumerate(cosine_sim[hotel_idx]))
                        
                        if sim_scores:
                            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                            top_sim_indices, top_sim_values = [], []
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
                            if 'Star_Rating' in top_competitors.columns: cols_to_show.append('Star_Rating')
                            if score_col and score_col in top_competitors.columns: cols_to_show.append(score_col)
                            cols_to_show.append('Cosine_Similarity')
                            
                            for crit in ['Cleanliness', 'Service', 'Location', 'Facilities', 'Value_for_money']:
                                if crit in top_competitors.columns and crit not in cols_to_show:
                                    cols_to_show.append(crit)
                                    
                            st.dataframe(top_competitors[cols_to_show], use_container_width=True, hide_index=True)

                # THỐNG KÊ KHÁCH HÀNG TAB
                with tab3:
                    st.subheader("👥 Phân tích Tệp Khách Hàng & Tính Mùa Vụ")
                    
                    import plotly.express as px
                    
                    if not hotel_comments.empty:
                        # 1. Tìm đúng cột chứa thời gian (Giữ nguyên logic của bạn)
                        possible_date_cols = [c for c in hotel_comments.columns if any(k in c.lower() for k in ['date', 'ngày', 'ngay', 'time', 'created', 'review_date'])]
                        date_col = possible_date_cols[0] if possible_date_cols else None
                        
                        has_date_data = False
                        
                        if date_col:
                            # =========================================================
                            # BÊ NGUYÊN ĐOẠN CODE GỐC CỦA BẠN VÀO ĐÂY
                            # Ví dụ như đoạn thay thế chuỗi bằng regex bạn đã làm:
                            cleaned_date = hotel_comments[date_col].astype(str)\
                                .str.lower()\
                                .str.replace(r'tháng|thg|month', '/', regex=True)\
                                .str.replace(r'năm|year', '/', regex=True)\
                                .str.replace(r'đã đánh giá vào|ngày|reviewed', '', regex=True)
                            
                            # Gán lại vào cột theo đúng code cũ của bạn
                            hotel_comments['parsed_date'] = pd.to_datetime(cleaned_date, errors='coerce')
                            hotel_comments['Year'] = hotel_comments['parsed_date'].dt.year
                            # =========================================================

                            # Kiểm tra xem sau khi chạy code của bạn, đã có dữ liệu năm chưa
                            if hotel_comments['Year'].notna().any():
                                has_date_data = True

                        # --- HIỂN THỊ BIỂU ĐỒ ---
                        c1, c2, c3 = st.columns(3)
                        
                        with c1:
                            st.markdown("<p style='text-align: center; font-weight: bold;'>Reviews per Year</p>", unsafe_allow_html=True)
                            if has_date_data:
                                # Code vẽ biểu đồ đường theo dữ liệu đã parse thành công của bạn
                                yearly_counts = hotel_comments['Year'].dropna().astype(int).value_counts().sort_index().reset_index()
                                yearly_counts.columns = ['Year', 'Count']
                                
                                fig_year = px.line(yearly_counts, x='Year', y='Count', markers=True)
                                fig_year.update_traces(line_color='#eb891a', marker=dict(color='#eb891a', size=8))
                                fig_year.update_layout(
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    margin=dict(l=30, r=20, t=20, b=30), font=dict(color="white")
                                )
                                st.plotly_chart(fig_year, use_container_width=True)
                            else:
                                st.warning("⚠️ Không tìm thấy dữ liệu thời gian.")

                        # Các cột c2 (Quốc tịch) và c3 (Loại nhóm) giữ nguyên như cũ...
                        # ...
                                
                        with c2:
                            st.markdown("<p style='text-align: center; font-weight: bold;'>Top Nationalities</p>", unsafe_allow_html=True)
                            nat_cols = [c for c in hotel_comments.columns if any(k in c.lower() for k in ['national', 'quốc tịch', 'country', 'quoc_tich'])]
                            nat_col = nat_cols[0] if nat_cols else None
                            if nat_col:
                                nat_counts = hotel_comments[nat_col].value_counts().head(5).reset_index()
                                nat_counts.columns = ['Nationality', 'Count']
                                fig_nat = px.bar(nat_counts.sort_values('Count'), x='Count', y='Nationality', orientation='h')
                                fig_nat.update_traces(marker_color='#eb891a')
                                fig_nat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
                                st.plotly_chart(fig_nat, use_container_width=True)
                            else:
                                st.info("Không có dữ liệu quốc tịch.")
                                
                        with c3:
                            st.markdown("<p style='text-align: center; font-weight: bold;'>Group Type</p>", unsafe_allow_html=True)
                            group_cols = [c for c in hotel_comments.columns if any(k in c.lower() for k in ['group', 'nhóm', 'type', 'traveler'])]
                            group_col = group_cols[0] if group_cols else None
                            if group_col:
                                group_counts = hotel_comments[group_col].value_counts().head(5).reset_index()
                                group_counts.columns = ['Group', 'Count']
                                fig_group = px.bar(group_counts, x='Group', y='Count')
                                fig_group.update_traces(marker_color='#508b6f')
                                fig_group.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
                                st.plotly_chart(fig_group, use_container_width=True)
                            else:
                                st.info("Không có dữ liệu loại nhóm.")

                # REVIEW TAB
                with tab4:
                    if not hotel_comments.empty:
                        st.markdown("#### Review gần đây")
                        sample_comments = hotel_comments.head(5)
                        for _, row in sample_comments.iterrows():
                            r_user = str(row.get("Reviewer Name", row.get("Reviewer ID", "Khách hàng Agoda")))
                            r_score = row.get("Score", score_val)
                            r_title = str(row.get("Title", "Đánh giá dịch vụ"))
                            r_text = str(row.get("Review_Content", row.get("Body", "")))
                            
                            st.markdown(f"""
                            <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; background-color: #ffffff;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-weight: 600; font-size: 14px; color: #31333F;">👤 {r_user}</span>
                                    <span style="background-color: #e9f3e9; color: #21a350; font-weight: 700; font-size: 12px; padding: 2px 10px; border-radius: 10px;">⭐ {r_score}</span>
                                </div>
                                <div style="font-weight: 600; font-size: 14px; color: #31333F; margin-top: 4px;">{r_title}</div>
                                <div style="font-size: 13px; color: #50525C; margin-top: 4px;">{r_text}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Chưa có bài đánh giá chi tiết nào cho khách sạn này.")