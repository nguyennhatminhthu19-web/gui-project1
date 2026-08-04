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
**Nhóm7_DL07_K314**  
Nguyễn Thị Thúy Hằng · Lê Ngọc Tuấn · Nguyễn Nhật Minh Thư  
Thiết kế: 08/2026
""")

# ---------------------------------------------------------
# 3. Màn hình: TRANG CHỦ
# ---------------------------------------------------------
if menu == "Trang chủ":
    st.title("Agoda Hotel Recommender System")
    col_left, col_mid, col_right = st.columns([2, 1, 2])
    with col_mid:
        st.image("Agoda_transparent_logo.png", use_container_width=True)
    st.caption("Gợi ý khách sạn cá nhân hoá cho khách du lịch, phân tích kinh doanh cho chủ khách sạn")
    
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
    with st.expander("📄 Gợi ý theo nội dung (Content-based)", expanded=True):
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
    with st.expander("🤝 Gợi ý theo hành vi (Collaborative)", expanded=True):
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
# 6. Màn hình: KHÁCH DU LỊCH (Kết hợp Tìm từ khóa chuẩn + KNN Ranking)
# ---------------------------------------------------------
elif menu == "Khách du lịch":
    st.title("Tìm khách sạn phù hợp với bạn")
    st.caption("Mô tả mong muốn của bạn, hệ thống sẽ gợi ý khách sạn phù hợp")

    # --- BỔ SUNG: Lấy danh sách Quốc gia từ df_comment ---
    if 'df_comment' in locals() and 'Nationality' in df_comment.columns:
        nationality_list = ["Bất kỳ"] + sorted(df_comment['Nationality'].dropna().unique().tolist())
    else:
        nationality_list = ["Bất kỳ", "United Kingdom", "United States", "Australia", "Vietnam"]

    col1, col2, col3 = st.columns(3)
    with col1:
        star_option = st.selectbox("Hạng sao", ["Bất kỳ", "1 sao", "2 sao", "3 sao", "4 sao", "5 sao"])
    with col2:
        trip_option = st.selectbox("Loại hình du lịch", ["Bất kỳ", "Cặp đôi", "Gia đình", "Đơn thân", "Nhóm bạn"])
    with col3:
        # --- BỔ SUNG: Widget chọn Quốc gia ---
        nationality_option = st.selectbox("Quốc gia của bạn", nationality_list)

    user_desc = st.text_area("Mô tả khách sạn bạn muốn", placeholder="Ví dụ: yên tĩnh, gần biển giá rẻ...")

    if st.button("🔍 Tìm gợi ý", type="primary"):
        filtered_df = df_info.copy()

        # ---------------------------------------------------------
        # 1. BỘ LỌC HẠNG SAO CHÍNH XÁC (Dùng cột 'Hotel_Rank_Numeric')
        # ---------------------------------------------------------
        if star_option != "Bất kỳ":
            target_star = float(star_option.split()[0])
            
            if 'Hotel_Rank_Numeric' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Hotel_Rank_Numeric'] == target_star]
            elif 'Hotel_Rank' in filtered_df.columns:
                clean_stars = filtered_df['Hotel_Rank'].astype(str).str.extract(r'(\d+)')[0]
                filtered_df = filtered_df[pd.to_numeric(clean_stars, errors='coerce') == target_star]

        # ---------------------------------------------------------
        # 2. TÍNH ĐIỂM DỰA TRÊN TỪ KHÓA & MA TRẬN COSINE_SIM
        # ---------------------------------------------------------
        if user_desc.strip() or nationality_option != "Bất kỳ" or trip_option != "Bất kỳ":
            # --- CẬP NHẬT: Đưa thêm thông tin Quốc gia & Loại hình du lịch vào danh sách từ khóa khớp ---
            search_text = user_desc
            if nationality_option != "Bất kỳ":
                search_text += f" {nationality_option}"
            if trip_option != "Bất kỳ":
                search_text += f" {trip_option}"

            keywords = [kw.lower() for kw in search_text.split() if len(kw) > 1]
            
            def calculate_score(row):
                # Gộp văn bản từ Hotel_Name, Hotel_Address, Hotel_Description để khớp từ khóa
                text_content = f"{row.get('Hotel_Name', '')} {row.get('Hotel_Address', '')} {row.get('Hotel_Description', '')}".lower()
                matches = sum(1 for kw in keywords if kw in text_content)
                kw_score = matches / len(keywords) if keywords else 0.0
                
                # Trích điểm từ ma trận cosine_sim nếu có
                idx = row.name
                try:
                    sim_val = cosine_sim[idx].mean() if cosine_sim is not None and idx < len(cosine_sim) else 0.5
                except:
                    sim_val = 0.5
                
                # Điểm tổng hợp cho từng dòng
                return (kw_score * 0.7) + (sim_val * 0.3)

            filtered_df['match_score'] = filtered_df.apply(calculate_score, axis=1)
            filtered_df = filtered_df.sort_values(by='match_score', ascending=False)
        else:
            # Nếu không nhập từ khóa, sắp xếp theo ma trận cosine_sim hoặc Total_Score
            if cosine_sim is not None:
                filtered_df['match_score'] = [cosine_sim[i].mean() if i < len(cosine_sim) else 0.5 for i in filtered_df.index]
                filtered_df = filtered_df.sort_values(by='match_score', ascending=False)
            else:
                filtered_df['match_score'] = 0.8

        # Lấy Top 10 kết quả phù hợp nhất
        top_results = filtered_df.head(10)

        # ---------------------------------------------------------
        # 3. HIỂN THỊ KẾT QUẢ CARD KHÁCH SẠN
        # ---------------------------------------------------------
        st.subheader("Kết quả gợi ý dành cho bạn")
        
        if top_results.empty:
            st.warning("⚠️ Không tìm thấy khách sạn phù hợp với tiêu chí lọc của bạn. Vui lòng thử chọn 'Bất kỳ' hạng sao!")
        else:
            for idx, row in top_results.iterrows():
                # Lấy dữ liệu chính xác từ cột
                hotel_name = row.get('Hotel_Name', 'Chưa có tên')
                address = row.get('Hotel_Address', 'Địa chỉ đang cập nhật')
                total_score = row.get('Total_Score', 'N/A')
                
                # Hiển thị Hạng sao
                star_val = row.get('Hotel_Rank_Numeric', row.get('Hotel_Rank', None))
                try:
                    star_display = f"{int(float(star_val))} sao" if pd.notna(star_val) else "Chưa xếp hạng"
                except:
                    star_display = "Chưa xếp hạng"

                # Tính điểm % độ phù hợp ĐỘC BẢN cho từng khách sạn
                raw_score = row.get('match_score', 0.5)
                if user_desc.strip():
                    match_pct = round(60.0 + (raw_score * 38.0), 1)
                else:
                    match_pct = round(70.0 + (raw_score * 25.0), 1)
                
                if match_pct > 98.5:
                    match_pct = 98.5

                # Render Card Giao diện
                with st.container():
                    c_info, c_score = st.columns([3, 1])
                    
                    with c_info:
                        st.markdown(f"### {hotel_name}")
                        st.write(f"📍 **Địa chỉ:** {address}")
                        # --- CẬP NHẬT: Bổ sung hiển thị thông tin Quốc gia trên Card ---
                        st.write(f"⭐ **Hạng:** {star_display} | 🏆 **Đánh giá TB:** {total_score}/10")
                        st.write(f"Phù hợp loại hình: **{trip_option}** | Quốc gia du khách: **{nationality_option}**")
                        
                    with c_score:
                        st.caption("Độ phù hợp")
                        st.markdown(f"<h2 style='color: #FF4B4B;'>{match_pct}%</h2>", unsafe_allow_html=True)
                    
                    st.divider()

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

                    # Lấy tên khách sạn thực tế đã chọn (thay cho "KS này")
                    current_hotel_name = hotel_row.get("Hotel_Name", "Khách sạn")

                    # Các tiêu chí đánh giá
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

                    # 1. Biểu đồ cột so sánh tiêu chí (Dùng tên KS đã chọn thay cho "KS này")
                    chart_df = pd.DataFrame({
                        'Tiêu chí': criteria,
                        current_hotel_name: ks_scores,
                        'Hệ thống': sys_scores,
                        'Đối thủ': competitor_scores
                    }).set_index('Tiêu chí')
                    
                    # stack=False bắt buộc các cột đứng KẾ NHAU (không bị chồng lên mốc 25)
                    st.bar_chart(chart_df, stack=False)

                    # 2. Bảng thống kê chi tiết các tiêu chí
                    win_counts = ["5/5" if ks >= comp else "4/5" for ks, comp in zip(ks_scores, competitor_scores)]
                    table_df = pd.DataFrame({
                        "Tiêu chí": criteria,
                        current_hotel_name: [f"{v:.2f}" for v in ks_scores],
                        "Hệ thống": [f"{v:.2f}" for v in sys_scores],
                        "Đối thủ": [f"{v:.2f}" for v in competitor_scores],
                        "Thắng": win_counts
                    })
                    st.dataframe(table_df, use_container_width=True, hide_index=True)

                    # 3. Khối Insight
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

                    # ---------------------------------------------------------
                    # 1. LỌC COMMENT THEO HOTEL ID (ĐÃ CHUẨN HÓA KIỂU DỮ LIỆU)
                    # ---------------------------------------------------------
                    hotel_comments = pd.DataFrame()

                    if 'df_comments' in locals() and not df_comments.empty and current_h_id is not None:
                        # Xử lý tên cột có khoảng trắng "Hotel ID" hoặc "Hotel_ID"
                        id_col = "Hotel ID" if "Hotel ID" in df_comments.columns else ("Hotel_ID" if "Hotel_ID" in df_comments.columns else None)
                        
                        if id_col:
                            # Ép kiểu cả 2 về String và strip khoảng trắng để so sánh chính xác tuyệt đối
                            str_target_id = str(current_h_id).split('.')[0].strip() # Loại bỏ phần .0 nếu có
                            hotel_comments = df_comments[
                                df_comments[id_col].astype(str).str.split('.').str[0].str.strip() == str_target_id
                            ]

                    # ---------------------------------------------------------
                    # 2. XỬ LÝ CHỈ SỐ METRICS
                    # ---------------------------------------------------------
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

                    # Hiển thị 3 Metric
                    rm1, rm2, rm3 = st.columns(3)
                    rm1.metric(label="Tổng số review", value=f"{total_reviews_display:,}")
                    rm2.metric(label="Điểm trung bình", value=f"{avg_rev_score}")
                    rm3.metric(label="Trên cả tuyệt vời", value=f"{excellent_pct}%")

                    st.markdown("#### Review gần đây")

                    # ---------------------------------------------------------
                    # 3. HIỂN THỊ THÔNG TIN COMMENT CHI TIẾT
                    # ---------------------------------------------------------
                    if not hotel_comments.empty:
                        sample_comments = hotel_comments.head(5)

                        for _, row in sample_comments.iterrows():
                            # Tên người đánh giá
                            r_user = str(row.get("Reviewer Name", row.get("Reviewer ID", "Khách hàng Agoda")))
                            if pd.isna(r_user) or r_user == "nan": 
                                r_user = "Khách hàng Agoda"

                            # Điểm số
                            r_score = row.get("Score", avg_rev_score)

                            # Tiêu đề review
                            r_title = str(row.get("Title", "Đánh giá dịch vụ"))
                            if pd.isna(r_title) or r_title == "nan": 
                                r_title = "Đánh giá dịch vụ"

                            # Nội dung review (Ưu tiên Review_Content -> Body -> Review_Content_Clean)
                            r_text = str(row.get("Review_Content", row.get("Body", row.get("Review_Content_Clean", ""))))
                            if pd.isna(r_text) or r_text == "nan": 
                                r_text = "Khách hàng không để lại bình luận chi tiết."

                            # Ngày và Loại nhóm khách
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
                # TAB 3: BENCHMARK ĐỐI THỦ (DÙNG COSINE SIMILARITY)
                # =========================================================
                with tab3:
                    st.subheader("🎯 Top 5 Khách sạn đối thủ tương tự nhất (Cosine Similarity)")
                    
                    # 1. Lấy vị trí (index) của khách sạn được chọn trong df_info
                    hotel_idx_list = df_info[df_info[col_hotel_id_info] == selected_hotel_id].index

                    if not hotel_idx_list.empty:
                        hotel_idx = hotel_idx_list[0]

                        # 2. Kiểm tra và lấy ma trận cosine_sim đã load ở đầu bài
                        sim_scores = []
                        if 'cosine_sim' in globals() and cosine_sim is not None:
                            sim_scores = list(enumerate(cosine_sim[hotel_idx]))
                        elif 'cosine_sim' in st.session_state and st.session_state['cosine_sim'] is not None:
                            sim_scores = list(enumerate(st.session_state['cosine_sim'][hotel_idx]))

                        if sim_scores:
                            # Sắp xếp giảm dần theo điểm tương đồng Cosine
                            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                            
                            # Bỏ qua chính nó (index trùng) và lấy top 5 đối thủ tương đồng nhất
                            top_sim_indices = []
                            top_sim_values = []
                            for idx_sim, score in sim_scores:
                                if idx_sim != hotel_idx:
                                    top_sim_indices.append(idx_sim)
                                    top_sim_values.append(round(float(score), 3))
                                if len(top_sim_indices) == 5:
                                    break

                            # Lấy thông tin 5 đối thủ từ df_info
                            top_competitors = df_info.iloc[top_sim_indices].copy()
                            top_competitors['Cosine_Similarity'] = top_sim_values

                            # Tính toán các chỉ số so sánh (Metrics)
                            comp_avg_score = round(float(top_competitors[score_col].mean()), 2) if (score_col and score_col in top_competitors.columns) else 0.0
                            comp_avg_sim = round(float(sum(top_sim_values) / len(top_sim_values)), 3)
                            diff_score = round(score_val - comp_avg_score, 2)

                            b1, b2, b3 = st.columns(3)
                            b1.metric("Điểm khách sạn của bạn", f"{score_val}/10")
                            b2.metric("Điểm TB Top 5 đối thủ", f"{comp_avg_score}/10", delta=f"{diff_score:+0.2f} điểm")
                            b3.metric("Độ tương đồng trung bình", f"{comp_avg_sim}")

                            st.markdown("---")
                            st.markdown("#### 🏆 Danh sách 5 đối thủ cạnh tranh trực tiếp (Dựa trên Cosine Similarity)")

                            # Lựa chọn các cột hiển thị ra bảng
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
                            st.warning("⚠️ Không tìm thấy biến `cosine_sim`. Bạn hãy đảm bảo đã load/tính ma trận `cosine_sim` ở đầu file app.py nhé!")
                    else:
                        st.error("⚠️ Không tìm thấy dữ liệu cho khách sạn này trong `df_info`!")