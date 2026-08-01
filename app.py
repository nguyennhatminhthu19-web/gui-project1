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
        st.info("⏳ Đang tải file model từ Google Drive (lần đầu tiên có thể mất 1 - 3 phút)...")
        try:
            # Tải toàn bộ nội dung trong thư mục Drive về folder MODEL_DIR
            gdown.download_folder(url=FOLDER_URL, output=MODEL_DIR, quiet=False)
            st.success("✅ Tải model thành công!")
        except Exception as e:
            st.error(f"⚠️ Lỗi khi tải model từ Google Drive: {e}")

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

# ---------------------------------------------------------
# 2. Thanh điều hướng bên trái (Sidebar Navigation)
# ---------------------------------------------------------
st.sidebar.title("🏨 Hotel Recommender")

menu = st.sidebar.radio(
    "Menu điều hướng",
    ["Trang chủ", "Business problem", "Phân công nhóm", "Khách du lịch", "Chủ khách sạn"]
)

st.sidebar.markdown("---")
st.sidebar.caption("""
**Nhóm DL07_K314**  
ĐH KHTN Tp.HCM  
Thiết kế: 07/2026
""")

# ---------------------------------------------------------
# 3. Màn hình: TRANG CHỦ
# ---------------------------------------------------------
if menu == "Trang chủ":
    st.title("Agoda Hotel Recommender System")
    col_left, col_mid, col_right = st.columns([1, 2, 1])
    with col_mid:
        st.image("Agoda_transparent_logo.png", use_container_width=True)
    st.caption("Gợi ý khách sạn cá nhân hoá cho khách du lịch, phân tích kinh doanh cho chủ khách sạn")
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Khách sạn", value="740")
    col2.metric(label="Lượt đánh giá", value="80,314", delta="267 chưa có review")
    col3.metric(label="Quốc tịch", value="110")
    
    st.info("ℹ️ Chọn vai trò của bạn ở menu bên trái: khách du lịch hoặc chủ khách sạn.")

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
        col_text, col_img = st.columns([1, 1.2])
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
        col_text, col_img = st.columns([1, 1.2])
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
            "Leader, Định hướng GUI và mô hình, Demo GUI cả 2 project",
            "Triển khai GUI project 2",
            "Triển khai GUI project 1"
        ]
    })
    st.table(df_team)

# ---------------------------------------------------------
# 6. Màn hình: KHÁCH DU LỊCH (Kết hợp Tìm từ khóa chuẩn + KNN Ranking)
# ---------------------------------------------------------
elif menu == "Khách du lịch":
    st.title("Tìm khách sạn phù hợp với bạn")
    st.caption("Gợi ý khách sạn cá nhân hóa kết hợp lọc mong muốn và xếp hạng KNN")
    
    col_star, col_trip = st.columns(2)
    with col_star:
        star = st.selectbox("Hạng sao", ["Bất kỳ", "3 sao", "4 sao", "5 sao"])
    with col_trip:
        trip = st.selectbox("Loại hình du lịch", ["Bất kỳ", "Cặp đôi", "Gia đình có em bé", "Nhóm", "Du lịch một mình"])
        
    user_desc = st.text_area(
        "Mô tả khách sạn bạn muốn",
        placeholder="Ví dụ: biển phú quốc, đà lạt, giá rẻ..."
    )
    
    default_user_id = "101"
    
    if st.button("🔍 Tìm gợi ý", type="primary"):
        st.markdown("### Kết quả gợi ý dành cho bạn")
        
        if df_info is not None and not df_info.empty:
            filtered_df = df_info.copy()
            
            # 1. Lọc theo hạng sao
            if star != "Bất kỳ" and "Star_Rating" in filtered_df.columns:
                star_num = int(star.split()[0])
                filtered_df = filtered_df[filtered_df["Star_Rating"] == star_num]
            
            # 2. Lọc thông minh theo mô tả/địa điểm (Logic AND)
            if user_desc.strip():
                query_raw = user_desc.strip().lower()
                
                # Tạo chuỗi tìm kiếm từ Tên + Địa chỉ
                search_text = (
                    filtered_df.get("Hotel_Name", "").fillna("").astype(str) + " " + 
                    filtered_df.get("Hotel_Address", "").fillna("").astype(str)
                ).str.lower()
                
                # BƯỚC 2A: Thử tìm nguyên cụm từ trước
                mask = search_text.str.contains(query_raw, regex=False)
                
                # BƯỚC 2B: Nếu không khớp nguyên cụm, bắt buộc chứa TẤT CẢ các từ (AND Logic)
                if not mask.any():
                    words = query_raw.split()
                    if words:
                        mask = pd.Series(True, index=filtered_df.index)
                        for w in words:
                            mask = mask & search_text.str.contains(w, regex=False)
                
                # Áp dụng bộ lọc nếu tìm thấy kết quả
                if mask.any():
                    filtered_df = filtered_df[mask]
                else:
                    st.info("💡 Không tìm thấy khách sạn khớp chính xác từ khóa, hiển thị gợi ý tổng quan.")

            if filtered_df.empty:
                st.warning("⚠️ Không tìm thấy khách sạn nào phù hợp với bộ lọc.")
            else:
                # 3. Dùng KNN để xếp hạng (Rank) tập kết quả đã lọc
                if knn_model is not None:
                    try:
                        hotel_id_col = "Hotel_ID" if "Hotel_ID" in filtered_df.columns else filtered_df.columns[0]
                        
                        predicted_scores = []
                        for h_id in filtered_df[hotel_id_col]:
                            pred = knn_model.predict(uid=str(default_user_id), iid=str(h_id))
                            predicted_scores.append(pred.est)
                        
                        filtered_df["Predicted_Score"] = predicted_scores
                        top_results = filtered_df.sort_values(by="Predicted_Score", ascending=False).head(5)
                    except Exception as e:
                        st.warning(f"⚠️ Lỗi tính điểm KNN ({e}), hiển thị danh sách mặc định.")
                        top_results = filtered_df.head(5)
                else:
                    top_results = filtered_df.head(5)
                
                # 4. Hiển thị kết quả
                for _, row in top_results.iterrows():
                    h_name = row.get("Hotel_Name", "Tên khách sạn")
                    h_address = row.get("Hotel_Address", "Địa chỉ")
                    
                    # Xử lý hiển thị Hạng sao
                    raw_star = row.get("Star_Rating")
                    h_star = int(raw_star) if (pd.notna(raw_star) and str(raw_star).replace('.','').isdigit()) else "N/A"
                    
                    # Xử lý hiển thị Điểm đánh giá
                    raw_score = row.get("Total_Score", row.get("Score"))
                    h_score = round(float(raw_score), 1) if pd.notna(raw_score) else "Chưa có"
                    
                    pred_val = round(row.get("Predicted_Score", 0), 2)
                    
                    with st.container(border=True):
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.subheader(h_name)
                            st.caption(f"📍 **Địa chỉ:** {h_address} | ⭐ **Hạng:** {h_star} sao | 🏆 **Đánh giá TB:** {h_score}/10")
                            st.caption(f"Phù hợp loại hình: **{trip}**")
                        with c2:
                            st.metric(label="KNN Score", value=f"{pred_val}/10")
        else:
            st.error("Chưa nạp được dữ liệu `hotel_info_clean.csv`!")

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
                # TAB 1: OVERVIEW (LẤY TỪ HTML MOCKUP)
                # =========================================================
                with tab1:
                    st.caption("So với trung bình hệ thống & đối thủ trực tiếp theo từng tiêu chí")

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

                    # 1. Biểu đồ cột so sánh tiêu chí
                    chart_df = pd.DataFrame({
                        'Tiêu chí': criteria,
                        'KS này': ks_scores,
                        'Hệ thống': sys_scores,
                        'Đối thủ': competitor_scores
                    }).set_index('Tiêu chí')
                    
                    st.bar_chart(chart_df)

                    # 2. Bảng thống kê chi tiết các tiêu chí
                    win_counts = ["5/5" if ks >= comp else "4/5" for ks, comp in zip(ks_scores, competitor_scores)]
                    table_df = pd.DataFrame({
                        "Tiêu chí": criteria,
                        "KS này": [f"{v:.2f}" for v in ks_scores],
                        "Hệ thống": [f"{v:.2f}" for v in sys_scores],
                        "Đối thủ": [f"{v:.2f}" for v in competitor_scores],
                        "Thắng": win_counts
                    })
                    st.dataframe(table_df, use_container_width=True, hide_index=True)

                    # 3. Khối Insight
                    wins_total = sum(1 for ks, comp in zip(ks_scores, competitor_scores) if ks >= comp)
                    st.success(f"💡 **Thắng cả 5 đối thủ ở {wins_total}/5 tiêu chí** — khách sạn thể hiện rất tốt trong nhóm cạnh tranh trực tiếp.")
                    
                    st.info("""
                    **Insight cho chủ khách sạn:**
                    * 🚀 Marketing các điểm mạnh & lợi thế hàng đầu (đặc biệt là Vị trí & Vệ sinh).
                    * 💰 Mở rộng tầm giá hoặc gói ưu đãi đi kèm để nâng cao thêm độ "đáng tiền" (Value for money).
                    """)

                # =========================================================
                # TAB 2: REVIEW (LẤY TỪ HTML MOCKUP)
                # =========================================================
                with tab2:
                    comment_score_col = "Score" if (not hotel_comments.empty and "Score" in hotel_comments.columns) else ("Review_Score" if (not hotel_comments.empty and "Review_Score" in hotel_comments.columns) else None)
                    
                    if not hotel_comments.empty and comment_score_col:
                        avg_rev_score = round(hotel_comments[comment_score_col].mean(), 1)
                        excellent_count = (hotel_comments[comment_score_col] >= 8.5).sum()
                        excellent_pct = int(round((excellent_count / len(hotel_comments)) * 100))
                    else:
                        avg_rev_score = score_val
                        excellent_pct = 74

                    # 3 Metric của Tab Review
                    rm1, rm2, rm3 = st.columns(3)
                    rm1.metric(label="Tổng số review", value=f"{total_reviews:,}")
                    rm2.metric(label="Điểm trung bình", value=f"{avg_rev_score}")
                    rm3.metric(label="Trên cả tuyệt vời", value=f"{excellent_pct}%")

                    st.markdown("#### Review gần đây (mẫu)")

                    # Hiển thị các comment đầu tiên dạng Card
                    if not hotel_comments.empty:
                        sample_comments = hotel_comments.head(5)
                        
                        reviewer_col = "User_Name" if "User_Name" in sample_comments.columns else ("User_ID" if "User_ID" in sample_comments.columns else None)
                        title_col = "Title" if "Title" in sample_comments.columns else None
                        text_col = "Comment" if "Comment" in sample_comments.columns else ("Review_Text" if "Review_Text" in sample_comments.columns else sample_comments.columns[-1])
                        date_col = "Date" if "Date" in sample_comments.columns else None
                        trip_col = "Trip_Type" if "Trip_Type" in sample_comments.columns else None

                        for _, row in sample_comments.iterrows():
                            r_user = str(row.get(reviewer_col, "Khách hàng Agoda"))
                            r_score = row.get(comment_score_col, 9.0)
                            r_title = str(row.get(title_col, "Trải nghiệm tuyệt vời")) if title_col and pd.notna(row.get(title_col)) else "Đánh giá dịch vụ"
                            r_text = str(row.get(text_col, "Khách sạn sạch sẽ, dịch vụ rất tốt."))
                            r_date = str(row.get(date_col, "Mới đây")) if date_col else "Gần đây"
                            r_trip = str(row.get(trip_col, "Khách du lịch")) if trip_col else "Cặp đôi"

                            st.markdown(f"""
                            <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; background-color: #ffffff;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-weight: 600; font-size: 14px; color: #31333F;">👤 {r_user}</span>
                                    <span style="background-color: #e9f3e9; color: #21a350; font-weight: 700; font-size: 12px; padding: 2px 10px; border-radius: 10px;">⭐ {r_score}</span>
                                </div>
                                <div style="font-size: 12px; color: #787A84; margin: 4px 0;">{r_trip} · {r_date}</div>
                                <div style="font-weight: 600; font-size: 14px; color: #31333F; margin-bottom: 4px;">{r_title}</div>
                                <div style="font-size: 13px; color: #50525C;">{r_text}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        sample_reviews_mock = [
                            {"user": "Nguyễn T. — Việt Nam", "score": 9.2, "meta": "Cặp đôi · 12 tháng 6 2026", "title": "Vị trí tuyệt vời, gần biển", "body": "Phòng sạch sẽ, nhân viên nhiệt tình, sẽ quay lại lần sau."},
                            {"user": "Kim S. — Hàn Quốc", "score": 8.8, "meta": "Gia đình có em bé · 03 tháng 6 2026", "title": "Dịch vụ tốt cho gia đình", "body": "Nhân viên hỗ trợ nhiệt tình khi có trẻ nhỏ, bữa sáng đa dạng."},
                            {"user": "John P. — Hoa Kỳ", "score": 7.5, "meta": "Một mình · 28 tháng 5 2026", "title": "Ổn nhưng phòng hơi nhỏ", "body": "Vị trí thuận tiện di chuyển, tuy nhiên phòng nhỏ hơn mong đợi."}
                        ]
                        for r in sample_reviews_mock:
                            st.markdown(f"""
                            <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; background-color: #ffffff;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-weight: 600; font-size: 14px; color: #31333F;">👤 {r['user']}</span>
                                    <span style="background-color: #e9f3e9; color: #21a350; font-weight: 700; font-size: 12px; padding: 2px 10px; border-radius: 10px;">⭐ {r['score']}</span>
                                </div>
                                <div style="font-size: 12px; color: #787A84; margin: 4px 0;">{r['meta']}</div>
                                <div style="font-weight: 600; font-size: 14px; color: #31333F; margin-bottom: 4px;">{r['title']}</div>
                                <div style="font-size: 13px; color: #50525C;">{r['body']}</div>
                            </div>
                            """, unsafe_allow_html=True)

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