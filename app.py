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
    st.title("Dashboard chủ khách sạn")
    
    # Quản lý trạng thái đăng nhập bằng Session State
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        
    if not st.session_state["logged_in"]:
        st.caption("Đăng nhập để xem phân tích khách sạn của bạn")
        username = st.text_input("Tên đăng nhập", placeholder="chukhachsan01")
        password = st.text_input("Mật khẩu", type="password", placeholder="••••••••")
        
        if st.button("Đăng nhập", type="primary"):
            st.session_state["logged_in"] = True
            st.rerun()
            
    else:
        hotel_list = df_info["Hotel_Name"].unique() if (df_info is not None and "Hotel_Name" in df_info.columns) else ["Sunrise Beach Hotel", "Central Garden Hotel", "Riverside Boutique"]
        
        current_hotel = st.selectbox("Khách sạn của bạn", hotel_list)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Điểm tổng", "8.6")
        m2.metric("Lượt đánh giá", "312")
        m3.metric("So với TB hệ thống", "+0.4", delta="▲ cao hơn trung bình")
        
        tab_overview, tab_review, tab_benchmark = st.tabs(["Overview", "Review", "Benchmark đối thủ"])
        
        # --- TAB OVERVIEW ---
        with tab_overview:
            st.caption("So với trung bình hệ thống & đối thủ trực tiếp theo từng tiêu chí")
            
            categories = ['Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money']
            ks_score = [9.40, 8.90, 8.90, 8.70, 8.70]
            sys_score = [8.27, 8.12, 8.30, 7.90, 8.18]
            comp_score = [8.09, 8.24, 8.44, 7.90, 8.34]
            
            # Vẽ biểu đồ cột nhóm bằng Plotly
            fig = go.Figure(data=[
                go.Bar(name='KS này', x=categories, y=ks_score, marker_color='#0B5ED7'),
                go.Bar(name='Hệ thống', x=categories, y=sys_score, marker_color='#B4B2A9'),
                go.Bar(name='Đối thủ', x=categories, y=comp_score, marker_color='#FAC775')
            ])
            fig.update_layout(barmode='group', yaxis_range=[0, 10], height=380, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            df_overview = pd.DataFrame({
                "Tiêu chí": categories,
                "KS này": ks_score,
                "Hệ thống": sys_score,
                "Đối thủ": comp_score,
                "Thắng": ["5/5", "5/5", "5/5", "5/5", "4/5"]
            })
            st.dataframe(df_overview, hide_index=True, use_container_width=True)
            
            st.success("Thắng cả 5 đối thủ ở 4/5 tiêu chí — khách sạn thể hiện rất tốt trong nhóm cạnh tranh trực tiếp.")
            
            with st.container(border=True):
                st.markdown("**Insight cho chủ khách sạn:**")
                st.markdown("- Marketing các điểm mạnh & lợi thế")
                st.markdown("- Mở rộng tầm giá để nâng cao độ 'đáng tiền'")

        # --- TAB REVIEW ---
        with tab_review:
            r1, r2, r3 = st.columns(3)
            r1.metric("Tổng số review", "312")
            r2.metric("Điểm trung bình", "8.6")
            r3.metric("Trên cả tuyệt vời", "74%")
            
            st.subheader("Review gần đây (mẫu)")
            
            reviews = [
                {"name": "Nguyễn T. — Việt Nam", "score": "9.2", "meta": "Cặp đôi · 12 tháng 6 2026", "title": "Vị trí tuyệt vời, gần biển", "body": "Phòng sạch sẽ, nhân viên nhiệt tình, sẽ quay lại lần sau."},
                {"name": "Kim S. — Hàn Quốc", "score": "8.8", "meta": "Gia đình có em bé · 03 tháng 6 2026", "title": "Dịch vụ tốt cho gia đình", "body": "Nhân viên hỗ trợ nhiệt tình khi có trẻ nhỏ, bữa sáng đa dạng."},
                {"name": "John P. — Hoa Kỳ", "score": "7.5", "meta": "Một mình · 28 tháng 5 2026", "title": "Ổn nhưng phòng hơi nhỏ", "body": "Vị trí thuận tiện di chuyển, tuy nhiên phòng nhỏ hơn mong đợi."}
            ]
            
            for r in reviews:
                with st.container(border=True):
                    head_col, score_col = st.columns([4, 1])
                    head_col.markdown(f"**{r['name']}**")
                    score_col.markdown(f"🟢 **{r['score']}**")
                    st.caption(r["meta"])
                    st.markdown(f"**{r['title']}**")
                    st.write(r["body"])

        # --- TAB BENCHMARK ---
        with tab_benchmark:
            b_col1, b_col2 = st.columns([2, 1])
            
            with b_col1:
                st.subheader("Top-5 khách sạn tương tự nhất (Content-based / Cosine Similarity)")
                
                selected_hotel_name = current_hotel
                
                # Tính toán đối thủ trực tiếp bằng Cosine Similarity
                if cosine_sim is not None and df_info is not None and "Hotel_Name" in df_info.columns:
                    try:
                        idx = df_info[df_info['Hotel_Name'] == selected_hotel_name].index[0]
                        
                        sim_scores = list(enumerate(cosine_sim[idx]))
                        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                        
                        top_indices = [i[0] for i in sim_scores[1:6]]
                        top_sim_values = [round(i[1], 3) for i in sim_scores[1:6]]
                        
                        df_comp = df_info.iloc[top_indices][['Hotel_Name', 'Star_Rating', 'Total_Score']].copy()
                        df_comp['Similarity'] = top_sim_values
                        df_comp.columns = ['Đối thủ', 'Hạng', 'Total Score', 'Similarity']
                        
                        st.dataframe(df_comp, hide_index=True, use_container_width=True)
                    except Exception as e:
                        st.info("Hiển thị dữ liệu đối thủ mẫu do chưa khớp tên trong dataset:")
                        df_comp_mock = pd.DataFrame({
                            "Đối thủ": ["Angella Hotel Nha Trang", "Dubai Nha Trang Hotel", "Khách sạn V Nha Trang", "ALPHA BIRD NHA TRANG", "Khách sạn Sochi"],
                            "Hạng": ["4.5 sao", "5 sao", "5 sao", "4 sao", "4 sao"],
                            "Total Score": [8.5, 8.2, 8.1, 8.0, 7.9],
                            "Similarity": [0.853, 0.853, 0.849, 0.839, 0.838]
                        })
                        st.dataframe(df_comp_mock, hide_index=True, use_container_width=True)
                
                st.caption("*Sử dụng ma trận Cosine Similarity (Content-based) để xác định đối thủ cạnh tranh trực tiếp.*")
                
            with b_col2:
                with st.container(border=True):
                    st.subheader("Beats # Competitors")
                    st.write("Location: **5/5**")
                    st.write("Cleanliness: **5/5**")
                    st.write("Service: **5/5**")
                    st.write("Facilities: **5/5**")
                    st.write("Value_for_money: **4/5**")