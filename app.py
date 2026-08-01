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
# 7. Màn hình: CHỦ KHÁCH SẠN (Chia 3 Tab Insight)
# ---------------------------------------------------------
elif menu == "Chủ khách sạn":
    st.title("📊 Phân tích & Insight dành cho Chủ Khách Sạn")
    st.caption("Báo cáo hiệu suất kinh doanh, so sánh đối thủ cạnh tranh và phân tích ý kiến phản hồi từ khách hàng.")

    if df_info is None or df_comments is None or df_info.empty:
        st.error("⚠️ Chưa nạp được đầy đủ dữ liệu từ `hotel_info_clean.csv` hoặc `hotel_comments_clean.csv`!")
    else:
        # Xác định các cột định danh
        col_hotel_id_info = "Hotel_ID" if "Hotel_ID" in df_info.columns else df_info.columns[0]
        col_hotel_name = "Hotel_Name" if "Hotel_Name" in df_info.columns else "Tên khách sạn"
        col_hotel_id_comments = "Hotel_ID" if "Hotel_ID" in df_comments.columns else df_comments.columns[0]

        # 1. Chọn khách sạn cần phân tích
        hotel_list = sorted(df_info[col_hotel_name].dropna().unique())
        selected_hotel_name = st.selectbox("🏨 Chọn khách sạn của bạn để xem báo cáo:", hotel_list)

        if selected_hotel_name:
            # Lấy thông tin khách sạn chọn
            hotel_row = df_info[df_info[col_hotel_name] == selected_hotel_name].iloc[0]
            selected_hotel_id = hotel_row[col_hotel_id_info]

            # Lọc bình luận tương ứng
            hotel_comments = df_comments[df_comments[col_hotel_id_comments] == selected_hotel_id]

            st.markdown("---")

            # Tạo 3 TAB giao diện
            tab1, tab2, tab3 = st.tabs([
                "🏨 1. Tổng quan khách sạn", 
                "📊 2. Benchmark đối thủ", 
                "💬 3. Phân tích Review"
            ])

            # =========================================================
            # TAB 1: TỔNG QUAN KHÁCH SẠN
            # =========================================================
            with tab1:
                st.subheader(f"Thông tin chung: {selected_hotel_name}")
                
                c1, c2, c3, c4 = st.columns(4)

                # Hạng sao
                raw_star = hotel_row.get("Star_Rating")
                star_val = int(raw_star) if (pd.notna(raw_star) and str(raw_star).replace('.', '').isdigit()) else 0
                c1.metric(label="Hạng sao", value=f"{star_val} ⭐" if star_val > 0 else "N/A")

                # Điểm trung bình
                raw_score = hotel_row.get("Total_Score", hotel_row.get("Score"))
                score_val = round(float(raw_score), 1) if pd.notna(raw_score) else 0.0
                c2.metric(label="Điểm trung bình", value=f"{score_val}/10" if score_val > 0 else "Chưa có")

                # Tổng số review
                c3.metric(label="Tổng lượt đánh giá", value=f"{len(hotel_comments):,} lượt")

                # Khách hàng
                user_col = "User_ID" if "User_ID" in hotel_comments.columns else None
                num_users = hotel_comments[user_col].nunique() if user_col and not hotel_comments.empty else len(hotel_comments)
                c4.metric(label="Số lượt khách đánh giá", value=f"{num_users:,}")

                st.markdown(f"📍 **Địa chỉ:** {hotel_row.get('Hotel_Address', 'N/A')}")
                
                st.markdown("#### 📌 Tóm tắt thuộc tính & Tiện ích")
                col_left, col_right = st.columns(2)
                with col_left:
                    st.info(f"• **Mã khách sạn (Hotel ID):** `{selected_hotel_id}`")
                with col_right:
                    st.info(f"• **Trạng thái dữ liệu:** Đã đồng bộ giữa Info ({len(df_info)} KS) & Comments ({len(df_comments)} Reviews)")

            # =========================================================
            # TAB 2: BENCHMARK ĐỐI THỦ & SO SÁNH ĐIỂM
            # =========================================================
            with tab2:
                st.subheader("📊 So sánh vị thế với toàn thị trường / cùng phân khúc")
                
                # Tính điểm trung bình toàn hệ thống
                score_col = "Total_Score" if "Total_Score" in df_info.columns else ("Score" if "Score" in df_info.columns else None)
                
                if score_col:
                    avg_market_score = round(df_info[score_col].mean(), 2)
                    
                    # Lọc đối thủ cùng hạng sao
                    same_star_df = df_info[df_info["Star_Rating"] == star_val] if star_val > 0 else df_info
                    avg_star_score = round(same_star_df[score_col].mean(), 2)

                    b1, b2, b3 = st.columns(3)
                    
                    diff_market = round(score_val - avg_market_score, 2)
                    diff_star = round(score_val - avg_star_score, 2)

                    b1.metric("Điểm khách sạn của bạn", f"{score_val}/10")
                    b2.metric("Trung bình toàn thị trường", f"{avg_market_score}/10", delta=f"{diff_market} điểm")
                    b3.metric(f"Trung bình phân khúc {star_val} sao", f"{avg_star_score}/10", delta=f"{diff_star} điểm")

                    st.markdown("---")
                    st.markdown("#### 🏆 Top 5 khách sạn cùng phân khúc sao có điểm cao nhất")
                    top_competitors = same_star_df.sort_values(by=score_col, ascending=False).head(5)
                    display_cols = [c for c in [col_hotel_name, "Star_Rating", score_col, "Hotel_Address"] if c in top_competitors.columns]
                    st.dataframe(top_competitors[display_cols], use_container_width=True)
                else:
                    st.warning("⚠️ Không tìm thấy cột chứa điểm đánh giá tổng thể trong `hotel_info_clean.csv`!")

            # =========================================================
            # TAB 3: PHÂN TÍCH REVIEW & BÌNH LUẬN
            # =========================================================
            with tab3:
                st.subheader("💬 Phân tích đánh giá chi tiết từ khách hàng")

                if hotel_comments.empty:
                    st.info("ℹ️ Khách sạn này hiện chưa có nhận xét/bình luận nào trong hệ thống.")
                else:
                    st.write(f"Hiển thị **{len(hotel_comments)}** bình luận gần nhất:")

                    # Ẩn bớt cột Hotel_ID khi hiển thị table cho gọn
                    show_comment_cols = [c for c in hotel_comments.columns if c != col_hotel_id_comments]
                    st.dataframe(
                        hotel_comments[show_comment_cols] if show_comment_cols else hotel_comments,
                        use_container_width=True,
                        height=400
                    )