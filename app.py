import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import os
import gdown

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================

st.set_page_config(
    page_title="Tea Doctor AI - Chẩn đoán bệnh chè",
    page_icon="🍃",
    layout="wide"
)

# ⚠️ QUAN TRỌNG: Thay file ID Google Drive của từng model vào đây
# Cách lấy file ID: mở link share Drive, copy phần sau /d/ và trước /view
# Ví dụ: https://drive.google.com/file/d/1ABC_XYZ_FILE_ID/view -> lấy "1ABC_XYZ_FILE_ID"
DRIVE_IDS = {
    "ResNet": "1mRXyMWGamodNwXv0Br2W7Ba7p0AX7XX1",
    "VGG16":  "1_psRep79o6O4OJWxeNCR6myVmLgUO6pZ",
    "ViT":    "1S7peGdbToyxn1Qrgz3u0d5BBRGvrK7Kt",
}

# Tên file lưu local sau khi download
MODEL_FILENAMES = {
    "ResNet": "resnet_model_optimized.tflite",
    "VGG16":  "vgg_model_optimized.tflite",
    "ViT":    "vit_model_optimized.tflite",
}

# Cấu hình trọng số cho Ensemble (Soft Voting)
# Tổng cộng lại = 1.0
ENSEMBLE_WEIGHTS = {
    "ResNet": 0.5,
    "VGG16":  0.35,
    "ViT":    0.15
}

# Danh sách tên bệnh (Phải đúng thứ tự output của model lúc train)
CLASS_NAMES = [
    "Anthracnose (Thán thư)",
    "Algal leaf (Tảo ký sinh)",
    "Bird eye spot (Đốm mắt cua)",
    "Brown blight (Cháy lá nâu)",
    "Gray blight (Cháy lá xám)",
    "Healthy (Khỏe mạnh)",
    "Red leaf spot (Đốm lá đỏ)",
    "White spot (Đốm trắng)"
]

IMG_SIZE = (256, 256)

# ==========================================
# 2. HÀM XỬ LÝ BACKEND (CORE AI)
# ==========================================

def download_model_if_needed(model_name):
    """Download model từ Google Drive nếu chưa có local"""
    filename = MODEL_FILENAMES[model_name]
    drive_id = DRIVE_IDS[model_name]

    if not os.path.exists(filename):
        with st.spinner(f"Đang tải model {model_name} về... (chỉ lần đầu)"):
            url = f"https://drive.google.com/uc?id={drive_id}"
            gdown.download(url, filename, quiet=False)

    return filename

@st.cache_resource
def load_tflite_interpreter(model_name):
    try:
        model_path = download_model_if_needed(model_name)

        if model_name == "ViT":
            # ViT dùng SELECT_TF_OPS, cần load bằng tensorflow đầy đủ
            interpreter = tf.lite.Interpreter(
                model_path=model_path,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.AUTO
            )
        else:
            interpreter = tf.lite.Interpreter(model_path=model_path)

        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ Lỗi load model {model_name}: {e}")
        return None

def preprocess_image(image: Image.Image):
    """Chuẩn hóa ảnh đầu vào cho model"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_norm = img_array / 255.0
    img_input = np.expand_dims(img_norm, axis=0)
    return img_input

def run_inference(interpreter, img_input):
    """Chạy dự đoán trên 1 interpreter"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

def predict_logic(mode, img_input):
    """Điều phối logic dự đoán (Đơn lẻ hoặc Ensemble)"""

    if mode == "Ensemble (All 3)":
        int_resnet = load_tflite_interpreter("ResNet")
        int_vgg    = load_tflite_interpreter("VGG16")
        int_vit    = load_tflite_interpreter("ViT")

        if not (int_resnet and int_vgg and int_vit):
            return None

        p1 = run_inference(int_resnet, img_input)
        p2 = run_inference(int_vgg,    img_input)
        p3 = run_inference(int_vit,    img_input)

        final_probs = (p1 * ENSEMBLE_WEIGHTS["ResNet"] +
                       p2 * ENSEMBLE_WEIGHTS["VGG16"]  +
                       p3 * ENSEMBLE_WEIGHTS["ViT"])
        return final_probs

    else:
        interpreter = load_tflite_interpreter(mode)
        if interpreter:
            return run_inference(interpreter, img_input)
        return None

# ==========================================
# 3. GIAO DIỆN NGƯỜI DÙNG (FRONTEND)
# ==========================================

st.markdown("<h1 style='text-align:center; color:#2E7D32;'>🍃 Tea Doctor - Chẩn Đoán Bệnh Chè</h1>", unsafe_allow_html=True)

# --- SIDEBAR: CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Chọn Mô Hình AI")
    selected_mode = st.radio(
        "Thuật toán xử lý:",
        ("Ensemble (All 3)", "ResNet", "VGG16", "ViT")
    )

    st.info(f"Đang dùng chế độ: **{selected_mode}**")
    if selected_mode == "Ensemble (All 3)":
        st.markdown("""
        **Cơ chế Soft Voting:**
        * ResNet: 50%
        * VGG16: 35%
        * ViT: 15%
        """)
        st.caption("Đây là chế độ chính xác nhất.")

# --- MÀN HÌNH CHÍNH ---
tab1, tab2 = st.tabs(["🔍 Phân Tích Hình Ảnh", "ℹ️ Hướng Dẫn"])

with tab1:
    col_input, col_result = st.columns([1, 1.5], gap="large")

    with col_input:
        st.subheader("1. Tải ảnh lên")
        uploaded_file = st.file_uploader("Chọn ảnh lá chè (JPG, PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh đầu vào", use_container_width=True)
            process_btn = st.button("🚀 Chẩn đoán bệnh", type="primary", use_container_width=True)

    with col_result:
        if uploaded_file and process_btn:
            st.subheader("2. Kết quả phân tích")

            progress_bar = st.progress(0, text="Đang khởi tạo model...")

            img_input = preprocess_image(image)
            progress_bar.progress(30, text=f"Đang phân tích bằng {selected_mode}...")

            probs = predict_logic(selected_mode, img_input)
            progress_bar.progress(80, text="Đang tổng hợp dữ liệu...")

            if probs is not None:
                pred_idx = np.argmax(probs)
                final_class = CLASS_NAMES[pred_idx]
                confidence = float(np.max(probs))

                progress_bar.progress(100, text="Hoàn tất!")
                time.sleep(0.5)
                progress_bar.empty()

                if "Healthy" in final_class:
                    st.success(f"### 🌱 Tình trạng: {final_class}")
                else:
                    st.error(f"### 🍂 Phát hiện bệnh: {final_class}")

                st.metric("Độ tin cậy (Confidence)", f"{confidence:.4%}")

                st.caption("Chi tiết xác suất các lớp bệnh:")
                st.bar_chart(dict(zip(CLASS_NAMES, probs)), color="#4CAF50")
            else:
                progress_bar.empty()
                st.error("❌ Lỗi: Không thể tải model. Vui lòng kiểm tra File ID Google Drive trong code.")

        elif not uploaded_file:
            st.info("👈 Vui lòng tải ảnh lên từ cột bên trái để bắt đầu.")

with tab2:
    st.markdown("""
    ### Hướng dẫn sử dụng Tea Doctor
    1. **Chọn Model:** Ở thanh bên trái, chọn thuật toán bạn muốn sử dụng. Khuyên dùng **Ensemble** để có độ chính xác cao nhất.
    2. **Tải Ảnh:** Nhấn nút "Browse files" để tải ảnh lá chè bị bệnh lên.
    3. **Chẩn đoán:** Nhấn nút "Chẩn đoán bệnh" và đợi hệ thống phân tích.
    4. **Xem kết quả:** Hệ thống sẽ trả về tên bệnh, độ tin cậy.
    """)
