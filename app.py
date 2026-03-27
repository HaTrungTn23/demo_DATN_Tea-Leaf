import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import google.generativeai as genai
import time

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================

st.set_page_config(
    page_title="Tea Doctor AI - Chẩn đoán bệnh chè",
    page_icon="🍃",
    layout="wide"
)

# ⚠️ QUAN TRỌNG: Thay đường dẫn đến 3 file .tflite trên máy của bạn vào đây
MODEL_PATHS = {
    "ResNet": r"C:\Users\kaios\benh_la_tra\resnet_model_optimized.tflite",  # Sửa đường dẫn file ResNet
    "VGG16":  r"C:\Users\kaios\benh_la_tra\vgg_model_optimized.tflite",     # Sửa đường dẫn file VGG16
    "ViT":    r"C:\Users\kaios\benh_la_tra\vit_model_optimized.tflite"        # Sửa đường dẫn file ViT
}

# Cấu hình trọng số cho Ensemble (Soft Voting)
# Tổng cộng lại = 1.0
ENSEMBLE_WEIGHTS = {
    "ResNet": 0.5,
    "VGG16":  0.35,
    "ViT":    0.15  # Ưu tiên ViT hơn một chút vì thường chính xác hơn
}

# Danh sách tên bệnh (Phải đúng thứ tự output của model lúc train)
CLASS_NAMES = [
    'Anthracnose (Thán thư)', 
    'algal leaf (Đốm tảo)', 
    'bird eye spot (Đốm mắt chim)', 
    'brown blight (Héo nâu)', 
    'gray light (Thối xám)', 
    'green mirid bug (Bọ mù xanh)', 
    'healthy (Lá khỏe mạnh)', 
    'helopeltis (Bọ xít muỗi)', 
    'red leaf spot (Đốm lá đỏ)', 
    'red spider (Nhện đỏ)', 
    'tea red scab (Vảy đỏ)', 
    'white spot (Đốm trắng)'
]

IMG_SIZE = (256, 256)

# Cấu hình API Gemini (Lấy từ secrets.toml)
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    HAS_API_KEY = True
except Exception:
    HAS_API_KEY = False

# ==========================================
# 2. HÀM XỬ LÝ BACKEND (CORE AI)
# ==========================================

@st.cache_resource
def load_tflite_interpreter(model_path):
    """Load TFLite Model và cache vào RAM"""
    try:
        # Load file tflite
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors() # Cấp phát bộ nhớ
        return interpreter
    except Exception as e:
        st.error(f"❌ Lỗi load model tại: {model_path}\nChi tiết: {e}")
        return None

def preprocess_image(image: Image.Image):
    """Chuẩn hóa ảnh đầu vào cho model"""
    # 1. Chuyển sang RGB (đề phòng ảnh PNG trong suốt)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # 2. Resize về kích thước (256, 256)
    image = image.resize(IMG_SIZE)
    
    # 3. Chuyển sang mảng Numpy và chuẩn hóa [0, 1]
    img_array = np.array(image, dtype=np.float32)
    img_norm = img_array / 255.0
    
    # 4. Thêm chiều batch: (256, 256, 3) -> (1, 256, 256, 3)
    img_input = np.expand_dims(img_norm, axis=0)
    return img_input

def run_inference(interpreter, img_input):
    """Chạy dự đoán trên 1 interpreter"""
    # Lấy thông tin cổng vào/ra
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Đưa dữ liệu vào model
    interpreter.set_tensor(input_details[0]['index'], img_input)
    
    # Chạy tính toán
    interpreter.invoke()
    
    # Lấy kết quả ra
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0] # Trả về mảng xác suất 1 chiều

def predict_logic(mode, img_input):
    """Điều phối logic dự đoán (Đơn lẻ hoặc Ensemble)"""
    
    # --- TRƯỜNG HỢP 1: ENSEMBLE (KẾT HỢP 3 MODEL) ---
    if mode == "Ensemble (All 3)":
        # Load cả 3 model
        int_resnet = load_tflite_interpreter(MODEL_PATHS["ResNet"])
        int_vgg = load_tflite_interpreter(MODEL_PATHS["VGG16"])
        int_vit = load_tflite_interpreter(MODEL_PATHS["ViT"])
        
        # Nếu thiếu 1 trong 3 file thì báo lỗi
        if not (int_resnet and int_vgg and int_vit):
            return None
        
        # Lấy xác suất từng model
        p1 = run_inference(int_resnet, img_input)
        p2 = run_inference(int_vgg, img_input)
        p3 = run_inference(int_vit, img_input)
        
        # Tính trung bình cộng có trọng số (Soft Voting)
        final_probs = (p1 * ENSEMBLE_WEIGHTS["ResNet"] + 
                       p2 * ENSEMBLE_WEIGHTS["VGG16"] + 
                       p3 * ENSEMBLE_WEIGHTS["ViT"])
        return final_probs

    # --- TRƯỜNG HỢP 2: MODEL ĐƠN LẺ ---
    else:
        # Load model tương ứng với lựa chọn
        interpreter = load_tflite_interpreter(MODEL_PATHS[mode])
        if interpreter:
            return run_inference(interpreter, img_input)
        return None

def get_gemini_advice(disease_name):
    """Gọi Gemini để lấy lời khuyên"""
    if not HAS_API_KEY:
        yield "⚠️ Chưa cấu hình API Key. Vui lòng thêm vào secrets.toml"
        return

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
            Bạn là kỹ sư nông nghiệp Việt Nam. Hãy tư vấn về bệnh trên cây chè: "{disease_name}".
            Yêu cầu định dạng Markdown đẹp:
            1. **Nguyên nhân**: Ngắn gọn.
            2. **Dấu hiệu**: Mô tả nhanh.
            3. **Cách trị bệnh**: Ưu tiên biện pháp sinh học, an toàn, hiệu quả.
            4. **Phòng ngừa**: Cách canh tác.
            Viết ngắn gọn, súc tích (khoảng 300 từ).
        """
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"⚠️ Lỗi kết nối AI: {str(e)}"

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

    # Cột Trái: Upload và Hiển thị ảnh
    with col_input:
        st.subheader("1. Tải ảnh lên")
        uploaded_file = st.file_uploader("Chọn ảnh lá chè (JPG, PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh đầu vào", use_container_width=True)
            
            # Nút bấm xử lý
            process_btn = st.button("🚀 Chẩn đoán bệnh", type="primary", use_container_width=True)

    # Cột Phải: Kết quả
    with col_result:
        if uploaded_file and process_btn:
            st.subheader("2. Kết quả phân tích")
            
            # Hiệu ứng loading
            progress_bar = st.progress(0, text="Đang khởi tạo model...")
            
            # Bước 1: Tiền xử lý
            img_input = preprocess_image(image)
            progress_bar.progress(30, text=f"Đang phân tích bằng {selected_mode}...")
            
            # Bước 2: Dự đoán (Inference)
            probs = predict_logic(selected_mode, img_input)
            progress_bar.progress(80, text="Đang tổng hợp dữ liệu...")
            
            if probs is not None:
                # Tìm class có điểm cao nhất
                pred_idx = np.argmax(probs)
                final_class = CLASS_NAMES[pred_idx]
                confidence = float(np.max(probs))
                
                progress_bar.progress(100, text="Hoàn tất!")
                time.sleep(0.5)
                progress_bar.empty()

                # Hiển thị thẻ kết quả
                if "Healthy" in final_class:
                    st.success(f"### 🌱 Tình trạng: {final_class}")
                else:
                    st.error(f"### 🍂 Phát hiện bệnh: {final_class}")
                
                st.metric("Độ tin cậy (Confidence)", f"{confidence:.2%}")

                # Biểu đồ xác suất
                st.caption("Chi tiết xác suất các lớp bệnh:")
                st.bar_chart(dict(zip(CLASS_NAMES, probs)), color="#4CAF50")

                # Bước 3: AI Tư vấn
                st.divider()
                st.subheader("💡 Bác sĩ AI Tư Vấn")
                with st.spinner("Đang kết nối chuyên gia nông nghiệp..."):
                    advice_box = st.container(height=300)
                    stream = get_gemini_advice(final_class)
                    advice_box.write_stream(stream)
            else:
                progress_bar.empty()
                st.error("❌ Lỗi: Không thể chạy mô hình. Vui lòng kiểm tra lại đường dẫn file .tflite trong code.")

        elif not uploaded_file:
            st.info("👈 Vui lòng tải ảnh lên từ cột bên trái để bắt đầu.")

with tab2:
    st.markdown("""
    ### Hướng dẫn sử dụng Tea Doctor
    1. **Chọn Model:** Ở thanh bên trái, chọn thuật toán bạn muốn sử dụng. Khuyên dùng **Ensemble** để có độ chính xác cao nhất.
    2. **Tải Ảnh:** Nhấn nút "Browse files" để tải ảnh lá chè bị bệnh lên.
    3. **Chẩn đoán:** Nhấn nút "Chẩn đoán bệnh" và đợi hệ thống phân tích.
    4. **Xem kết quả:** Hệ thống sẽ trả về tên bệnh, độ tin cậy và hướng dẫn cách điều trị từ chuyên gia AI.
    """)