import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import time
import urllib.request
import hashlib
import base64
from io import BytesIO
from gtts import gTTS
from concurrent.futures import ThreadPoolExecutor

# --- 1. 페이지 기본 설정 및 컴팩트 레이아웃 / 수평 일렬 정렬 CSS ---
st.set_page_config(page_title="🎉 축제 닮은꼴 부스", layout="wide")

st.markdown("""
    <style>
    .block-container {
        padding-top: 0.2rem !important;
        padding-bottom: 0rem !important;
        max-width: 85% !important;
    }
    div[data-testid="stCameraInput"] video,
    div[data-testid="stCameraInput"] img {
        transform: scaleX(-1) !important;
    }
    header, footer { visibility: hidden; }

    /* 화면 진동 */
    @keyframes superShake {
        0% { transform: translate(0, 0) rotate(0deg); }
        20% { transform: translate(-4px, 3px) rotate(-1deg); }
        40% { transform: translate(4px, -3px) rotate(1deg); }
        60% { transform: translate(-3px, 2px) rotate(0deg); }
        80% { transform: translate(3px, -2px) rotate(0deg); }
        100% { transform: translate(0, 0) rotate(0deg); }
    }

    .shake-effect {
        animation: superShake 0.5s ease-in-out;
    }

    /* 메인 1위 불꽃 네온 박스 */
    @keyframes superFire {
        0% { box-shadow: 0 0 10px #ff4500; border-color: #ff4500; }
        50% { box-shadow: 0 0 20px #ff0000, 0 0 35px #ffd700; border-color: #ffd700; }
        100% { box-shadow: 0 0 10px #ff4500; border-color: #ff4500; }
    }

    .fire-box {
        border: 3px solid #ff4500;
        border-radius: 8px;
        padding: 2px;
        animation: superFire 0.6s infinite alternate;
        background: linear-gradient(180deg, rgba(255,0,0,0.25) 0%, rgba(255,140,0,0.1) 100%);
        text-align: center;
    }

    .custom-divider {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, rgba(255,69,0,0), rgba(255,69,0,0.8), rgba(255,69,0,0));
        margin: 4px 0;
    }

    /* 🎯 2위/3위 하단 여백 및 바닥 정렬 조절 (싱크로율 일렬 맞춤) */
    [data-testid="stHorizontalBlock"] {
        align-items: flex-end !important;
    }

    .side-podium-box {
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; margin-bottom: 0px;'>🎉 나랑 닮은 꼴 찾기 부스</h3>", unsafe_allow_html=True)

# --- 2. OpenCV 모델 안전 로드 ---
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

@st.cache_resource
def load_face_cascade():
    cascade_path = cv2.data.haarcascades + CASCADE_FILENAME
    if not os.path.exists(cascade_path) or os.path.getsize(cascade_path) < 1000:
        cascade_path = CASCADE_FILENAME
        if not os.path.exists(cascade_path) or os.path.getsize(cascade_path) < 1000:
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, cascade_path)

    cascade = cv2.CascadeClassifier(cascade_path)
    return cascade

face_cascade = load_face_cascade()

# --- 3. 비교 대상 데이터베이스 ---
TARGETS = {
    "웃는 쿼카": {"path": "quokka.jpg", "desc": "보기만 해도 기분이 좋아지는 긍정 바이러스!"},
    "장난꾸러기 고양이": {"path": "cat.jpg", "desc": "도도해 보이지만 끌리는 치명적 매력!"},
    "친절한 리트리버": {"path": "dog.jpg", "desc": "누구에게나 호감을 주는 순수한 에너지!"}
}

# --- 4. 유틸리티 함수 ---
def crop_to_square(img, size=(120, 120)):
    return ImageOps.fit(img, size, Image.Resampling.LANCZOS)

def get_user_face_feature(image_np):
    if face_cascade is None or face_cascade.empty():
        return np.array([image_np.shape[1], image_np.shape[0], image_np.shape[1] / float(image_np.shape[0])])
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            return np.array([w, h, w / float(h)])
    except Exception:
        pass
    return np.array([image_np.shape[1], image_np.shape[0], image_np.shape[1] / float(image_np.shape[0])])

def process_target(item_with_userfeat_and_img):
    item, u_feat, u_img_bytes = item_with_userfeat_and_img
    name, info = item
    
    if not os.path.exists(info["path"]):
        hash_val = int(hashlib.md5(name.encode()).hexdigest(), 16) % 30
        return (float(hash_val), name)
    
    try:
        target_img = Image.open(info["path"])
        target_np = np.array(target_img)
        
        hsv_user = cv2.cvtColor(cv2.resize(np.array(Image.open(u_img_bytes)), (100, 100)), cv2.COLOR_RGB2HSV)
        hsv_target = cv2.cvtColor(cv2.resize(target_np, (100, 100)), cv2.COLOR_RGB2HSV)
        
        hist_u = cv2.calcHist([hsv_user], [0, 1], None, [18, 25], [0, 180, 0, 256])
        hist_t = cv2.calcHist([hsv_target], [0, 1], None, [18, 25], [0, 180, 0, 256])
        
        cv2.normalize(hist_u, hist_u, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_t, hist_t, 0, 1, cv2.NORM_MINMAX)
        
        sim_score = cv2.compareHist(hist_u, hist_t, cv2.HISTCMP_CORREL)
        hash_val = int(hashlib.md5((name + str(u_feat[0])).encode()).hexdigest(), 16) % 15
        final_score = float(sim_score) * 100 + hash_val
        
        return (final_score, name)
    except Exception:
        hash_val = int(hashlib.md5(name.encode()).hexdigest(), 16) % 30
        return (float(hash_val), name)

# 🔊 로컬 drumroll.mp3 오디오 자동 재생
def play_custom_drumroll():
    audio_path = "drumroll.mp3"
    if os.path.exists(audio_path):
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            b64_audio = base64.b64encode(audio_bytes).decode()
            js_code = f"""
            <script>
            var audio = new Audio("data:audio/mp3;base64,{b64_audio}");
            audio.play().catch(function(e){{ console.log(e); }});
            </script>
            """
            st.components.v1.html(js_code, height=0)

# 🔊 1등 이름만 딱 자동으로 읽어주는 gTTS 내장 함수
def play_name_tts(text_name):
    tts = gTTS(text=text_name, lang='ko')
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    b64_audio = base64.b64encode(fp.read()).decode()
    js_code = f"""
    <script>
    var audio = new Audio("data:audio/mp3;base64,{b64_audio}");
    audio.play().catch(function(e){{ console.log(e); }});
    </script>
    """
    st.components.v1.html(js_code, height=0)

# 💥 초강력 폭발 Confetti 연출
def trigger_explosive_effect():
    js_code = """
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
    <script>
    var duration = 2.0 * 1000;
    var end = Date.now() + duration;

    (function frame() {
      confetti({ particleCount: 10, angle: 60, spread: 70, origin: { x: 0 } });
      confetti({ particleCount: 10, angle: 120, spread: 70, origin: { x: 1 } });
      if (Date.now() < end) { requestAnimationFrame(frame); }
    })();
    </script>
    """
    st.components.v1.html(js_code, height=0)

# --- 5. 웹캠 UI ---
img_buffer = st.camera_input("웹캠 화면", key="webcam")

if img_buffer is not None:
    user_img = Image.open(img_buffer)
    user_np = np.array(user_img)

    user_np = cv2.flip(user_np, 1)
    user_img = Image.fromarray(user_np)

    user_feat = get_user_face_feature(user_np)

    tasks = [(item, user_feat, img_buffer) for item in TARGETS.items()]
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_target, tasks))

    valid_results = [r for r in results if r is not None]
    valid_results.sort(key=lambda x: x[0], reverse=True)

    target_names = [r[1] for r in valid_results]
    default_scores = [92, 79, 64]
    
    scored_results = []
    for i in range(3):
        name = target_names[i] if i < len(target_names) else list(TARGETS.keys())[i]
        scored_results.append((name, default_scores[i]))

    # --- 🥁 [5.5초 드럼롤 재생 대기] ---
    waiting_placeholder = st.empty()
    with waiting_placeholder.container():
        st.markdown("<h4 style='text-align: center; color: #FF4500;'>🥁 과연 나랑 가장 닮은 꼴은?! 🥁</h4>", unsafe_allow_html=True)
        col_w1, col_w2, col_w3 = st.columns([1.2, 1, 1.2])
        with col_w2:
            st.image(crop_to_square(user_img, (140, 140)), use_container_width=True)

    play_custom_drumroll()
    time.sleep(5.5)
    waiting_placeholder.empty()

    # --- 💥 [5.5초 뒤 결과 발표] ---
    top1_name, top1_sim = scored_results[0]
    top2_name, top2_sim = scored_results[1]
    top3_name, top3_sim = scored_results[2]

    top1_info, top2_info, top3_info = TARGETS[top1_name], TARGETS[top2_name], TARGETS[top3_name]

    # 폭발 연출 및 1위 이름만 자동 TTS 재생
    trigger_explosive_effect()
    play_name_tts(top1_name)

    # 🏆 시상대 레이아웃
    col_2nd, col_main, col_3rd = st.columns([1.0, 1.1, 1.0])

    # 🥈 2위
    with col_2nd:
        st.markdown("<div class='side-podium-box'>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style='text-align: center;'>
                <p style='margin:0; font-weight:bold;'>🥈 2위</p>
                <p style='margin:2px 0; font-size: 0.9rem;'><b>{top2_name}</b></p>
            </div>
        """, unsafe_allow_html=True)
        if os.path.exists(top2_info["path"]):
            img = Image.open(top2_info["path"])
            st.image(crop_to_square(img, (100, 100)), use_container_width=True)
        st.markdown(f"<p style='text-align: center; font-size: 0.85rem; font-weight: bold; margin: 0;'>싱크로율: {top2_sim}%</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 📸 내 얼굴 + 🔥 1위 (중앙)
    with col_main:
        st.markdown("<p style='text-align: center; margin: 0; font-weight:bold; font-size: 0.9rem;'>📸 내 얼굴</p>", unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns([1, 2, 1])
        with col_m2:
            st.image(crop_to_square(user_img, (90, 90)), use_container_width=True)

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        st.markdown(f"""
            <div class="fire-box shake-effect">
                <p style='margin:0; color: #FF4500; font-size: 0.85rem; font-weight:bold;'>🔥 1위 🔥</p>
                <p style='margin:0; color: #FFFFFF; font-size: 1rem;'><b>{top1_name}</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        if os.path.exists(top1_info["path"]):
            img = Image.open(top1_info["path"])
            st.image(crop_to_square(img, (120, 120)), use_container_width=True)
        st.markdown(f"<p style='text-align: center; font-size: 0.9rem; font-weight: bold; margin: 2px 0;'>🔥 싱크로율: {top1_sim}%</p>", unsafe_allow_html=True)

    # 🥉 3위
    with col_3rd:
        st.markdown("<div class='side-podium-box'>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style='text-align: center;'>
                <p style='margin:0; font-weight:bold;'>🥉 3위</p>
                <p style='margin:2px 0; font-size: 0.9rem;'><b>{top3_name}</b></p>
            </div>
        """, unsafe_allow_html=True)
        if os.path.exists(top3_info["path"]):
            img = Image.open(top3_info["path"])
            st.image(crop_to_square(img, (100, 100)), use_container_width=True)
        st.markdown(f"<p style='text-align: center; font-size: 0.85rem; font-weight: bold; margin: 0;'>싱크로율: {top3_sim}%</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)