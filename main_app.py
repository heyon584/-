import base64
import os
import gtts
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# ==========================================
# 1. 페이지 기본 설정 및 스타일 (포디움 레이아웃 CSS)
# ==========================================
st.set_page_config(
    page_title="닮은꼴 동물 찾기 부스", page_icon="🐾", layout="centered"
)

st.markdown(
    """
    <style>
    /* 전체 배경 및 폰트 설정 */
    .main {
        background-color: #f8f9fa;
    }
    
    /* 포디움(시상대) 전체 컨테이너 */
    .podium-container {
        display: flex;
        justify-content: center;
        align-items: flex-end; /* 하단 라인 정렬 */
        gap: 15px;
        margin-top: 25px;
        margin-bottom: 25px;
    }
    
    /* 포디움 개별 카드의 공통 카드 스타일 */
    .podium-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    /* 1등 전용 스타일 (가장 크고 높게) */
    .podium-rank-1 {
        width: 38%;
        border: 3px solid #ffd700;
        background: linear-gradient(180deg, #ffffff 0%, #fffdf0 100%);
        transform: translateY(-10px);
        z-index: 2;
    }
    
    /* 2등, 3등 스타일 (1등 옆에 하단 맞춤) */
    .podium-rank-2 {
        width: 28%;
        border: 2px solid #c0c0c0;
        z-index: 1;
    }
    .podium-rank-3 {
        width: 28%;
        border: 2px solid #cd7f32;
        z-index: 1;
    }
    
    /* 이미지 둥글게 처리 및 배지 스타일 */
    .podium-img {
        width: 100%;
        max-width: 120px;
        height: 120px;
        object-fit: cover;
        border-radius: 50%;
        margin-bottom: 10px;
    }
    
    .badge {
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 12px;
        color: white;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    .badge-1 { background-color: #ffd700; color: #000; }
    .badge-2 { background-color: #c0c0c0; }
    .badge-3 { background-color: #cd7f32; }
    
    .score-text {
        font-size: 1.1rem;
        font-weight: 800;
        color: #ff4b4b;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. 얼굴 인식 및 캐시 로드
# ==========================================
@st.cache_resource
def load_face_cascade():
    cascade_path = (
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade


face_cascade = load_face_cascade()


# ==========================================
# 3. 이미지 특징 분석 및 유사도 계산 함수
# ==========================================
def extract_color_hist(img_pil):
    """이미지의 HSV 컬러 히스토그램 추출"""
    img_np = np.array(img_pil.convert("RGB"))
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def calculate_similarity(face_crop_pil, target_img_path):
    """얼굴 크롭 이미지와 대상 동물 이미지 간 유사도(0~100%) 계산"""
    if not os.path.exists(target_img_path):
        return 0.0

    target_pil = Image.open(target_img_path)

    # 1. 히스토그램 유사도
    h1 = extract_color_hist(face_crop_pil)
    h2 = extract_color_hist(target_pil)
    hist_sim = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    hist_sim = max(0, hist_sim)  # 음수 보정

    # 2. 크기 및 밝기 기반 임의 변동 가산 (부스 재미 요소)
    np.random.seed(int(face_crop_pil.size[0] + face_crop_pil.size[1]))
    random_factor = np.random.uniform(0.6, 0.95)

    final_score = (hist_sim * 40) + (random_factor * 60)
    return round(min(99.9, max(50.0, final_score)), 1)


# ==========================================
# 4. 이미지 Base64 변환 (HTML 삽입용)
# ==========================================
def img_to_base64(img_path_or_pil):
    if isinstance(img_path_or_pil, str):
        if not os.path.exists(img_path_or_pil):
            return ""
        with open(img_path_or_pil, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    else:
        import io

        buffered = io.BytesIO()
        img_path_or_pil.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ==========================================
# 5. 음성(TTS) 및 드럼롤 자동 재생 생성 함수
# ==========================================
def play_audio_and_tts(text_to_speak, audio_path="drumroll.mp3"):
    # 1. 드럼롤 오디오 B64
    drumroll_b64 = ""
    if os.path.exists(audio_path):
        with open(audio_path, "rb") as f:
            drumroll_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 2. gTTS 음성 파일 생성
    tts = gtts.gTTS(text=text_to_speak, lang="ko")
    tts_file = "temp_tts.mp3"
    tts.save(tts_file)

    with open(tts_file, "rb") as f:
        tts_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 3. 순차 재생 JavaScript 코드 (드럼롤 후 TTS 실행)
    html_code = f"""
        <audio id="drumroll" src="data:audio/mp3;base64,{drumroll_b64}"></audio>
        <audio id="tts" src="data:audio/mp3;base64,{tts_b64}"></audio>
        <script>
            var drum = document.getElementById('drumroll');
            var tts = document.getElementById('tts');
            
            if(drum && "{drumroll_b64}" !== "") {{
                drum.play().catch(e => console.log("Autoplay blocked:", e));
                drum.onended = function() {{
                    tts.play().catch(e => console.log("TTS Autoplay blocked:", e));
                }};
            }} else {{
                tts.play().catch(e => console.log("TTS Autoplay blocked:", e));
            }}
        </script>
    """
    st.components.v1.html(html_code, height=0)


# ==========================================
# 6. 메인 앱 화면 구성
# ==========================================
st.title("🐾 닮은꼴 동물 찾기 부스")
st.write("카메라로 얼굴을 촬영하여 나만의 닮은꼴 동물 순위를 확인하세요!")

# 비교 대상 동물 데이터 세팅
animals = [
    {"name": "쿼카", "file": "quokka.jpg"},
    {"name": "고양이", "file": "cat.jpg"},
    {"name": "강아지", "file": "dog.jpg"},
]

# 카메라 입력
camera_image = st.camera_input("카메라를 바라보고 촬영 버튼을 눌러주세요!")

if camera_image:
    img = Image.open(camera_image)
    img_np = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 얼굴 감지
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        st.warning(
            "⚠️ 얼굴을 인식하지 못했습니다. 조명이 밝은 곳에서 다시 촬영해 주세요."
        )
    else:
        # 가장 큰 얼굴 선택
        (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        face_crop = img.crop((x, y, x + w, y + h))

        # 동물별 유사도 계산
        results = []
        for animal in animals:
            score = calculate_similarity(face_crop, animal["file"])
            results.append(
                {
                    "name": animal["name"],
                    "score": score,
                    "file": animal["file"],
                    "b64": img_to_base64(animal["file"]),
                }
            )

        # 점수 순으로 정렬 (1등, 2등, 3등)
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        rank_1 = results[0]
        rank_2 = results[1]
        rank_3 = results[2]

        st.subheader("🏆 닮은꼴 분석 결과")

        # HTML 시상대(Podium) 레이아웃 렌더링
        podium_html = f"""
        <div class="podium-container">
            <!-- 2등 (왼쪽) -->
            <div class="podium-card podium-rank-2">
                <div class="badge badge-2">2위</div>
                <img class="podium-img" src="data:image/jpeg;base64,{rank_2['b64']}">
                <div style="font-weight:bold; font-size:1rem;">{rank_2['name']}</div>
                <div class="score-text">{rank_2['score']}%</div>
            </div>
            
            <!-- 1등 (중앙, 가장 높게) -->
            <div class="podium-card podium-rank-1">
                <div class="badge badge-1">👑 1위</div>
                <img class="podium-img" src="data:image/jpeg;base64,{rank_1['b64']}">
                <div style="font-weight:bold; font-size:1.2rem;">{rank_1['name']}</div>
                <div class="score-text" style="font-size:1.4rem;">{rank_1['score']}%</div>
            </div>
            
            <!-- 3등 (오른쪽) -->
            <div class="podium-card podium-rank-3">
                <div class="badge badge-3">3위</div>
                <img class="podium-img" src="data:image/jpeg;base64,{rank_3['b64']}">
                <div style="font-weight:bold; font-size:1rem;">{rank_3['name']}</div>
                <div class="score-text">{rank_3['score']}%</div>
            </div>
        </div>
        """
        st.markdown(podium_html, unsafe_allow_html=True)

        # 결과 TTS 안내 및 효과음 실행
        announce_text = f"축하합니다! 당신은 {rank_1['score']}% 확률로 {rank_1['name']}와 가장 닮았습니다!"
        play_audio_and_tts(announce_text)