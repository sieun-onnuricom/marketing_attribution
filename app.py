"""
마케팅 기여도 분석 도구
- 브랜드별 분석
- lag 자동 선택 (1, 3, 5, 7일)
- 매체별 단독R² + 전체R²
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 페이지 설정
# ==========================================
st.set_page_config(
    page_title="마케팅 기여도 분석",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 색상 팔레트 — 푸른 계열 통일
BG = "#0F1419"
SURFACE = "#1A2332"
SURFACE_2 = "#2A3548"
BORDER = "#3D4A5F"
TEXT = "#F1F5F9"
TEXT_DIM = "#CBD5E1"
TEXT_MUTED = "#94A3B8"
NAVY = "#1E3A5F"
NAVY_LIGHT = "#3B5B85"
ACCENT = "#A5C9E8"           # 연한 하늘색 (분석/다운로드 버튼)
ACCENT_DEEP = "#5B8FBF"      # 진한 하늘 (호버)
HIGHLIGHT_BG = "#1E3A5F"     # HIGH 강조 (네이비)
SUCCESS_BG = "#264870"       # 분석 성공 (네이비 톤, 초록 제거)

# 내부 고정값
MIN_DATA_DAYS = 30
MIN_SOLO_DAYS = 20
LAG_CANDIDATES = [1, 3, 5, 7]

# CSS
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    
    .stApp, .stApp * {{
        color: {TEXT} !important;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT} !important;
        font-weight: 600;
    }}
    
    /* H4 — subheader(h3)보다 작고 본문보다 크게 */
    .stMarkdown h4 {{
        font-size: 1.05rem !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.6rem !important;
        color: {ACCENT} !important;
    }}
    
    /* 사이드바 */
    section[data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{
        color: {TEXT} !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {{
        color: {TEXT} !important;
    }}
    section[data-testid="stSidebar"] strong {{
        color: {TEXT} !important;
    }}
    
    /* 사이드바 코드 박스 — 줄간격 충분 */
    section[data-testid="stSidebar"] [data-testid="stCodeBlock"],
    section[data-testid="stSidebar"] pre {{
        background-color: {BG} !important;
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 14px !important;
        margin: 8px 0;
    }}
    section[data-testid="stSidebar"] [data-testid="stCodeBlock"] code,
    section[data-testid="stSidebar"] pre code,
    section[data-testid="stSidebar"] [data-testid="stCodeBlock"] * {{
        background-color: transparent !important;
        color: {TEXT} !important;
        font-size: 0.82rem !important;
        line-height: 1.8 !important;
        white-space: pre !important;
    }}
    
    /* 메트릭 */
    [data-testid="stMetric"] {{
        background-color: {SURFACE};
        padding: 18px 20px;
        border-radius: 10px;
        border: 1px solid {BORDER};
    }}
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
        font-size: 0.85rem !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {TEXT} !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }}
    
    /* 분석 실행 버튼 — 강조 */
    .stButton > button {{
        background-color: {ACCENT};
        color: {BG} !important;
        border: 1px solid {ACCENT};
        font-weight: 700;
        padding: 12px 32px;
        border-radius: 8px;
        font-size: 1rem;
    }}
    .stButton > button:hover {{
        background-color: {TEXT};
        border-color: {TEXT};
        color: {BG} !important;
    }}
    .stButton > button * {{
        color: {BG} !important;
    }}
    
    /* 다운로드 버튼 — 강조 */
    .stDownloadButton > button {{
        background-color: {ACCENT};
        color: {BG} !important;
        border: 1px solid {ACCENT};
        font-weight: 700;
        padding: 12px 32px;
        border-radius: 8px;
        font-size: 1rem;
    }}
    .stDownloadButton > button:hover {{
        background-color: {TEXT};
        border-color: {TEXT};
        color: {BG} !important;
    }}
    .stDownloadButton > button * {{
        color: {BG} !important;
    }}
    
    /* 파일 업로더 — 네이비 강조 */
    [data-testid="stFileUploader"] {{
        background-color: {NAVY};
        padding: 24px;
        border-radius: 10px;
        border: 1px dashed {NAVY_LIGHT};
    }}
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] div {{
        color: {TEXT} !important;
    }}
    [data-testid="stFileUploader"] button {{
        background-color: {ACCENT} !important;
        color: {BG} !important;
        border: 1px solid {ACCENT} !important;
        font-weight: 600;
    }}
    [data-testid="stFileUploader"] button * {{
        color: {BG} !important;
    }}
    [data-testid="stFileUploader"] button:hover {{
        background-color: {TEXT} !important;
        border-color: {TEXT} !important;
    }}
    
    /* 업로드된 파일 카드 — 파일명/크기 잘 보이게 */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFileData"],
    [data-testid="stFileUploaderFileName"],
    .uploadedFile,
    .uploadedFileData,
    .uploadedFileName {{
        background-color: {SURFACE_2} !important;
        color: {TEXT} !important;
    }}
    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploaderFileData"] *,
    [data-testid="stFileUploaderFileName"] *,
    .uploadedFile *,
    .uploadedFileData *,
    .uploadedFileName * {{
        color: {TEXT} !important;
        opacity: 1 !important;
    }}
    [data-testid="stFileUploaderFile"] svg,
    [data-testid="stFileUploaderDeleteBtn"] svg {{
        fill: {TEXT} !important;
        color: {TEXT} !important;
        opacity: 1 !important;
    }}
    [data-testid="stFileUploaderDeleteBtn"] {{
        background-color: transparent !important;
    }}
    [data-testid="stFileUploaderDeleteBtn"]:hover {{
        background-color: {NAVY_LIGHT} !important;
    }}
    
    /* 파일 업로더 영어 안내 텍스트 숨김 (Drag and drop, Limit 200MB 등) */
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] *,
    .css-9emcfa, .css-1uixxvy {{
        display: none !important;
    }}
    /* dropzone에서 버튼만 남기기 위해 정렬 조정 */
    [data-testid="stFileUploaderDropzone"] {{
        justify-content: center !important;
        padding: 12px !important;
    }}
    
    /* 데이터프레임 우상단 툴바 — 박스 완전 제거 */
    [data-testid="stElementToolbar"],
    [data-testid="stElementToolbarButton"],
    [data-testid="stElementToolbarButtonGroup"],
    [data-testid="stDataFrameToolbar"],
    div[data-testid="stElementToolbar"],
    div[data-testid="stElementToolbarButton"] {{
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    [data-testid="stElementToolbar"] button,
    [data-testid="stElementToolbarButton"] button,
    [data-testid="stElementToolbarButtonGroup"] button,
    [data-testid="stDataFrameToolbar"] button {{
        background-color: {SURFACE_2} !important;
        border: none !important;
        box-shadow: none !important;
        color: {TEXT} !important;
        width: 32px !important;
        height: 32px !important;
        padding: 6px !important;
        margin: 2px !important;
        border-radius: 50% !important;
        transition: background-color 0.15s;
        opacity: 0.7 !important;
    }}
    [data-testid="stElementToolbar"] button:hover,
    [data-testid="stElementToolbarButton"] button:hover,
    [data-testid="stElementToolbarButtonGroup"] button:hover,
    [data-testid="stDataFrameToolbar"] button:hover {{
        background-color: {NAVY_LIGHT} !important;
        opacity: 1 !important;
    }}
    /* SVG 아이콘 — filter로 흰색 변환 */
    [data-testid="stElementToolbar"] svg,
    [data-testid="stElementToolbarButton"] svg,
    [data-testid="stElementToolbarButtonGroup"] svg,
    [data-testid="stDataFrameToolbar"] svg,
    [data-testid="stDataFrame"] [class*="toolbar"] svg,
    [data-testid="stDataFrame"] [class*="Toolbar"] svg {{
        filter: brightness(0) invert(1) !important;
        opacity: 1 !important;
    }}
    /* 데이터프레임 내부 추가 selector (Streamlit 새 버전 대응) */
    [data-testid="stDataFrame"] [class*="toolbar"],
    [data-testid="stDataFrame"] [class*="Toolbar"] {{
        background-color: transparent !important;
        border: none !important;
    }}
    
    /* 탭 */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent !important;
        color: {TEXT_MUTED} !important;
        font-weight: 500;
        padding: 12px 20px;
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {TEXT} !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {ACCENT} !important;
        border-bottom: 2px solid {ACCENT} !important;
    }}
    
    /* 데이터프레임 */
    [data-testid="stDataFrame"] {{
        background-color: {SURFACE};
        border-radius: 10px;
        border: 1px solid {BORDER};
    }}
    
    /* 알림 — 외부 사각형 박스 제거, 내부 둥근 컨테이너만 표시 */
    div[data-testid="stAlert"] {{
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }}
    div[data-testid="stAlertContainer"],
    [data-baseweb="notification"],
    div[data-testid="stAlert"] > div {{
        background-color: {SUCCESS_BG} !important;
        border: 1px solid {NAVY_LIGHT} !important;
        border-left: 4px solid {ACCENT} !important;
        border-radius: 8px !important;
        color: {TEXT} !important;
    }}
    div[data-testid="stAlert"] *,
    div[data-testid="stAlertContainer"] *,
    [data-baseweb="notification"] * {{
        color: {TEXT} !important;
        background-color: transparent !important;
    }}
    div[data-testid="stAlert"] svg,
    div[data-testid="stAlertContainer"] svg {{
        fill: {ACCENT} !important;
        color: {ACCENT} !important;
    }}
    
    /* 인풋 */
    .stNumberInput input,
    .stTextInput input,
    .stTextArea textarea {{
        background-color: {SURFACE} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
    }}
    
    /* 캡션 */
    [data-testid="stCaptionContainer"] * {{
        color: {TEXT_MUTED} !important;
    }}
    
    /* expander */
    [data-testid="stExpander"] {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}
    [data-testid="stExpander"] summary {{
        color: {TEXT} !important;
    }}
    
    hr {{ border-color: {BORDER} !important; }}
    
    /* 마크다운 테이블 */
    .stMarkdown table {{
        background-color: {SURFACE};
        border-collapse: collapse;
        border-radius: 8px;
        overflow: hidden;
    }}
    .stMarkdown th {{
        background-color: {NAVY};
        color: {TEXT} !important;
        font-weight: 600;
        padding: 10px 14px;
        border: 1px solid {BORDER};
    }}
    .stMarkdown td {{
        color: {TEXT} !important;
        padding: 10px 14px;
        border: 1px solid {BORDER};
        background-color: {SURFACE};
    }}
    
    /* 마크다운 인라인 코드 (`코드`) */
    .stMarkdown code,
    .stMarkdown p code,
    .stMarkdown li code,
    .stMarkdown td code {{
        background-color: {SURFACE_2} !important;
        color: {ACCENT} !important;
        padding: 2px 8px !important;
        border-radius: 4px;
        font-size: 0.85rem;
        border: 1px solid {BORDER};
    }}
    
    /* 마크다운 코드 블록 (```코드```) */
    .stMarkdown pre {{
        background-color: {SURFACE_2} !important;
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 14px 18px !important;
    }}
    .stMarkdown pre code {{
        background-color: transparent !important;
        color: {TEXT} !important;
        border: none !important;
        padding: 0 !important;
        font-size: 0.85rem;
    }}
    .stMarkdown pre * {{
        color: {TEXT} !important;
    }}
    
    /* 일반 코드 블록 */
    [data-testid="stCodeBlock"] {{
        background-color: {SURFACE_2} !important;
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}
    [data-testid="stCodeBlock"] * {{
        color: {TEXT} !important;
        background-color: transparent !important;
    }}
    
    /* 헤더 배지 */
    .header-brand-tag {{
        display: inline-block;
        background-color: {NAVY};
        color: {TEXT} !important;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 4px 12px;
        border-radius: 12px;
        letter-spacing: 0.5px;
    }}
    
    /* 헤더 영역 — 일부 요소만 숨김 (사이드바 토글 유지) */
    [data-testid="stStatusWidget"] {{ display: none !important; }}
    .stDeployButton {{ display: none !important; }}
    footer {{ visibility: hidden !important; }}
    
    /* 헤더 배경 다크 톤으로 */
    header[data-testid="stHeader"] {{
        background-color: {BG} !important;
    }}
    header[data-testid="stHeader"] * {{
        color: {TEXT} !important;
    }}
    header[data-testid="stHeader"] svg {{
        fill: {TEXT} !important;
        color: {TEXT} !important;
    }}
    
    /* 사이드바 토글 버튼(축소/확장) 보이게 */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    button[kind="header"] {{
        background-color: {SURFACE_2} !important;
        color: {TEXT} !important;
    }}
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="collapsedControl"] svg,
    button[kind="header"] svg {{
        fill: {TEXT} !important;
    }}
    
    /* 메인 영역 padding 확보 (sticky 푸터에 가리지 않게) */
    [data-testid="stMain"],
    section.main,
    .main {{
        padding-bottom: 80px !important;
    }}
    
    /* sticky 푸터 */
    .app-footer-sticky {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: {SURFACE};
        border-top: 1px solid {BORDER};
        padding: 10px 28px;
        font-size: 0.82rem;
        color: {TEXT_DIM} !important;
        z-index: 100;
        text-align: center;
    }}
    .app-footer-sticky span {{
        color: {TEXT_DIM} !important;
        margin: 0 10px;
    }}
    .app-footer-sticky strong {{
        color: {TEXT} !important;
    }}
    .app-footer-sticky .sep {{
        color: {BORDER} !important;
        margin: 0 8px;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 헤더
# ==========================================
st.markdown(f"<div style='margin-bottom: 8px;'><span class='header-brand-tag'>온누리커뮤니케이션</span></div>", unsafe_allow_html=True)
st.markdown("<h1 style='margin-top: 0; margin-bottom: 4px;'>마케팅 기여도 분석</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{TEXT_MUTED} !important; margin-top:0;'>청정일 베이스라인 + 매체별 회귀 가중치 산출</p>", unsafe_allow_html=True)

# 푸터 (sticky, 모든 화면에서 보이도록 페이지 상단에서 한 번만 렌더)
st.markdown("""
<div class="app-footer-sticky">
    <span><strong>마케팅 기여도 분석 도구</strong></span>
    <span class="sep">·</span>
    <span>2026-05-27</span>
    <span class="sep">·</span>
    <span>박시은</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 사이드바
# ==========================================
st.sidebar.header("분석 정보")

DEFAULT_BRAND_FEATURES = {
    '란시노': ['숏폼조회수', '사이다조회수'],
    '라쉼': ['숏폼조회수', '사이다조회수'],
    '뉴트리딥': ['1~3위조회수'],
    '포벰브': ['1~3위조회수']
}

MEDIA_ORDER = ['1~3위조회수', '숏폼조회수', '사이다조회수']

st.sidebar.markdown("**매체별 lag 정책**")
st.sidebar.caption(
    "1~3위조회수: lag 없음 (당일만)\n"
    "숏폼/사이다: 1, 3, 5, 7일 자동 선택"
)

st.sidebar.markdown("**분석 기준값 (고정)**")
st.sidebar.caption("최소 데이터 30일 / 매체 단독일 최소 20개")

st.sidebar.markdown("---")
st.sidebar.markdown("**필수 컬럼**")
st.sidebar.code(
    "날짜\n브랜드\n제품\n메타제외매출\n숏폼조회수\n사이다조회수\n1~3위조회수",
    language=None
)

# ==========================================
# 파일 업로드 (xlsx 또는 csv)
# ==========================================
uploaded_file = st.file_uploader(
    "일간데이터 업로드 (xlsx 또는 csv)",
    type=["xlsx", "csv"],
    key="main_data"
)
uploaded_cida = st.file_uploader(
    "기획사이다_raw 업로드 (선택, xlsx 또는 csv)",
    type=["xlsx", "csv"],
    key="cida_raw",
    help="조회수가 비어있는 콘텐츠는 같은 채널의 평균 조회수로 보정합니다."
)


def find_header_row(xl, sheet, required_keywords):
    """필수 컬럼 키워드가 가장 많이 매칭되는 행을 헤더로 자동 선택 (0~5행 시도)."""
    best_header = 0
    best_match_count = -1
    for h in range(6):
        try:
            df_try = pd.read_excel(xl, sheet_name=sheet, header=h, nrows=1)
            cols_lower = [str(c).lower().strip() for c in df_try.columns]
            match_count = sum(
                1 for kw in required_keywords
                if any(kw.lower() in c for c in cols_lower)
            )
            if match_count > best_match_count:
                best_match_count = match_count
                best_header = h
        except Exception:
            continue
    return best_header


def read_data_file(file, target_sheet=None, required_keywords=None):
    """xlsx/csv 자동 감지 + 시트 매칭 + 헤더 행 자동 감지."""
    import unicodedata
    
    name = file.name.lower()
    if name.endswith('.xlsx'):
        xl = pd.ExcelFile(file)
        all_sheets = xl.sheet_names
        
        # 1. 시트 매칭
        used_sheet = None
        if target_sheet is None:
            used_sheet = all_sheets[0]
        elif target_sheet in all_sheets:
            used_sheet = target_sheet
        else:
            def normalize(s):
                return unicodedata.normalize('NFC', str(s)).strip()
            target_norm = normalize(target_sheet)
            for s in all_sheets:
                if normalize(s) == target_norm:
                    used_sheet = s
                    break
            if used_sheet is None:
                for s in all_sheets:
                    if target_norm in normalize(s) or normalize(s) in target_norm:
                        used_sheet = s
                        break
            if used_sheet is None:
                used_sheet = all_sheets[0]
        
        # 2. 헤더 행 자동 감지
        if required_keywords:
            header_row = find_header_row(xl, used_sheet, required_keywords)
        else:
            header_row = 0
        
        df = pd.read_excel(xl, sheet_name=used_sheet, header=header_row)
        return df, used_sheet, all_sheets
    else:
        return pd.read_csv(file, thousands=','), None, None


if uploaded_file is None:
    st.info("일간데이터 파일을 업로드하세요 (xlsx 또는 csv).")
    st.stop()

try:
    df, used_sheet, all_sheets = read_data_file(
        uploaded_file,
        target_sheet='일간데이터',
        required_keywords=['날짜', '브랜드', '제품', '메타제외매출']
    )
    if all_sheets is not None and used_sheet != '일간데이터':
        st.warning(
            f"'일간데이터' 시트를 찾지 못해 '{used_sheet}' 시트를 사용합니다. "
            f"파일 내 전체 시트: {all_sheets}"
        )
except Exception as e:
    st.error(f"파일 읽기 실패: {e}")
    st.stop()

required_cols = ['날짜', '브랜드', '제품', '메타제외매출']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"필수 컬럼 누락: {missing}")
    st.info(f"현재 시트 컬럼: {df.columns.tolist()}")
    st.stop()

for c in MEDIA_ORDER:
    if c not in df.columns:
        df[c] = 0
        st.warning(f"'{c}' 컬럼 없음. 0으로 처리.")

df = df.dropna(subset=['날짜'])
df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
df = df.dropna(subset=['날짜'])

df_products = df[df['제품'] != '전체'].sort_values(by=['브랜드', '제품', '날짜']).copy()

# 브랜드별 집계
df_brand = df_products.groupby(['날짜', '브랜드'], as_index=False).agg({
    '메타제외매출': 'sum',
    '숏폼조회수': 'sum',
    '사이다조회수': 'sum',
    '1~3위조회수': 'sum',
}).sort_values(by=['브랜드', '날짜'])

# ==========================================
# 기획사이다_raw 보정 (선택적)
# ==========================================
cida_correction_info = None
if uploaded_cida is not None:
    try:
        df_cida, used_cida_sheet, cida_all_sheets = read_data_file(
            uploaded_cida,
            target_sheet='기획사이다_raw',
            required_keywords=['게시일자', '브랜드', '채널', '조회수', 'url']
        )
        if cida_all_sheets is not None and used_cida_sheet != '기획사이다_raw':
            st.warning(
                f"'기획사이다_raw' 시트를 찾지 못해 '{used_cida_sheet}' 시트를 사용합니다. "
                f"파일 내 전체 시트: {cida_all_sheets}"
            )
    except Exception as e:
        st.error(f"기획사이다_raw 파일 읽기 실패: {e}")
        df_cida = None
    
    if df_cida is not None:
        # 컬럼명 정규화 (소문자 url로 통일)
        col_rename = {}
        for c in df_cida.columns:
            c_str = str(c).strip()
            if c_str.lower() == 'url':
                col_rename[c] = 'url'
        if col_rename:
            df_cida = df_cida.rename(columns=col_rename)
        
        cida_required = ['게시일자', '브랜드', '제품', '채널', '조회수', 'url']
        cida_missing = [c for c in cida_required if c not in df_cida.columns]
        
        if cida_missing:
            st.warning(
                f"기획사이다_raw 필수 컬럼 누락: {cida_missing}. 보정 건너뜀.\n"
                f"현재 컬럼: {df_cida.columns.tolist()}"
            )
        else:
            df_cida['게시일자'] = pd.to_datetime(df_cida['게시일자'], errors='coerce')
            df_cida = df_cida.dropna(subset=['게시일자', '브랜드', '제품', '채널'])
            df_cida['조회수'] = pd.to_numeric(df_cida['조회수'], errors='coerce')
            df_cida['url'] = df_cida['url'].astype(str).str.strip()
            
            # url 있고 조회수 비어있는 행 = 보정 대상
            has_url = (df_cida['url'] != '') & (df_cida['url'].str.lower() != 'nan')
            empty_views = df_cida['조회수'].isna() | (df_cida['조회수'] == 0)
            need_correction = has_url & empty_views
            n_need = int(need_correction.sum())
            
            # (제품, 채널) 조합별 평균 조회수
            valid_rows = df_cida[has_url & ~empty_views]
            combo_avg = valid_rows.groupby(['제품', '채널'])['조회수'].mean()
            
            # 보정 대상 행: (제품, 채널) 조합 평균이 있으면 적용, 없으면 0 (보정 안 함)
            df_cida['보정조회수'] = 0.0
            n_combo_matched = 0
            n_combo_missed = 0
            missed_combos = []
            for idx in df_cida[need_correction].index:
                product = df_cida.at[idx, '제품']
                channel = df_cida.at[idx, '채널']
                key = (product, channel)
                if key in combo_avg.index:
                    df_cida.at[idx, '보정조회수'] = combo_avg[key]
                    n_combo_matched += 1
                else:
                    n_combo_missed += 1
                    missed_combos.append(f"{product}/{channel}")
            
            # 브랜드 + 게시일자별 보정 조회수 합산 (보정된 행만)
            cida_daily = df_cida[df_cida['보정조회수'] > 0].groupby(
                ['게시일자', '브랜드'], as_index=False
            )['보정조회수'].sum()
            cida_daily.columns = ['날짜', '브랜드', '사이다조회수_보정']
            
            # df_brand에 병합. 원본은 그대로 두고, 보정값 합산한 '사이다조회수_보정후' 별도 컬럼 생성
            df_brand = df_brand.merge(cida_daily, on=['날짜', '브랜드'], how='left')
            df_brand['사이다조회수_보정'] = df_brand['사이다조회수_보정'].fillna(0)
            df_brand['사이다조회수_보정후'] = df_brand['사이다조회수'] + df_brand['사이다조회수_보정']
            # 사이다조회수 원본은 유지 (청정일 정의에 사용)
            
            # 보정 못 한 조합 안내
            unique_missed = sorted(set(missed_combos)) if missed_combos else []
            
            cida_correction_info = {
                'total_rows': len(df_cida),
                'corrected': n_combo_matched,
                'skipped': n_combo_missed,
                'combos_used': len(combo_avg),
                'missed_combos': unique_missed,
                'brands_affected': sorted(cida_daily['브랜드'].unique().tolist()) if len(cida_daily) > 0 else [],
                'total_corrected_views': int(df_brand['사이다조회수_보정'].sum()),
            }
            
            df_brand = df_brand.drop(columns=['사이다조회수_보정'])

df_brand['요일'] = df_brand['날짜'].dt.day_name()
df_brand['청정일'] = (df_brand[MEDIA_ORDER].sum(axis=1) == 0)
# 사이다조회수_보정후 컬럼이 없으면 (사이다 미업로드) 원본 복사
if '사이다조회수_보정후' not in df_brand.columns:
    df_brand['사이다조회수_보정후'] = df_brand['사이다조회수']

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 행수", f"{len(df):,}")
col2.metric("브랜드 단위 행수", f"{len(df_brand):,}")
col3.metric("브랜드 수", df_brand['브랜드'].nunique())
col4.metric("제품 수", df_products['제품'].nunique())

latest_date = df_brand['날짜'].max()
earliest_date = df_brand['날짜'].min()
st.caption(f"데이터 기간: {earliest_date.strftime('%Y-%m-%d')} ~ {latest_date.strftime('%Y-%m-%d')}")

if cida_correction_info is not None:
    info = cida_correction_info
    brands_str = ', '.join(info['brands_affected']) if info['brands_affected'] else '없음'
    st.info(
        f"기획사이다 보정 — 결측 조회수 {info['corrected']}건을 "
        f"(제품×채널) 조합 {info['combos_used']}개의 평균치로 보정 → "
        f"총 {info['total_corrected_views']:,}회 추가 합산 "
        f"(영향 브랜드: {brands_str})"
    )
    
    if info['skipped'] > 0:
        with st.expander(f"보정 못 한 행 {info['skipped']}건 — (제품×채널) 조합 없음", expanded=False):
            st.caption(
                f"해당 (제품, 채널) 조합으로 조회수 데이터가 한 건도 없어 평균을 산출할 수 없는 케이스입니다. "
                f"이 행들은 보정 없이 그대로 두며, 해당 일자의 사이다조회수가 과소 측정될 수 있습니다."
            )
            missed_preview = info['missed_combos'][:30]
            st.code('\n'.join(missed_preview))
            if len(info['missed_combos']) > 30:
                st.caption(f"...외 {len(info['missed_combos']) - 30}개")

# 분석 실행 버튼 + 완료 메시지 한 줄
btn_col, info_col = st.columns([1, 4])
with btn_col:
    run_analysis = st.button("분석 실행", type="primary", use_container_width=True)
with info_col:
    result_placeholder = st.empty()

if not run_analysis:
    st.stop()


# ==========================================
# 헬퍼 함수
# ==========================================
# 매체별 lag 정책: 1~3위조회수는 lag 없음 (당일 상위노출 조회수 그대로)
# 숏폼/사이다는 누적 효과 있어 lag 자동 선택
MEDIA_LAG_POLICY = {
    '1~3위조회수': 'none',      # lag 미적용
    '숏폼조회수': 'auto',       # 1, 3, 5, 7 중 최적 자동 선택
    '사이다조회수': 'auto',
}


def fit_regression_with_lag(group_data, feature, lag_max):
    """특정 매체 + lag 1~lag_max 단순 회귀. R², 계수, 샘플수 반환."""
    df_local = group_data.copy()
    features = [feature]
    for lag in range(1, lag_max + 1):
        col = f'{feature}_lag{lag}'
        df_local[col] = df_local[feature].shift(lag)
        features.append(col)
    
    reg_data = df_local[['Y'] + features].dropna()
    if len(reg_data) < 20:
        return None, None, None
    
    X = reg_data[features]
    y = reg_data['Y']
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    r2 = model.score(X, y)
    coefs = dict(zip(features, model.coef_))
    return r2, coefs, len(reg_data)


def analyze_group(group, features_base, brand_label):
    """단일 브랜드 그룹 분석"""
    group = group.copy().reset_index(drop=True)
    
    result = {
        '브랜드': brand_label,
        '청정일수': 0,
        '청정일_요일커버': 0,
        '오가닉_분포등급': '-',
        '최근청정일_경과(일)': '-',
        '오가닉_최신성등급': '-',
        '오가닉매출': 0,
        '오가닉_월': 0, '오가닉_화': 0, '오가닉_수': 0, '오가닉_목': 0,
        '오가닉_금': 0, '오가닉_토': 0, '오가닉_일': 0,
        '오가닉모드': '-',
        '기저매출_7일vs30일(%)': '-',
        '사이다보정': '-',
        '시즌보정': '-',
        '요일패턴_최대': '-',
        '월패턴_최대': '-',
        '회귀샘플수': 0,
        '전체R²': '-',
    }
    for m in MEDIA_ORDER:
        result[f'{m}_가중치'] = '-'
        result[f'{m}_최적lag'] = '-'
        result[f'{m}_단독R²'] = '-'
        result[f'{m}_단독일수'] = 0
    result['분석상태'] = '대기'
    
    if len(group) < MIN_DATA_DAYS:
        result['분석상태'] = f'데이터부족({len(group)}일)'
        return result
    
    if group['메타제외매출'].sum() == 0:
        result['분석상태'] = '매출 0'
        return result
    
    # 청정일 분석
    clean_days = group[group['청정일']].copy()
    clean_days['매출_조정'] = clean_days['메타제외매출'].clip(lower=0)
    n_clean = len(clean_days)
    
    if n_clean == 0:
        result['분석상태'] = '청정일 없음'
        return result
    
    organic_avg = clean_days['매출_조정'].mean()
    weekdays_covered = clean_days['요일'].nunique()
    latest_clean = clean_days['날짜'].max()
    latest_data = group['날짜'].max()
    days_since_clean = (latest_data - latest_clean).days
    
    # 요일별 오가닉매출 산출 (표본 5개 이상인 요일만, 미만은 전체 평균 fallback)
    weekday_names = ['월', '화', '수', '목', '금', '토', '일']
    clean_days['요일num'] = clean_days['날짜'].dt.dayofweek
    wd_count_full = clean_days.groupby('요일num').size().to_dict()
    wd_avg_full = clean_days.groupby('요일num')['매출_조정'].mean().to_dict()
    
    organic_by_weekday = {}
    organic_by_weekday_source = {}  # '요일별' 또는 '전체평균'
    MIN_SAMPLE_WD = 5
    for d in range(7):
        n = wd_count_full.get(d, 0)
        v = wd_avg_full.get(d, np.nan)
        if n >= MIN_SAMPLE_WD and not pd.isna(v):
            organic_by_weekday[d] = round(v, 0)
            organic_by_weekday_source[d] = '요일별'
        else:
            organic_by_weekday[d] = round(organic_avg, 0)
            organic_by_weekday_source[d] = '전체평균'
    
    if weekdays_covered == 7:
        org_dist_grade = 'HIGH'
    elif weekdays_covered >= 5:
        org_dist_grade = 'MEDIUM'
    else:
        org_dist_grade = 'LOW'
    
    if days_since_clean <= 30:
        org_recency_grade = 'HIGH'
    elif days_since_clean <= 90:
        org_recency_grade = 'MEDIUM'
    else:
        org_recency_grade = 'LOW'
    
    # 요일별 오가닉 컬럼들
    wd_organic_cols = {}
    for d in range(7):
        wd_organic_cols[f'오가닉_{weekday_names[d]}'] = organic_by_weekday[d]
    
    result.update({
        '청정일수': n_clean,
        '오가닉매출': round(organic_avg, 0),
        '청정일_요일커버': weekdays_covered,
        '오가닉_분포등급': org_dist_grade,
        '최근청정일_경과(일)': days_since_clean,
        '오가닉_최신성등급': org_recency_grade,
        **wd_organic_cols,
    })
    
    # 기저매출 변동률
    group['매출_음수제거'] = group['메타제외매출'].clip(lower=0)
    group['평균_7일'] = group['매출_음수제거'].rolling(7, min_periods=3).mean().shift(1)
    group['평균_30일'] = group['매출_음수제거'].rolling(30, min_periods=10).mean().shift(1)
    
    recent = group.dropna(subset=['평균_7일', '평균_30일']).tail(30)
    if len(recent) > 0 and recent['평균_30일'].mean() > 0:
        baseline_dev_pct = ((recent['평균_7일'].mean() - recent['평균_30일'].mean()) / recent['평균_30일'].mean()) * 100
    else:
        baseline_dev_pct = 0
    result['기저매출_7일vs30일(%)'] = round(baseline_dev_pct, 1)
    
    # ==========================================
    # ==========================================
    # 사이다 보정 적용/미적용 자동 비교 (사이다 사용 브랜드만)
    # ==========================================
    use_cida_correction = False
    if '사이다조회수' in features_base and '사이다조회수_보정후' in group.columns:
        # 두 버전 모두 R² 측정
        def measure_r2_with_sai(sai_col):
            """주어진 사이다 컬럼으로 lag 1~7 모두 X 변수에 추가 후 R²"""
            tmp_g = group.copy()
            # X 변수 구성
            xs = []
            for f in features_base:
                if f == '사이다조회수':
                    tmp_g['_사이다'] = tmp_g[sai_col]
                    xs.append('_사이다')
                    if MEDIA_LAG_POLICY.get(f) == 'auto':
                        for lag in range(1, max(LAG_CANDIDATES) + 1):
                            tmp_g[f'_사이다_lag{lag}'] = tmp_g['_사이다'].shift(lag)
                            xs.append(f'_사이다_lag{lag}')
                else:
                    xs.append(f)
                    if MEDIA_LAG_POLICY.get(f) == 'auto':
                        for lag in range(1, max(LAG_CANDIDATES) + 1):
                            tmp_g[f'{f}_lag{lag}'] = tmp_g[f].shift(lag)
                            xs.append(f'{f}_lag{lag}')
            
            # Y_raw 기준 (시즌 보정 전)
            tmp_g['_Y'] = tmp_g['메타제외매출'] - organic_avg
            reg = tmp_g[['_Y'] + xs].dropna()
            if len(reg) < 20:
                return -1
            m = LinearRegression()
            m.fit(reg[xs], reg['_Y'])
            return m.score(reg[xs], reg['_Y'])
        
        r2_원본 = measure_r2_with_sai('사이다조회수')
        r2_보정 = measure_r2_with_sai('사이다조회수_보정후')
        
        if r2_보정 > r2_원본:
            # 보정 적용: 사이다조회수 컬럼 자체를 보정값으로 교체
            group['사이다조회수'] = group['사이다조회수_보정후']
            use_cida_correction = True
            result['사이다보정'] = f'적용 (R² {r2_원본:.3f}→{r2_보정:.3f})'
        else:
            # 원본 유지
            result['사이다보정'] = f'미적용 (보정 시 R² {r2_보정:.3f} < 원본 {r2_원본:.3f})'
    elif '사이다조회수' not in features_base:
        result['사이다보정'] = 'N/A (사이다 미사용 브랜드)'
    else:
        result['사이다보정'] = '미적용 (보정 데이터 없음)'
    
    # ==========================================
    # 시즌성 보정 (요일/월 계수) — 안정화 적용
    # ==========================================
    season_info = {
        'weekday_coef': {},
        'month_coef': {},
    }
    
    # 기본 Y (보정 전, 단일 오가닉)
    group['Y_raw'] = group['메타제외매출'] - organic_avg
    
    # 요일별 오가닉 Y
    group['요일num'] = group['날짜'].dt.dayofweek
    group['Y_wd'] = group['메타제외매출'] - group['요일num'].map(organic_by_weekday).fillna(organic_avg)
    
    apply_season = False
    
    if n_clean >= 14:
        clean_days['요일num'] = clean_days['날짜'].dt.dayofweek
        clean_days['월'] = clean_days['날짜'].dt.month
        overall_avg = clean_days['매출_조정'].mean()
        
        if overall_avg > 0:
            # 표본 수 충분한 요일/월만 계수 적용 (5개 미만이면 1.0)
            MIN_SAMPLE = 5
            wd_count = clean_days.groupby('요일num').size()
            mo_count = clean_days.groupby('월').size()
            wd_avg = clean_days.groupby('요일num')['매출_조정'].mean()
            mo_avg = clean_days.groupby('월')['매출_조정'].mean()
            
            wd_coef_raw = (wd_avg / overall_avg).to_dict()
            mo_coef_raw = (mo_avg / overall_avg).to_dict()
            
            # outlier 클립 (0.6 ~ 1.7) + 표본 부족 시 1.0
            def safe_coef(coef_dict, count_dict, full_range, lo=0.6, hi=1.7):
                result_dict = {}
                for k in full_range:
                    n = count_dict.get(k, 0)
                    if n < MIN_SAMPLE:
                        result_dict[k] = 1.0
                    else:
                        v = coef_dict.get(k, 1.0)
                        if pd.isna(v):
                            result_dict[k] = 1.0
                        else:
                            result_dict[k] = max(lo, min(hi, v))
                return result_dict
            
            wd_coef = safe_coef(wd_coef_raw, wd_count.to_dict(), range(7))
            mo_coef = safe_coef(mo_coef_raw, mo_count.to_dict(), range(1, 13))
            
            # 보정 효과 사전 평가 (회귀로 R² 비교는 뒤에서)
            # 일단 계수 적용해본 Y_adjusted 산출
            group['요일num'] = group['날짜'].dt.dayofweek
            group['월'] = group['날짜'].dt.month
            group['시즌계수'] = (
                group['요일num'].map(wd_coef).fillna(1.0) *
                group['월'].map(mo_coef).fillna(1.0)
            )
            group['조정매출'] = group['메타제외매출'] / group['시즌계수']
            
            clean_days['시즌계수'] = (
                clean_days['요일num'].map(wd_coef).fillna(1.0) *
                clean_days['월'].map(mo_coef).fillna(1.0)
            )
            organic_adjusted = (clean_days['매출_조정'] / clean_days['시즌계수']).mean()
            group['Y_adj'] = group['조정매출'] - organic_adjusted
            
            apply_season = True
            season_info['weekday_coef'] = wd_coef
            season_info['month_coef'] = mo_coef
    
    # 시즌 보정 적용/미적용 비교 — 전체 회귀 사전 평가
    def quick_r2_full(y_col):
        """모든 매체 X 변수로 다중회귀 R² 빠르게 산출"""
        all_x = list(features_base)
        gtmp = group.copy()
        for f in features_base:
            if MEDIA_LAG_POLICY.get(f) == 'auto':
                # 최대 lag=7까지 모든 lag(1~7) 변수로 추가 (fit_regression_with_lag와 동일 방식)
                for lag in range(1, max(LAG_CANDIDATES) + 1):
                    gtmp[f'{f}_lag{lag}'] = gtmp[f].shift(lag)
                    all_x.append(f'{f}_lag{lag}')
        reg = gtmp[[y_col] + all_x].dropna()
        if len(reg) < 20:
            return -1
        m = LinearRegression()
        m.fit(reg[all_x], reg[y_col])
        return m.score(reg[all_x], reg[y_col])
    
    # 3가지 Y 비교: 단일 오가닉 / 요일별 오가닉 / (조건부) 시즌 보정
    r2_단일 = quick_r2_full('Y_raw')
    r2_요일별 = quick_r2_full('Y_wd')
    r2_시즌 = quick_r2_full('Y_adj') if apply_season else -999
    
    # 최적 선택
    candidates = [
        ('단일', r2_단일, 'Y_raw'),
        ('요일별', r2_요일별, 'Y_wd'),
    ]
    if apply_season:
        candidates.append(('시즌보정', r2_시즌, 'Y_adj'))
    
    best_mode, best_r2, best_y_col = max(candidates, key=lambda x: x[1])
    group['Y'] = group[best_y_col]
    
    result['오가닉모드'] = f"{best_mode} (R²={best_r2:.3f})"
    
    # 비교 결과 시즌보정 컬럼에 요약
    if apply_season:
        if best_mode == '시즌보정':
            wd_max_key = max(season_info['weekday_coef'], key=season_info['weekday_coef'].get)
            mo_max_key = max(season_info['month_coef'], key=season_info['month_coef'].get)
            result['시즌보정'] = f'적용 (단일 {r2_단일:.3f} / 요일별 {r2_요일별:.3f} / 시즌 {r2_시즌:.3f})'
            result['요일패턴_최대'] = f"{weekday_names[wd_max_key]} ({season_info['weekday_coef'][wd_max_key]:.2f})"
            result['월패턴_최대'] = f"{mo_max_key}월 ({season_info['month_coef'][mo_max_key]:.2f})"
        else:
            result['시즌보정'] = f'미채택 (단일 {r2_단일:.3f} / 요일별 {r2_요일별:.3f} / 시즌 {r2_시즌:.3f})'
    else:
        if n_clean < 14:
            result['시즌보정'] = f'미적용 (청정일 {n_clean}개 < 14)'
        else:
            result['시즌보정'] = '미적용 (오가닉 0)'
    
    # 매체별 lag 정책 적용
    media_best = {}
    for f in features_base:
        policy = MEDIA_LAG_POLICY.get(f, 'auto')
        
        if policy == 'none':
            # lag 없이 당일만으로 회귀
            r2, coefs, n = fit_regression_with_lag(group, f, 0)
            if r2 is None:
                media_best[f] = {'lag': 0, 'r2': -1, 'coefs': {}, 'n': 0}
                result[f'{f}_최적lag'] = '-'
            else:
                media_best[f] = {'lag': 0, 'r2': r2, 'coefs': coefs, 'n': n}
                result[f'{f}_최적lag'] = 0  # lag 0 = 당일만
        else:
            # lag 1, 3, 5, 7 중 R² 최대 자동 선택
            best = {'lag': 1, 'r2': -1, 'coefs': {}, 'n': 0}
            for lag_try in LAG_CANDIDATES:
                r2, coefs, n = fit_regression_with_lag(group, f, lag_try)
                if r2 is None:
                    continue
                if r2 > best['r2']:
                    best = {'lag': lag_try, 'r2': r2, 'coefs': coefs, 'n': n}
            media_best[f] = best
            result[f'{f}_최적lag'] = best['lag']
        
        # 가중치: 당일 + 모든 lag 계수 합 (음수는 0)
        total = media_best[f]['coefs'].get(f, 0)
        for lag in range(1, media_best[f]['lag'] + 1):
            total += media_best[f]['coefs'].get(f'{f}_lag{lag}', 0)
        result[f'{f}_가중치'] = max(0, round(total, 2))
    
    # 전체 다중회귀
    all_features = []
    group_full = group.copy()
    for f in features_base:
        all_features.append(f)
        best_lag = media_best[f]['lag']
        for lag in range(1, best_lag + 1):
            col = f'{f}_lag{lag}'
            if col not in group_full.columns:
                group_full[col] = group_full[f].shift(lag)
            all_features.append(col)
    
    reg_data = group_full[['Y'] + all_features].dropna(subset=['Y'] + all_features)
    
    if len(reg_data) < 20:
        result['회귀샘플수'] = len(reg_data)
        result['분석상태'] = f'회귀샘플 부족 ({len(reg_data)})'
        return result
    
    X = reg_data[all_features]
    y = reg_data['Y']
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    total_r2 = model.score(X, y)
    
    result['전체R²'] = round(total_r2, 3)
    result['회귀샘플수'] = len(reg_data)
    
    # 매체별 단독R²
    for f in features_base:
        best_lag = media_best[f]['lag']
        other_media = [m for m in features_base if m != f]
        if other_media:
            mask = (reg_data[f] > 0)
            for om in other_media:
                mask = mask & (reg_data[om] == 0)
            solo_data = reg_data[mask]
        else:
            solo_data = reg_data[reg_data[f] > 0]
        
        n_solo = len(solo_data)
        result[f'{f}_단독일수'] = n_solo
        
        if n_solo < MIN_SOLO_DAYS:
            result[f'{f}_단독R²'] = '단독일 부족'
            continue
        
        solo_features = [f] + [f'{f}_lag{lag}' for lag in range(1, best_lag + 1)]
        X_solo = solo_data[solo_features]
        y_solo = solo_data['Y']
        m_solo = LinearRegression(fit_intercept=True)
        m_solo.fit(X_solo, y_solo)
        r2_solo = m_solo.score(X_solo, y_solo)
        result[f'{f}_단독R²'] = round(r2_solo, 3)
    
    result['분석상태'] = '분석성공'
    return result


# ==========================================
# 브랜드별 분석
# ==========================================
results = []
progress = st.progress(0)
status = st.empty()

all_brands = list(df_brand.groupby('브랜드'))

for idx, (brand, group) in enumerate(all_brands):
    progress.progress((idx + 1) / len(all_brands))
    status.text(f"분석 중: {brand}")
    
    if brand not in DEFAULT_BRAND_FEATURES:
        continue
    
    result = analyze_group(group, DEFAULT_BRAND_FEATURES[brand], brand)
    results.append(result)

progress.empty()
status.empty()

if len(results) == 0:
    st.warning("분석 결과 없음.")
    st.stop()

result_df = pd.DataFrame(results)

ordered_cols = [
    '브랜드',
    '청정일수', '청정일_요일커버', '오가닉_분포등급',
    '최근청정일_경과(일)', '오가닉_최신성등급',
    '오가닉매출',
    '오가닉_월', '오가닉_화', '오가닉_수', '오가닉_목', '오가닉_금', '오가닉_토', '오가닉_일',
    '오가닉모드',
    '기저매출_7일vs30일(%)',
    '사이다보정', '시즌보정', '요일패턴_최대', '월패턴_최대',
    '회귀샘플수', '전체R²',
]
for m in MEDIA_ORDER:
    ordered_cols.append(f'{m}_가중치')
    ordered_cols.append(f'{m}_최적lag')
    ordered_cols.append(f'{m}_단독R²')
    ordered_cols.append(f'{m}_단독일수')
ordered_cols.append('분석상태')
ordered_cols = [c for c in ordered_cols if c in result_df.columns]
result_df = result_df[ordered_cols]

success_n = sum(result_df['분석상태'] == '분석성공')
result_placeholder.success(f"분석 완료 — 전체 {len(result_df)}개 브랜드 / 분석성공 {success_n}개")

# ==========================================
# 색상 스타일 — HIGH만 강조
# ==========================================
def color_grade(val):
    if val == 'HIGH':
        return f'background-color: {HIGHLIGHT_BG}; color: {TEXT}; font-weight: 600'
    return ''

def color_r2(val):
    if val == '-' or val == '단독일 부족' or val is None or pd.isna(val):
        return ''
    try:
        v = float(val)
    except:
        return ''
    if v >= 0.5:
        return f'background-color: {HIGHLIGHT_BG}; color: {TEXT}; font-weight: 600'
    return ''

def color_status(val):
    if val == '분석성공':
        return f'background-color: {HIGHLIGHT_BG}; color: {TEXT}; font-weight: 600'
    return ''

# ==========================================
# 결과 탭
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["가중치 요약", "전체 결과", "용어 정의", "산정 기준"])

with tab1:
    st.subheader("매체별 가중치 (브랜드 단위)")
    st.caption("시트에 적용할 핵심 가중치입니다. 단독R²로 신뢰도 확인 후 사용하세요.")
    
    summary_cols = ['브랜드', '오가닉매출', '전체R²']
    for m in MEDIA_ORDER:
        summary_cols += [f'{m}_가중치', f'{m}_최적lag', f'{m}_단독R²']
    summary_cols.append('분석상태')
    summary_cols = [c for c in summary_cols if c in result_df.columns]
    
    style2 = result_df[summary_cols].style
    style2 = style2.map(color_r2, subset=[c for c in summary_cols if 'R²' in c])
    style2 = style2.map(color_status, subset=['분석상태'])
    st.dataframe(style2, use_container_width=True, height=400)

with tab2:
    st.subheader("전체 분석 결과")
    st.caption("청정일 분포, 회귀샘플수 등 상세 정보 포함")
    
    style = result_df.style
    style = style.map(color_grade, subset=['오가닉_분포등급', '오가닉_최신성등급'])
    r2_cols = [c for c in result_df.columns if 'R²' in c]
    if r2_cols:
        style = style.map(color_r2, subset=r2_cols)
    style = style.map(color_status, subset=['분석상태'])
    st.dataframe(style, use_container_width=True, height=600)

with tab3:
    st.subheader("용어 정의")
    glossary_df = pd.DataFrame([
        ['오가닉매출', '광고/마케팅 없이 발생하는 매출', '청정일 매출 평균으로 추정'],
        ['기저매출', '직전 7일 광고 없이 발생하는 매출', '변동플래그용 (참고)'],
        ['마케팅기여매출', '당일매출 - 오가닉매출', '양수만 측정'],
        ['콘텐츠기여매출', '마케팅기여매출 × 채널비중 × 채널내 콘텐츠조회수비중', ''],
        ['마케팅추정ROAS', '콘텐츠기여매출 / 비용', ''],
        ['변동플래그', '기저매출 ± 표준편차 + 15% 허들 통과', '시트 조건부서식 대상'],
    ], columns=['용어', '정의', '비고'])
    st.dataframe(glossary_df, use_container_width=True, hide_index=True, height=300)
    
    st.markdown("### 등급 기준")
    st.markdown("""
| 등급 항목 | HIGH | MEDIUM | LOW |
|---|---|---|---|
| 오가닉_분포등급 | 7요일 모두 청정일 있음 | 5~6요일 | 4요일 이하 |
| 오가닉_최신성등급 | 최근 청정일 30일 이내 | 31~90일 | 90일 초과 |
| R² (회귀 설명력) | 0.5 이상 | 0.3 ~ 0.5 | 0.1 ~ 0.3 (낮음) |
""")

with tab4:
    st.subheader("산정 기준")
    st.markdown(f"""
#### 1. 분석 단위: 브랜드별

같은 브랜드의 모든 제품 데이터를 일자 단위로 합산해 분석합니다.

#### 2. 사이다조회수 보정 (선택, 기획사이다_raw 업로드 시)

기획사이다_raw 시트에 콘텐츠 발행은 기록되었으나 조회수가 비어있는 행이 많아 사이다조회수가 과소 측정되는 문제를 보정합니다.

**보정 로직**:
- 기획사이다_raw 시트에서 url 값이 채워져 있고 조회수가 비어있는 행 식별
- 같은 **(제품 × 채널) 조합**의 조회수가 채워진 행들의 **평균 조회수**를 그 행의 추정 조회수로 가정
- 추정 조회수를 그 행의 **브랜드 + 게시일자** 기준으로 합산
- 메인 일간데이터의 기존 사이다조회수에 **추가** 합산 (기존 조회수는 그대로 유지, 보정분만 더함)

**왜 (제품 × 채널) 조합인가**:
- 같은 채널이라도 제품에 따라 반응이 다름 (예: 같은 카페에서 라쉼 징크앰플 vs 라쉼 펌핑밤은 평균 조회수 차이 큼)
- 단순 채널 평균보다 정밀

**필수 컬럼**: 게시일자, 브랜드, 제품, 채널, 조회수, url

**(제품 × 채널) 조합이 없는 경우**:
- 결측 행의 (제품, 채널) 조합으로 조회수 데이터가 한 건도 없으면 평균 산출 불가
- 해당 행은 **보정 없이 그대로 둠**
- 화면 상단 안내 메시지 + expander에 미매칭 조합 목록 표시
- 그 일자의 사이다조회수가 과소 측정될 수 있음을 인지하고 활용

**중복 합산 방지**: 조회수가 이미 채워진 행은 보정 대상에서 제외 (메인 일간데이터에 이미 반영되어 있다고 가정)

**자동 적용/미적용 비교**:
보정 적용이 항상 도움 되는 건 아닙니다 (실제 매출 영향 없던 콘텐츠를 평균값으로 채우면 노이즈가 될 수도). 따라서 각 브랜드별로:

1. 보정 미적용 R² 측정
2. 보정 적용 R² 측정
3. 더 높은 쪽 자동 선택

결과의 `사이다보정` 컬럼:
- `적용 (R² 0.030→0.111)` — 보정으로 R² 상승, 채택
- `미적용 (보정 시 R² 0.319 < 원본 0.512)` — 원본이 더 좋아 미채택
- `N/A` — 사이다 안 쓰는 브랜드 (뉴트리딥/포벰브)

#### 3. 오가닉매출 산정 (단일 + 요일별 + 시즌보정 3가지)

브랜드별로 다음 3가지 베이스라인을 모두 계산하고, **R² 가장 높은 방식을 자동 선택**합니다.

**(A) 단일 오가닉**
- `오가닉매출 = AVG(MAX(0, 청정일 메타제외매출))`
- 모든 날에 같은 베이스라인 적용

**(B) 요일별 오가닉**
- 요일별로 청정일 매출 평균을 별도 산출 (월/화/.../일 각각)
- `Y(그날) = 메타제외매출 - 오가닉(그날 요일)`
- 요일 표본 5건 미만이면 단일 오가닉으로 fallback
- 결과 컬럼: `오가닉_월`, `오가닉_화`, ... `오가닉_일`

**(C) 시즌 보정 (요일계수 × 월계수)**
- 청정일 데이터에서 요일/월 계수 산출 → 매출을 정규화
- 상세는 아래 시즌 보정 항목 참조

**자동 선택**:
3가지로 각각 회귀 학습 후 R² 가장 높은 것을 채택. 결과의 `오가닉모드` 컬럼에 어떤 모드가 선택됐는지 표시.

**음수 매출 처리**: 환불 등 음수 매출은 0으로 클램프 (3가지 모두)

#### 4. 시즌성 보정 (요일/월 패턴)

뇌피셜이 아닌 **데이터 기반**으로 요일·월별 매출 패턴을 자동 추출해 회귀 분석에 반영합니다.

**보정 로직**:

1. **청정일 데이터**(마케팅 영향 없는 날)에서 요일별·월별 평균 매출 산출
2. 각 요일/월 평균 ÷ 전체 평균 = **시즌 계수** (예: 월요일 1.15배, 6월 0.92배)
3. 회귀 분석 시 매출에서 시즌 효과 제거:
   - `조정매출 = 실매출 / (요일계수 × 월계수)`
   - `Y = 조정매출 - 오가닉(조정)`
4. 결과적으로 매체 가중치는 **시즌 영향 제외한 순수 마케팅 효과**만 측정

**예시**:
- 라쉼 청정일 데이터에서 월요일 매출 평균 = 전체 평균의 1.20배
- 5/13(월) 실제 매출 100만원 → 조정매출 = 100 / 1.20 = 83.3만원
- 회귀에서 사용되는 Y값은 시즌 효과 빼고 마케팅 순효과만 반영

**적용 조건**: 청정일 14개 이상일 때만 적용 (그 미만이면 패턴 신뢰 어려움)

결과의 `시즌보정`, `요일패턴_최대`, `월패턴_최대` 컬럼으로 실제 적용 여부와 패턴 확인 가능.

#### 5. 회귀분석 + 매체별 lag 정책

- **Y값**: 조정매출 - 오가닉매출 (시즌 보정 후)
- **X값**: 매체별 당일 조회수 (+ 매체에 따라 lag 변수)
- **가중치**: (당일 계수 + 1~최적lag일 전 계수들) 합, 음수면 0

**매체별 lag 정책**:

| 매체 | lag 정책 | 이유 |
|---|---|---|
| 1~3위조회수 | **lag 없음 (당일만)** | 상위노출 조회수는 해당일 기준 측정값 (누적 X) |
| 숏폼조회수 | **lag 자동 (1/3/5/7 중 R² 최대)** | 누적 조회수, 시차 효과 존재 |
| 사이다조회수 | **lag 자동 (1/3/5/7 중 R² 최대)** | 누적 조회수, 시차 효과 존재 |

결과의 `매체_최적lag` 컬럼:
- `0` = lag 없음 (당일만 사용)
- `1/3/5/7` = 자동 선택된 lag 일수 (당일 + N일 전까지 포함)

#### 6. R² 두 종류

- **전체R²**: 모든 매체 + 각자 최적 lag 동시 학습 모델 설명력
- **매체별 단독R²**: 그 매체만 단독 진행된 날의 데이터만으로 계산
- 단독일 < {MIN_SOLO_DAYS}개면 "단독일 부족" 표시

#### 7. 시트 연동 — 가중치 적용 방법

엑셀 다운로드 후 가중치_마스터 시트를 Google Sheets로 가져와 연동합니다.

**Step 1. 가중치 마스터 시트 임포트**
- 다운로드한 `가중치_마스터.xlsx` 파일을 마스터 시트로 사용
- 또는 별도 탭 `가중치_마스터`에 결과 복붙

**Step 2. 일간데이터 시트에 컬럼 추가**

| 컬럼 | 수식 |
|---|---|
| 오가닉매출 | `=VLOOKUP($C2, '가중치_마스터'!$A:$G, 7, FALSE)` |
| 마케팅기여매출 | `=MAX(0, [메타제외매출] - [오가닉매출])` |

**Step 3. 채널비중 산출 (매체별 가중치 × 당일 조회수)**

| 채널 | 점수 수식 |
|---|---|
| 숏폼 점수 | `[숏폼조회수] × [숏폼_가중치]` |
| 사이다 점수 | `[사이다조회수] × [사이다_가중치]` |
| 1~3위 점수 | `[1~3위조회수] × [1~3위_가중치]` |
| 가중합 | `숏폼점수 + 사이다점수 + 1~3위점수` |

**Step 4. 채널별 기여매출**

```
숏폼 기여매출 = 마케팅기여매출 × (숏폼 점수 / 가중합)
사이다 기여매출 = 마케팅기여매출 × (사이다 점수 / 가중합)
1~3위 기여매출 = 마케팅기여매출 × (1~3위 점수 / 가중합)
```

**Step 5. 콘텐츠 기여매출 (콘텐츠 단위로 쪼개기)**

```
콘텐츠 기여매출 = 채널 기여매출 × (콘텐츠 조회수 / 채널 당일 총 조회수)
```

**Step 6. 마케팅 추정 ROAS**

```
마케팅 추정 ROAS = 콘텐츠 기여매출 / 콘텐츠 비용
```

#### 8. 실무 적용 가이드

- **R² ≥ 0.5 브랜드**: 회귀 가중치 신뢰 가능 → 그대로 시트 적용
- **R² 0.3 ~ 0.5 브랜드**: 참고 사용 + 정성 판단 병행
- **R² < 0.3 브랜드**: 가중치 신뢰 낮음 → 단순 조회수 비율로 대체 권장
""")

# 결과 xlsx 다운로드
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    result_df.to_excel(writer, sheet_name='가중치_마스터', index=False)
    glossary_export = pd.DataFrame([
        ['오가닉매출', '광고/마케팅 없이 발생하는 매출', '청정일 매출 평균으로 추정'],
        ['기저매출', '직전 7일 광고 없이 발생하는 매출', '변동플래그용 (참고)'],
        ['마케팅기여매출', '당일매출 - 오가닉매출', '양수만 측정'],
        ['콘텐츠기여매출', '마케팅기여매출 × 채널비중 × 채널내 콘텐츠조회수비중', ''],
        ['마케팅추정ROAS', '콘텐츠기여매출 / 비용', ''],
        ['변동플래그', '기저매출 ± 표준편차 + 15% 허들 통과', ''],
    ], columns=['용어', '정의', '비고'])
    glossary_export.to_excel(writer, sheet_name='용어집', index=False)

st.download_button(
    label="결과 엑셀 다운로드 (.xlsx)",
    data=buffer.getvalue(),
    file_name="가중치_마스터.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown("---")