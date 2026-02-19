import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import re
import json
import difflib

# API 키 설정
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----데이터 전처리-----

## 카테고리 자동분류

RULES = {
    "식비": ["식당", "카페", "커피", "치킨", "피자", "분식", "김밥", "국밥", "버거", "제과", "베이커리", "배달", "도시락"],
    "금융/보험": ["보험", "손보", "화재", "생명", "대출", "이자"],
    "의료/건강": ["병원", "의원", "약국", "치과", "한의원", "검진", "헬스", "필라테스","GYM"],
    "교통": ["주차", "택시", "버스", "지하철", "주유", "정비", "톨게이트", "고속도로"],
    "쇼핑": ["마트", "편의점", "백화점", "아울렛", "다이소", "올리브영","쿠팡","쿠팡(쿠페이)"],
    "주거/통신": ["관리비", "월세", "가스", "전기", "수도", "통신", "인터넷", "kt", "skt", "유플러스"],
    "구독": ["넷플릭스", "netflix", "유튜브", "youtube", "멜론", "spotify", "애플", "google one"],
    "문화/여가": ["영화", "cgv", "메가박스", "롯데시네마", "공연", "전시", "여행", "숙박", "호텔", "놀이공원","투어"],
}
CATEGORIES = ["식비","교통","쇼핑","주거/통신","구독","의료/건강","문화/여가","금융/보험","기타"]

## 컬럼명 정규화
def _norm_col(x: str) -> str:
    """컬럼명 비교용 정규화: 소문자, 공백/특수문자 제거"""
    s = "" if x is None else str(x)
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9a-zA-Z가-힣_]", "", s)
    return s

def normalize_text(x):
    s = "" if pd.isna(x) else str(x)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

## 규칙을 기반한 카테고리
def rule_category(description: str) -> str:
    d = normalize_text(description)
    if not d:
        return "미분류"
    for cat, keywords in RULES.items():
        for kw in keywords:
            if normalize_text(kw) in d:
                return cat
    return "미분류"

## AI 카테고리
def ai_category_batch(descriptions, api_key, model="gpt-4o-mini"):
    descriptions = list(descriptions)
    if not descriptions:
        return {}

    client = OpenAI(api_key=api_key)
    prompt = f"""
너는 카드 지출의 설명(description)을 지출 카테고리로 분류한다.
카테고리는 반드시 아래 중 하나만 사용:
{", ".join(CATEGORIES)}

규칙:
- 애매하면 "기타"
- 출력은 JSON만. 예: {{"가맹점A":"식비","가맹점B":"교통"}}
- 키는 입력 description 문자열 그대로 사용

description 목록:
{json.dumps(descriptions, ensure_ascii=False)}
""".strip()

    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    txt = res.choices[0].message.content.strip()

    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        start, end = txt.find("{"), txt.rfind("}")
        data = json.loads(txt[start:end+1])

    out = {}
    for k, v in data.items():
        out[k] = v if v in CATEGORIES else "기타"
    return out

def build_category(df, api_key, desc_col="description"):
    ### 1. 규칙
    df["category_rule"] = df[desc_col].apply(rule_category)

    ### 2. 미분류만 AI
    targets = df.loc[df["category_rule"]=="미분류", desc_col].dropna().astype(str).unique()
    ai_map = ai_category_batch(targets, api_key=api_key)

    ### 3. 최종
    df["category"] = df.apply(
        lambda r: r["category_rule"] if r["category_rule"]!="미분류"
        else ai_map.get(str(r[desc_col]), "기타"),
        axis=1
    )
    return df

## 컬럼명 자동 매핑
STANDARD_COLS = [
    "date", "amount", "category", "description",
    "sub_category", "payment_method", "is_fixed",
    "installment_type", "installment_months"
]
REQUIRED_COLS = ["date", "amount", "category", "description"]

## 동의어
SYNONYMS = {
    "date": [
        "date","거래일","거래일자","승인일","승인일자","결제일","결제일자","사용일","사용일자","이용일","이용일자","날짜","일자"
    ],
    "amount": [
        "amount","금액","결제금액","사용금액","이용금액","승인금액","청구금액","지출","지출액","금액원","원금액","국내이용금액"
    ],
    "category": [
        "category","카테고리","분류","대분류","상위카테고리","지출분류"
    ],
    "description": [
        "description","내역","거래내역","사용내역","이용내역","가맹점","가맹점명","상호","상호명","내용","적요","메모","상품명","이용하신곳"
    ],
    "sub_category": [
        "sub_category","subcategory","세부카테고리","소분류","하위카테고리","세부분류"
    ],
    "payment_method": [
        "payment_method","결제수단","지불수단","지불수단","카드종류","수단"
    ],
    "is_fixed": [
        "is_fixed","고정비","고정지출","정기","정기결제","고정여부","고정비여부"
    ],
    "installment_type": [
        "installment_type","할부유형","결제방식","일시불할부","할부구분","할부여부","결제유형","결제방법"
    ],
    "installment_months": [
        "installment_months","할부개월","할부개월수","할부기간","할부개월수","개월","할부월","할부"
    ],
}

def auto_map_columns(df: pd.DataFrame):
    """
    df의 컬럼을 표준 컬럼으로 자동 매핑.
    - 1차: SYNONYMS 사전 기반
    - 2차: difflib 유사도 기반(사전에 없는 경우)
    반환: (df_renamed, mapping, dropped_cols)
    """
    original_cols = list(df.columns)
    norm_to_orig = { _norm_col(c): c for c in original_cols }

    ### 사전 기반 매핑
    mapping = {}
    used_original = set()

    for std, syns in SYNONYMS.items():
        ### 표준명 자체도 동의어처럼 취급
        candidates = [_norm_col(x) for x in ([std] + syns)]
        hit = None
        for cand in candidates:
            if cand in norm_to_orig:
                hit = norm_to_orig[cand]
                break
        if hit and hit not in used_original:
            mapping[hit] = std
            used_original.add(hit)

    ### 유사도 기반 보완
    remaining_std = [c for c in STANDARD_COLS if c not in set(mapping.values())]
    remaining_orig = [c for c in original_cols if c not in mapping]

    for std in remaining_std:
        
        orig_norms = [_norm_col(c) for c in remaining_orig]
        
        target = _norm_col(std)
        matches = difflib.get_close_matches(target, orig_norms, n=1, cutoff=0.86)
        if matches:
            norm_hit = matches[0]
            orig_hit = norm_to_orig[norm_hit]
            if orig_hit not in mapping:
                mapping[orig_hit] = std
                remaining_orig.remove(orig_hit)

    ### rename 적용
    df2 = df.rename(columns=mapping).copy()

    ## 표준 스키마 이외 drop
    dropped = [c for c in df2.columns if c not in STANDARD_COLS]
    df2 = df2.drop(columns=dropped, errors="ignore")

    return df2, mapping, dropped

## 타입 정리
def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    ### date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    ### amount
    if "amount" in df.columns:
        df["amount"] = (
            df["amount"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("원", "", regex=False)
            .str.strip()
        )
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df.dropna(subset=["amount"])

    ### is_fixed
    if "is_fixed" in df.columns:
        def _to_bool(x):
            if pd.isna(x):
                return False
            s = str(x).strip().lower()
            if s in ["true", "1", "y", "yes", "t", "고정", "정기", "o", "ㅇ"]:
                return True
            if s in ["false", "0", "n", "no", "f", "x", "비고정", "일회", "", "nan"]:
                return False
            return False

        df["is_fixed"] = df["is_fixed"].apply(_to_bool)


    ### installment_type / installment_months
    def normalize_installment(row):
        itype = row.get("installment_type", pd.NA)
        months = row.get("installment_months", pd.NA)

        ### installment_type에 숫자가 있는 경우
        if not pd.isna(itype):
            s = str(itype).strip()
            if s.isdigit():
                m = int(s)
                if m <= 1:
                    return "일시불", pd.NA
                else:
                    return "할부", m

        ### 기존 텍스트 기반 판단
        s_low = str(itype).lower() if not pd.isna(itype) else ""

        if "할부" in s_low or "install" in s_low:
            m = pd.to_numeric(months, errors="coerce")
            return "할부", m if (pd.notna(m) and m > 1) else pd.NA

        ### 기본값
        return "일시불", pd.NA


    ### 컬럼 있으면 적용
    if "installment_type" in df.columns:
        df[["installment_type", "installment_months"]] = df.apply(
            normalize_installment,
            axis=1,
            result_type="expand"
        )


    ### category / description / string 계열
    for col in ["category", "description", "sub_category", "payment_method"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["", "nan", "None"]), col] = pd.NA


    return df


## 최종 전처리 파이프라인
def preprocess_any_expense_df(df: pd.DataFrame, api_key: str):
    """
    1) 컬럼 자동 매핑 + drop
    2) 타입 강제
    3) 필수 컬럼 보정/검사
    4) category 없거나 비어있으면 생성(규칙→미분류만 AI)
    5) month/year_month 생성
    반환: df_final, report(dict)
    """
    report = {}

    df1, mapping, dropped = auto_map_columns(df)
    report["column_mapping"] = mapping
    report["dropped_columns"] = dropped
    report["columns_after_mapping"] = list(df1.columns)

    df2 = coerce_types(df1)

    ### 필수 컬럼 존재 확인
    missing_required = [c for c in ["date","amount","description"] if c not in df2.columns]
    if missing_required:
        report["missing_columns"] = missing_required
        report["error_type"] = "missing_required_columns"
        return None, report

    ### category 처리: 없거나 비어있으면 생성
    if "category" not in df2.columns:
        df2["category"] = pd.NA

    ### category 비어있거나 미분류면 생성
    need_cat = df2["category"].isna() | (df2["category"].astype(str).str.strip() == "") | (df2["category"] == "미분류")
    if need_cat.any():
        tmp = df2.copy()
        ### build_category는 description 기반으로 category 생성하기 때문에 전체를 넣고 덮어쓰기
        tmp = build_category(tmp, api_key=api_key, desc_col="description")
        df2.loc[need_cat, "category"] = tmp.loc[need_cat, "category"]

    ### category 값 표준화
    df2["category"] = df2["category"].apply(lambda x: x if x in CATEGORIES else "기타")

    ### month/year_month 생성
    df2["month"] = df2["date"].dt.to_period("M").astype(str)
    df2["year_month"] = df2["date"].dt.strftime("%Y-%m")

    ### 컬럼 순서 정렬
    ordered = [c for c in STANDARD_COLS if c in df2.columns] + ["month","year_month"]
    df2 = df2[ordered]

    report["rows_final"] = len(df2)
    return df2, report


# 페이지 설정
st.set_page_config(
    page_title="💰 개인 지출 분석",
    page_icon="💰",
    layout="wide"
)

st.title("💰 개인 지출 분석")

# Session State 초기화
if 'df' not in st.session_state:
    st.session_state.df = None
if "df_processed" not in st.session_state:
    st.session_state.df_processed = None
if "prep_report" not in st.session_state:
    st.session_state.prep_report = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

# 사이드바 - 파일 업로드 (기존 UI 유지)
with st.sidebar:
    st.header("📁 데이터 업로드")
    uploaded_file = st.file_uploader("파일 업로드", type=['csv', 'xlsx', 'xls'])

# 파일 업로드 처리
if uploaded_file is not None:
    ## 새 파일이 업로드되었을 때만 처리
    if (not st.session_state.file_uploaded) or (st.session_state.get('file_name') != uploaded_file.name):
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df_raw = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df_raw = pd.read_csv(uploaded_file, encoding='cp949')
            else:
                df_raw = pd.read_excel(uploaded_file)

            ### Session State에 저장 (원본)
            st.session_state.df = df_raw
            st.session_state.file_uploaded = True
            st.session_state.file_name = uploaded_file.name

            st.session_state.df_processed = None
            st.session_state.prep_report = None

        except Exception as e:
            st.error(f"오류: {e}")


if st.session_state.df is not None:
    try:
        ## 전처리 1회만 실행
        if st.session_state.df_processed is None:
            df, prep_report = preprocess_any_expense_df(
                st.session_state.df,
                api_key=st.secrets["OPENAI_API_KEY"]
            )

            ## 전처리 실패 안내
            if df is None:
                st.error("❌ 필수 컬럼이 없어 전처리를 진행할 수 없어요.")
                st.markdown(
                    f"""
        **누락된 필수 컬럼:** `{', '.join(prep_report['missing_columns'])}`

        👉 파일에 **거래일 / 금액 / 내역**에 해당하는 정보가 있는지 확인해주세요.

        컬럼명은 정확히 일치하지 않아도 괜찮아요.  
        예를 들어,  
        - 이용일 → `거래일`, `결제일`, `승인일자`
        - 거래금액 → `금액`, `사용금액`, `결제금액`
        - 이용하신곳 → `거래내역`, `사용내역`, `가맹점명`
        """
                )

                with st.expander("🔎 자동 매핑 결과 보기"):
                    st.write(prep_report["column_mapping"])
                st.stop()

            ## 성공이면 session_state에 저장
            st.session_state.df_processed = df
            st.session_state.prep_report = prep_report

            st.success(f"✅ 전처리 완료! ({prep_report['rows_final']}건)")

        ## 이미 전처리 했으면 저장된 것 사용
        df = st.session_state.df_processed
        prep_report = st.session_state.prep_report


        col_map, col_drop = st.columns(2)

        with col_map:
            with st.expander("🧩 컬럼 자동 매핑 결과"):
                st.write(prep_report["column_mapping"])

        with col_drop:
            with st.expander("🗑️ 삭제된 컬럼"):
                st.write(prep_report["dropped_columns"])

        with st.expander("📋 전처리된 데이터 미리보기"):
            st.dataframe(df.head(10))

    except Exception:
        st.stop()

else:
    st.info("👈 왼쪽 사이드바에서 파일을 업로드해주세요.")
 
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")

    ## 카드로 핵심 기능 설명
    c1, c2, c3 = st.columns(3) 
        
    with c1: 
        st.markdown("## 🧹 자동 전처리") 
        st.caption("거래일/금액/내역 컬럼을 자동 인식하고 정리해요.") 
        
    with c2: 
        st.markdown("## 🧩 매핑 결과 리포트") 
        st.caption("원본 컬럼이 어떤 필드로 매핑됐는지 보여줘요.") 
        
    with c3: 
        st.markdown("## 📊 분석 & 인사이트") 
        st.caption("월별/카테고리별 분석과 AI 요약을 제공해요.") 
    
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")


    st.markdown("## 🚀 사용 방법") 
    st.markdown(
        """ 
        1. 왼쪽 사이드바에서 **CSV/Excel** 파일 업로드 
        2. 자동 전처리 완료 후, **미리보기/매핑 결과** 확인 
        3. 분석/리포트 생성 버튼으로 결과 확인 
        """ 
        )

    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    
    tab1, tab2 = st.tabs(["📄 업로드 예시", "❓ FAQ"])

    with tab1:
        st.markdown(
            """
            **파일에 이런 형태가 들어있으면 좋아요**
            - 거래일: 2026-02-03 / 2026.02.03 / 2026/02/03
            - 금액: 15000 / -15000(환불) 등
            - 내역: 스타벅스 아메리카노, 쿠팡, 지하철 등
            """
        )

    with tab2:
        st.markdown(
            """
            **Q. 컬럼명이 꼭 '거래일/금액/내역'이어야 하나요?**  
            A. 아니요! 비슷한 의미면 자동으로 매핑해요.

            **Q. 업로드한 파일은 저장되나요?**  
            A. 아니요! 따로 저장되지는 않아요.

            **Q. 카테고리별 월 누적 지출은 몇 달까지 확인 할 수 있나요?**  
            A. 최대 6달까지 확인 가능합니다!
            
            """
        )

# 사이드바
if st.session_state.get("df") is not None and 'df' in dir():

    with st.sidebar:
        st.header("🔍 필터")

        ## 기간 필터
        df_filtered = df.copy()

        if 'date' in df_filtered.columns:
            min_date = df_filtered['date'].min().date()
            max_date = df_filtered['date'].max().date()

            date_range = st.date_input(
                "기간 선택",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                start_date, end_date = date_range
                df_filtered = df_filtered[
                    (df_filtered['date'].dt.date >= start_date) &
                    (df_filtered['date'].dt.date <= end_date)
                ]

        ## 카테고리 필터
        if 'category' in df_filtered.columns:
            categories = sorted(df_filtered['category'].dropna().unique().tolist())
            selected_categories = st.multiselect(
                "카테고리 선택",
                options=categories,
                default=categories
            )
            if selected_categories:
                df_filtered = df_filtered[df_filtered['category'].isin(selected_categories)]
            else:
                df_filtered = df_filtered.iloc[0:0]

        ## 일시불/할부 필터
        if 'installment_type' in df_filtered.columns:
            pay_types = sorted(
                df_filtered['installment_type']
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_pay_types = st.multiselect(
                "결제 방식 선택 (일시불/할부)",
                options=pay_types,
                default=pay_types
            )

            if selected_pay_types:
                df_filtered = df_filtered[
                    df_filtered['installment_type'].astype(str).isin(selected_pay_types)
                ]
            else:
                df_filtered = df_filtered.iloc[0:0]

        ## 금액 슬라이드 필터
        if 'amount' in df_filtered.columns:
            min_amt = int(df_filtered['amount'].min())
            max_amt = int(df_filtered['amount'].max())

            selected_range = st.slider(
                "결제 금액 범위",
                min_value=min_amt,
                max_value=max_amt,
                value=(min_amt, max_amt),
                step=1000
            )

            st.markdown(
                f"선택 범위 : **{selected_range[0]:,}원 ~ {selected_range[1]:,}원**"
            )


            df_filtered = df_filtered[
                df_filtered['amount'].between(selected_range[0], selected_range[1])
            ]


    # 핵심 지표 카드
    st.markdown("### 📊 핵심 지표")
    col1, col2, col3, col4 = st.columns(4)

    total_expense = df_filtered['amount'].sum()

    # 월평균 지출 계산
    monthly_sum = (
        df_filtered
        .groupby(df_filtered['date'].dt.to_period('M'))['amount']
        .sum()
    )
    monthly_avg_expense = monthly_sum.mean()

    max_expense = df_filtered['amount'].max()
    transaction_count = len(df_filtered)

    col1.metric("💵 총 지출", f"{total_expense:,.0f}원")
    col2.metric("📆 월평균 지출", f"{monthly_avg_expense:,.0f}원")
    col3.metric("📈 최대 단일 지출", f"{max_expense:,.0f}원")
    col4.metric("🧾 거래 건수", f"{transaction_count}건")

    
    st.markdown("---")
    

    # 차트 영역
    col_left, col_right = st.columns(2)
    
    ## 도넛차트
    with col_left:
        st.markdown("### 🥧 지출 구성")

        donut_view = st.segmented_control(
            "도넛 보기",
            ["카테고리", "일시불/할부"],
            default=st.session_state.get("donut_view", "카테고리"),
            key="donut_view",
            label_visibility="collapsed"
        )

        if donut_view == "카테고리":
            if "category" in df_filtered.columns:
                category_sum = (
                    df_filtered.groupby("category")["amount"]
                    .sum()
                    .reset_index()
                )

                fig_pie = px.pie(
                    category_sum,
                    values="amount",
                    names="category",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(
                    textposition="inside",
                    textinfo="percent+label"
                )
                st.plotly_chart(fig_pie, use_container_width=True, key="donut_chart_category")
            else:
                st.info("category 컬럼이 없어요.")

        else:
            df_i = df_filtered.copy()

            metric_mode = st.radio(
                "기준 선택",
                ["금액", "건수"],
                horizontal=True,
                key="donut_metric_mode"
            )

            ### 할부/일시불 분류
            if "installment_months" in df_i.columns:
                months = pd.to_numeric(df_i["installment_months"], errors="coerce").fillna(0)
                df_i["pay_type"] = (months > 0).map({True: "할부", False: "일시불"})
            elif "installment_type" in df_i.columns:
                s = df_i["installment_type"].fillna("").astype(str).str.strip()
                df_i["pay_type"] = (~s.isin(["일시불", "0", "0개월", "일괄"])).map({True: "할부", False: "일시불"})
            else:
                st.info("할부 관련 컬럼이 없어요.")
                st.stop()

            ### 집계: 금액, 건수
            if metric_mode == "금액":
                pay_stat = df_i.groupby("pay_type")["amount"].sum().reset_index()
                value_col = "amount"
            else:
                pay_stat = df_i.groupby("pay_type").size().reset_index(name="count")
                value_col = "count"

            fig_pay = px.pie(
                pay_stat,
                values=value_col,
                names="pay_type",
                hole=0.4,
                color_discrete_map={"일시불": "#4C78A8", "할부": "#43AECF"}
            )
            fig_pay.update_traces(textposition="inside", textinfo="percent+label")

            st.plotly_chart(fig_pay, use_container_width=True, key="donut_chart_pay")
    

    ## 바 차트
    with col_right:
        st.markdown("### 📊 카테고리별 월 누적 지출")

        if {'category', 'year_month'}.issubset(df_filtered.columns):
            category_month_sum = (
                df_filtered
                .groupby(['category', 'year_month'])['amount']
                .sum()
                .reset_index()
            )

            recent_months = sorted(df_filtered['year_month'].unique())[-6:]
            category_month_sum = category_month_sum[
                category_month_sum['year_month'].isin(recent_months)
            ]


            fig_bar = px.bar(
                category_month_sum,
                x='category',
                y='amount',
                color='year_month'
            )

            fig_bar.update_layout(
                xaxis_title="카테고리",
                yaxis_title="지출 금액 (원)",
                barmode='stack',
                legend_title="월"
            )

            st.plotly_chart(fig_bar, use_container_width=True)
    

    ## 라인차트
    trend_view = st.segmented_control(
        "추이 보기",
        ["월별", "주별", "요일별", "일별"],
        default=st.session_state.get("trend_view", "월별"),
        key="trend_view",
        label_visibility="collapsed"
    )
    

    def draw_line(df, x_col, x_title):
        summary = df.groupby(x_col)['amount'].sum().reset_index()
        fig = px.line(summary, x=x_col, y='amount', markers=True)
        fig.update_layout(
            xaxis_title=x_title,
            yaxis_title="지출 금액 (원)"
        )
        st.plotly_chart(fig, use_container_width=True, key=f"trend_chart_{trend_view}")


    ### 월별
    if trend_view == "월별":
        st.markdown("### 📈 월별 지출 추이")

        df_m = df_filtered.copy()
        df_m['year_month'] = df_m['date'].dt.strftime('%Y-%m')

        draw_line(df_m, 'year_month', '월')

    ### 주별 (1~5주)
    elif trend_view == "주별":
        st.markdown("### 📈 주별 지출 추이")

        df_w = df_filtered.copy()
        df_w['week'] = ((df_w['date'].dt.day - 1) // 7) + 1
        df_w['week'] = df_w['week'].clip(1, 5)
        df_w['week_label'] = df_w['week'].astype(str) + "주"

        draw_line(df_w, 'week_label', '주')

    ### 일별 (1~31일)
    elif trend_view == "일별":
        st.markdown("### 📊 일별 지출 막대그래프 ")

        df_d = df_filtered.copy()

        df_d["date"] = pd.to_datetime(df_d["date"], errors="coerce")
        df_d = df_d.dropna(subset=["date"])


        df_d["ym"] = df_d["date"].dt.to_period("M").astype(str)
        ym_list = sorted(df_d["ym"].unique())
        selected_ym = st.selectbox("월 선택", ym_list, index=len(ym_list) - 1, key="daily_selected_ym")

        df_m = df_d[df_d["ym"] == selected_ym].copy()

        df_m["date_only"] = df_m["date"].dt.normalize()

        ### 일별 합계 + 거래건수
        daily = (
            df_m.groupby("date_only", as_index=False)
            .agg(amount=("amount", "sum"), tx_count=("amount", "size"))
            .rename(columns={"date_only": "date"})
            .sort_values("date")
        )

        ### hover용 요일
        order = ["일", "월", "화", "수", "목", "금", "토"]
        daily["dow_num"] = daily["date"].dt.dayofweek  # 월=0..일=6
        daily["weekday"] = daily["dow_num"].map(lambda x: order[(x + 1) % 7])

        ### 주말 여부(토=5, 일=6)
        daily["is_weekend"] = daily["dow_num"].isin([5, 6])

        daily["day_type"] = daily["is_weekend"].map(
            {True: "주말", False: "평일"})

        ### 최고지출일
        max_idx = daily["amount"].idxmax() if len(daily) else None
        max_row = daily.loc[max_idx] if max_idx is not None else None

        ### 막대그래프
        fig = px.bar(
            daily,
            x="date",
            y="amount",
            color="day_type",
            labels={
                "date": "날짜",
                "amount": "지출 금액(원)",
                "is_weekend": "구분"
            },
            title=f"{selected_ym} 일별 지출",
            color_discrete_map={
                "평일": "#4C78A8",
                "주말": "#E45756"
            },
            ### hover에 추가로 보여줄 컬럼들
            hover_data={
                "weekday": True,
                "tx_count": True,
                "is_weekend": False,
                "dow_num": False
            }
        )

        fig.update_traces(
            hovertemplate=(
                "날짜: %{x|%Y-%m-%d}<br>"
                "요일: %{customdata[0]}<br>"
                "지출: %{y:,.0f}원<br>"
                "거래건수: %{customdata[1]}건"
                "<extra></extra>"
            )
        )

        ### 최고지출일 표시(점 + 텍스트)
        if max_row is not None:
            fig.add_scatter(
                x=[max_row["date"]],
                y=[max_row["amount"]],
                mode="markers+text",
                text=[f"💥 최고 {max_row['amount']:,.0f}원"],
                textposition="top center",
                marker=dict(size=10, color="black"),
                showlegend=False,
                hoverinfo="skip"
            )

        ### 레이아웃
        fig.update_layout(
            xaxis=dict(tickformat="%d일", tickangle=-45),
            bargap=0.15,
            legend_title_text=""
        )

        st.plotly_chart(fig, use_container_width=True, key=f"trend_chart_{trend_view}")

    ### 요일별 (일~토)
    elif trend_view == "요일별":
        st.markdown("### 🔥 요일별 지출 히트맵")

        order = ["일", "월", "화", "수", "목", "금", "토"]

        df_hm = df_filtered.copy()

        df_hm["weekday"] = df_hm["date"].dt.dayofweek.map(lambda x: order[(x + 1) % 7])
        df_hm["weekday"] = pd.Categorical(df_hm["weekday"], categories=order, ordered=True)

        iso = df_hm["date"].dt.isocalendar()
        df_hm["week_key"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

        pivot = (
            df_hm.groupby(["week_key", "weekday"])["amount"]
            .mean()
            .reset_index()
            .pivot(index="weekday", columns="week_key", values="amount")
            .fillna(0)
        )

        fig = px.imshow(
            pivot,
            aspect="auto",
            labels=dict(x="주차", y="요일", color="지출(원)"),
        )

        ### hover에 넣을 날짜 매트릭스
        week_cols = list(pivot.columns)
        weekday_rows = list(pivot.index)

        weekday_to_iso_u = {"월": 1, "화": 2, "수": 3, "목": 4, "금": 5, "토": 6, "일": 7}

        date_matrix = []
        for wd in weekday_rows:
            u = weekday_to_iso_u[wd]
            row_dates = []
            for wk in week_cols:
                year, week = wk.split("-W")
                d = pd.to_datetime(f"{year}-W{week}-{u}", format="%G-W%V-%u")
                row_dates.append(d.strftime("%Y-%m-%d"))
            date_matrix.append(row_dates)

        customdata = np.array(date_matrix)

        fig.update_traces(
            customdata=customdata,
            hovertemplate=(
                "날짜: %{customdata}<br>"
                "요일: %{y}<br>"
                "지출: %{z:,.0f}원"
                "<extra></extra>"
            )
        )

        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))

        st.plotly_chart(fig, use_container_width=True, key=f"trend_chart_{trend_view}")


    ### 슬로프 차트
    st.markdown("### 🔀 카테고리별 두 달 비교")

    months = sorted(df_filtered['year_month'].dropna().astype(str).unique())
    if len(months) < 2:
        st.info("비교하려면 최소 2달의 데이터가 필요해요.")
    else:
        ### 월선택
        left, right = st.columns([1.25, 1])

        with left:
            c1, c2 = st.columns(2)
            with c1:
                month1 = st.selectbox("월1", months, index=max(0, len(months) - 2), key="compare_m1")
            with c2:
                month2 = st.selectbox("월2", months, index=max(0, len(months) - 1), key="compare_m2")

        if month1 == month2:
            st.warning("월1과 월2는 서로 다르게 선택해주세요.🥹")
        else:
            base = df_filtered[df_filtered['year_month'].astype(str).isin([month1, month2])].copy()
            base = base.dropna(subset=['category'])

            pivot = (
                base.groupby(['category', 'year_month'])['amount']
                .sum()
                .reset_index()
                .pivot(index='category', columns='year_month', values='amount')
                .fillna(0)
            )

            for m in [month1, month2]:
                if m not in pivot.columns:
                    pivot[m] = 0

            pivot = pivot[[month1, month2]]
            pivot['diff'] = pivot[month2] - pivot[month1]
            pivot = pivot.sort_values('diff', ascending=False)

            long_df = (
                pivot[[month1, month2]]
                .reset_index()
                .melt(id_vars='category', var_name='month', value_name='amount')
            )

            ### 슬로프 차트
            with left:
                st.markdown(f"#### 📉 {month1} → {month2}")

                fig = px.line(
                    long_df,
                    x='category',
                    y='amount', 
                    color='month',
                    line_group='category',
                    markers=True
                )
                fig.update_layout(
                    xaxis_title="카테고리",
                    yaxis_title="지출 금액 (원)",
                    legend_title="월"
                )
                st.plotly_chart(fig, use_container_width=True)

            ### 변화 요약
            with right:
                st.markdown("#### 🧾 두달간 변화 요약")

                top_n = 10
                show_df = pivot.reset_index().copy()


                def badge(diff: float) -> str:
                    if diff > 0:
                        return f"<span style='color:#d32f2f; font-weight:700;'>▲ {diff:+,}원</span>"
                    elif diff < 0:
                        return f"<span style='color:#2e7d32; font-weight:700;'>▼ {diff:+,}원</span>"
                    else:
                        return f"<span style='color:#616161; font-weight:700;'>■ {diff:+,}원</span>"

                show_df = show_df.sort_values('diff', ascending=False).head(top_n)

                for i, row in show_df.iterrows():
                    cat = row['category']
                    m1v = int(row[month1])
                    m2v = int(row[month2])
                    diff = int(row['diff'])

                    line_col, detail_col = st.columns([0.78, 0.22])

                    with line_col:
                        st.markdown(
                            f"**{cat}**&nbsp;&nbsp;{badge(diff)}",
                            unsafe_allow_html=True
                        )

                    with detail_col:

                        try:
                            with st.popover("상세", use_container_width=True):
                                st.write(f"- {month1} 사용금액: **{m1v:,.0f}원**")
                                st.write(f"- {month2} 사용금액: **{m2v:,.0f}원**")
                                st.write(f"- 차이: **{diff:+,.0f}원**")
                        except Exception:
                            with st.expander("상세"):
                                st.write(f"- {month1} 사용금액: **{m1v:,.0f}원**")
                                st.write(f"- {month2} 사용금액: **{m2v:,.0f}원**")
                                st.write(f"- 차이: **{diff:+,.0f}원**")


# 데이터 요약 통계
def generate_expense_summary(df):
    """지출 데이터 요약 통계 생성 + 기간(개월) + 월평균 포함"""
    summary = {
        'total': df['amount'].sum(),
        'average': df['amount'].mean(),
        'max': df['amount'].max(),
        'min': df['amount'].min(),
        'count': len(df),
    }

    ## 기간(개월 수) 계산: df_filtered 기준으로 계산됨
    if 'date' in df.columns:
    
        months_count = int(df['date'].dt.to_period('M').nunique())
        months_count = max(months_count, 1)

        summary['months_count'] = months_count
        summary['period_start'] = str(df['date'].min().date())
        summary['period_end'] = str(df['date'].max().date())

        ### 월평균 총지출
        summary['monthly_avg_total'] = summary['total'] / months_count
    else:
        summary['months_count'] = 1
        summary['period_start'] = ""
        summary['period_end'] = ""
        summary['monthly_avg_total'] = summary['total']

    ## 카테고리별 통계 + 월평균(카테고리)
    if 'category' in df.columns:
        category_stats = df.groupby('category')['amount'].agg(['sum', 'count']).reset_index()
        category_stats['percentage'] = (category_stats['sum'] / summary['total'] * 100).round(1)

        ### 카테고리 월평균 추가
        category_stats['monthly_avg'] = (category_stats['sum'] / summary['months_count']).round(0)

        summary['category_breakdown'] = category_stats.to_dict('records')

    ## 월별 통계
    if 'year_month' in df.columns:
        monthly_stats = df.groupby('year_month')['amount'].sum().to_dict()
        summary['monthly'] = monthly_stats

    return summary


# OpenAI 클라이언트 생성
# AI 인사이트 함수
def get_ai_insights(summary_data):
    """AI 인사이트 생성"""

    ## 기간 정보 (없으면 1개월)
    months = summary_data.get("months_count", 1)
    monthly_avg_total = summary_data.get(
        "monthly_avg_total",
        summary_data["total"] / max(months, 1)
    )

    ## 카테고리 breakdown 문자열 생성 (총액 + 월평균)
    category_text = ""
    if "category_breakdown" in summary_data:
        for item in summary_data["category_breakdown"]:
            monthly_avg = item.get(
                "monthly_avg",
                item["sum"] / max(months, 1)
            )

            category_text += (
                f"- {item['category']}: "
                f"총 {item['sum']:,.0f}원 ({item['percentage']}%), "
                f"월평균 {monthly_avg:,.0f}원\n"
            )

    prompt = f"""
    
당신은 개인 재무 전문가입니다. 아래 지출 데이터를 분석하고 실용적인 인사이트와 조언을 제공해주세요.

⚠️ 중요 규칙 (반드시 지키세요)
- 이 데이터는 **1개월치가 아니라 총 {months}개월치 데이터**입니다.
- 모든 제안/권장 예산/절약 금액은 반드시 **월평균(총액 ÷ {months}) 기준**으로만 계산하세요.
- 절대 {months}개월치 총액을 다음 달 1개월 예산으로 제시하지 마세요.
- 모든 인사이트에는 반드시 근거 수치(비중 %, 월평균 금액, 거래 건수 등)를 포함하세요.
- 단순 요약이 아닌, 해석과 의미가 포함된 인사이트만 제시하세요.
- 반드시 아래 분석 요청에 작성한 순서대로 작성해주세요.
- 절약 금액은 반드시 100원 단위로 반올림하여 표시하세요.
- 예산 조정 우선순위이기 때문에 후순위는 아예 제안하지 말라는 뜻은 아닙니다. 필요할 경우는 제시하세요. 
- 권장 예산 조정 우선순위: (1) 변동비 → (2) 반복 지출(구독) → (3) 고정비(주거/통신/보험은 큰 변경 제안 금지, 대신 절감 아이디어 제시) 


[지출 요약 - {months}개월 기준]
- 총 지출: {summary_data['total']:,.0f}원
- 월평균 총 지출: {monthly_avg_total:,.0f}원
- 최대 단일 지출: {summary_data['max']:,.0f}원
- 거래 건수: {summary_data['count']}건

[카테고리별 지출 (총액 + 월평균)]
{category_text}

[분석 요청]
1. 지출 패턴에서 주목할 점 2~3가지
- 전체 규모, 지출 집중도, 최대 단일 지출, 거래 건수의 영향 등을 작성하세요.
2. 개선이 필요한 소비 부문
- 개선이 필요한 이유 정확한 근거를 들어 제시하세요.
3. 다음 달 권장 예산 (카테고리별, 월 기준)
- 월 절약 목표가 월평균 총지출의 30%를 초과하지 않도록 하세요. (해당 내용은 공개하지 마세요)
- 반드시 아래 표 형식으로 제시:
  | 카테고리 | 현재 월평균 | 권장 월예산 | 월 절약액 | 실행 팁(1줄) |
- 모든 카테고리가 나올 필요는 없습니다. 절약 할 필요가 있는 카테고리만 표시합니다.
4. 절약액 적금 시뮬레이션 (3/6/12개월)
- 저축 예상 금액은 아래 공식대로 계산하세요.
  월이율 r = 0.03/12
  n개월 후 적립금 FV = 월저축액*S * (((1+r)^n - 1)/r)
- 3/6/12개월별로 알려주고 결과는 원 단위로 반올림해서 제시하세요.
- 공식은 보여주지 말고 계산에만 사용하세요. 연 3% 이자, 월복리, 매월 말 납입 적금으로 계산했다는 것만 언급해주세요.
- 절약액 적금 시뮬레이션은 반드시 표 형식으로 제시하세요:
  | 기간 | 예상 적립금 | 총 납입원금 | 이자 수익 |
5. 한 줄 결론
- 핵심 수치 1개 이상을 포함하고, 다음 행동 방향이 담긴 문장으로 작성하세요.
- 감성적인 표현은 최소화하고 데이터 기반으로 작성하세요.

친근하고 이해하기 쉬운 말투로 작성해주세요.
가독성 좋게 작성해주세요.
구체적인 수치를 포함해서 실행 가능한 조언을 해주세요.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 친절한 개인 재무 전문가입니다. 모든 예산과 절약 금액은 반드시 월 기준으로 계산합니다."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"AI 분석 중 오류가 발생했습니다: {e}"


# 월간 리포트
def generate_monthly_report(df, insights=None):
    """월간 리포트 마크다운 생성"""

    total_spend = df["amount"].sum()
    max_spend = df["amount"].max()
    count_tx = df["amount"].count()
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
        monthly_avg_total = (
            df.dropna(subset=["date"])
            .assign(month=df["date"].dt.to_period("M"))
            .groupby("month")["amount"]
            .sum()
            .mean()
        )
    else:
        ### date 컬럼이 없으면 전체 평균으로 대체
        monthly_avg_total = df["amount"].mean()
    
    report = f"""

#📊 월간 지출 리포트

생성일: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

---

## 📈 지출 요약

| 항목 | 금액 |
|------|------|
| 총 지출 | {total_spend:,.0f}원 |
| 월평균 지출 | {monthly_avg_total:,.0f}원 |
| 최대 단일 지출 | {max_spend:,.0f}원 |
| 거래 건수 | {count_tx}건 |

---

## 🏷️ 카테고리별 지출

"""
    
    if 'category' in df.columns:
        category_sum = df.groupby('category')['amount'].sum().sort_values(ascending=False)
        total = category_sum.sum()               
        
        report += "| 카테고리 | 금액 | 비율 |\n"
        report += "|----------|------|------|\n"
        for cat, amount in category_sum.items():
            percentage = (amount / total * 100)
            report += f"| {cat} | {amount:,.0f}원 | {percentage:.1f}% |\n"
    
        desc_col = None
        for c in ["description", "content", "memo", "note", "item", "place"]:
            if c in df.columns:
                desc_col = c
                break


        ### 고정비 리스트
        report += "\n---\n\n## 🧾 고정비 리스트\n\n"

        fixed_col = "is_fixed"

        fixed_df = df[df[fixed_col] == True].copy()

        if fixed_df.empty:
            report += "고정비로 표시된 내역이 없어요. (`is_fixed`가 True인 행)\n"
        else:
            fixed_df["date"] = pd.to_datetime(fixed_df["date"], errors="coerce")

            group_keys = ["category"]
            if desc_col:
                group_keys.append(desc_col)

            fixed_summary = (
                fixed_df.groupby(group_keys, dropna=False)["amount"]
                .agg(total="sum", count="size", avg="mean")
                .reset_index()
                .sort_values("total", ascending=False)
            )

            fixed_total = fixed_df["amount"].sum()
            report += f"- 고정비 총합: **{fixed_total:,.0f}원**\n\n"

            report += "| 카테고리 | 항목 | 월 합계 | 발생 횟수 | 1회 평균 금액 |\n"
            report += "|----------|------|---------|-----------|----------------|\n"

            for _, row in fixed_summary.iterrows():
                cat = row["category"]
                item = row[desc_col] if desc_col else "-"
                report += (
                    f"| {cat} | {item} | "
                    f"{row['total']:,.0f}원 | "
                    f"{int(row['count'])}회 | "
                    f"{row['avg']:,.0f}원 |\n"
                )

        ### 지출 집중일
        report += "\n---\n\n## 🔥 지출 집중일\n\n"

        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        tmp = tmp.dropna(subset=["date"]).copy()

        if tmp.empty:
            report += "지출 집중일을 계산할 수 있는 유효한 날짜가 없어요.\n"
        else:
            tmp["day"] = tmp["date"].dt.date
            daily_sum = tmp.groupby("day")["amount"].sum().sort_values(ascending=False)

            focus_day = daily_sum.index[0]
            focus_amt = float(daily_sum.iloc[0])

            report += f"- 가장 많이 지출한 날: **{focus_day}**\n"
            report += f"- 해당일 총 지출: **{focus_amt:,.0f}원**\n\n"

            day_df = tmp[tmp["day"] == focus_day].copy()

            cols_day = ["date", "category"]
            if desc_col:
                cols_day.insert(2, desc_col)
            cols_day.append("amount")

            day_df = day_df.sort_values("amount", ascending=False).head(5)

            report += "### 📌 해당일 지출 상세 (상위 5건)\n\n"
            report += "| 시간 | 카테고리 | 내용 | 금액 |\n"
            report += "|------|----------|------|------|\n"

            for _, row in day_df[cols_day].iterrows():
                time_str = row["date"].strftime("%H:%M")
                cat = row["category"]
                desc_value = row[desc_col] if desc_col else "-"
                report += f"| {time_str} | {cat} | {desc_value} | {row['amount']:,.0f}원 |\n"


    report += "\n---\n\n## 💡 상위 5개 지출\n\n"
    
    desc_col = None
    for c in ["description", "content", "memo", "note", "item", "place"]:
        if c in df.columns:
            desc_col = c
            break

    cols = ["date", "category", "amount"]
    if desc_col:
        cols.insert(2, desc_col)  

    top5 = df.nlargest(5, "amount")[cols]

    report += "| 날짜 | 카테고리 | 내용 | 금액 |\n"
    report += "|------|----------|------|------|\n"
    for _, row in top5.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else '-'
        desc_value = row[desc_col] if desc_col else "-"
        report += f"| {date_str} | {row['category']} | {desc_value} | {row['amount']:,.0f}원 |\n"
    
    if insights:
        report += f"\n---\n\n## 🤖 AI 인사이트\n\n{insights}\n"
    
    return report


# Streamlit UI에서 사용
if uploaded_file is not None:

    st.markdown("---")
    st.markdown("### 🤖 AI 분석 인사이트")

    if st.button("🔍 AI 분석 시작", type="primary"):
        with st.spinner("AI가 지출 패턴을 분석하고 있습니다..."):
            summary = generate_expense_summary(df_filtered)
            insights = get_ai_insights(summary)

            st.markdown(insights)
            st.session_state['last_insights'] = insights

    ## 이전 분석 결과 표시
    if 'last_insights' in st.session_state:
        with st.expander("📝 이전 분석 결과 보기"):
            st.markdown(st.session_state['last_insights'])

    ## 월간리포트 생성
    st.markdown("---")
    st.markdown("### 📋 월간 리포트")

    if st.button("📄 리포트 생성"):
        insights = st.session_state.get('last_insights', None)
        report = generate_monthly_report(df_filtered, insights)

        st.markdown(report)
        
        st.markdown("---")
        st.download_button(
            label="📥 리포트 다운로드 (Markdown)",
            data=report,
            file_name=f"expense_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )