import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import re
import json
import difflib

# -----데이터 전처리-----

## 카테고리 자동분류

RULES = {
    "식비": ["식당", "카페", "커피", "치킨", "피자", "분식", "김밥", "국밥", "버거", "제과", "베이커리", "배달", "도시락"],
    "금융/보험": ["보험", "손보", "화재", "생명", "대출", "이자"],
    "의료/건강": ["병원", "의원", "약국", "치과", "한의원", "검진", "헬스", "필라테스","GYM"],
    "교통": ["주차", "택시", "버스", "지하철", "주유", "정비", "톨게이트", "고속도로"],
    "쇼핑": ["마트", "편의점", "백화점", "아울렛", "다이소", "올리브영"],
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

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
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
        "amount","금액","결제금액","사용금액","이용금액","승인금액","청구금액","지출","지출액","금액원","원금액"
    ],
    "category": [
        "category","카테고리","분류","대분류","상위카테고리","지출분류"
    ],
    "description": [
        "description","내역","거래내역","사용내역","이용내역","가맹점","가맹점명","상호","상호명","내용","적요","메모","상품명"
    ],
    "sub_category": [
        "sub_category","subcategory","세부카테고리","소분류","하위카테고리","세부분류"
    ],
    "payment_method": [
        "payment_method","결제수단","결제방법","지불수단","지불방법","카드종류","수단"
    ],
    "is_fixed": [
        "is_fixed","고정비","고정지출","정기","정기결제","고정여부","고정비여부"
    ],
    "installment_type": [
        "installment_type","할부유형","결제방식","일시불할부","할부구분","할부여부","결제유형"
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
            if s in ["true","1","y","yes","t","고정","정기","o","ㅇ"]:
                return True
            if s in ["false","0","n","no","f","x","비고정","일회","", "nan"]:
                return False
            return False
        df["is_fixed"] = df["is_fixed"].apply(_to_bool)

    ### installment_type
    if "installment_type" in df.columns:
        def _norm_inst(x):
            if pd.isna(x): return "일시불"
            s = str(x).strip()
            s_low = s.lower()
            
            if "할부" in s or "install" in s_low:
                return "할부"
            if s in ["0","일시불","single","one-time","일시"]:
                return "일시불"
            
            return "일시불"
        df["installment_type"] = df["installment_type"].apply(_norm_inst)

    ### installment_months
    if "installment_months" in df.columns:
        df["installment_months"] = pd.to_numeric(df["installment_months"], errors="coerce")
        ### 할부가 아닌데 개월수 있으면 정리
        if "installment_type" in df.columns:
            df.loc[df["installment_type"] != "할부", "installment_months"] = pd.NA

    ### category/description/string 계열
    for col in ["category","description","sub_category","payment_method"]:
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
        raise ValueError(f"필수 컬럼 매핑 실패: {missing_required} (원본 컬럼명을 SYNONYMS에 추가해줘야 함)")

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