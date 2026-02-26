import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import random
import re
import json
import difflib

from utils.preprocess import preprocess_any_expense_df
from utils.sample_data_code import make_sample_expense_data
from utils.ai_and_report import generate_expense_summary, get_ai_insights, generate_monthly_report

# API 키 설정
api_key = st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

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

    # 업로드 전 선택값을 세션에 저장
    if "use_ai_pref" not in st.session_state:
        st.session_state.use_ai_pref = False

    if uploaded_file is None:
        st.session_state.use_ai_pref = st.toggle(
            "카테고리 AI 자동 보정 사용",
            value=st.session_state.use_ai_pref
        )
    
    st.markdown("---")

    if uploaded_file is None:
        st.header("📥 샘플 데이터가 필요하신가요?")

        @st.cache_data
        def get_sample_csv_bytes(seed) -> bytes:
            df = make_sample_expense_data(seed=seed)
            return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

        st.download_button(
            label="샘플 CSV 다운로드",
            data=get_sample_csv_bytes(random.randint(0, 100000)),
            file_name="sample_expense_data.csv",
            mime="text/csv"
        )

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
            st.session_state.df_raw = df_raw
            st.session_state.file_uploaded = True
            st.session_state.file_name = uploaded_file.name

            st.session_state.df_processed = None
            st.session_state.prep_report = None

        except Exception as e:
            st.error(f"오류: {e}")

api_key = st.secrets.get("OPENAI_API_KEY")


if uploaded_file is not None and st.session_state.get("df_raw") is not None:
    try:
        drop_non_standard = st.toggle("표준 컬럼 외 컬럼 삭제", value=True)
        use_ai = bool(st.session_state.get("use_ai_pref", False))
        ## 전처리 트리거(파일+옵션) 고정
        file_sig = (uploaded_file.name, uploaded_file.size)
        proc_sig = (file_sig, drop_non_standard, use_ai)

        ## proc_sig가 바뀐 경우에만 전처리 다시 수행
        if st.session_state.get("proc_sig") != proc_sig:
            st.session_state.df_processed = None
            st.session_state.prep_report = None
            st.session_state.proc_sig = proc_sig

        ## 전처리 1회만 실행
        if st.session_state.df_processed is None:
            df, prep_report = preprocess_any_expense_df(
            st.session_state.df_raw,
            api_key=api_key,
            use_ai=use_ai,
            drop_non_standard=drop_non_standard
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

            type_report = prep_report.get("type_coerce_report", {})
            dropped_total = type_report.get("rows_dropped_types_total", 0)

            if dropped_total > 0:
                st.warning(
                    f"⚠️ 날짜/금액을 읽을 수 없는 데이터 {dropped_total}건이 제외됐어요. "
                    f"(전: {type_report.get('rows_before_types')} → "
                    f"후: {type_report.get('rows_after_types')})"
                )
                st.caption(
                    "예: 날짜 형식이 다르거나, 금액에 문자/기호가 섞여 있는 경우입니다."
                )

                st.caption(
                    f"- 날짜 확인 불가: {type_report.get('date_parse_failed', 0)}건 / "
                    f"금액 확인 불가: {type_report.get('amount_parse_failed', 0)}건"
                )

        ## 이미 전처리 했으면 저장된 것 사용
        df = st.session_state.df_processed
        prep_report = st.session_state.get("prep_report", {})


        col_map, col_drop = st.columns(2)

        with col_map:
            with st.expander("🧩 컬럼 자동 매핑 결과"):
                st.write(prep_report["column_mapping"])

        with col_drop:
            with st.expander("🗑️ 삭제된 컬럼"):
                st.write(prep_report.get("dropped_columns", []))

        with st.expander("📋 전처리된 데이터 미리보기"):
            st.dataframe(df.head(10))

    except Exception as e:
        st.error(f"처리 중 오류: {e}")
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

            **Q. “표준 컬럼 외 컬럼 삭제”를 껐는데도 왜 ‘삭제된 컬럼’ 항목에 컬럼 이름이 표시되나요?**  
            A. 토글을 꺼도 표시되는 컬럼은 실제로 삭제된 것이 아니라, 표준 컬럼과 매칭되지 않은 컬럼을 안내용으로 보여주는 목록일 뿐입니다.  
            데이터 미리보기에서 삭제되지 않은 것을 확인할 수 있습니다.

            **Q. 월간 리포트를 다운로드 할 때 확장자가 무엇인가요?**  
            A. 마크다운(.md)과 텍스트(.txt) 중 선택 가능합니다!

            """
        )

# 사이드바
if st.session_state.df_processed is not None:
    df = st.session_state.df_processed

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

            existing_types = set(pay_stat["pay_type"])

            if "일시불" not in existing_types and "할부" not in existing_types:
                st.warning("데이터가 없습니다.")
                st.stop()

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


# Streamlit UI에서 사용
if st.session_state.df_processed is not None:
    df = st.session_state.df_processed
    df_filtered = df.copy()

    st.markdown("---")
    st.markdown("### 🤖 AI 분석 인사이트")

    if st.button("🔍 AI 분석 시작", type="primary"):
        with st.spinner("AI가 지출 패턴을 분석하고 있습니다..."):
            df = st.session_state.df_processed
            summary = generate_expense_summary(df_filtered)
            insights = get_ai_insights(summary, api_key=st.secrets.get("OPENAI_API_KEY"))

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
        insights = st.session_state.get("last_insights", None)
        st.session_state["monthly_report"] = generate_monthly_report(df_filtered, insights)

    report = st.session_state.get("monthly_report", None)
    if report:
        st.markdown(report)
        st.markdown("---")

        st.download_button(
            label="📥 Markdown 다운로드 (.md)",
            data=report,
            file_name=f"expense_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )

        st.download_button(
            label="📥 Text 다운로드 (.txt)",
            data=report,
            file_name=f"expense_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )