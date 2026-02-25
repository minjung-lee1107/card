import pandas as pd
import streamlit as st
from openai import OpenAI



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
def get_ai_insights(summary_data, api_key=None):
    """AI 인사이트 생성"""

    if not api_key:
        return "⚠️ OPENAI_API_KEY가 설정되지 않아 AI 분석을 건너뛰었어요."
    client = OpenAI(api_key=api_key)

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
- 월 절약액은 반드시 100원 단위로 반올림하여 표시하세요.
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

        if fixed_col not in df.columns:
            df[fixed_col] = False

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
