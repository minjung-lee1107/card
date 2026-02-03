import pandas as pd
import random
from datetime import datetime, timedelta

NUM_ROWS = 300
START_DATE = datetime(2025, 2, 1)
END_DATE = datetime(2026, 2, 1)
# 고정비 파악과 월별 분석, 전월대비 분석을 위해 1년으로 설정

CATEGORY_RULES = {
    "식비": ["식당", "카페", "배달", "장보기"],
    "교통비": ["대중교통", "택시", "주유", "주차"],
    "쇼핑": ["의류", "잡화", "온라인 쇼핑"],
    "주거/통신": ["월세", "관리비", "통신비", "인터넷"],
    "구독": ["넷플릭스", "유튜브", "멜론"],
    "의료/건강": ["병원", "약국", "헬스장"],
    "문화/여가": ["영화", "공연", "여행", "취미"],
    "금융/보험": ["보험", "대출"],
    "교육": ["학원", "강의", "도서"]
}

PAYMENT_METHODS = ["카드", "현금"]

FIXED_CATEGORIES = ["주거/통신", "구독", "교육", "금융/보험"]
# 고정비 판단

INSTALLMENT_CATEGORIES = ["쇼핑", "교육", "의료/건강", "금융/보험"]
INSTALLMENT_MONTHS_OPTIONS = list(range(2, 13))
INSTALLMENT_MIN_AMOUNT = 50_000

def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

def random_amount(category):
    if category == "식비":
        amount = random.randint(5_000, 150_000)
    elif category == "교통비":
        amount = random.randint(1_500, 100_000)
    elif category == "주거/통신":
        amount = random.randint(50_000, 800_000)
    elif category == "구독":
        amount = random.randint(9_000, 30_000)
    elif category == "쇼핑":
        amount = random.randint(10_000, 200_000)
    elif category == "금융/보험":
        amount = random.randint(50_000, 500_000)
    else:
        amount = random.randint(5_000, 300_000)

    # 10원 단위로 반올림 후 정수 반환
    return int(round(amount, -1))

# 현실적인 범위 설정

rows = []

for _ in range(NUM_ROWS):
    category = random.choice(list(CATEGORY_RULES.keys()))
    sub_category = random.choice(CATEGORY_RULES[category])

    amount = random_amount(category)
    payment_method = random.choice(PAYMENT_METHODS)

    # 날짜 설정 (고정비는 월 초에 몰리도록)
    if category in FIXED_CATEGORIES:
        date = datetime(START_DATE.year, random.randint(1, 12), 1)
    else:
        date = random_date(START_DATE, END_DATE)

    # 할부 여부 판단
    if (
        payment_method == "카드"
        and category in INSTALLMENT_CATEGORIES
        and amount >= INSTALLMENT_MIN_AMOUNT
        and random.random() < 0.1   # 10% 확률로 할부
    ):
        installment_type = "할부"
        installment_months = random.choice(INSTALLMENT_MONTHS_OPTIONS)
        memo = f"{installment_months}개월 할부"
    else:
        installment_type = "일시불"
        installment_months = None
        memo = ""

    rows.append({
        "date": date.strftime("%Y-%m-%d"),
        "amount": amount,
        "category": category,
        "sub_category": sub_category,
        "description": f"{sub_category} 관련 지출",
        "payment_method": payment_method,
        "is_fixed": category in FIXED_CATEGORIES,
        "memo": memo,
        "installment_type": installment_type,
        "installment_months": installment_months
    })

df = pd.DataFrame(rows)
df.to_csv("sample_expense_data.csv", index=False, encoding="utf-8-sig")

print("sample_expense_data.csv 생성 완료")
