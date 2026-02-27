import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Optional

NUM_ROWS = 300
START_DATE = datetime(2025, 2, 1)
END_DATE = datetime(2026, 1, 31)

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

MERCHANT_RULES = {
    "식비": {
        "식당": ["김밥천국", "한솥도시락", "맘스터치", "홍콩반점", "서브웨이"],
        "카페": ["스타벅스", "이디야", "투썸플레이스", "메가커피", "컴포즈커피"],
        "배달": ["배달의민족", "쿠팡이츠", "요기요"],
        "장보기": ["이마트", "홈플러스", "롯데마트", "GS25", "CU"]
    },
    "교통비": {
        "대중교통": ["교통카드", "티머니"],
        "택시": ["카카오T", "UT", "일반택시"],
        "주유": ["SK주유소", "GS칼텍스", "S-OIL", "현대오일뱅크"],
        "주차": ["카카오T주차", "모두의주차장", "공영주차장"]
    },
    "쇼핑": {
        "의류": ["무신사", "유니클로", "H&M", "지그재그"],
        "잡화": ["다이소", "올리브영", "아트박스"],
        "온라인 쇼핑": ["쿠팡", "네이버쇼핑", "11번가", "G마켓", "SSG"]
    },
    "주거/통신": {
        "월세": ["월세이체"],
        "관리비": ["아파트관리비"],
        "통신비": ["SKT", "KT", "LG U+"],
        "인터넷": ["KT인터넷", "SK브로드밴드", "LG U+인터넷"]
    },
    "구독": {
        "넷플릭스": ["넷플릭스"],
        "유튜브": ["유튜브프리미엄"],
        "멜론": ["멜론"]
    },
    "의료/건강": {
        "병원": ["동네내과", "정형외과", "치과", "피부과"],
        "약국": ["동네약국", "온누리약국", "메디팜"],
        "헬스장": ["스포애니", "헬스보이짐", "동네헬스장"]
    },
    "문화/여가": {
        "영화": ["CGV", "롯데시네마", "메가박스"],
        "공연": ["인터파크티켓", "예스24티켓"],
        "여행": ["야놀자", "여기어때", "아고다", "에어비앤비"],
        "취미": ["교보문고", "문구점", "클래스101"]
    },
    "금융/보험": {
        "보험": ["현대해상", "삼성화재", "KB손해보험"],
        "대출": ["대출이자", "은행자동이체"]
    },
    "교육": {
        "학원": ["영어학원", "수학학원", "코딩학원"],
        "강의": ["인프런", "패스트캠퍼스", "클래스101"],
        "도서": ["교보문고", "YES24", "알라딘"]
    }
}

PAYMENT_METHODS = ["카드", "현금"]
FIXED_CATEGORIES = ["주거/통신", "구독", "금융/보험"]
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

    return int(round(amount, -1))


def get_merchant(category, sub_category):
    try:
        return random.choice(MERCHANT_RULES[category][sub_category])
    except KeyError:
        return f"{sub_category}"


def make_sample_expense_data(
    num_rows: int = NUM_ROWS,
    start_date: datetime = START_DATE,
    end_date: datetime = END_DATE,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Streamlit에서 import해서 쓰기 위한 샘플 데이터 생성 함수"""
    if seed is not None:
        random.seed(seed)

    rows = []

    for _ in range(num_rows):
        category = random.choice(list(CATEGORY_RULES.keys()))
        sub_category = random.choice(CATEGORY_RULES[category])

        amount = random_amount(category)
        payment_method = random.choice(PAYMENT_METHODS)

        if category in FIXED_CATEGORIES:
            date = datetime(start_date.year, random.randint(1, 12), 1)
        else:
            date = random_date(start_date, end_date)

        if (
            payment_method == "카드"
            and category in INSTALLMENT_CATEGORIES
            and amount >= INSTALLMENT_MIN_AMOUNT
            and random.random() < 0.3
        ):
            installment_type = "할부"
            installment_months = random.choice(INSTALLMENT_MONTHS_OPTIONS)
        else:
            installment_type = "일시불"
            installment_months = None

        merchant = get_merchant(category, sub_category)

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "amount": amount,
            "category": category,
            "sub_category": sub_category,
            "description": merchant,
            "payment_method": payment_method,
            "is_fixed": category in FIXED_CATEGORIES,
            "installment_type": installment_type,
            "installment_months": installment_months
        })

    return pd.DataFrame(rows)


def save_sample_csv(path: str = "sample_expense_data.csv") -> str:
    """로컬/스크립트 실행용: CSV 파일로 저장"""
    df = make_sample_expense_data()
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


if __name__ == "__main__":
    out = save_sample_csv("sample_expense_data.csv")
    print(f"{out} 생성 완료")
