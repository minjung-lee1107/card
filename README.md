# AI 기반 개인 지출 분석 대시보드

Streamlit과 OpenAI API를 활용한 자동 지출 분류 및 소비 패턴 분석 앱


## 주요 기능

- 📂 CSV / Excel 파일 업로드
- 🧹 자동 전처리 (컬럼 정규화, 날짜/금액 파싱)
- 🤖 AI 기반 카테고리 자동 분류
- 📊 카테고리별 지출 시각화 (도넛 / 막대 / 추이 분석)
- 💡 AI 인사이트 제공
- 📝 월간 소비 리포트 자동 생성


## Tech Stack

- Python
- Streamlit
- Pandas / Numpy
- Plotly
- OpenAI API


## Project Structure


개인지출분석/
├── app.py                      # 메인 Streamlit 앱
├── sample_data_code.py         # sample data 생성 코드
├── requirements.txt            # 의존성 목록
├── README.md                   # 프로젝트 설명
├── .gitignore                  # Git 제외 파일
├── runtime.txt                 # 배포 환경 Python 버전 명시
├── Planning/
│   ├── 01_data_planning.pdf    # 프로젝트 기획서
│   ├── 02_data_specification.md # 데이터 명세서
│   └── 03_app_structure_tree.md # 앱 구조 트리
├── data/
│   └── sample__expense_data.csv # 샘플 데이터
└── utils/
    ├── __init__.py             # 패키지 초기화 파일
    ├── preprocess.py           # 데이터 처리 함수
    └── ai_and_report.py        # AI 분석, 월간 리포트 함수


## How to Run

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
