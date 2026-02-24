```markdown
## How to Run

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
```

## 📁 Project Structure


개인지출분석/
│
├── app.py                 # 메인 Streamlit 앱
├── sample_data_code.py   # sample data 생성 코드
├── requirements.txt       # 의존성 목록
├── README.md             # 프로젝트 설명
├── .gitignore            # Git 제외 파일
│
├── Planning/
│   ├─ 01_data_planning.pdf       # 프로젝트 기획서
│   ├─ 02_data_specification.md   # 데이터 명세서 
│   └─ 03_app_structure_tree.md   # 앱 구조 트리
│
├── data/
│   └─ sample__expense_data.csv   # 샘플 데이터
│
└── utils/
    ├─ __init__.py
    ├─ preprocess.py            # 데이터 처리 함수
    └─ ai_and_report.py         # AI 분석, 월간리포트 함수
