# 타이타닉 생존자 예측 서비스

머신러닝을 활용한 타이타닉 생존자 예측 웹 서비스입니다. LightGBM 모델을 사용하여 승객 정보를 입력하면 생존 확률을 예측합니다.

## 주요 기능

- 웹 인터페이스를 통한 생존 예측
- 승객 정보 입력 폼 제공
- 생존/사망 확률 시각화
- REST API 엔드포인트 제공
- LightGBM 기반 머신러닝 모델

## 시스템 요구사항

- Python 3.8 이상
- Git

## 설치 및 실행 방법

### 1. 저장소 복제

```bash
git clone https://github.com/venture21/autoML_202511.git
cd autoML_202511
```

### 2. 가상환경 생성 (권장)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

설치되는 주요 패키지:
- Flask 3.0.0 (웹 프레임워크)
- pandas 2.1.3 (데이터 처리)
- numpy 1.26.2 (수치 연산)
- scikit-learn 1.3.2 (머신러닝 유틸리티)
- lightgbm 4.1.0 (예측 모델)

### 4. 서비스 실행

```bash
python app.py
```

서버가 정상적으로 시작되면 다음과 같은 메시지가 표시됩니다:
```
============================================================
타이타닉 생존 예측 서비스
============================================================
서버 시작: http://127.0.0.1:5000
============================================================
```

### 5. 웹 브라우저에서 접속

웹 브라우저를 열고 다음 주소로 접속합니다:
```
http://127.0.0.1:5000
```
또는
```
http://localhost:5000
```

## 사용 방법

### 웹 인터페이스 사용

1. 메인 페이지에서 승객 정보 입력:
   - 이름 (Name)
   - 객실 등급 (Pclass): 1등급, 2등급, 3등급
   - 성별 (Sex): 남성(male), 여성(female)
   - 나이 (Age)
   - 형제자매/배우자 수 (SibSp)
   - 부모/자녀 수 (Parch)
   - 요금 (Fare)
   - 객실 번호 (Cabin) - 선택사항
   - 승선 항구 (Embarked): S(Southampton), C(Cherbourg), Q(Queenstown)

2. "예측하기" 버튼 클릭

3. 결과 페이지에서 확인:
   - 생존 예측 결과
   - 생존 확률
   - 사망 확률
   - 입력된 승객 정보 요약

### REST API 사용

API 엔드포인트: `POST /api/predict`

**요청 예시:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Name": "John Doe",
    "Pclass": "1",
    "Sex": "male",
    "Age": "30",
    "SibSp": "0",
    "Parch": "0",
    "Fare": "50",
    "Cabin": "C23",
    "Embarked": "S"
  }'
```

**응답 예시:**
```json
{
  "success": true,
  "survived": 1,
  "probability_death": 0.35,
  "probability_survival": 0.65,
  "title": "Mr"
}
```

## 프로젝트 구조

```
autoML_202511/
├── app.py                      # Flask 웹 애플리케이션 메인 파일
├── preprocess.py               # 데이터 전처리 스크립트
├── model_lgbm.py              # LightGBM 모델 학습 스크립트
├── model_ensemble.py          # 앙상블 모델 스크립트
├── compare_models.py          # 모델 비교 스크립트
├── requirements.txt           # 필요한 패키지 목록
├── lgbm_titanic_model.pkl    # 학습된 모델 파일
├── train.csv                 # 원본 훈련 데이터
├── train_preprocessed.csv    # 전처리된 훈련 데이터
├── templates/                # HTML 템플릿
│   ├── index.html           # 메인 페이지
│   ├── result.html          # 결과 페이지
│   └── error.html           # 에러 페이지
└── README.md                 # 프로젝트 설명서
```

## 모델 재학습

데이터를 업데이트하여 모델을 재학습하려면:

1. 데이터 전처리:
```bash
python preprocess.py
```

2. 모델 학습:
```bash
python model_lgbm.py
```

3. 모델 비교 (선택사항):
```bash
python compare_models.py
```

## 서비스 종료

실행 중인 서비스를 종료하려면:
- 터미널에서 `Ctrl + C` 를 누릅니다

## 문제 해결

### 모듈을 찾을 수 없다는 오류
```bash
pip install -r requirements.txt
```
명령어로 필요한 패키지를 다시 설치합니다.

### 포트가 이미 사용 중이라는 오류
`app.py` 파일의 마지막 줄에서 포트 번호를 변경합니다:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # 5000 -> 5001로 변경
```

### 모델 파일을 찾을 수 없다는 오류
`lgbm_titanic_model.pkl` 파일이 프로젝트 루트 디렉토리에 있는지 확인하고, 없다면 `python model_lgbm.py`를 실행하여 모델을 생성합니다.

## 라이선스

이 프로젝트는 교육 목적으로 만들어졌습니다.

## 기여

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!