# 타이타닉 생존 예측 서비스 - 개발자 가이드

## 목차
1. [시스템 아키텍처](#시스템-아키텍처)
2. [프로젝트 구조](#프로젝트-구조)
3. [데이터 전처리 파이프라인](#데이터-전처리-파이프라인)
4. [모델 상세](#모델-상세)
5. [API 명세](#api-명세)
6. [코드 상세 설명](#코드-상세-설명)
7. [확장 가이드](#확장-가이드)
8. [트러블슈팅](#트러블슈팅)

---

## 시스템 아키텍처

### 개요
본 시스템은 Flask 기반의 경량 웹 애플리케이션으로, LightGBM 모델을 사용하여 타이타닉 승객의 생존 여부를 예측합니다.

```
┌─────────────────┐
│   웹 브라우저    │
│  (사용자 입력)   │
└────────┬────────┘
         │ HTTP Request
         ▼
┌─────────────────┐
│  Flask Server   │
│   (app.py)      │
├─────────────────┤
│  • 라우팅       │
│  • 데이터 검증   │
│  • 전처리       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LightGBM Model  │
│  (.pkl 파일)    │
├─────────────────┤
│  • 예측 수행    │
│  • 확률 계산    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HTML Template  │
│  (Jinja2)       │
├─────────────────┤
│  • 결과 렌더링   │
└─────────────────┘
```

### 기술 스택
- **Backend**: Flask 3.0.0
- **ML Framework**: LightGBM 4.1.0
- **Data Processing**: pandas 2.1.3, numpy 1.26.2
- **Frontend**: HTML5, CSS3, Jinja2 템플릿
- **Model**: scikit-learn 1.3.2 (전처리 및 평가)

---

## 프로젝트 구조

```
autoML_202511/
├── app.py                          # Flask 애플리케이션 메인 파일
├── model_lgbm.py                   # 모델 학습 스크립트
├── preprocess.py                   # 데이터 전처리 스크립트
├── lgbm_titanic_model.pkl          # 학습된 모델 파일
├── train.csv                       # 원본 학습 데이터
├── train_preprocessed.csv          # 전처리된 학습 데이터
├── requirements.txt                # Python 패키지 의존성
├── README_Flask_Service.md         # 사용자 가이드
├── DEVELOPER_GUIDE.md              # 개발자 가이드 (본 문서)
└── templates/                      # Jinja2 HTML 템플릿
    ├── index.html                  # 메인 입력 폼
    ├── result.html                 # 예측 결과 페이지
    └── error.html                  # 에러 페이지
```

---

## 데이터 전처리 파이프라인

### 1. 원본 데이터 특성
```python
# train.csv 원본 컬럼
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
```

### 2. 전처리 과정 (preprocess.py)

#### 2.1 피처 엔지니어링

**Title 추출**
```python
# Name에서 호칭 추출
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# 희귀한 호칭 통합
{
    'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
    'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona' → 'Other'
    'Mlle', 'Ms' → 'Miss'
    'Mme' → 'Mrs'
}
```

**Deck 추출**
```python
# Cabin의 첫 글자를 Deck으로 사용
df['Deck'] = df['Cabin'].str[0].fillna('U')  # U = Unknown
```

**가족 관련 피처**
```python
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
```

#### 2.2 결측치 처리

| 컬럼 | 처리 방법 |
|------|-----------|
| Age | Title별 중앙값으로 대체 |
| Fare | Pclass별 중앙값으로 대체 |
| Embarked | 최빈값(mode)으로 대체 |
| Cabin | 'U' (Unknown)으로 대체 |

#### 2.3 인코딩

```python
# 범주형 변수를 숫자로 변환
encoding_map = {
    'Sex': {'male': 0, 'female': 1},
    'Title': {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5},
    'Deck': {'U': 0, 'C': 1, 'E': 2, 'G': 3, 'D': 4, 'A': 5, 'B': 6, 'F': 7, 'T': 8},
    'Embarked': {'S': 0, 'C': 1, 'Q': 2}
}
```

#### 2.4 스케일링

```python
# StandardScaler 적용
# Age와 Fare를 평균 0, 표준편차 1로 정규화
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
```

### 3. 최종 특성 (9개)

```python
features = [
    'Pclass',      # 객실 등급 (1, 2, 3)
    'Sex',         # 성별 (0: 남성, 1: 여성)
    'Age',         # 나이 (스케일링됨)
    'Fare',        # 운임 (스케일링됨)
    'Embarked',    # 승선 항구 (0: S, 1: C, 2: Q)
    'Title',       # 호칭 (1-5)
    'Deck',        # 객실 위치 (0-8)
    'FamilySize',  # 가족 크기 (1~)
    'IsAlone'      # 혼자 여부 (0 or 1)
]
```

---

## 모델 상세

### LightGBM 하이퍼파라미터

```python
model = lgb.LGBMClassifier(
    objective='binary',              # 이진 분류
    metric='binary_logloss',         # 손실 함수
    n_estimators=500,                # 최대 트리 개수
    learning_rate=0.05,              # 학습률
    num_leaves=40,                   # 리프 노드 개수
    max_depth=7,                     # 트리 최대 깊이
    min_child_samples=20,            # 리프 노드 최소 샘플 수
    subsample=0.8,                   # 행 샘플링 비율
    colsample_bytree=0.8,            # 열 샘플링 비율
    random_state=42,                 # 재현성
    verbose=-1                       # 로그 출력 없음
)
```

### 조기 종료 (Early Stopping)

```python
# 50회 반복 동안 성능 개선이 없으면 학습 중단
callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
```

### 모델 성능 지표

학습 완료 후 다음 지표들이 계산됩니다:
- **Accuracy**: 전체 정확도
- **Precision**: 생존 예측의 정밀도
- **Recall**: 실제 생존자를 찾아낸 비율
- **F1-Score**: Precision과 Recall의 조화 평균
- **ROC-AUC**: 분류 성능 종합 지표

### 피처 중요도

모델은 다음과 같은 순서로 특성의 중요도를 학습합니다:
1. Sex (성별)
2. Title (호칭)
3. Fare (운임)
4. Age (나이)
5. Pclass (객실 등급)
6. FamilySize (가족 크기)
7. 기타...

---

## API 명세

### 1. 웹 인터페이스 엔드포인트

#### GET /
메인 페이지 - 입력 폼 렌더링

**응답**: HTML 페이지 (index.html)

---

#### POST /predict
폼 데이터를 받아 예측 수행

**요청 형식**: `application/x-www-form-urlencoded`

**요청 파라미터**:
```python
{
    'name': str,        # 이름 (호칭 포함) - 필수
    'pclass': int,      # 객실 등급 (1-3) - 필수
    'sex': str,         # 성별 ('male' or 'female') - 필수
    'age': float,       # 나이 - 필수
    'sibsp': int,       # 형제/배우자 수 - 필수
    'parch': int,       # 부모/자녀 수 - 필수
    'fare': float,      # 운임 - 필수
    'cabin': str,       # 객실 번호 - 선택
    'embarked': str     # 승선 항구 ('C', 'Q', 'S') - 필수
}
```

**응답**: HTML 페이지 (result.html 또는 error.html)

**result.html에 전달되는 데이터**:
```python
{
    'survived': int,                      # 0 or 1
    'probability_death': float,           # 0.0 ~ 1.0
    'probability_survival': float,        # 0.0 ~ 1.0
    'passenger_info': {
        'name': str,
        'title': str,
        'pclass': int,
        'sex': str,
        'age': float,
        'family_size': int
    }
}
```

---

### 2. RESTful API 엔드포인트

#### POST /api/predict
JSON 형식으로 예측 요청

**요청 헤더**:
```
Content-Type: application/json
```

**요청 본문**:
```json
{
    "Name": "Mr. John Smith",
    "Pclass": "1",
    "Sex": "male",
    "Age": "30",
    "SibSp": "1",
    "Parch": "0",
    "Fare": "50.0",
    "Cabin": "C85",
    "Embarked": "S"
}
```

**성공 응답** (200 OK):
```json
{
    "success": true,
    "survived": 1,
    "probability_death": 0.234,
    "probability_survival": 0.766,
    "title": "Mr"
}
```

**실패 응답** (400 Bad Request):
```json
{
    "success": false,
    "error": "오류 메시지"
}
```

**cURL 예제**:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Name": "Mrs. Elizabeth Smith",
    "Pclass": "1",
    "Sex": "female",
    "Age": "25",
    "SibSp": "0",
    "Parch": "1",
    "Fare": "100",
    "Cabin": "C85",
    "Embarked": "C"
  }'
```

**Python 예제**:
```python
import requests

url = 'http://localhost:5000/api/predict'
data = {
    "Name": "Mr. John Smith",
    "Pclass": "1",
    "Sex": "male",
    "Age": "30",
    "SibSp": "1",
    "Parch": "0",
    "Fare": "50.0",
    "Cabin": "C85",
    "Embarked": "S"
}

response = requests.post(url, json=data)
result = response.json()
print(f"생존 확률: {result['probability_survival']*100:.1f}%")
```

---

## 코드 상세 설명

### app.py 주요 함수

#### 1. 모델 및 스케일링 파라미터 로드

```python
# 저장된 모델 로드
with open('lgbm_titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 전처리된 학습 데이터에서 스케일링 파라미터 계산
train_df = pd.read_csv('train_preprocessed.csv')
age_mean = train_df['Age'].mean()
age_std = train_df['Age'].std()
fare_mean = train_df['Fare'].mean()
fare_std = train_df['Fare'].std()
```

**중요**: 스케일링 파라미터는 학습 시 사용된 값과 동일해야 합니다.

---

#### 2. extract_title() 함수

```python
def extract_title(name):
    """이름에서 호칭 추출

    Args:
        name (str): 전체 이름 (예: "Mr. John Smith")

    Returns:
        str: 표준화된 호칭 ('Mr', 'Miss', 'Mrs', 'Master', 'Other')

    예제:
        >>> extract_title("Mr. John Smith")
        'Mr'
        >>> extract_title("Miss. Jane Doe")
        'Miss'
        >>> extract_title("Dr. Robert Brown")
        'Other'
    """
```

**처리 로직**:
1. 정규표현식으로 호칭 추출: ` ([A-Za-z]+)\.`
2. 희귀 호칭을 'Other'로 통합
3. 유사 호칭 정규화 (Mlle→Miss, Mme→Mrs)

---

#### 3. preprocess_input() 함수

```python
def preprocess_input(data):
    """사용자 입력 데이터를 모델 입력 형식으로 전처리

    Args:
        data (dict): 폼에서 받은 원본 데이터

    Returns:
        tuple: (features DataFrame, title)
            - features: 모델 입력용 특성 데이터프레임
            - title: 추출된 호칭

    처리 순서:
        1. Title 추출 및 인코딩
        2. Sex 인코딩
        3. Deck 추출 및 인코딩
        4. Embarked 인코딩
        5. FamilySize, IsAlone 계산
        6. Age, Fare 스케일링
        7. DataFrame 생성 (모델 학습 시 순서와 동일)
    """
```

**스케일링 공식**:
```python
scaled_value = (original_value - mean) / std
```

**특성 순서** (중요!):
모델 학습 시 사용된 순서와 정확히 일치해야 합니다.
```python
['Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
 'Title', 'Deck', 'FamilySize', 'IsAlone']
```

---

#### 4. predict() 라우트

```python
@app.route('/predict', methods=['POST'])
def predict():
    """예측 수행 및 결과 렌더링

    플로우:
        1. 폼 데이터 수집
        2. preprocess_input()로 전처리
        3. 모델 예측 (predict, predict_proba)
        4. 결과 딕셔너리 생성
        5. result.html 템플릿 렌더링

    예외 처리:
        - 모든 예외는 error.html로 리다이렉트
    """
```

---

### HTML 템플릿

#### index.html 구조

```html
<form action="/predict" method="POST">
    <!-- 입력 필드들 -->
    <input type="text" name="name" required>
    <select name="pclass" required>
    <!-- ... -->
    <button type="submit">예측하기</button>
</form>
```

**CSS 특징**:
- Flexbox 레이아웃
- 그라디언트 배경
- 반응형 디자인 (`@media` 쿼리)
- 호버 효과 및 트랜지션

---

#### result.html 구조

**Jinja2 조건문**:
```html
{% if result.survived == 1 %}
    <h1 class="survived">생존 예측</h1>
{% else %}
    <h1 class="died">사망 예측</h1>
{% endif %}
```

**확률 시각화**:
```html
<div class="probability-bar">
    <div class="probability-fill survival-bar"
         style="width: {{ result.probability_survival * 100 }}%">
        {{ "%.1f" | format(result.probability_survival * 100) }}%
    </div>
</div>
```

**Jinja2 필터 사용**:
- `{{ "%.1f" | format(value) }}`: 소수점 1자리 포맷팅

---

## 확장 가이드

### 1. 새로운 특성 추가하기

**Step 1**: `preprocess.py` 수정
```python
# 새로운 특성 생성
df['NewFeature'] = df['ExistingColumn'].apply(custom_function)
```

**Step 2**: `app.py`의 `preprocess_input()` 수정
```python
new_feature_value = calculate_new_feature(data)
features = pd.DataFrame({
    # ... 기존 특성들
    'NewFeature': [new_feature_value]
})
```

**Step 3**: 모델 재학습
```bash
python3 preprocess.py
python3 model_lgbm.py
```

---

### 2. 다른 모델로 교체하기

**XGBoost 예제**:
```python
import xgboost as xgb

# 모델 학습
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# 저장
with open('xgb_titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

**app.py 수정**:
```python
# 모델 로드 부분만 변경
with open('xgb_titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

---

### 3. 데이터베이스 연동

**SQLite 예제**:
```python
import sqlite3
from datetime import datetime

@app.route('/predict', methods=['POST'])
def predict():
    # ... 예측 수행

    # 예측 결과 저장
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions
        (timestamp, name, prediction, probability)
        VALUES (?, ?, ?, ?)
    ''', (datetime.now(), data['Name'], prediction, probability))
    conn.commit()
    conn.close()

    return render_template('result.html', result=result)
```

---

### 4. 배치 예측 기능 추가

```python
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """CSV 파일 업로드로 배치 예측"""
    file = request.files['file']
    df = pd.read_csv(file)

    predictions = []
    for _, row in df.iterrows():
        features, title = preprocess_input(row.to_dict())
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]
        predictions.append({
            'name': row['Name'],
            'survived': int(pred),
            'probability': float(prob)
        })

    return jsonify(predictions)
```

---

### 5. 모델 버전 관리

```python
import os
from datetime import datetime

# 모델 저장 시 타임스탬프 추가
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = f'models/lgbm_titanic_{timestamp}.pkl'

os.makedirs('models', exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# 메타데이터 저장
metadata = {
    'timestamp': timestamp,
    'accuracy': accuracy,
    'features': X.columns.tolist(),
    'hyperparameters': model.get_params()
}
import json
with open(f'models/metadata_{timestamp}.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

### 6. 로깅 추가

```python
import logging
from logging.handlers import RotatingFileHandler

# 로거 설정
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
app.logger.addHandler(handler)

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info(f"Prediction request received: {request.form}")
    try:
        # ... 예측 로직
        app.logger.info(f"Prediction successful: {prediction}")
    except Exception as e:
        app.logger.error(f"Prediction failed: {str(e)}")
        raise
```

---

### 7. CORS 설정 (다른 도메인에서 API 호출 허용)

```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 모든 도메인 허용

# 또는 특정 도메인만 허용
CORS(app, resources={r"/api/*": {"origins": "http://example.com"}})
```

---

## 트러블슈팅

### 1. 모델 로드 실패

**에러**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'lgbm_titanic_model.pkl'
```

**해결**:
- 현재 디렉토리 확인: `pwd`
- 모델 파일 존재 확인: `ls -la *.pkl`
- `app.py`와 같은 디렉토리에 모델 파일이 있는지 확인

---

### 2. 스케일링 파라미터 불일치

**증상**: 예측 결과가 이상하게 나옴

**원인**: `train_preprocessed.csv`가 없거나 다른 데이터

**해결**:
```bash
# 전처리 다시 실행
python3 preprocess.py

# app.py 재시작
python3 app.py
```

---

### 3. 포트 충돌

**에러**:
```
OSError: [Errno 98] Address already in use
```

**해결**:
```bash
# 5000번 포트 사용 중인 프로세스 찾기
lsof -i :5000

# 프로세스 종료
kill -9 <PID>

# 또는 다른 포트 사용
app.run(port=5001)
```

---

### 4. 템플릿 렌더링 실패

**에러**:
```
jinja2.exceptions.TemplateNotFound: index.html
```

**해결**:
- `templates/` 폴더가 `app.py`와 같은 디렉토리에 있는지 확인
- 파일 이름 및 확장자 확인 (대소문자 구분)

---

### 5. 인코딩 문제

**에러**:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**해결**:
```python
# CSV 읽을 때 인코딩 명시
df = pd.read_csv('train.csv', encoding='utf-8')

# 또는 다른 인코딩 시도
df = pd.read_csv('train.csv', encoding='cp949')
```

---

### 6. 메모리 부족

**증상**: 서버가 느려지거나 크래시

**해결**:
```python
# 모델을 전역 변수로 한 번만 로드 (현재 구현)
# app.py 시작 시 로드됨 ✓

# 큰 데이터 처리 시 청크 사용
df = pd.read_csv('large_file.csv', chunksize=1000)
for chunk in df:
    process(chunk)
```

---

### 7. WSGI 서버 배포 시 이슈

**개발 서버 (Flask 내장)**:
```python
# 개발 환경에서만 사용
app.run(debug=True)
```

**프로덕션 서버 (Gunicorn)**:
```bash
# 설치
pip install gunicorn

# 실행
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**프로덕션 서버 (uWSGI)**:
```bash
# 설치
pip install uwsgi

# 실행
uwsgi --http :5000 --wsgi-file app.py --callable app --processes 4 --threads 2
```

---

## 성능 최적화

### 1. 모델 예측 캐싱

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_predict(feature_tuple):
    """동일한 입력에 대해 결과 캐싱"""
    features = pd.DataFrame([feature_tuple])
    return model.predict(features)[0]
```

### 2. 비동기 처리

```python
from flask import Flask
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.route('/predict', methods=['POST'])
def predict():
    # 무거운 작업을 백그라운드로
    future = executor.submit(perform_prediction, data)
    result = future.result()
    return render_template('result.html', result=result)
```

---

## 보안 고려사항

### 1. 입력 검증

```python
from wtforms import Form, StringField, validators

class PredictionForm(Form):
    name = StringField('Name', [validators.Length(min=2, max=100)])
    age = FloatField('Age', [validators.NumberRange(min=0, max=120)])
    # ...
```

### 2. Rate Limiting

```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # ...
```

### 3. 환경 변수 사용

```python
import os
from dotenv import load_dotenv

load_dotenv()

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
MODEL_PATH = os.getenv('MODEL_PATH', 'lgbm_titanic_model.pkl')
```

---

## 테스트

### 단위 테스트 예제

```python
import unittest

class TestPreprocessing(unittest.TestCase):
    def test_extract_title(self):
        self.assertEqual(extract_title("Mr. John Smith"), "Mr")
        self.assertEqual(extract_title("Miss. Jane Doe"), "Miss")
        self.assertEqual(extract_title("Dr. Robert Brown"), "Other")

    def test_preprocess_input(self):
        data = {
            'Name': 'Mr. Test User',
            'Pclass': '1',
            'Sex': 'male',
            'Age': '30',
            'SibSp': '0',
            'Parch': '0',
            'Fare': '50',
            'Embarked': 'S',
            'Cabin': ''
        }
        features, title = preprocess_input(data)
        self.assertEqual(len(features.columns), 9)
        self.assertEqual(title, 'Mr')

if __name__ == '__main__':
    unittest.test()
```

---

## 참고 자료

- [Flask 공식 문서](https://flask.palletsprojects.com/)
- [LightGBM 공식 문서](https://lightgbm.readthedocs.io/)
- [Jinja2 템플릿 가이드](https://jinja.palletsprojects.com/)
- [scikit-learn 전처리](https://scikit-learn.org/stable/modules/preprocessing.html)

---

## 라이선스 및 기여

본 프로젝트는 교육 목적으로 제작되었습니다.

**기여 방법**:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|-----------|
| 1.0.0 | 2025-11-18 | 초기 버전 릴리스 |

---

**마지막 업데이트**: 2025-11-18
**작성자**: Claude Code
**문서 버전**: 1.0.0
