from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import re

app = Flask(__name__)

# 모델 로드
with open('lgbm_titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 전처리를 위한 스케일링 파라미터 (훈련 데이터에서 계산된 값)
# train_preprocessed.csv에서 계산된 평균과 표준편차
train_df = pd.read_csv('train_preprocessed.csv')
age_mean = train_df['Age'].mean()
age_std = train_df['Age'].std()
fare_mean = train_df['Fare'].mean()
fare_std = train_df['Fare'].std()

def extract_title(name):
    """이름에서 호칭 추출"""
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        title = title_search.group(1)
        # 희귀한 호칭들을 'Other'로 통합
        if title in ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
            return 'Other'
        elif title == 'Mlle' or title == 'Ms':
            return 'Miss'
        elif title == 'Mme':
            return 'Mrs'
        return title
    return 'Other'

def preprocess_input(data):
    """입력 데이터 전처리"""
    # Title 추출
    title = extract_title(data['Name'])

    # Title 매핑
    title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
    title_encoded = title_map.get(title, 5)

    # Sex 매핑
    sex_encoded = 1 if data['Sex'] == 'female' else 0

    # Deck 추출 (Cabin이 없으면 'U')
    cabin = data.get('Cabin', '')
    deck = cabin[0] if cabin else 'U'
    deck_map = {'U': 0, 'C': 1, 'E': 2, 'G': 3, 'D': 4, 'A': 5, 'B': 6, 'F': 7, 'T': 8}
    deck_encoded = deck_map.get(deck, 0)

    # Embarked 매핑
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    embarked_encoded = embarked_map.get(data['Embarked'], 0)

    # FamilySize와 IsAlone 계산
    family_size = int(data['SibSp']) + int(data['Parch']) + 1
    is_alone = 1 if family_size == 1 else 0

    # Age와 Fare 스케일링
    age_scaled = (float(data['Age']) - age_mean) / age_std
    fare_scaled = (float(data['Fare']) - fare_mean) / fare_std

    # 특성 배열 생성 (모델 학습 시 사용된 순서와 동일하게)
    # ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'Deck', 'FamilySize', 'IsAlone']
    features = pd.DataFrame({
        'Pclass': [int(data['Pclass'])],
        'Sex': [sex_encoded],
        'Age': [age_scaled],
        'Fare': [fare_scaled],
        'Embarked': [embarked_encoded],
        'Title': [title_encoded],
        'Deck': [deck_encoded],
        'FamilySize': [family_size],
        'IsAlone': [is_alone]
    })

    return features, title

@app.route('/')
def home():
    """홈페이지"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """예측 수행"""
    try:
        # 폼 데이터 수집
        data = {
            'Name': request.form['name'],
            'Pclass': request.form['pclass'],
            'Sex': request.form['sex'],
            'Age': request.form['age'],
            'SibSp': request.form['sibsp'],
            'Parch': request.form['parch'],
            'Fare': request.form['fare'],
            'Cabin': request.form.get('cabin', ''),
            'Embarked': request.form['embarked']
        }

        # 전처리
        features, title = preprocess_input(data)

        # 예측
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]

        # 결과 준비
        result = {
            'survived': int(prediction),
            'probability_death': float(prediction_proba[0]),
            'probability_survival': float(prediction_proba[1]),
            'passenger_info': {
                'name': data['Name'],
                'title': title,
                'pclass': int(data['Pclass']),
                'sex': data['Sex'],
                'age': float(data['Age']),
                'family_size': int(data['SibSp']) + int(data['Parch']) + 1
            }
        }

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API 엔드포인트 (JSON 형식)"""
    try:
        data = request.json

        # 전처리
        features, title = preprocess_input(data)

        # 예측
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]

        return jsonify({
            'success': True,
            'survived': int(prediction),
            'probability_death': float(prediction_proba[0]),
            'probability_survival': float(prediction_proba[1]),
            'title': title
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    print("=" * 60)
    print("타이타닉 생존 예측 서비스")
    print("=" * 60)
    print("서버 시작: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
