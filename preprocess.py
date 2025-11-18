import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. 데이터 불러오기
print("train.csv 파일을 읽는 중입니다...")
df = pd.read_csv('train.csv')
print("파일을 성공적으로 읽었습니다.")
print(f"데이터 크기: {df.shape}")

# 2. 피처 엔지니어링: Name에서 Title 추출
print("피처 엔지니어링: 'Name'에서 'Title' 추출 중...")
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# 희귀한 호칭들을 'Other'로 통합하고, 유사한 호칭들을 정리합니다.
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
print("'Title' 피처 생성 완료.")

# 3. 결측치 처리
print("결측치 처리 중...")
# Embarked의 결측치는 최빈값으로 채웁니다.
embarked_mode = df['Embarked'].mode()[0]
df['Embarked'].fillna(embarked_mode, inplace=True)

# Fare의 결측치는 Pclass별 중앙값으로 채웁니다.
if df['Fare'].isnull().any():
    fare_median_by_pclass = df.groupby('Pclass')['Fare'].transform('median')
    df['Fare'].fillna(fare_median_by_pclass, inplace=True)

# Age의 결측치는 Title별 Age 중앙값으로 채웁니다.
age_median_by_title = df.groupby('Title')['Age'].transform('median')
df['Age'].fillna(age_median_by_title, inplace=True)
# 그래도 남은 Age 결측치가 있다면 전체 Age 중앙값으로 채웁니다.
if df['Age'].isnull().any():
    df['Age'].fillna(df['Age'].median(), inplace=True)
print("결측치 처리가 완료되었습니다.")

# 4. 피처 엔지니어링: Cabin에서 Deck 추출
print("피처 엔지니어링: 'Cabin'에서 'Deck' 추출 중...")
# Cabin의 첫 글자를 Deck으로 사용하고, 결측치는 'U' (Unknown)으로 지정합니다.
df['Deck'] = df['Cabin'].str[0].fillna('U')
print("'Deck' 피처 생성 완료.")

# 5. 피처 엔지니어링: FamilySize 생성
print("피처 엔지니어링: 'FamilySize' 생성 중...")
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print("'FamilySize' 피처 생성 완료.")

# 6. 피처 엔지니어링: IsAlone 생성
print("피처 엔지니어링: 'IsAlone' 생성 중...")
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
print("'IsAlone' 피처 생성 완료.")

# 7. 불필요한 컬럼 제거
print("불필요한 컬럼 제거 중...")
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
print("컬럼 제거 완료.")

# 8. 범주형 데이터 수치화
print("범주형 데이터 수치화 중...")
# Sex, Title, Deck, Embarked 컬럼을 숫자형으로 변환합니다.
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
df['Title'] = df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}).astype(int)
df['Deck'] = df['Deck'].map({'U': 0, 'C': 1, 'E': 2, 'G': 3, 'D': 4, 'A': 5, 'B': 6, 'F': 7, 'T': 8}).astype(int)
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
print("데이터 수치화 완료.")

# 9. 스케일링
print("Age와 Fare 컬럼 스케일링 중...")
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
print("스케일링 완료.")

# 10. 최종 데이터 저장
output_file = 'train_preprocessed.csv'
print(f"전처리된 데이터를 '{output_file}' 파일로 저장 중입니다...")
df.to_csv(output_file, index=False)
print(f"성공적으로 '{output_file}' 파일을 생성했습니다.")

# 최종 데이터 정보 출력
print("\n--- 전처리 후 데이터 샘플 ---")
print(df.head())
print("\n--- 컬럼별 결측치 현황 ---")
print(df.isnull().sum())
print("\n--- 데이터 정보 요약 ---")
df.info()
