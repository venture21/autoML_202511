import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

print("=" * 60)
print("LightGBM 타이타닉 생존 예측 모델")
print("=" * 60)

# 1. 데이터 불러오기
print("\n[1/6] 데이터 로딩 중...")
df = pd.read_csv('train_preprocessed.csv')
print(f"데이터 크기: {df.shape}")
print(f"생존률: {df['Survived'].mean():.2%}")

# 2. 특성(X)과 타겟(y) 분리
print("\n[2/6] 특성과 타겟 분리 중...")
X = df.drop('Survived', axis=1)
y = df['Survived']
print(f"특성 개수: {X.shape[1]}")
print(f"특성 목록: {X.columns.tolist()}")

# 3. 학습/검증 데이터 분리 (stratify로 클래스 비율 유지)
print("\n[3/6] 학습/검증 데이터 분리 중...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"학습 데이터: {X_train.shape}")
print(f"검증 데이터: {X_test.shape}")

# 4. LightGBM 모델 학습
print("\n[4/6] LightGBM 모델 학습 중...")
model = lgb.LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=40,
    max_depth=7,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='logloss',
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)
print(f"최적 반복 횟수: {model.best_iteration_}")

# 5. 예측 및 성능 평가
print("\n[5/6] 모델 성능 평가 중...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 상세 성능 지표
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 60)
print("모델 성능 지표")
print("=" * 60)
print(f"정확도 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"정밀도 (Precision): {precision:.4f}")
print(f"재현율 (Recall):    {recall:.4f}")
print(f"F1 Score:          {f1:.4f}")
print(f"ROC-AUC Score:     {roc_auc:.4f}")

# 교차 검증 점수
print("\n5-Fold 교차 검증 점수:")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV 점수: {cv_scores}")
print(f"평균 CV 점수: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 혼동 행렬
print("\n혼동 행렬:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\n실제 사망자 중 정확히 예측: {cm[0,0]}/{cm[0].sum()} ({cm[0,0]/cm[0].sum()*100:.1f}%)")
print(f"실제 생존자 중 정확히 예측: {cm[1,1]}/{cm[1].sum()} ({cm[1,1]/cm[1].sum()*100:.1f}%)")

# 분류 리포트
print("\n상세 분류 리포트:")
print(classification_report(y_test, y_pred, target_names=['사망', '생존']))

# 6. 피처 중요도 분석
print("\n[6/6] 피처 중요도 분석 중...")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 60)
print("피처 중요도 TOP 10")
print("=" * 60)
for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:15s}: {row['importance']:6.0f}")

# 시각화
print("\n시각화 생성 중...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 피처 중요도
axes[0, 0].barh(feature_importance['feature'], feature_importance['importance'])
axes[0, 0].set_xlabel('Importance')
axes[0, 0].set_title('Feature Importance')
axes[0, 0].invert_yaxis()

# 2. 혼동 행렬
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=['사망', '생존'], yticklabels=['사망', '생존'])
axes[0, 1].set_title('Confusion Matrix')
axes[0, 1].set_ylabel('True Label')
axes[0, 1].set_xlabel('Predicted Label')

# 3. 예측 확률 분포
axes[1, 0].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.5, label='사망', color='red')
axes[1, 0].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.5, label='생존', color='blue')
axes[1, 0].set_xlabel('Predicted Probability')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Prediction Probability Distribution')
axes[1, 0].legend()

# 4. 성능 지표 요약
metrics_data = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'ROC-AUC': roc_auc
}
axes[1, 1].bar(metrics_data.keys(), metrics_data.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
axes[1, 1].set_ylim([0, 1])
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Model Performance Metrics')
axes[1, 1].axhline(y=0.8, color='r', linestyle='--', alpha=0.3)
for i, (key, value) in enumerate(metrics_data.items()):
    axes[1, 1].text(i, value + 0.02, f'{value:.3f}', ha='center')

plt.tight_layout()
plt.savefig('lgbm_model_results.png', dpi=300, bbox_inches='tight')
print("시각화 저장 완료: lgbm_model_results.png")

# 모델 저장
print("\n모델 저장 중...")
with open('lgbm_titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("모델 저장 완료: lgbm_titanic_model.pkl")

print("\n" + "=" * 60)
print("모델 학습 완료!")
print("=" * 60)
