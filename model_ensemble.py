import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

print("=" * 60)
print("앙상블 모델 (LightGBM + Random Forest + Logistic Regression)")
print("=" * 60)

# 1. 데이터 로딩
print("\n[1/7] 데이터 로딩 중...")
df = pd.read_csv('train_preprocessed.csv')
print(f"데이터 크기: {df.shape}")
print(f"생존률: {df['Survived'].mean():.2%}")

# 2. 특성과 타겟 분리
print("\n[2/7] 특성과 타겟 분리 중...")
X = df.drop('Survived', axis=1)
y = df['Survived']
print(f"특성 개수: {X.shape[1]}")

# 3. 학습/검증 데이터 분리
print("\n[3/7] 학습/검증 데이터 분리 중...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"학습 데이터: {X_train.shape}")
print(f"검증 데이터: {X_test.shape}")

# 4. 개별 모델 정의
print("\n[4/7] 개별 모델 정의 중...")

# LightGBM
lgbm_model = lgb.LGBMClassifier(
    objective='binary',
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

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# Logistic Regression
lr_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42
)

print("✓ LightGBM 모델")
print("✓ Random Forest 모델")
print("✓ Logistic Regression 모델")

# 5. Voting Classifier (Soft Voting)
print("\n[5/7] Voting Classifier 학습 중...")
voting_clf = VotingClassifier(
    estimators=[
        ('lgbm', lgbm_model),
        ('rf', rf_model),
        ('lr', lr_model)
    ],
    voting='soft',
    n_jobs=-1
)

voting_clf.fit(X_train, y_train)
print("✓ Voting Classifier 학습 완료")

# 6. Stacking Classifier
print("\n[6/7] Stacking Classifier 학습 중...")
stacking_clf = StackingClassifier(
    estimators=[
        ('lgbm', lgbm_model),
        ('rf', rf_model),
        ('lr', lr_model)
    ],
    final_estimator=LogisticRegression(random_state=42),
    cv=5,
    n_jobs=-1
)

stacking_clf.fit(X_train, y_train)
print("✓ Stacking Classifier 학습 완료")

# 7. 성능 평가
print("\n[7/7] 모델 성능 평가 중...")

# Voting Classifier 평가
y_pred_voting = voting_clf.predict(X_test)
y_pred_proba_voting = voting_clf.predict_proba(X_test)[:, 1]

voting_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_voting),
    'precision': precision_score(y_test, y_pred_voting),
    'recall': recall_score(y_test, y_pred_voting),
    'f1': f1_score(y_test, y_pred_voting),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_voting)
}

# Stacking Classifier 평가
y_pred_stacking = stacking_clf.predict(X_test)
y_pred_proba_stacking = stacking_clf.predict_proba(X_test)[:, 1]

stacking_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_stacking),
    'precision': precision_score(y_test, y_pred_stacking),
    'recall': recall_score(y_test, y_pred_stacking),
    'f1': f1_score(y_test, y_pred_stacking),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_stacking)
}

# 결과 출력
print("\n" + "=" * 60)
print("Voting Classifier 성능 지표")
print("=" * 60)
print(f"정확도 (Accuracy):  {voting_metrics['accuracy']:.4f} ({voting_metrics['accuracy']*100:.2f}%)")
print(f"정밀도 (Precision): {voting_metrics['precision']:.4f}")
print(f"재현율 (Recall):    {voting_metrics['recall']:.4f}")
print(f"F1 Score:          {voting_metrics['f1']:.4f}")
print(f"ROC-AUC Score:     {voting_metrics['roc_auc']:.4f}")

# 교차 검증
print("\n5-Fold 교차 검증 점수:")
cv_scores_voting = cross_val_score(voting_clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f"CV 점수: {cv_scores_voting}")
print(f"평균 CV 점수: {cv_scores_voting.mean():.4f} (+/- {cv_scores_voting.std() * 2:.4f})")

# 혼동 행렬
cm_voting = confusion_matrix(y_test, y_pred_voting)
print("\n혼동 행렬:")
print(cm_voting)
print(f"\n실제 사망자 중 정확히 예측: {cm_voting[0,0]}/{cm_voting[0].sum()} ({cm_voting[0,0]/cm_voting[0].sum()*100:.1f}%)")
print(f"실제 생존자 중 정확히 예측: {cm_voting[1,1]}/{cm_voting[1].sum()} ({cm_voting[1,1]/cm_voting[1].sum()*100:.1f}%)")

print("\n상세 분류 리포트:")
print(classification_report(y_test, y_pred_voting, target_names=['사망', '생존']))

print("\n" + "=" * 60)
print("Stacking Classifier 성능 지표")
print("=" * 60)
print(f"정확도 (Accuracy):  {stacking_metrics['accuracy']:.4f} ({stacking_metrics['accuracy']*100:.2f}%)")
print(f"정밀도 (Precision): {stacking_metrics['precision']:.4f}")
print(f"재현율 (Recall):    {stacking_metrics['recall']:.4f}")
print(f"F1 Score:          {stacking_metrics['f1']:.4f}")
print(f"ROC-AUC Score:     {stacking_metrics['roc_auc']:.4f}")

# 교차 검증
print("\n5-Fold 교차 검증 점수:")
cv_scores_stacking = cross_val_score(stacking_clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f"CV 점수: {cv_scores_stacking}")
print(f"평균 CV 점수: {cv_scores_stacking.mean():.4f} (+/- {cv_scores_stacking.std() * 2:.4f})")

# 혼동 행렬
cm_stacking = confusion_matrix(y_test, y_pred_stacking)
print("\n혼동 행렬:")
print(cm_stacking)
print(f"\n실제 사망자 중 정확히 예측: {cm_stacking[0,0]}/{cm_stacking[0].sum()} ({cm_stacking[0,0]/cm_stacking[0].sum()*100:.1f}%)")
print(f"실제 생존자 중 정확히 예측: {cm_stacking[1,1]}/{cm_stacking[1].sum()} ({cm_stacking[1,1]/cm_stacking[1].sum()*100:.1f}%)")

print("\n상세 분류 리포트:")
print(classification_report(y_test, y_pred_stacking, target_names=['사망', '생존']))

# 시각화
print("\n시각화 생성 중...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Voting - 혼동 행렬
sns.heatmap(cm_voting, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Death', 'Survived'], yticklabels=['Death', 'Survived'])
axes[0, 0].set_title('Voting - Confusion Matrix')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')

# 2. Voting - 예측 확률 분포
axes[0, 1].hist(y_pred_proba_voting[y_test == 0], bins=30, alpha=0.5, label='Death', color='red')
axes[0, 1].hist(y_pred_proba_voting[y_test == 1], bins=30, alpha=0.5, label='Survived', color='blue')
axes[0, 1].set_xlabel('Predicted Probability')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Voting - Probability Distribution')
axes[0, 1].legend()

# 3. Voting - 성능 지표
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metrics_values = [voting_metrics['accuracy'], voting_metrics['precision'],
                  voting_metrics['recall'], voting_metrics['f1'], voting_metrics['roc_auc']]
axes[0, 2].bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
axes[0, 2].set_ylim([0, 1])
axes[0, 2].set_ylabel('Score')
axes[0, 2].set_title('Voting - Performance Metrics')
axes[0, 2].axhline(y=0.8, color='r', linestyle='--', alpha=0.3)
for i, value in enumerate(metrics_values):
    axes[0, 2].text(i, value + 0.02, f'{value:.3f}', ha='center')

# 4. Stacking - 혼동 행렬
sns.heatmap(cm_stacking, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0],
            xticklabels=['Death', 'Survived'], yticklabels=['Death', 'Survived'])
axes[1, 0].set_title('Stacking - Confusion Matrix')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

# 5. Stacking - 예측 확률 분포
axes[1, 1].hist(y_pred_proba_stacking[y_test == 0], bins=30, alpha=0.5, label='Death', color='red')
axes[1, 1].hist(y_pred_proba_stacking[y_test == 1], bins=30, alpha=0.5, label='Survived', color='blue')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Stacking - Probability Distribution')
axes[1, 1].legend()

# 6. Stacking - 성능 지표
metrics_values_stacking = [stacking_metrics['accuracy'], stacking_metrics['precision'],
                           stacking_metrics['recall'], stacking_metrics['f1'], stacking_metrics['roc_auc']]
axes[1, 2].bar(metrics_names, metrics_values_stacking, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
axes[1, 2].set_ylim([0, 1])
axes[1, 2].set_ylabel('Score')
axes[1, 2].set_title('Stacking - Performance Metrics')
axes[1, 2].axhline(y=0.8, color='r', linestyle='--', alpha=0.3)
for i, value in enumerate(metrics_values_stacking):
    axes[1, 2].text(i, value + 0.02, f'{value:.3f}', ha='center')

plt.tight_layout()
plt.savefig('ensemble_model_results.png', dpi=300, bbox_inches='tight')
print("시각화 저장 완료: ensemble_model_results.png")

# 모델 저장
print("\n모델 저장 중...")
with open('voting_classifier_model.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)
print("Voting Classifier 저장 완료: voting_classifier_model.pkl")

with open('stacking_classifier_model.pkl', 'wb') as f:
    pickle.dump(stacking_clf, f)
print("Stacking Classifier 저장 완료: stacking_classifier_model.pkl")

# 최종 요약
print("\n" + "=" * 60)
print("앙상블 모델 학습 완료!")
print("=" * 60)
print(f"\nVoting Classifier 정확도:   {voting_metrics['accuracy']:.4f}")
print(f"Stacking Classifier 정확도: {stacking_metrics['accuracy']:.4f}")
print(f"\n최고 성능 모델: {'Voting' if voting_metrics['accuracy'] > stacking_metrics['accuracy'] else 'Stacking'}")
