import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import time

print("=" * 70)
print("모델 성능 비교: LightGBM vs Voting Ensemble vs Stacking Ensemble")
print("=" * 70)

# 1. 데이터 로딩
print("\n[1/4] 데이터 로딩 중...")
df = pd.read_csv('train_preprocessed.csv')
X = df.drop('Survived', axis=1)
y = df['Survived']

# 학습/검증 데이터 분리 (동일한 random_state 사용)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"학습 데이터: {X_train.shape}, 검증 데이터: {X_test.shape}")

# 2. 모델 정의
print("\n[2/4] 모델 정의 중...")

models = {}

# LightGBM
models['LightGBM'] = lgb.LGBMClassifier(
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

# Voting Classifier
lgbm_base = lgb.LGBMClassifier(
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

rf_base = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

lr_base = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42
)

models['Voting Ensemble'] = VotingClassifier(
    estimators=[
        ('lgbm', lgbm_base),
        ('rf', rf_base),
        ('lr', lr_base)
    ],
    voting='soft',
    n_jobs=-1
)

# Stacking Classifier
models['Stacking Ensemble'] = StackingClassifier(
    estimators=[
        ('lgbm', lgb.LGBMClassifier(objective='binary', n_estimators=500, learning_rate=0.05,
                                    num_leaves=40, max_depth=7, random_state=42, verbose=-1)),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)),
        ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ],
    final_estimator=LogisticRegression(random_state=42),
    cv=5,
    n_jobs=-1
)

print(f"총 {len(models)}개 모델 준비 완료")

# 3. 모델 학습 및 평가
print("\n[3/4] 모델 학습 및 평가 중...")

results = []

for model_name, model in models.items():
    print(f"\n{'='*70}")
    print(f"모델: {model_name}")
    print(f"{'='*70}")

    # 학습 시간 측정
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 성능 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # 교차 검증
    print(f"교차 검증 중... (5-Fold CV)")
    cv_start = time.time()
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cv_time = time.time() - cv_start
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # 결과 저장
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'CV Mean': cv_mean,
        'CV Std': cv_std,
        'Training Time (s)': training_time,
        'CV Time (s)': cv_time
    })

    # 출력
    print(f"정확도 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"정밀도 (Precision): {precision:.4f}")
    print(f"재현율 (Recall):    {recall:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"ROC-AUC Score:     {roc_auc:.4f}")
    print(f"CV Mean:           {cv_mean:.4f} (+/- {cv_std*2:.4f})")
    print(f"학습 시간:         {training_time:.2f}초")
    print(f"CV 시간:           {cv_time:.2f}초")

# 4. 결과 비교 및 시각화
print(f"\n{'='*70}")
print("최종 결과 비교")
print(f"{'='*70}")

results_df = pd.DataFrame(results)
results_df = results_df.round(4)

print("\n")
print(results_df.to_string(index=False))

# 최고 성능 모델
best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
best_accuracy = results_df['Accuracy'].max()
print(f"\n🏆 최고 성능 모델: {best_model} (정확도: {best_accuracy:.4f})")

# 시각화
print("\n[4/4] 시각화 생성 중...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 정확도 비교
ax1 = fig.add_subplot(gs[0, :])
x_pos = np.arange(len(results_df))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax1.bar(x_pos, results_df['Accuracy'], color=colors, alpha=0.8, edgecolor='black')
ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results_df['Model'], fontsize=11)
ax1.set_ylim([0.75, 0.88])
ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% Baseline')
ax1.grid(axis='y', alpha=0.3)
ax1.legend()

for i, (bar, val) in enumerate(zip(bars, results_df['Accuracy'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.4f}\n({val*100:.2f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2. 다중 지표 비교
ax2 = fig.add_subplot(gs[1, 0])
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.25

for i, model_name in enumerate(results_df['Model']):
    values = results_df.loc[results_df['Model'] == model_name, metrics].values[0]
    ax2.bar(x + i*width, values, width, label=model_name, alpha=0.8, color=colors[i])

ax2.set_xlabel('Metrics', fontsize=11, fontweight='bold')
ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
ax2.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
ax2.set_xticks(x + width)
ax2.set_xticklabels(metrics, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 1.0])

# 3. CV Mean vs Accuracy
ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(results_df['CV Mean'], results_df['Accuracy'], s=300, c=colors, alpha=0.6, edgecolors='black', linewidths=2)
for i, model in enumerate(results_df['Model']):
    ax3.annotate(model, (results_df.loc[i, 'CV Mean'], results_df.loc[i, 'Accuracy']),
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
ax3.plot([0.75, 0.88], [0.75, 0.88], 'r--', alpha=0.5, label='Perfect Match')
ax3.set_xlabel('CV Mean Accuracy', fontsize=11, fontweight='bold')
ax3.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
ax3.set_title('Cross-Validation vs Test Accuracy', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. 학습 시간 비교
ax4 = fig.add_subplot(gs[1, 2])
bars = ax4.barh(results_df['Model'], results_df['Training Time (s)'], color=colors, alpha=0.8, edgecolor='black')
ax4.set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
ax4.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, results_df['Training Time (s)'])):
    ax4.text(val + 0.1, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}s', va='center', fontsize=10, fontweight='bold')

# 5. 성능-시간 트레이드오프
ax5 = fig.add_subplot(gs[2, 0])
scatter = ax5.scatter(results_df['Training Time (s)'], results_df['Accuracy'],
                     s=results_df['ROC-AUC']*500, c=colors, alpha=0.6, edgecolors='black', linewidths=2)
for i, model in enumerate(results_df['Model']):
    ax5.annotate(model, (results_df.loc[i, 'Training Time (s)'], results_df.loc[i, 'Accuracy']),
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
ax5.set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax5.set_title('Performance vs Training Time\n(Bubble size = ROC-AUC)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. CV 점수 분포
ax6 = fig.add_subplot(gs[2, 1])
bars = ax6.bar(results_df['Model'], results_df['CV Mean'], yerr=results_df['CV Std']*2,
              color=colors, alpha=0.8, edgecolor='black', capsize=10)
ax6.set_ylabel('CV Mean Accuracy', fontsize=11, fontweight='bold')
ax6.set_title('Cross-Validation Accuracy (Mean ± 2*Std)', fontsize=12, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, results_df['CV Mean'])):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 0.005,
             f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 7. Precision vs Recall
ax7 = fig.add_subplot(gs[2, 2])
ax7.scatter(results_df['Recall'], results_df['Precision'], s=300, c=colors, alpha=0.6, edgecolors='black', linewidths=2)
for i, model in enumerate(results_df['Model']):
    ax7.annotate(model, (results_df.loc[i, 'Recall'], results_df.loc[i, 'Precision']),
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
ax7.set_xlabel('Recall', fontsize=11, fontweight='bold')
ax7.set_ylabel('Precision', fontsize=11, fontweight='bold')
ax7.set_title('Precision vs Recall Trade-off', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

plt.suptitle('Comprehensive Model Comparison: LightGBM vs Ensemble Models',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
print("시각화 저장 완료: model_comparison_results.png")

# 결과 CSV 저장
results_df.to_csv('model_comparison_results.csv', index=False)
print("결과 CSV 저장 완료: model_comparison_results.csv")

print("\n" + "=" * 70)
print("모델 비교 완료!")
print("=" * 70)

# 승자 분석
print("\n📊 상세 분석:")
print(f"  • 가장 높은 정확도:     {results_df.loc[results_df['Accuracy'].idxmax(), 'Model']} ({results_df['Accuracy'].max():.4f})")
print(f"  • 가장 높은 ROC-AUC:    {results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']} ({results_df['ROC-AUC'].max():.4f})")
print(f"  • 가장 높은 F1-Score:   {results_df.loc[results_df['F1-Score'].idxmax(), 'Model']} ({results_df['F1-Score'].max():.4f})")
print(f"  • 가장 빠른 학습 시간:  {results_df.loc[results_df['Training Time (s)'].idxmin(), 'Model']} ({results_df['Training Time (s)'].min():.2f}초)")
print(f"  • 가장 안정적인 CV:     {results_df.loc[results_df['CV Std'].idxmin(), 'Model']} (Std: {results_df['CV Std'].min():.4f})")

# 추천
print("\n💡 추천:")
accuracy_diff = results_df['Accuracy'].max() - results_df['Accuracy'].min()
if accuracy_diff < 0.01:  # 1% 미만 차이
    fastest_model = results_df.loc[results_df['Training Time (s)'].idxmin(), 'Model']
    print(f"  모델들의 성능 차이가 작으므로, 가장 빠른 '{fastest_model}' 모델을 추천합니다.")
else:
    print(f"  최고 성능인 '{best_model}' 모델을 추천합니다.")
