#!/usr/bin/env python3

import sys
import os

# Добавляем корень проекта в путь
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from src.config import RANDOM_SEED
from src.features.profiles import build_developer_profiles, add_advanced_features
from src.data.loader import load_data
from src.data.cleaner import clean_data, clean_text


# ============================================================================
# МЕТРИКИ РАНЖИРОВАНИЯ
# ============================================================================

def precision_at_k(y_true, y_score, k=5):
    """Precision@k - доля релевантных среди топ-k."""
    sorted_indices = np.argsort(y_score)[::-1]
    return np.sum(y_true[sorted_indices[:k]] == 1) / k


def recall_at_k(y_true, y_score, k=5):
    """Recall@k - доля релевантных в топ-k среди всех релевантных."""
    sorted_indices = np.argsort(y_score)[::-1]
    relevant_in_top_k = np.sum(y_true[sorted_indices[:k]] == 1)
    total_relevant = np.sum(y_true == 1)
    return relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0


def mean_reciprocal_rank(y_true, y_score):
    """MRR - обратная позиция первого релевантного."""
    sorted_indices = np.argsort(y_score)[::-1]
    for i, idx in enumerate(sorted_indices):
        if y_true[idx] == 1:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(y_true, y_score, k=5):
    """NDCG@k - нормализованная дисконтированная кумулятивная выгода."""
    sorted_indices = np.argsort(y_score)[::-1]
    dcg = sum(1.0 / np.log2(i + 2) for i, idx in enumerate(sorted_indices[:k]) if y_true[idx] == 1)
    ideal_sorted = np.argsort(y_true)[::-1]
    idcg = sum(1.0 / np.log2(i + 2) for i, idx in enumerate(ideal_sorted[:k]) if y_true[idx] == 1)
    return dcg / idcg if idcg > 0 else 0.0


def ranking_metrics(y_true, y_score, k=5):
    """Все метрики ранжирования."""
    return {
        'precision@5': precision_at_k(y_true, y_score, k=5),
        'recall@5': recall_at_k(y_true, y_score, k=5),
        'mrr': mean_reciprocal_rank(y_true, y_score),
        'ndcg@5': ndcg_at_k(y_true, y_score, k=5)
    }


# ============================================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================================

print("=" * 80)
print("МОДЕЛИРОВАНИЕ: РАНЖИРОВАНИЕ РАЗРАБОТЧИКОВ")
print("=" * 80)

print("\n[1/8] Загрузка данных...")
employees, jira_raw, gitlab_raw = load_data('data/raw')
print(f"  Employees: {employees.shape}")
print(f"  Jira: {jira_raw.shape}")
print(f"  GitLab: {gitlab_raw.shape}")


# ============================================================================
# КОРРЕКТНОЕ РАЗДЕЛЕНИЕ (ДО ОЧИСТКИ И ПРОФИЛЕЙ)
# ============================================================================

print("\n[2/8] Разделение на train/test (ДО очистки и профилей)...")

# Сначала делим сырые данные
unique_issues = jira_raw['issue_rk'].unique()
train_issues, test_issues = train_test_split(
    unique_issues, test_size=0.2, random_state=RANDOM_SEED
)

jira_train_raw = jira_raw[jira_raw['issue_rk'].isin(train_issues)].copy()
jira_test_raw = jira_raw[jira_raw['issue_rk'].isin(test_issues)].copy()

print(f"  Train задачи: {len(jira_train_raw)}")
print(f"  Test задачи: {len(jira_test_raw)}")


# ============================================================================
# ОЧИСТКА ДАННЫХ
# ============================================================================

print("\n[3/8] Очистка данных...")

# Очищаем train и test отдельно, но используя одни и те же функции
employees, jira_train, gitlab_train = clean_data(
    employees.copy(), jira_train_raw.copy(), gitlab_raw.copy()
)

# Для test данных - только очистка, без удаления строк из train
jira_test = jira_test_raw.copy()
jira_test['issue_desc_clean'] = jira_test['issue_desc'].apply(clean_text)

# Унификация логинов
jira_test['assignee_login_nm'] = jira_test['assignee_login_nm'].str.lower().str.strip()

print(f"  Train после очистки: {jira_train.shape}")
print(f"  Test после очистки: {jira_test.shape}")


# ============================================================================
# СОЗДАНИЕ ПРОФИЛЕЙ (ТОЛЬКО НА TRAIN)
# ============================================================================

print("\n[4/8] Создание профилей разработчиков (ТОЛЬКО на train)...")

# ВАЖНО: Для предотвращения leakage используем ТОЛЬКО признаки сотрудников
# БЕЗ агрегированной истории задач (jira_issues_count, commits_count)
# Иначе модель будет "читать" таргет из признаков

profiles = build_developer_profiles(employees, jira_train, gitlab_train, jira_train=jira_train)
profiles = add_advanced_features(profiles)

# Удаляем признаки, которые напрямую зависят от количества задач
# (они создают data leakage с таргетом is_active)
leakage_features = ['jira_issues_count', 'commits_count', 'jira_text', 'gitlab_text', 
                    'developer_profile_text', 'profile_words_count', 'projects_count', 'repos_count']
profiles = profiles.drop(columns=[c for c in leakage_features if c in profiles.columns], errors='ignore')

print(f"  Профили (после удаления leakage): {profiles.shape}")


# ============================================================================
# СОЗДАНИЕ ТАРГЕТА
# ============================================================================

print("\n[5/8] Создание таргета...")

# Таргет: активный разработчик в train (>= 3 задач)
developer_activity = jira_train.groupby('assignee_login_nm').size().reset_index(name='total_tasks')
developer_activity.columns = ['login', 'total_tasks']
threshold = 3
developer_activity['is_active'] = (developer_activity['total_tasks'] >= threshold).astype(int)

profiles = profiles.merge(developer_activity[['login', 'total_tasks', 'is_active']], on='login', how='left')
profiles['total_tasks'] = profiles['total_tasks'].fillna(0).astype(int)
profiles['is_active'] = profiles['is_active'].fillna(0).astype(int)

print(f"  Распределение таргета:")
print(f"    Активных (1): {profiles['is_active'].sum()}")
print(f"    Неактивных (0): {(profiles['is_active'] == 0).sum()}")
print(f"    Дисбаланс: {profiles['is_active'].mean():.2%}")


# ============================================================================
# ПОДГОТОВКА ПРИЗНАКОВ
# ============================================================================

print("\n[6/8] Подготовка признаков...")

# Базовые признаки (без истории задач - только характеристики сотрудников)
baseline_features = ['work_experience_day_cnt']

# Полные признаки (БЕЗ leakage признаков - только характеристики сотрудников)
full_features = [
    'work_experience_day_cnt',
    'is_senior', 'is_junior',
    'log_experience'
]

# Кодирование категориальных признаков
le_spec = LabelEncoder()
profiles['specialization_encoded'] = le_spec.fit_transform(
    profiles['specialization_nm'].fillna('Unknown')
)
full_features.append('specialization_encoded')

le_pos = LabelEncoder()
profiles['position_encoded'] = le_pos.fit_transform(
    profiles['position_nm'].fillna('Unknown')
)
full_features.append('position_encoded')

# Данные
X_baseline = profiles[baseline_features].fillna(0).values
X_full = profiles[full_features].fillna(0).values
y = profiles['is_active'].values

# Масштабирование
scaler = StandardScaler()
X_baseline_scaled = scaler.fit_transform(X_baseline)
X_full_scaled = scaler.fit_transform(X_full)

# Разделение на train/test для валидации моделей
X_train, X_test, y_train, y_test = train_test_split(
    X_full_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

print(f"  Baseline признаков: {len(baseline_features)}")
print(f"  Полных признаков: {len(full_features)}")
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")


# ============================================================================
# ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================================================

print("\n[7/8] Обучение моделей...")

results = []

# --- BASELINE ---
print("\n" + "-" * 60)
print("BASELINE: Logistic Regression (3 признака)")
print("-" * 60)

lr_baseline = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, class_weight='balanced')
cv_scores = cross_val_score(lr_baseline, X_baseline_scaled, y, cv=5, scoring='roc_auc')
lr_baseline.fit(X_baseline_scaled, y)
y_pred_proba = lr_baseline.predict_proba(X_baseline_scaled)[:, 1]
metrics = ranking_metrics(y, y_pred_proba)

print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"  Precision@5: {metrics['precision@5']:.4f}")
print(f"  Recall@5: {metrics['recall@5']:.4f}")
print(f"  MRR: {metrics['mrr']:.4f}")
print(f"  NDCG@5: {metrics['ndcg@5']:.4f}")

results.append({
    'model': 'Logistic Regression (baseline)',
    'features': len(baseline_features),
    'roc_auc_cv': cv_scores.mean(),
    'roc_auc_cv_std': cv_scores.std(),
    'precision@5': metrics['precision@5'],
    'recall@5': metrics['recall@5'],
    'mrr': metrics['mrr'],
    'ndcg@5': metrics['ndcg@5']
})


# --- ОСНОВНЫЕ МОДЕЛИ ---
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED),
    'KNN': KNeighborsClassifier(n_neighbors=10, weights='distance'),
}

trained_models = {}

for name, model in models.items():
    print(f"\n{name}:")
    
    # Кросс-валидация
    cv_scores = cross_val_score(model, X_full_scaled, y, cv=5, scoring='roc_auc')
    
    # Обучение на train
    model.fit(X_train, y_train)
    
    # Предсказания на test
    y_pred_test = model.predict_proba(X_test)[:, 1]
    metrics_test = ranking_metrics(y_test, y_pred_test)
    roc_test = roc_auc_score(y_test, y_pred_test)
    
    results.append({
        'model': name,
        'features': len(full_features),
        'roc_auc_cv': cv_scores.mean(),
        'roc_auc_cv_std': cv_scores.std(),
        'roc_auc_test': roc_test,
        'precision@5': metrics_test['precision@5'],
        'recall@5': metrics_test['recall@5'],
        'mrr': metrics_test['mrr'],
        'ndcg@5': metrics_test['ndcg@5']
    })
    trained_models[name] = model
    
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  Test ROC-AUC: {roc_test:.4f}")
    print(f"  Test Recall@5: {metrics_test['recall@5']:.4f}")


# --- АНСАМБЛИ ---
print("\n" + "-" * 60)
print("АНСАМБЛИ")
print("-" * 60)

# Voting
voting_clf = VotingClassifier(
    estimators=[
        ('rf', trained_models['Random Forest']),
        ('gb', trained_models['Gradient Boosting']),
        ('lr', trained_models['Logistic Regression'])
    ],
    voting='soft'
)

cv_scores = cross_val_score(voting_clf, X_full_scaled, y, cv=5, scoring='roc_auc')
voting_clf.fit(X_train, y_train)
y_pred_test = voting_clf.predict_proba(X_test)[:, 1]
metrics_test = ranking_metrics(y_test, y_pred_test)
roc_test = roc_auc_score(y_test, y_pred_test)

results.append({
    'model': 'Voting Ensemble',
    'features': len(full_features),
    'roc_auc_cv': cv_scores.mean(),
    'roc_auc_cv_std': cv_scores.std(),
    'roc_auc_test': roc_test,
    'precision@5': metrics_test['precision@5'],
    'recall@5': metrics_test['recall@5'],
    'mrr': metrics_test['mrr'],
    'ndcg@5': metrics_test['ndcg@5']
})

print(f"\nVoting Ensemble:")
print(f"  CV ROC-AUC: {cv_scores.mean():.4f}")
print(f"  Test ROC-AUC: {roc_test:.4f}")


# Stacking
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', trained_models['Random Forest']),
        ('gb', trained_models['Gradient Boosting']),
        ('lr', trained_models['Logistic Regression'])
    ],
    final_estimator=LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
    cv=5,
    passthrough=True
)

cv_scores = cross_val_score(stacking_clf, X_full_scaled, y, cv=5, scoring='roc_auc')
stacking_clf.fit(X_train, y_train)
y_pred_test = stacking_clf.predict_proba(X_test)[:, 1]
metrics_test = ranking_metrics(y_test, y_pred_test)
roc_test = roc_auc_score(y_test, y_pred_test)

results.append({
    'model': 'Stacking Ensemble',
    'features': len(full_features),
    'roc_auc_cv': cv_scores.mean(),
    'roc_auc_cv_std': cv_scores.std(),
    'roc_auc_test': roc_test,
    'precision@5': metrics_test['precision@5'],
    'recall@5': metrics_test['recall@5'],
    'mrr': metrics_test['mrr'],
    'ndcg@5': metrics_test['ndcg@5']
})

print(f"\nStacking Ensemble:")
print(f"  CV ROC-AUC: {cv_scores.mean():.4f}")
print(f"  Test ROC-AUC: {roc_test:.4f}")


# --- GRID SEARCH ---
print("\n" + "-" * 60)
print("GRID SEARCH (гиперпараметры)")
print("-" * 60)

# Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5]
}

print("\nRandom Forest...")
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1),
    param_grid_rf,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_rf.fit(X_train, y_train)

y_pred_test = grid_rf.best_estimator_.predict_proba(X_test)[:, 1]
metrics_test = ranking_metrics(y_test, y_pred_test)
roc_test = roc_auc_score(y_test, y_pred_test)

results.append({
    'model': 'Random Forest (tuned)',
    'features': len(full_features),
    'roc_auc_cv': grid_rf.best_score_,
    'roc_auc_cv_std': 0,
    'roc_auc_test': roc_test,
    'precision@5': metrics_test['precision@5'],
    'recall@5': metrics_test['recall@5'],
    'mrr': metrics_test['mrr'],
    'ndcg@5': metrics_test['ndcg@5']
})

print(f"  Лучшие параметры: {grid_rf.best_params_}")
print(f"  CV ROC-AUC: {grid_rf.best_score_:.4f}")
print(f"  Test ROC-AUC: {roc_test:.4f}")


# Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

print("\nGradient Boosting...")
grid_gb = GridSearchCV(
    GradientBoostingClassifier(random_state=RANDOM_SEED),
    param_grid_gb,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_gb.fit(X_train, y_train)

y_pred_test = grid_gb.best_estimator_.predict_proba(X_test)[:, 1]
metrics_test = ranking_metrics(y_test, y_pred_test)
roc_test = roc_auc_score(y_test, y_pred_test)

results.append({
    'model': 'Gradient Boosting (tuned)',
    'features': len(full_features),
    'roc_auc_cv': grid_gb.best_score_,
    'roc_auc_cv_std': 0,
    'roc_auc_test': roc_test,
    'precision@5': metrics_test['precision@5'],
    'recall@5': metrics_test['recall@5'],
    'mrr': metrics_test['mrr'],
    'ndcg@5': metrics_test['ndcg@5']
})

print(f"  Лучшие параметры: {grid_gb.best_params_}")
print(f"  CV ROC-AUC: {grid_gb.best_score_:.4f}")
print(f"  Test ROC-AUC: {roc_test:.4f}")


# ============================================================================
# PCA ЭКСПЕРИМЕНТЫ
# ============================================================================

print("\n[8/8] PCA (уменьшение размерности)...")

pca_results = []
n_features = len(full_features)

for n_comp in [2, 3, 4, None]:
    # Ограничиваем количество компонент
    n_comp_safe = min(n_comp, n_features) if n_comp else n_features
    
    if n_comp is None:
        X_transformed = X_full_scaled
        label = 'All features'
    else:
        pca_temp = PCA(n_components=n_comp_safe, random_state=RANDOM_SEED)
        X_transformed = pca_temp.fit_transform(X_full_scaled)
        label = f'PCA ({n_comp_safe})'

    # Разделение
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_transformed, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Обучение
    rf_temp = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=RANDOM_SEED,
        n_jobs=-1, class_weight='balanced'
    )
    rf_temp.fit(X_tr, y_tr)
    roc_auc_temp = roc_auc_score(y_te, rf_temp.predict_proba(X_te)[:, 1])

    pca_results.append({
        'method': label,
        'n_components': n_comp_safe,
        'roc_auc_test': roc_auc_temp
    })

    print(f"  {label}: Test ROC-AUC = {roc_auc_temp:.4f}")


# ============================================================================
# ИТОГОВАЯ ТАБЛИЦА
# ============================================================================

print("\n" + "=" * 80)
print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('roc_auc_cv', ascending=False).reset_index(drop=True)

display_cols = ['model', 'features', 'roc_auc_cv', 'roc_auc_test', 'recall@5', 'ndcg@5']
print(results_df[display_cols].to_string(index=False))


# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================

print("\n" + "=" * 80)
print("ВИЗУАЛИЗАЦИЯ")
print("=" * 80)

# Сравнение моделей
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# CV ROC-AUC
axes[0].barh(results_df['model'], results_df['roc_auc_cv'], 
             xerr=results_df['roc_auc_cv_std'] * 2, color='steelblue', capsize=3)
axes[0].set_xlabel('ROC-AUC (5-fold CV)')
axes[0].set_title('Сравнение по CV ROC-AUC')
axes[0].grid(axis='x', alpha=0.3)
axes[0].set_xlim(0.5, 1.0)
plt.setp(axes[0].get_yticklabels(), rotation=45, ha='right')

# Test ROC-AUC
axes[1].barh(results_df['model'], results_df['roc_auc_test'], color='coral')
axes[1].set_xlabel('ROC-AUC (Test)')
axes[1].set_title('Сравнение по Test ROC-AUC')
axes[1].grid(axis='x', alpha=0.3)
axes[1].set_xlim(0.5, 1.0)
plt.setp(axes[1].get_yticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('presentation/model_comparison.png', dpi=150, bbox_inches='tight')
print("  Сохранено: presentation/model_comparison.png")

# PCA результаты
plt.figure(figsize=(10, 6))
plt.plot(range(len(pca_results)), [r['roc_auc_test'] for r in pca_results], 
         marker='o', linewidth=2, markersize=8)
plt.xticks(range(len(pca_results)), [r['method'] for r in pca_results], rotation=45, ha='right')
plt.xlabel('Метод')
plt.ylabel('Test ROC-AUC')
plt.title('Влияние уменьшения размерности на качество')
plt.grid(alpha=0.3)
plt.ylim(0.4, 1.0)
plt.tight_layout()
plt.savefig('presentation/pca_results.png', dpi=150, bbox_inches='tight')
print("  Сохранено: presentation/pca_results.png")


# Важность признаков
rf_model = trained_models.get('Random Forest')
if rf_model and hasattr(rf_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': full_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'].head(10), importance_df['importance'].head(10), 
             color='steelblue')
    plt.xlabel('Важность признака')
    plt.title('Топ-10 важных признаков (Random Forest)')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('presentation/feature_importance.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: presentation/feature_importance.png")


# ============================================================================
# СОХРАНЕНИЕ
# ============================================================================

print("\n" + "=" * 80)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 80)

# Таблицы
results_df.to_csv('data/processed/model_comparison.csv', index=False)
print("  data/processed/model_comparison.csv")

pd.DataFrame(pca_results).to_csv('data/processed/pca_experiments.csv', index=False)
print("  data/processed/pca_experiments.csv")

# Модели
import joblib
best_model = grid_gb.best_estimator_ if 'grid_gb' in dir() else rf_model
joblib.dump(best_model, 'models/best_model.joblib')
print("  models/best_model.joblib")

joblib.dump(scaler, 'models/scaler.joblib')
print("  models/scaler.joblib")

joblib.dump(le_spec, 'models/label_encoder_spec.joblib')
print("  models/label_encoder_spec.joblib")

joblib.dump(le_pos, 'models/label_encoder_pos.joblib')
print("  models/label_encoder_pos.joblib")


# ============================================================================
# ЛУЧШАЯ МОДЕЛЬ
# ============================================================================

print("\n" + "=" * 80)
print("ЛУЧШАЯ МОДЕЛЬ (по CV ROC-AUC)")
print("=" * 80)

best = results_df.iloc[0]
print(f"  Модель: {best['model']}")
print(f"  CV ROC-AUC: {best['roc_auc_cv']:.4f} (+/- {best['roc_auc_cv_std'] * 2:.4f})")
print(f"  Test ROC-AUC: {best['roc_auc_test']:.4f}")
print(f"  Recall@5: {best['recall@5']:.4f}")
print(f"  NDCG@5: {best['ndcg@5']:.4f}")
