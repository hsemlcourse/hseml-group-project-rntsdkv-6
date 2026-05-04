import pandas as pd


def build_developer_profiles(
    employees: pd.DataFrame,
    jira: pd.DataFrame,
    gitlab: pd.DataFrame,
    jira_train: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Построить профили разработчиков на основе их активности.

    Args:
        employees: DataFrame с сотрудниками
        jira: DataFrame с Jira-задачами (должен содержать issue_desc_clean)
        gitlab: DataFrame с GitLab-коммитами (должен содержать commit_title_clean)
        jira_train: DataFrame с Jira-задачами для train (если None, используется jira).

    Returns:
        DataFrame с профилями разработчиков
    """
    jira_for_profile = jira_train if jira_train is not None else jira

    # Агрегация Jira по исполнителям
    jira_profile = (
        jira_for_profile.groupby("assignee_login_nm")
        .agg({
            "issue_rk": "count",
            "issue_desc_clean": lambda x: " ".join(x),
            "project_full_nm": lambda x: x.nunique()
        })
        .reset_index()
    )
    jira_profile.columns = ["login", "jira_issues_count", "jira_text", "projects_count"]

    # Агрегация GitLab по авторам (с дополнительными признаками)
    gitlab_profile = (
        gitlab.groupby("author_login")
        .agg({
            "commit_hash": "count",
            "commit_title_clean": lambda x: " ".join(x),
            "repo": lambda x: x.nunique()
        })
        .reset_index()
    )
    gitlab_profile.columns = ["login", "commits_count", "gitlab_text", "repos_count"]

    # Объединение с сотрудниками
    profiles = employees.merge(jira_profile, on="login", how="left")
    profiles = profiles.merge(gitlab_profile, on="login", how="left")

    # Заполнение пропусков
    profiles["jira_issues_count"] = profiles["jira_issues_count"].fillna(0).astype(int)
    profiles["commits_count"] = profiles["commits_count"].fillna(0).astype(int)
    profiles["projects_count"] = profiles["projects_count"].fillna(0).astype(int)
    profiles["repos_count"] = profiles["repos_count"].fillna(0).astype(int)
    profiles["jira_text"] = profiles["jira_text"].fillna("")
    profiles["gitlab_text"] = profiles["gitlab_text"].fillna("")

    # Объединённый текстовый профиль
    profiles["developer_profile_text"] = (
        profiles["jira_text"] + " " + profiles["gitlab_text"]
    ).str.strip()

    # Количество слов в профиле
    profiles["profile_words_count"] = profiles["developer_profile_text"].apply(
        lambda x: len(x.split())
    )

    return profiles


def add_advanced_features(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить продвинутые признаки для ранжирования.

    Args:
        profiles: DataFrame с профилями разработчиков

    Returns:
        DataFrame с дополнительными признаками
    """
    df = profiles.copy()

    # 1. Интенсивность активности (задачи на единицу опыта)
    df["tasks_per_experience"] = df["jira_issues_count"] / (df["work_experience_day_cnt"] / 365)

    # 2. Интенсивность коммитов
    df["commits_per_experience"] = df["commits_count"] / (df["work_experience_day_cnt"] / 365)

    # 3. Универсальность (проекты + репозитории)
    df["versatility_score"] = df["projects_count"] + df["repos_count"]

    # 4. Плотность текста (слов на задачу)
    df["text_density"] = df["profile_words_count"] / df["jira_issues_count"].replace(0, 1)

    # 5. Бинарные признаки для позиций
    df["is_senior"] = df["position_nm"].str.contains(
        "Ведущий|Старший|Главный|Руководитель", case=False, na=False
    ).astype(int)

    df["is_junior"] = df["position_nm"].str.contains(
        "Младший|Стажёр|Junior", case=False, na=False
    ).astype(int)

    # 6. Логарифмические признаки
    import numpy as np
    df["log_experience"] = np.log1p(df["work_experience_day_cnt"])
    df["log_jira_count"] = np.log1p(df["jira_issues_count"])
    df["log_commits_count"] = np.log1p(df["commits_count"])

    return df
