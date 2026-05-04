import pandas as pd


def build_developer_profiles(
    employees: pd.DataFrame, jira: pd.DataFrame, gitlab: pd.DataFrame
) -> pd.DataFrame:
    """
    Построить профили разработчиков на основе их активности.

    Args:
        employees: DataFrame с сотрудниками
        jira: DataFrame с Jira-задачами (должен содержать issue_desc_clean)
        gitlab: DataFrame с GitLab-коммитами (должен содержать commit_title_clean)

    Returns:
        DataFrame с профилями разработчиков
    """
    # Агрегация Jira по исполнителям
    jira_profile = (
        jira.groupby("assignee_login_nm")
        .agg({"issue_rk": "count", "issue_desc_clean": lambda x: " ".join(x)})
        .reset_index()
    )
    jira_profile.columns = ["login", "jira_issues_count", "jira_text"]

    # Агрегация GitLab по авторам
    gitlab_profile = (
        gitlab.groupby("author_login")
        .agg({"commit_hash": "count", "commit_title_clean": lambda x: " ".join(x)})
        .reset_index()
    )
    gitlab_profile.columns = ["login", "commits_count", "gitlab_text"]

    # Объединение с сотрудниками
    profiles = employees.merge(jira_profile, on="login", how="left")
    profiles = profiles.merge(gitlab_profile, on="login", how="left")

    # Заполнение пропусков
    profiles["jira_issues_count"] = profiles["jira_issues_count"].fillna(0).astype(int)
    profiles["commits_count"] = profiles["commits_count"].fillna(0).astype(int)
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
