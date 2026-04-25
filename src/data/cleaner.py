import re

import pandas as pd


def remove_missing(
    employees: pd.DataFrame, jira: pd.DataFrame, gitlab: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Удалить строки с пропусками в ключевых полях.

    Args:
        employees: DataFrame с сотрудниками
        jira: DataFrame с Jira-задачами
        gitlab: DataFrame с GitLab-коммитами

    Returns:
        Очищенные DataFrames
    """
    employees = employees.dropna(subset=["login"])
    jira = jira.dropna(subset=["issue_desc"])
    gitlab = gitlab.dropna(subset=["author_login"])

    return employees, jira, gitlab


def remove_duplicates(
    employees: pd.DataFrame, jira: pd.DataFrame, gitlab: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Удалить дубликаты.

    Args:
        employees: DataFrame с сотрудниками
        jira: DataFrame с Jira-задачами
        gitlab: DataFrame с GitLab-коммитами

    Returns:
        DataFrames без дубликатов
    """
    jira = jira.drop_duplicates(subset=["assignee_login_nm", "issue_rk"])
    gitlab = gitlab.drop_duplicates(subset=["commit_hash"])

    return employees, jira, gitlab


def unify_logins(
    employees: pd.DataFrame, jira: pd.DataFrame, gitlab: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Унифицировать форматы login (lowercase + strip)."""
    employees["login"] = employees["login"].str.lower().str.strip()
    jira["assignee_login_nm"] = jira["assignee_login_nm"].str.lower().str.strip()
    gitlab["author_login"] = gitlab["author_login"].str.lower().str.strip()

    return employees, jira, gitlab


def clean_text(text: str) -> str:
    """Очистить текст: lowercase, удаление спецсимволов, нормализация пробелов."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_texts(jira: pd.DataFrame, gitlab: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Очистить текстовые поля в датасетах."""
    jira = jira.copy()
    gitlab = gitlab.copy()

    jira["issue_desc_clean"] = jira["issue_desc"].apply(clean_text)
    gitlab["commit_title_clean"] = gitlab["commit_title"].apply(clean_text)

    return jira, gitlab


def handle_outliers(
    employees: pd.DataFrame, gitlab: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Обработать выбросы.

    Args:
        employees: DataFrame с сотрудниками
        gitlab: DataFrame с GitLab-коммитами

    Returns:
        Обработанные DataFrames
    """
    employees = employees.copy()
    gitlab = gitlab.copy()

    # Удаляем опыт < 1 дня (невозможен)
    employees = employees[employees["work_experience_day_cnt"] >= 1]

    # Удаляем мусорные заголовки коммитов
    trash_titles = ["wip", "fix", "upd", "test", "tmp", "merge", "revert"]
    gitlab = gitlab[~gitlab["commit_title_clean"].str.lower().isin(trash_titles)]

    return employees, gitlab


def clean_data(
    employees: pd.DataFrame, jira: pd.DataFrame, gitlab: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Полный пайплайн очистки данных."""
    # 1. Удаление пропусков
    employees, jira, gitlab = remove_missing(employees, jira, gitlab)

    # 2. Удаление дубликатов
    employees, jira, gitlab = remove_duplicates(employees, jira, gitlab)

    # 3. Унификация login
    employees, jira, gitlab = unify_logins(employees, jira, gitlab)

    # 4. Очистка текста
    jira, gitlab = clean_texts(jira, gitlab)

    # 5. Обработка выбросов
    employees, gitlab = handle_outliers(employees, gitlab)

    return employees, jira, gitlab
