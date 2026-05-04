from pathlib import Path

import pandas as pd


def load_employees(filepath: str) -> pd.DataFrame:
    """Загрузить данные о сотрудниках."""
    return pd.read_csv(filepath)


def load_jira(filepath: str) -> pd.DataFrame:
    """Загрузить данные о Jira-задачах."""
    return pd.read_csv(filepath)


def load_gitlab(filepath: str) -> pd.DataFrame:
    """Загрузить данные о GitLab-коммитах."""
    return pd.read_csv(filepath)


def load_data(data_dir: str = "../data/raw") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загрузить все датасеты.

    Returns:
        employees, jira, gitlab - DataFrame с данными
    """
    data_path = Path(data_dir)

    employees = load_employees(data_path / "employees.csv")
    jira = load_jira(data_path / "jira_issues.csv")
    gitlab = load_gitlab(data_path / "gitlab_commits.csv")

    return employees, jira, gitlab
