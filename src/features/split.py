import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_val_test_split(
    jira: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разделить данные на train/val/test по issue_rk.

    Args:
        jira: DataFrame с Jira-задачами
        test_size: Доля test выборки
        val_size: Доля validation выборки
        random_state: Seed для воспроизводимости

    Returns:
        jira_train, jira_val, jira_test
    """
    unique_issues = jira["issue_rk"].unique()

    # Split 60/20/20
    train_issues, temp_issues = train_test_split(
        unique_issues, test_size=test_size + val_size, random_state=random_state
    )
    val_issues, test_issues = train_test_split(
        temp_issues, test_size=val_size / (test_size + val_size), random_state=random_state
    )

    jira_train = jira[jira["issue_rk"].isin(train_issues)].copy()
    jira_val = jira[jira["issue_rk"].isin(val_issues)].copy()
    jira_test = jira[jira["issue_rk"].isin(test_issues)].copy()

    return jira_train, jira_val, jira_test
