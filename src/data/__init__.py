# Модуль обработки данных

from .cleaner import clean_data, handle_outliers, remove_duplicates, remove_missing
from .loader import load_data, load_employees, load_gitlab, load_jira

__all__ = [
    "load_data",
    "load_employees",
    "load_jira",
    "load_gitlab",
    "clean_data",
    "remove_missing",
    "remove_duplicates",
    "handle_outliers",
]
