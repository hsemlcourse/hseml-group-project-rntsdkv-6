RANDOM_SEED = 42  # Фиксированный seed для воспроизводимости

# Пути к данным
DATA_RAW = "../data/raw"
DATA_PROCESSED = "../data/processed"

# Файлы
EMPLOYEES_FILE = "employees.csv"
JIRA_ISSUES_FILE = "jira_issues.csv"
GITLAB_COMMITS_FILE = "gitlab_commits.csv"

# Колонки
EMPLOYEE_LOGIN_COL = "login"
JIRA_ASSIGNEE_COL = "assignee_login_nm"
JIRA_ISSUE_RK_COL = "issue_rk"
JIRA_DESC_COL = "issue_desc"
GITLAB_AUTHOR_COL = "author_login"
GITLAB_COMMIT_HASH_COL = "commit_hash"
GITLAB_TITLE_COL = "commit_title"
EXPERIENCE_COL = "work_experience_day_cnt"
POSITION_COL = "position_nm"
SPECIALIZATION_COL = "specialization_nm"
