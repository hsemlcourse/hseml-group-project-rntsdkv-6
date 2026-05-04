"""
Тесты для модулей предобработки данных из notebooks/preprocessing.ipynb
"""

import pandas as pd
from src.data.cleaner import (
    clean_data,
    clean_text,
    clean_texts,
    handle_outliers,
    remove_duplicates,
    remove_missing,
    unify_logins,
)
from src.data.loader import load_data, load_employees, load_gitlab, load_jira
from src.features.profiles import build_developer_profiles
from src.features.split import create_train_val_test_split

# ==================== Тесты для loader.py ====================


class TestLoaders:
    """Тесты функций загрузки данных."""

    def test_load_employees_structure(self, tmp_path):
        """Проверка структуры загруженных данных сотрудников."""
        csv_file = tmp_path / "employees.csv"
        csv_file.write_text(
            "masterid,login,last_nm,first_nm,position_nm\n"
            "1117,a.agashkov,Агашков,Андрей,Ведущий разработчик\n"
            "1147,v.ageev,Агеев,Владимир,Руководитель\n"
        )

        df = load_employees(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "login" in df.columns
        assert "position_nm" in df.columns

    def test_load_jira_structure(self, tmp_path):
        """Проверка структуры загруженных данных Jira."""
        csv_file = tmp_path / "jira_issues.csv"
        csv_file.write_text(
            "mdm_employee_rk,assignee_login_nm,issue_rk,issue_desc,project_full_nm,project_short_nm\n"
            "587302,v.uspenskiy,153,Описание задачи 1,Portal,PRT\n"
            "587303,v.uspenskiy,208,Описание задачи 2,Portal,PRT\n"
        )

        df = load_jira(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "issue_rk" in df.columns
        assert "assignee_login_nm" in df.columns

    def test_load_gitlab_structure(self, tmp_path):
        """Проверка структуры загруженных данных GitLab."""
        csv_file = tmp_path / "gitlab_commits.csv"
        csv_file.write_text(
            "tenant,repo,commit_hash,commit_title,author_login,date,jira_task\n"
            "cpd,Akira Broadcaster,abc123,INTERCOM-2469 add test,semen.panevin,2025-10-14 10:52:01+00:00,INTERCOM-2469\n"
            "cpd,Akira Gateway,def456,INTERCOM-2247 fix bug,m.a.kanatnikov,2025-08-08 07:48:13+00:00,INTERCOM-2247\n"
        )

        df = load_gitlab(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "commit_hash" in df.columns
        assert "author_login" in df.columns

    def test_load_data_returns_three_dataframes(self, tmp_path):
        """Проверка что load_data возвращает три DataFrame."""
        # Создаем тестовые файлы
        emp_file = tmp_path / "employees.csv"
        emp_file.write_text(
            "masterid,login,last_nm,first_nm,position_nm\n1,a,Фамилия,Имя,Должность\n"
        )

        jira_file = tmp_path / "jira_issues.csv"
        jira_file.write_text(
            "mdm_employee_rk,assignee_login_nm,issue_rk,issue_desc,project_full_nm,project_short_nm\n1,a,1,desc,Project,PRJ\n"
        )

        git_file = tmp_path / "gitlab_commits.csv"
        git_file.write_text(
            "tenant,repo,commit_hash,commit_title,author_login,date,jira_task\ncp,repo,hash,title,a,2025-01-01,task\n"
        )

        employees, jira, gitlab = load_data(str(tmp_path))

        assert isinstance(employees, pd.DataFrame)
        assert isinstance(jira, pd.DataFrame)
        assert isinstance(gitlab, pd.DataFrame)


# ==================== Тесты для cleaner.py ====================


class TestRemoveMissing:
    """Тесты функции удаления пропусков."""

    def test_remove_missing_employees(self):
        """Проверка удаления строк с пропусками в login."""
        employees = pd.DataFrame(
            {"login": ["user1", None, "user3"], "last_nm": ["Фамилия1", "Фамилия2", "Фамилия3"]}
        )
        jira = pd.DataFrame({"issue_desc": ["desc1", "desc2", "desc3"]})
        gitlab = pd.DataFrame({"author_login": ["a1", "a2", "a3"]})

        emp_clean, _, _ = remove_missing(employees, jira, gitlab)

        assert len(emp_clean) == 2
        assert None not in emp_clean["login"].values

    def test_remove_missing_jira(self):
        """Проверка удаления строк с пропусками в issue_desc."""
        employees = pd.DataFrame({"login": ["u1"]})
        jira = pd.DataFrame({"issue_desc": ["desc1", None, "desc3"], "issue_rk": [1, 2, 3]})
        gitlab = pd.DataFrame({"author_login": ["a1"]})

        _, jira_clean, _ = remove_missing(employees, jira, gitlab)

        assert len(jira_clean) == 2

    def test_remove_missing_gitlab(self):
        """Проверка удаления строк с пропусками в author_login."""
        employees = pd.DataFrame({"login": ["u1"]})
        jira = pd.DataFrame({"issue_desc": ["desc1"]})
        gitlab = pd.DataFrame({"author_login": ["a1", None, "a3"]})

        _, _, git_clean = remove_missing(employees, jira, gitlab)

        assert len(git_clean) == 2


class TestRemoveDuplicates:
    """Тесты функции удаления дубликатов."""

    def test_remove_duplicates_jira(self):
        """Проверка удаления дубликатов в Jira."""
        employees = pd.DataFrame({"login": ["u1"]})
        jira = pd.DataFrame(
            {
                "assignee_login_nm": ["u1", "u1", "u1"],
                "issue_rk": [1, 1, 2],  # Дубликат по (assignee_login_nm, issue_rk)
            }
        )
        gitlab = pd.DataFrame({"commit_hash": ["h1"]})

        _, jira_clean, _ = remove_duplicates(employees, jira, gitlab)

        assert len(jira_clean) == 2

    def test_remove_duplicates_gitlab(self):
        """Проверка удаления дубликатов в GitLab."""
        employees = pd.DataFrame({"login": ["u1"]})
        jira = pd.DataFrame({"assignee_login_nm": ["u1"], "issue_rk": [1]})
        gitlab = pd.DataFrame(
            {
                "commit_hash": ["h1", "h1", "h2"]  # Дубликат по commit_hash
            }
        )

        _, _, git_clean = remove_duplicates(employees, jira, gitlab)

        assert len(git_clean) == 2


class TestUnifyLogins:
    """Тесты функции унификации логинов."""

    def test_unify_logins_lowercase(self):
        """Проверка приведения к нижнему регистру."""
        employees = pd.DataFrame({"login": ["User1", "USER2", "user3"]})
        jira = pd.DataFrame({"assignee_login_nm": ["User1"]})
        gitlab = pd.DataFrame({"author_login": ["User1"]})

        emp_clean, jira_clean, git_clean = unify_logins(employees, jira, gitlab)

        assert emp_clean["login"].tolist() == ["user1", "user2", "user3"]
        assert jira_clean["assignee_login_nm"].iloc[0] == "user1"
        assert git_clean["author_login"].iloc[0] == "user1"

    def test_unify_logins_strip(self):
        """Проверка удаления пробелов."""
        employees = pd.DataFrame({"login": [" user1 ", "user2  ", "  user3"]})
        jira = pd.DataFrame({"assignee_login_nm": [" user1 "]})
        gitlab = pd.DataFrame({"author_login": [" user1 "]})

        emp_clean, jira_clean, git_clean = unify_logins(employees, jira, gitlab)

        assert emp_clean["login"].tolist() == ["user1", "user2", "user3"]


class TestCleanText:
    """Тесты функции очистки текста."""

    def test_clean_text_lowercase(self):
        """Проверка приведения к нижнему регистру."""
        assert clean_text("Hello World") == "hello world"

    def test_clean_text_remove_special_chars(self):
        """Проверка удаления спецсимволов."""
        assert clean_text("Hello! @World# $123") == "hello world 123"

    def test_clean_text_normalize_spaces(self):
        """Проверка нормализации пробелов."""
        assert clean_text("Hello    World") == "hello world"

    def test_clean_text_nan(self):
        """Проверка обработки NaN."""
        assert clean_text(float("nan")) == ""

    def test_clean_text_russian(self):
        """Проверка обработки русского текста."""
        assert clean_text("Привет Мир!") == "привет мир"


class TestCleanTexts:
    """Тесты функции очистки текстовых полей."""

    def test_clean_texts_creates_clean_columns(self):
        """Проверка создания очищенных колонок."""
        jira = pd.DataFrame({"issue_desc": ["Task 1!", "Task 2@"]})
        gitlab = pd.DataFrame({"commit_title": ["Fix bug#", "Add feature$"]})

        jira_clean, git_clean = clean_texts(jira, gitlab)

        assert "issue_desc_clean" in jira_clean.columns
        assert "commit_title_clean" in git_clean.columns
        assert jira_clean["issue_desc_clean"].iloc[0] == "task 1"
        assert git_clean["commit_title_clean"].iloc[0] == "fix bug"


class TestHandleOutliers:
    """Тесты функции обработки выбросов."""

    def test_handle_outliers_remove_invalid_experience(self):
        """Проверка удаления недопустимого опыта."""
        employees = pd.DataFrame(
            {
                "login": ["u1", "u2", "u3"],
                "work_experience_day_cnt": [0, 100, 500],  # 0 - недопустимо
            }
        )
        gitlab = pd.DataFrame({"commit_title_clean": ["fix", "add feature", "wip"]})

        emp_clean, _ = handle_outliers(employees, gitlab)

        assert len(emp_clean) == 2
        assert 0 not in emp_clean["work_experience_day_cnt"].values

    def test_handle_outliers_remove_trash_titles(self):
        """Проверка удаления мусорных заголовков коммитов."""
        employees = pd.DataFrame({"login": ["u1"], "work_experience_day_cnt": [100]})
        gitlab = pd.DataFrame(
            {
                "commit_title_clean": [
                    "wip",
                    "fix",
                    "upd",
                    "test",
                    "tmp",
                    "merge",
                    "revert",
                    "normal commit",
                ]
            }
        )

        _, git_clean = handle_outliers(employees, gitlab)

        assert len(git_clean) == 1
        assert git_clean["commit_title_clean"].iloc[0] == "normal commit"


class TestCleanData:
    """Тесты полного пайплайна очистки."""

    def test_clean_data_full_pipeline(self):
        """Проверка полного пайплайна очистки."""
        employees = pd.DataFrame(
            {"login": [" User1 ", None, "User2"], "work_experience_day_cnt": [0, 100, 200]}
        )
        jira = pd.DataFrame(
            {
                "assignee_login_nm": [" User1 ", "user1", "user2"],
                "issue_rk": [1, 1, 2],
                "issue_desc": ["Task!", None, "Task 2"],
            }
        )
        gitlab = pd.DataFrame(
            {
                "author_login": [" User1 ", "user2"],
                "commit_hash": ["h1", "h1"],
                "commit_title": ["WIP fix!", "Add feature@"],
            }
        )

        emp_clean, jira_clean, git_clean = clean_data(employees, jira, gitlab)

        # Проверка что данные очищены
        assert len(emp_clean) > 0
        assert len(jira_clean) > 0
        assert len(git_clean) > 0

        # Проверка что login унифицированы
        assert all(emp_clean["login"] == emp_clean["login"].str.lower())

        # Проверка что созданы очищенные текстовые поля
        assert "issue_desc_clean" in jira_clean.columns
        assert "commit_title_clean" in git_clean.columns


# ==================== Тесты для profiles.py ====================


class TestBuildDeveloperProfiles:
    """Тесты функции построения профилей разработчиков."""

    def test_build_developer_profiles_structure(self):
        """Проверка структуры выходных данных."""
        employees = pd.DataFrame(
            {"login": ["user1", "user2"], "position_nm": ["dev", "senior dev"]}
        )
        jira = pd.DataFrame(
            {
                "assignee_login_nm": ["user1", "user1", "user2"],
                "issue_rk": [1, 2, 3],
                "issue_desc_clean": ["task 1", "task 2", "task 3"],
            }
        )
        gitlab = pd.DataFrame(
            {
                "author_login": ["user1", "user2"],
                "commit_hash": ["h1", "h2"],
                "commit_title_clean": ["fix bug", "add feature"],
            }
        )

        profiles = build_developer_profiles(employees, jira, gitlab)

        assert isinstance(profiles, pd.DataFrame)
        assert len(profiles) == 2
        assert "jira_issues_count" in profiles.columns
        assert "commits_count" in profiles.columns
        assert "developer_profile_text" in profiles.columns
        assert "profile_words_count" in profiles.columns

    def test_build_developer_profiles_counts(self):
        """Проверка правильности подсчета активности."""
        employees = pd.DataFrame({"login": ["user1", "user2"]})
        jira = pd.DataFrame(
            {
                "assignee_login_nm": ["user1", "user1", "user1", "user2"],
                "issue_rk": [1, 2, 3, 4],
                "issue_desc_clean": ["t1", "t2", "t3", "t4"],
            }
        )
        gitlab = pd.DataFrame(
            {
                "author_login": ["user1", "user2", "user2"],
                "commit_hash": ["h1", "h2", "h3"],
                "commit_title_clean": ["c1", "c2", "c3"],
            }
        )

        profiles = build_developer_profiles(employees, jira, gitlab)

        user1 = profiles[profiles["login"] == "user1"].iloc[0]
        user2 = profiles[profiles["login"] == "user2"].iloc[0]

        assert user1["jira_issues_count"] == 3
        assert user1["commits_count"] == 1
        assert user2["jira_issues_count"] == 1
        assert user2["commits_count"] == 2

    def test_build_developer_profiles_missing_data(self):
        """Проверка обработки отсутствующих данных."""
        employees = pd.DataFrame({"login": ["user1", "user2"]})
        jira = pd.DataFrame(
            {"assignee_login_nm": ["user1"], "issue_rk": [1], "issue_desc_clean": ["task"]}
        )
        gitlab = pd.DataFrame(
            {"author_login": ["user2"], "commit_hash": ["h1"], "commit_title_clean": ["commit"]}
        )

        profiles = build_developer_profiles(employees, jira, gitlab)

        user1 = profiles[profiles["login"] == "user1"].iloc[0]
        user2 = profiles[profiles["login"] == "user2"].iloc[0]

        # Проверка заполнения пропусков нулями
        assert user1["jira_issues_count"] == 1
        assert user1["commits_count"] == 0
        assert user2["jira_issues_count"] == 0
        assert user2["commits_count"] == 1

    def test_build_developer_profiles_text_concatenation(self):
        """Проверка объединения текстов."""
        employees = pd.DataFrame({"login": ["user1"]})
        jira = pd.DataFrame(
            {
                "assignee_login_nm": ["user1", "user1"],
                "issue_rk": [1, 2],
                "issue_desc_clean": ["task one", "task two"],
            }
        )
        gitlab = pd.DataFrame(
            {
                "author_login": ["user1", "user1"],
                "commit_hash": ["h1", "h2"],
                "commit_title_clean": ["commit one", "commit two"],
            }
        )

        profiles = build_developer_profiles(employees, jira, gitlab)

        profile_text = profiles["developer_profile_text"].iloc[0]
        assert "task one" in profile_text
        assert "task two" in profile_text
        assert "commit one" in profile_text
        assert "commit two" in profile_text


# ==================== Тесты для split.py ====================


class TestCreateTrainValTestSplit:
    """Тесты функции разделения на train/val/test."""

    def test_split_returns_three_dataframes(self):
        """Проверка что функция возвращает три DataFrame."""
        jira = pd.DataFrame({"issue_rk": list(range(100)), "assignee_login_nm": ["user"] * 100})

        train, val, test = create_train_val_test_split(jira)

        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_split_sizes(self):
        """Проверка размеров выборок (60/20/20)."""
        jira = pd.DataFrame({"issue_rk": list(range(1000)), "assignee_login_nm": ["user"] * 1000})

        train, val, test = create_train_val_test_split(
            jira, test_size=0.2, val_size=0.2, random_state=42
        )

        # Проверка приблизительных пропорций
        total = len(jira)
        assert 0.55 <= len(train) / total <= 0.65
        assert 0.15 <= len(val) / total <= 0.25
        assert 0.15 <= len(test) / total <= 0.25

    def test_split_no_overlap(self):
        """Проверка отсутствия пересечений между выборками."""
        jira = pd.DataFrame({"issue_rk": list(range(100)), "assignee_login_nm": ["user"] * 100})

        train, val, test = create_train_val_test_split(jira, random_state=42)

        train_issues = set(train["issue_rk"])
        val_issues = set(val["issue_rk"])
        test_issues = set(test["issue_rk"])

        # Проверка отсутствия пересечений
        assert len(train_issues & val_issues) == 0
        assert len(train_issues & test_issues) == 0
        assert len(val_issues & test_issues) == 0

        # Проверка что все issue распределены
        assert len(train_issues | val_issues | test_issues) == len(jira)

    def test_split_reproducимость(self):
        """Проверка воспроизводимости разделения."""
        jira = pd.DataFrame({"issue_rk": list(range(100)), "assignee_login_nm": ["user"] * 100})

        train1, val1, test1 = create_train_val_test_split(jira, random_state=42)
        train2, val2, test2 = create_train_val_test_split(jira, random_state=42)

        assert list(train1["issue_rk"]) == list(train2["issue_rk"])
        assert list(val1["issue_rk"]) == list(val2["issue_rk"])
        assert list(test1["issue_rk"]) == list(test2["issue_rk"])

    def test_split_custom_sizes(self):
        """Проверка разделения с кастомными размерами."""
        jira = pd.DataFrame({"issue_rk": list(range(1000)), "assignee_login_nm": ["user"] * 1000})

        train, val, test = create_train_val_test_split(
            jira, test_size=0.3, val_size=0.2, random_state=42
        )

        total = len(jira)
        # Проверка что разделение работает с кастомными параметрами
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == total


# ==================== Интеграционные тесты ====================


class TestPreprocessingPipeline:
    """Интеграционные тесты полного пайплайна предобработки."""

    def test_full_pipeline(self, tmp_path):
        """Проверка полного пайплайна от загрузки до разделения."""
        # Создаем тестовые данные
        emp_file = tmp_path / "employees.csv"
        emp_file.write_text(
            "masterid,login,last_nm,first_nm,position_nm,work_experience_day_cnt\n"
            "1,user1,Фамилия1,Имя1,Developer,100\n"
            "2,user2,Фамилия2,Имя2,Senior Developer,200\n"
            "3,user3,Фамилия3,Имя3,Lead Developer,300\n"
        )

        jira_file = tmp_path / "jira_issues.csv"
        jira_file.write_text(
            "mdm_employee_rk,assignee_login_nm,issue_rk,issue_desc,project_full_nm,project_short_nm\n"
            "1,user1,1,Разработать модуль А,Project1,P1\n"
            "2,user2,2,Разработать модуль Б,Project1,P1\n"
            "1,user1,3,Исправить баг,Project2,P2\n"
            "3,user3,4,Провести код ревью,Project2,P2\n"
        )

        git_file = tmp_path / "gitlab_commits.csv"
        git_file.write_text(
            "tenant,repo,commit_hash,commit_title,author_login,date,jira_task\n"
            "t1,r1,h1,Fix bug in module A,user1,2025-01-01,1\n"
            "t1,r1,h2,Add feature B,user2,2025-01-02,2\n"
            "t1,r1,h3,Code review fixes,user3,2025-01-03,4\n"
        )

        # Загрузка
        employees, jira, gitlab = load_data(str(tmp_path))

        # Очистка
        employees, jira, gitlab = clean_data(employees, jira, gitlab)

        # Построение профилей
        profiles = build_developer_profiles(employees, jira, gitlab)

        # Разделение
        train, val, test = create_train_val_test_split(jira, random_state=42)

        # Проверки
        assert len(profiles) > 0
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert "developer_profile_text" in profiles.columns
        assert "profile_words_count" in profiles.columns
