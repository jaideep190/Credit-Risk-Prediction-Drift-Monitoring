TARGET_COL = "SeriousDlqin2yrs"

RAW_FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

DELINQUENCY_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

SENTINEL_CODES = [96, 98]


def apply_preprocessing(df, artifacts: dict):
    df = df.copy()

    df["MonthlyIncome_was_missing"] = df["MonthlyIncome"].isnull().astype(int)
    df["NumberOfDependents_was_missing"] = df["NumberOfDependents"].isnull().astype(int)

    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(artifacts["MonthlyIncome_median"])
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(artifacts["NumberOfDependents_median"])

    df["MonthlyIncome"] = df["MonthlyIncome"].clip(upper=artifacts["monthly_income_cap"])
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(
        upper=artifacts["revolving_utilization_cap"]
    )
    df["DebtRatio"] = df["DebtRatio"].clip(upper=artifacts["debt_ratio_cap"])

    for col in DELINQUENCY_COLS:
        df[col] = df[col].clip(upper=artifacts["delinquency_caps"][col])

    return df