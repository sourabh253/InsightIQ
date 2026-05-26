def dataset_quality_check(df):

    total_cells = df.shape[0] * df.shape[1]

    missing_values = df.isnull().sum().sum()

    duplicate_rows = df.duplicated().sum()

    missing_percentage = (
        missing_values / total_cells
    ) * 100

    quality_score = max(
        0,
        100 - missing_percentage - duplicate_rows
    )

    return {
        "Missing Values": missing_values,
        "Duplicate Rows": duplicate_rows,
        "Quality Score": round(quality_score, 2)
    }