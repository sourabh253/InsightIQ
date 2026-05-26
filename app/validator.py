import pandas as pd

REQUIRED_COLUMNS = [
    'InvoiceNo',
    'StockCode',
    'Description',
    'Quantity',
    'InvoiceDate',
    'UnitPrice',
    'CustomerID',
    'Country'
]

def validate_dataset(df):

    missing_columns = []

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            missing_columns.append(column)

    if len(missing_columns) > 0:
        return False, missing_columns

    return True, []