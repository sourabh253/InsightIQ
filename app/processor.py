import pandas as pd

def clean_data(df):

    # Remove null values
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # Convert InvoiceDate
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Create TotalPrice
    df['TotalPrice'] = (
        df['Quantity'] * df['UnitPrice']
    )

    return df