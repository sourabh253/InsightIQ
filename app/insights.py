def generate_basic_insights(df):

    insights = {}

    insights['Total Revenue'] = round(
        df['TotalPrice'].sum(), 2
    )

    insights['Total Orders'] = df['InvoiceNo'].nunique()

    insights['Total Customers'] = df['CustomerID'].nunique()

    insights['Top Product'] = (
        df.groupby('Description')['Quantity']
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    return insights