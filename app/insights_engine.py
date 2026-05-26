def generate_ai_insights(df):

    insights = []

    # Total Revenue
    total_revenue = round(df['TotalPrice'].sum(), 2)

    insights.append(
        f"Total platform revenue generated is ${total_revenue:,.0f}."
    )

    # Top Country
    top_country = (
        df.groupby('Country')['TotalPrice']
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    insights.append(
        f"{top_country} generated the highest revenue."
    )

    # Top Product
    top_product = (
        df.groupby('Description')['Quantity']
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    insights.append(
        f"The most demanded product is '{top_product}'."
    )

    # Average Order Value
    avg_order = (
        df.groupby('InvoiceNo')['TotalPrice']
        .sum()
        .mean()
    )

    insights.append(
        f"Average order value is ${avg_order:.2f}."
    )

    return insights