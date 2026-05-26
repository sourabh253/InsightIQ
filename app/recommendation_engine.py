import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def generate_recommendations(df, product_name):

    product_matrix = df.pivot_table(
        index='CustomerID',
        columns='Description',
        values='Quantity',
        fill_value=0
    )

    similarity = cosine_similarity(product_matrix.T)

    similarity_df = pd.DataFrame(
        similarity,
        index=product_matrix.columns,
        columns=product_matrix.columns
    )

    recommendations = (
        similarity_df[product_name]
        .sort_values(ascending=False)[1:6]
    )

    return recommendations