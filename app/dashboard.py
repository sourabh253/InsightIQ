import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="InsightIQ",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("InsightIQ — AI-Powered E-Commerce Intelligence Platform")

st.markdown("""
InsightIQ is an advanced AI-powered analytics platform that transforms raw e-commerce datasets into intelligent business insights using Machine Learning, Forecasting, Recommendation Systems, and Decision Analytics.
""")

# =========================================================
# REQUIRED COLUMNS
# =========================================================

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

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("InsightIQ")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "AI Insights",
        "Dataset Quality",
        "Customer Segmentation",
        "Recommendation System",
        "Sales Forecasting",
        "Business Recommendations",
        "Anomaly Detection"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### CSV Requirements

| Column | Data Type |
|---|---|
| InvoiceNo | String |
| StockCode | String |
| Description | String |
| Quantity | Integer |
| InvoiceDate | DateTime |
| UnitPrice | Float |
| CustomerID | Numeric |
| Country | String |

Example Date Format:
12/1/2010 8:26
""")

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Your E-Commerce CSV File",
    type=["csv"]
)

# =========================================================
# MAIN LOGIC
# =========================================================

if uploaded_file is not None:

    try:

        # =====================================================
        # LOAD DATA
        # =====================================================

        df = pd.read_csv(
            uploaded_file,
            encoding='latin1'
        )

        # =====================================================
        # COLUMN VALIDATION
        # =====================================================

        missing_columns = []

        for column in REQUIRED_COLUMNS:

            if column not in df.columns:
                missing_columns.append(column)

        if len(missing_columns) > 0:

            st.error(
                f"Missing Required Columns: {missing_columns}"
            )

            st.stop()

        st.success("Dataset Uploaded Successfully")

        # =====================================================
        # DATA CLEANING
        # =====================================================

        df = df.dropna()

        df = df.drop_duplicates()

        df['InvoiceDate'] = pd.to_datetime(
            df['InvoiceDate']
        )

        df['TotalPrice'] = (
            df['Quantity'] * df['UnitPrice']
        )

        # =====================================================
        # OVERVIEW PAGE
        # =====================================================

        if page == "Overview":

            st.header("Business Overview Dashboard")

            total_sales = df['TotalPrice'].sum()

            total_orders = df['InvoiceNo'].nunique()

            total_customers = df['CustomerID'].nunique()

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Revenue",
                f"${total_sales:,.0f}"
            )

            col2.metric(
                "Orders",
                total_orders
            )

            col3.metric(
                "Customers",
                int(total_customers)
            )

            st.divider()

            # =================================================
            # SALES TREND
            # =================================================

            st.subheader("Sales Trend Analysis")

            sales_trend = (
                df.groupby(
                    df['InvoiceDate'].dt.date
                )['TotalPrice']
                .sum()
                .reset_index()
            )

            fig1 = px.line(
                sales_trend,
                x='InvoiceDate',
                y='TotalPrice',
                title='Daily Sales Trend'
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )

            st.divider()

            # =================================================
            # TOP PRODUCTS
            # =================================================

            st.subheader("Top Selling Products")

            top_products = (
                df.groupby('Description')['Quantity']
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )

            fig2 = px.bar(
                top_products,
                x='Description',
                y='Quantity',
                title='Top 10 Selling Products'
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            st.divider()

            # =================================================
            # COUNTRY ANALYSIS
            # =================================================

            st.subheader("Top Revenue Countries")

            country_sales = (
                df.groupby('Country')['TotalPrice']
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )

            fig_country = px.bar(
                country_sales,
                x='Country',
                y='TotalPrice',
                title='Country Wise Revenue'
            )

            st.plotly_chart(
                fig_country,
                use_container_width=True
            )

        # =====================================================
        # AI INSIGHTS PAGE
        # =====================================================

        elif page == "AI Insights":

            st.header("AI Business Insights")

            insights = []

            total_revenue = round(
                df['TotalPrice'].sum(),
                2
            )

            insights.append(
                f"Total platform revenue generated is ${total_revenue:,.0f}."
            )

            top_country = (
                df.groupby('Country')['TotalPrice']
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )

            insights.append(
                f"{top_country} generated the highest revenue."
            )

            top_product = (
                df.groupby('Description')['Quantity']
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )

            insights.append(
                f"The most demanded product is '{top_product}'."
            )

            avg_order = (
                df.groupby('InvoiceNo')['TotalPrice']
                .sum()
                .mean()
            )

            insights.append(
                f"Average order value is ${avg_order:.2f}."
            )

            for idx, insight in enumerate(
                insights,
                start=1
            ):

                st.info(
                    f"AI Insight {idx}: {insight}"
                )

        # =====================================================
        # DATASET QUALITY PAGE
        # =====================================================

        elif page == "Dataset Quality":

            st.header("Dataset Quality Intelligence")

            total_cells = (
                df.shape[0] * df.shape[1]
            )

            missing_values = (
                df.isnull().sum().sum()
            )

            duplicate_rows = (
                df.duplicated().sum()
            )

            missing_percentage = (
                missing_values / total_cells
            ) * 100

            quality_score = max(
                0,
                100 - missing_percentage - duplicate_rows
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Missing Values",
                missing_values
            )

            col2.metric(
                "Duplicate Rows",
                duplicate_rows
            )

            col3.metric(
                "Quality Score",
                f"{round(quality_score, 2)}%"
            )

            st.divider()

            st.subheader("Dataset Summary")

            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")

            st.subheader("Column Data Types")

            st.dataframe(
                pd.DataFrame(
                    df.dtypes,
                    columns=['DataType']
                )
            )

        # =====================================================
        # CUSTOMER SEGMENTATION
        # =====================================================

        elif page == "Customer Segmentation":

            st.header("Customer Segmentation")

            customer_data = df.groupby(
                'CustomerID'
            ).agg({
                'TotalPrice': 'sum',
                'InvoiceNo': 'nunique',
                'Quantity': 'sum'
            }).reset_index()

            customer_data.columns = [
                'CustomerID',
                'TotalSpent',
                'TotalOrders',
                'TotalQuantity'
            ]

            features = customer_data[
                [
                    'TotalSpent',
                    'TotalOrders',
                    'TotalQuantity'
                ]
            ]

            scaler = StandardScaler()

            scaled_features = scaler.fit_transform(
                features
            )

            kmeans = KMeans(
                n_clusters=3,
                random_state=42
            )

            customer_data['Cluster'] = (
                kmeans.fit_predict(
                    scaled_features
                )
            )

            fig3 = px.scatter(
                customer_data,
                x='TotalSpent',
                y='TotalOrders',
                color=customer_data['Cluster'].astype(str),
                title='Customer Segments'
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

            st.dataframe(
                customer_data.head(20)
            )

        # =====================================================
        # RECOMMENDATION SYSTEM
        # =====================================================

        elif page == "Recommendation System":

            st.header("AI Product Recommendation Engine")

            st.markdown("""
Select a product to discover highly related products based on customer purchasing behavior.
""")

            product_matrix = df.pivot_table(
                index='CustomerID',
                columns='Description',
                values='Quantity',
                fill_value=0
            )

            similarity = cosine_similarity(
                product_matrix.T
            )

            similarity_df = pd.DataFrame(
                similarity,
                index=product_matrix.columns,
                columns=product_matrix.columns
            )

            product_list = sorted(
                df['Description']
                .dropna()
                .unique()
            )

            product_name = st.selectbox(
                "Choose Product",
                product_list
            )

            if st.button(
                "Generate Recommendations"
            ):

                recommendations = (
                    similarity_df[product_name]
                    .sort_values(ascending=False)[1:6]
                )

                st.subheader(
                    "Recommended Products"
                )

                for idx, product in enumerate(
                    recommendations.index,
                    start=1
                ):

                    similarity_score = round(
                        recommendations[product] * 100,
                        2
                    )

                    st.success(
                        f"{idx}. {product} | Similarity Score: {similarity_score}%"
                    )

        # =====================================================
        # SALES FORECASTING
        # =====================================================

        elif page == "Sales Forecasting":

            st.header("Sales Forecasting")

            daily_sales = (
                df.groupby(
                    df['InvoiceDate'].dt.date
                )['TotalPrice']
                .sum()
                .reset_index()
            )

            daily_sales.columns = [
                'Date',
                'Sales'
            ]

            daily_sales['Day'] = range(
                len(daily_sales)
            )

            X = daily_sales[['Day']]

            y = daily_sales['Sales']

            model = RandomForestRegressor(
                random_state=42
            )

            model.fit(X, y)

            future_days = pd.DataFrame({
                'Day': range(
                    len(daily_sales),
                    len(daily_sales) + 30
                )
            })

            future_predictions = model.predict(
                future_days
            )

            forecast_df = pd.DataFrame({
                'Day': future_days['Day'],
                'PredictedSales': future_predictions
            })

            fig4 = px.line(
                forecast_df,
                x='Day',
                y='PredictedSales',
                title='30-Day Future Sales Forecast'
            )

            st.plotly_chart(
                fig4,
                use_container_width=True
            )

            future_growth = (
                (
                    future_predictions[-1]
                    - future_predictions[0]
                )
                /
                future_predictions[0]
            ) * 100

            st.success(
                f"Predicted sales trend indicates approximately {future_growth:.2f}% growth over the next 30 days."
            )

            st.dataframe(
                forecast_df.head(10)
            )

        # =====================================================
        # BUSINESS RECOMMENDATIONS
        # =====================================================

        elif page == "Business Recommendations":

            st.header("AI Business Recommendations")

            top_country = (
                df.groupby('Country')['TotalPrice']
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )

            top_product = (
                df.groupby('Description')['Quantity']
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )

            st.success(
                f"Focus marketing campaigns in {top_country} region."
            )

            st.success(
                f"Increase inventory for '{top_product}' due to high demand."
            )

            st.success(
                "Target high-value customer segments with loyalty programs."
            )

            st.success(
                "Monitor anomaly transactions to reduce business risk."
            )

        # =====================================================
        # ANOMALY DETECTION
        # =====================================================

        elif page == "Anomaly Detection":

            st.header("AI Anomaly Detection System")

            anomaly_data = df[
                ['Quantity', 'UnitPrice']
            ]

            model = IsolationForest(
                contamination=0.02,
                random_state=42
            )

            df['Anomaly'] = model.fit_predict(
                anomaly_data
            )

            anomalies = df[
                df['Anomaly'] == -1
            ]

            st.metric(
                "Detected Anomalies",
                len(anomalies)
            )

            fig_anomaly = px.scatter(
                anomalies,
                x='Quantity',
                y='UnitPrice',
                color='Country',
                title='Detected Transaction Anomalies'
            )

            st.plotly_chart(
                fig_anomaly,
                use_container_width=True
            )

            st.dataframe(
                anomalies.head(20)
            )

        # =====================================================
        # DOWNLOAD DATASET
        # =====================================================

        csv = df.to_csv(
            index=False
        ).encode('utf-8')

        st.download_button(
            "Download Processed Dataset",
            csv,
            "processed_dataset.csv",
            "text/csv"
        )

    except Exception as e:

        st.error(
            f"Error Processing File: {e}"
        )

else:

    st.info(
        "Please upload a valid e-commerce CSV dataset to continue."
    )