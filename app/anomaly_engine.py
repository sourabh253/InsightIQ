from sklearn.ensemble import IsolationForest

def detect_anomalies(df):

    anomaly_data = df[['Quantity', 'UnitPrice']]

    model = IsolationForest(
        contamination=0.02,
        random_state=42
    )

    df['Anomaly'] = model.fit_predict(
        anomaly_data
    )

    anomalies = df[df['Anomaly'] == -1]

    return anomalies