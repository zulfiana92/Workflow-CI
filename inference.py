import requests
import json
import time
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import random

# =========================
# PROMETHEUS METRICS
# =========================
REQUEST_COUNT = Counter('ml_request_count', 'Total prediction requests')
REQUEST_LATENCY = Histogram('ml_request_latency_seconds', 'Prediction latency')
PREDICTION_0 = Counter('ml_prediction_no_disease', 'Predictions: No Disease')
PREDICTION_1 = Counter('ml_prediction_disease', 'Predictions: Disease')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Model accuracy score')
ACTIVE_REQUESTS = Gauge('ml_active_requests', 'Active requests')

# Set accuracy
MODEL_ACCURACY.set(0.8421)

# =========================
# INFERENCE
# =========================
def predict(data):
    url = "http://127.0.0.1:5001/invocations"
    headers = {"Content-Type": "application/json"}
    payload = {"dataframe_records": [data]}

    ACTIVE_REQUESTS.inc()
    start = time.time()

    try:
        REQUEST_COUNT.inc()
        response = requests.post(url, headers=headers, json=payload)
        latency = time.time() - start
        REQUEST_LATENCY.observe(latency)

        result = response.json()
        prediction = result['predictions'][0]

        if prediction == 0:
            PREDICTION_0.inc()
        else:
            PREDICTION_1.inc()

        print(f"Prediction: {'Disease' if prediction == 1 else 'No Disease'} (latency: {latency:.3f}s)")
        return prediction

    except Exception as e:
        print(f"Error: {e}")
    finally:
        ACTIVE_REQUESTS.dec()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Start Prometheus exporter di port 8000
    start_http_server(8000)
    print("✅ Prometheus metrics server running at http://localhost:8000")

    # Sample data untuk testing
    sample_data = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
        "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
        "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }

    print("🔄 Sending predictions every 5 seconds...")
    while True:
        predict(sample_data)
        time.sleep(5)