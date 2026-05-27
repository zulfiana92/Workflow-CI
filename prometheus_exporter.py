from prometheus_client import start_http_server, Counter, Histogram, Gauge

# =========================
# PROMETHEUS METRICS
# =========================
REQUEST_COUNT = Counter('ml_request_count', 'Total prediction requests')
REQUEST_LATENCY = Histogram('ml_request_latency_seconds', 'Prediction latency')
PREDICTION_0 = Counter('ml_prediction_no_disease', 'Predictions: No Disease')
PREDICTION_1 = Counter('ml_prediction_disease', 'Predictions: Disease')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Model accuracy score')
ACTIVE_REQUESTS = Gauge('ml_active_requests', 'Active requests')

MODEL_ACCURACY.set(0.8421)

if __name__ == "__main__":
    start_http_server(8000)
    print("✅ Prometheus exporter running at http://localhost:8000/metrics")
    import time
    while True:
        time.sleep(1)