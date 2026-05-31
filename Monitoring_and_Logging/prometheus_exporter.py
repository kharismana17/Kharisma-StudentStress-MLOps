import time
import mlflow.sklearn
import numpy as np
import pandas as pd
import psutil

from flask import Flask, request, jsonify

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    start_http_server
)

app = Flask(__name__)

DICODING_USER = "Kharisma Nur Aulia"
metric_prefix = DICODING_USER.lower().replace(" ", "_")

REQ_COUNTER = Counter(
    f"{metric_prefix}_total_inference_requests",
    "Total inference requests"
)

SUCCESS_COUNTER = Counter(
    f"{metric_prefix}_success_requests",
    "Successful requests"
)

ERROR_COUNTER = Counter(
    f"{metric_prefix}_error_requests",
    "Failed requests"
)

LATENCY_HIST = Histogram(
    f"{metric_prefix}_inference_latency_seconds",
    "Inference latency"
)

PRED_COUNTER = Counter(
    f"{metric_prefix}_predicted_stress_level",
    "Prediction count by class",
    ["class_label"]
)

CURRENT_REQUESTS = Gauge(
    f"{metric_prefix}_current_requests",
    "Current active requests"
)

REQUEST_SIZE = Histogram(
    f"{metric_prefix}_request_size_bytes",
    "Request payload size"
)

MODEL_LOAD_TIME = Gauge(
    f"{metric_prefix}_model_load_time_seconds",
    "Model loading time"
)

CPU_USAGE = Gauge(
    f"{metric_prefix}_cpu_usage_percent",
    "CPU usage percent"
)

MEMORY_USAGE = Gauge(
    f"{metric_prefix}_memory_usage_percent",
    "Memory usage percent"
)

PREDICTION_TIME = Histogram(
    f"{metric_prefix}_prediction_time_seconds",
    "Prediction execution time"
)


model_load_start = time.time()

try:
    model = mlflow.sklearn.load_model(
        "mlruns/0/303c7f037f654808951080faed0cc5b0/artifacts"
    )

    print("Model MLflow berhasil dimuat")

except Exception as e:

    print(f"Gagal memuat model MLflow: {e}")
    print("Menggunakan model dummy")

    from sklearn.ensemble import RandomForestClassifier

    X_dummy = np.random.rand(100, 20)
    y_dummy = np.random.randint(0, 3, 100)

    model = RandomForestClassifier()
    model.fit(X_dummy, y_dummy)

MODEL_LOAD_TIME.set(
    time.time() - model_load_start
)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "API berjalan",
        "model": "Student Stress Prediction"
    })

@app.route("/predict", methods=["POST"])
def predict():

    start_time = time.time()

    CURRENT_REQUESTS.inc()
    REQ_COUNTER.inc()

    try:

        data = request.get_json()

        if data is None:

            ERROR_COUNTER.inc()

            return jsonify({
                "status": "error",
                "message": "JSON tidak ditemukan"
            }), 400

        if "features" not in data:

            ERROR_COUNTER.inc()

            return jsonify({
                "status": "error",
                "message": "Key 'features' tidak ditemukan"
            }), 400

        REQUEST_SIZE.observe(
            len(str(data))
        )

        df_input = pd.DataFrame(
            [data["features"]]
        )

        pred_start = time.time()

        prediction = model.predict(
            df_input
        )[0]

        pred_duration = (
            time.time() - pred_start
        )

        PREDICTION_TIME.observe(
            pred_duration
        )

        PRED_COUNTER.labels(
            class_label=str(prediction)
        ).inc()

        SUCCESS_COUNTER.inc()

        CPU_USAGE.set(
            psutil.cpu_percent()
        )

        MEMORY_USAGE.set(
            psutil.virtual_memory().percent
        )

        total_duration = (
            time.time() - start_time
        )

        LATENCY_HIST.observe(
            total_duration
        )

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "latency_seconds":
                round(total_duration, 5)
        })

    except Exception as e:

        ERROR_COUNTER.inc()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:

        CURRENT_REQUESTS.dec()

# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    try:
        start_http_server(9000)

        print(
            "Prometheus Metrics berjalan di port 9000"
        )

    except OSError:

        print(
            "Port 9000 sudah digunakan"
        )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )