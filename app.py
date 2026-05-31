from fastapi import FastAPI, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import random

app = FastAPI()

# Menyiapkan 5 metriks berbeda secara aman
METRIC_PREDICTIONS = Counter('model_predictions_total', 'Total prediksi stres')
METRIC_STRESS_LEVEL = Gauge('student_stress_level_pred', 'Level stres mahasiswa terakhir')
METRIC_LATENCY = Histogram('model_inference_latency_seconds', 'Waktu inferensi model')
METRIC_MEMORY = Gauge('app_memory_usage_bytes', 'Penggunaan memori aplikasi')
METRIC_ERRORS = Counter('http_errors_total', 'Total error API')

@app.post("/predict")
def predict():
    start_time = time.time()
    
    # 5% peluang simulasi error untuk memicu metriks ke-5
    if random.random() < 0.05:
        METRIC_ERRORS.inc()
        return Response(status_code=500, content="Simulated Server Error")
        
    # Jalankan simulasi prediksi (0: Rendah, 1: Sedang, 2: Tinggi)
    pred_class = random.choice([0, 1, 2])
    time.sleep(random.uniform(0.05, 0.15)) # Simulasi jeda model ML
    
    # Update metriks
    METRIC_PREDICTIONS.inc()
    METRIC_STRESS_LEVEL.set(pred_class)
    METRIC_LATENCY.observe(time.time() - start_time)
    
    return {"status": "success", "stress_level": pred_class}

@app.get("/metrics")
def metrics():
    # Update dinamis untuk metriks memori
    METRIC_MEMORY.set(random.randint(200000000, 300000000))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)