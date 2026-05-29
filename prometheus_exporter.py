
from prometheus_client import Counter, start_http_server
import time

REQUEST_COUNT = Counter(
    'prediction_request_total',
    'Total Prediction'
)

print("Prometheus exporter running on port 8000")

start_http_server(8000)

try:
    while True:
        REQUEST_COUNT.inc()
        print("Metric updated")
        time.sleep(5)

except KeyboardInterrupt:
    print("Exporter stopped")
