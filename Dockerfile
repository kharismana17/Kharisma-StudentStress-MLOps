FROM python:3.10-slim

WORKDIR /app

# Salin file requirements terlebih dahulu agar memanfaatkan cache docker
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]