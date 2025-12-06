FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model files
COPY app.py .
COPY rf_model.pkl .
COPY scaler.pkl .

EXPOSE 5000

CMD ["python", "app.py"]
