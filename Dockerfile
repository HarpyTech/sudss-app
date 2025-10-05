FROM python:3.11-alpine

WORKDIR /app

# Install dependencies
COPY ./src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY ./src/app/ .

# Expose port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
