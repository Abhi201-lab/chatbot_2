FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirement.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy service sources from the build context into the image
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
