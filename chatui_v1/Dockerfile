FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirement.txt ./requirements.txt
# Install from the requirements file (now includes httpx)
RUN pip install --no-cache-dir -r requirements.txt

# copy service sources from the build context into the image
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
