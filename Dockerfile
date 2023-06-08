FROM alpine:3.11

FROM python:3.8

WORKDIR /app

COPY . /app

RUN pip --no-cache-dir install -r requirements.txt

ENTRYPOINT ["python3", "app.py"]