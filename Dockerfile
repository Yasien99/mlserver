FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
