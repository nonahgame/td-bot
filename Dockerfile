
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENV FLASK_APP=app.py
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
