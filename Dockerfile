# FROM huggingface/transformers-pytorch-cpu:latest
FROM python:3.9-slim
# FROM --platform=linux/arm64 huggingface/transformers-pytorch-cpu:latest
COPY ./app.py /app/
COPY ./models/model.onnx /app/models/ 
COPY ./inference_onnx.py /app/
COPY ./data.py /app/
COPY ./utils.py /app/
COPY ./requirements_prod.txt /app/
WORKDIR /app
RUN pip install -r requirements_prod.txt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]