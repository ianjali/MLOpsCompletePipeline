FROM python:3.9-slim
COPY ./app.py /app/
COPY ./models/model.onnx /app/models/ 
COPY ./inference_onnx.py /app/
COPY ./data.py /app/
COPY ./utils.py /app/
COPY ./requirements_prod.txt /app/
COPY ./cred.json /app/
COPY ./.dvc /app/
WORKDIR /app

RUN pip install "dvc[gdrive]"
RUN pip install -r requirements_prod.txt

# initialise dvc
RUN dvc init --no-scm
# configuring remote server in dvc
# RUN dvc remote add -f storage gdrive://[drive_id]
# RUN dvc remote modify storage gdrive_use_service_account true
# RUN dvc remote modify storage gdrive_service_account_json_file_path /app/cred.json
# adding the model to dvc
RUN cat .dvc/config
# pulling the trained model
# RUN dvc pull model.onnx

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]