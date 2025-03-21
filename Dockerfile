FROM python:3.9-slim
COPY ./app.py /app/
# COPY ./models/model.onnx /app/models/ 
COPY ./inference_onnx.py /app/
COPY ./data.py /app/
COPY ./utils.py /app/
COPY ./requirements_prod.txt /app/
# COPY ./cred.json /app/
COPY ./.dvc /app/
COPY ./dvcfiles/trained_model.dvc /app/dvcfiles/
WORKDIR /app


ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# aws credentials configuration
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY


# RUN pip install "dvc[gdrive]"
# install requirements
RUN pip install "dvc[s3]" 
RUN pip install -r requirements_prod.txt

# initialise dvc
RUN dvc init --no-scm
# configuring remote server in dvc
RUN dvc remote add -d model-store s3://models-dvc-1/trained_models/
# RUN dvc remote add -f storage gdrive://[drive_id]
# RUN dvc remote modify storage gdrive_use_service_account true
# RUN dvc remote modify storage gdrive_service_account_json_file_path /app/cred.json
# adding the model to dvc

RUN cat .dvc/config
# RUN cat .dvc/config
# pulling the trained model
RUN dvc pull dvcfiles/trained_model.dvc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]