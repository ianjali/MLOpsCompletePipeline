# CoLA (Corpus of Linguistic Acceptability) Model
This project implements a BERT-based model for the Corpus of Linguistic Acceptability (CoLA) task, which involves determining whether English sentences are grammatically acceptable.

## Setup Instructions
### 1. Create a Virtual Environment

```
# Create a new conda environment
conda create -n cola-env python=3.11
conda activate cola-env
```
### 2. Install Dependencies
```
# Install required packages
pip install -r requirements.txt
```
### 3. Run the Training
```
python trainer.py
```
This will:

Load the CoLA dataset from the GLUE benchmark
Train a BERT-based model for 5 epochs (unless early stopping is triggered)
Save model checkpoints and logs

### 4. Monitor Training (Optional)
Run this on terminal 
* Tensorboard
```
tensorboard --logdir=logs/cola
```
http://localhost:6006/ 
* Weights and Bias
 Create your accont and add your API Key to monitor


### 5. Inferencing
After training, update the model checkpoint path in the code and run
```
python inference.py
```

### 6. Hydra for model configurations
```
@hydra.main(config_path="./configs", config_name="config")
```
Hydra operates on top of OmegaConf, which is a YAML based hierarchical configuration system, with support for merging configurations from multiple sources (files, CLI argument, environment variables) providing a consistent API regardless of how the configuration was created.

### 7. DVC Data Version Control
There are many libraries which supports versioning of models and data. The prominent ones are:
* DVC
* DAGsHub
* Hub
* Modelstore
* ModelDB

Initialize 
* This command will create .dvc folder and .dvcignore file.
```
dvc init
```
* configure remote storage(Google Drive) to store trained models (or datasets).
 
```
dvc add models/best-checkpoint.ckpt
```
```
dvc push models/best-checkpoint.ckpt.dvc
```

### 8. Model Packaging - ONNX
ONNX defines a common set of operators - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.
Run 
```
python convert_model_to_onnx.py
```
to convert your model into ONNX format
Models in ONNX format can be easily deployed to various cloud platforms as well as to IoT devices.
### 9. API endpoints
Create endpoints using FastAPI 
Check app.py
Run
```
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

[http://localhost:8000/docs](http://localhost:8000/docs)

### 10. Model Packaging - Docker
For others to run the applications they have to set up the same environment as it was run on the host side which means a lot of manual configuration and installation of components
By containerizing/packaging the application, we can run the application on any cloud platform to get advantages of managed services and autoscaling and reliability, and many more.

The most prominent tool to do the packaging of application is Docker.

Docker is a container management tool, which packages the application code, configuration, and dependencies into a portable image that can be shared and run on any platform or system.
* A Docker File contains the list of commands to run which are necessary for the application to run (like dependencies, codes, command to run etc.)
The best part is all layers are cached and if we modified some instructions in the Dockerfile then during the build process it will just rebuild the changed layer.

* A Docker Image is a lightweight, standalone, executable package of software (built using dockerfile) that includes everything needed to run an application: code, runtime, system tools, system libraries, and settings.

* A Docker Container is an instance of Docker Image which contains the running application.


## Project Structure

data.py: Contains the DataModule class that handles data loading, preprocessing, and creating dataloaders
model.py: Implements the ColaModel class, a PyTorch Lightning module for the CoLA task
trainer.py: Main script that sets up training with appropriate callbacks and logging

## Model Architecture
The model uses a small BERT model (google/bert_uncased_L-2_H-128_A-2) with a linear classification head on top of the [CLS] token representation. It's configured to classify sentences into two categories: grammatically acceptable (1) or unacceptable (0).


### References 
* https://deep-learning-blogs.vercel.app/blog/mlops-project-setup-part1

