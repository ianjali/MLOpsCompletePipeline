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
## Project Structure

data.py: Contains the DataModule class that handles data loading, preprocessing, and creating dataloaders
model.py: Implements the ColaModel class, a PyTorch Lightning module for the CoLA task
trainer.py: Main script that sets up training with appropriate callbacks and logging

## Model Architecture
The model uses a small BERT model (google/bert_uncased_L-2_H-128_A-2) with a linear classification head on top of the [CLS] token representation. It's configured to classify sentences into two categories: grammatically acceptable (1) or unacceptable (0).


### References 
* https://deep-learning-blogs.vercel.app/blog/mlops-project-setup-part1

