# Topic Modeling  

This project involves training a topic modeling model, specifically BERTopic, to group text (answers) into 20 topics. A simple inference API is then created using FastAPI for testing the trained model. Below are detailed instructions on how to configure and run the project.

## Setup environment
1. Create a python environment:
   
   ```bash
    python -m venv env
   ```

2. Activate envrionment:
   ```
   cd env/Scripts
   activate
   ```

3. Install project dependencies:
   ```bash
    pip install -r requirements.txt
   ```


## Train BertTopic
1. Make sure that the dataset is placed in the correct folder `data/raw/dataset.csv`.
2. Activate the environment `env`.
   
      ```
   cd env/Scripts
   activate
   ```

3. Run the following command from the `src/` folder:
   
   ```
    python train.py
   ```
This will train a BertTopic model on the dataset using the 'best_answers' column. The trained model will be saved in the `data/models/bertopic/` folder.

## Run inference server
The inference API is created using **FastAPI**. To run the inference server, execute the following command from the `src/` folder:
```
uvicorn inference_server:app
```

Now, we can get the predictions from the api using `curl`:
```
curl -X 'POST' \
  'http://127.0.0.1:8000/predict_topic' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Try this software.  I didn'\''t check it out personally, but it claims to be capable of doing what you need it to."
}'
```

## What can be improved?
- **Model performance and evaluation**: Train different models and variations of those models on this dataset and compare their performances to pick the best models. One tricky part though is the fact that evaluating topic models can be rather difficult due to the somewhat subjective nature of evaluation. Some techniques such as cluster analysis, topics' similarties calculations, etc.. can be used to help ease up the process.  Moreover, hyperparameters such as the number of topics and the topics themselves would be defined depending on the project/client requirements and constraints.
- **Inference**:
    Inference speed optimization can be achieved in a variety of ways:
    - *Using GPUs*: This is the most straightforward way if we're using DL/ML models that support GPU operations. However, GPUs usually give rise to high operational costs, which, depending on the project/budget, can be unacceptable.
    - *Model optimizations*: This can be model dependent. One way is to use AI engines such as ONNX to speed up inference. This can be very useful when the use of GPUs is limited and the model needs to be run on CPUs as it offers reduced inference latency.
    - *Reducing model size*: Transformers models are usually more resource intensive, so choosing another model, such as LDA, can be beneficial in resource-constrained production environments.