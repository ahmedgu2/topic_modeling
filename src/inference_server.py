from bertopic import BERTopic
from fastapi import FastAPI
from typing import Dict
from Answer import Answer
import numpy as np

app = FastAPI()

# Create a global model to load it only once.
model = BERTopic.load("../data/models/bertopic", embedding_model='all-MiniLM-L6-v2')

@app.post("/predict_topic")
def predict(input: Answer) -> Dict:
    """
        Perform inference.

        Parameters:
            input: Input data for the model.

        Returns:
            Prediction topic.
    """
    input_dict = input.model_dump()
    topic, _ = model.transform(input_dict['text'])
    # Map topic id to topic name
    freq = model.get_topic_info()
    predicted_topic = freq[freq['Topic'] == topic[0]]['CustomName'].values[0]

    return {
        "prediction": predicted_topic
    }
