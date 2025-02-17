import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic


def init_model() -> BERTopic:
    """
        Initialize the BERTopic model and its components.

        Returns:
            BERTopic model
    """

    # Define the components of BERTopic.
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    representation_model = { 
        'KeyBERT': KeyBERTInspired()
    }
    # Use vectorizer for stop_words removal.
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

    # Create the topic model
    topic_model = BERTopic(
        language="english", 
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model, 
        calculate_probabilities=True, 
        verbose=True, 
        nr_topics=20
    )
    return topic_model


def train(topic_model: BERTopic, data: pd.DataFrame, save_model: bool) -> BERTopic:
    """
        Train and save the trained BERTopic model.

        Parameters:
            topic_model: The BERTopic model to be trained.
            data: Training data.
            save_model: Whether to save the trained model or not.
        
        Returns:
            topic_model: Trained BERTopic model.
    """
    topic_model.fit(data)
    
    # Re-label the topics for better and clearer readability
    keybert_topic_labels = {topic: " | ".join(list(zip(*values))[0][:3]) for topic, values in topic_model.topic_aspects_["KeyBERT"].items()}
    topic_model.set_topic_labels(keybert_topic_labels)
    
    if save_model:
        topic_model.save("../data/models/bertopic", save_ctfidf=True, save_embedding_model=True, serialization='safetensors')
    return topic_model


def preprocess_data(data: pd.DataFrame) -> pd.Series:
    """
        Clean and filter the dataset.

        Parameters:
            data: Raw dataset.
        
        Returns:
            data: Preprocessed data.
    """
    # Remove NaNs
    data = data[~data.best_answer.isna()]
    # We'll use the best_answer column for training 
    return data.best_answer
    

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data = pd.read_csv("../data/raw/dataset.csv", sep=';', index_col=0)
    # Preprocess data
    print("Processing data...")
    processed_data = preprocess_data(data)
    
    # Initilize model
    print("Initializing BERTopic model...")
    topic_model = init_model()
    # Train
    print("Training model...")
    train(topic_model, processed_data, save_model=True)
    print("Finished training!")