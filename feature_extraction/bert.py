import numpy as np
from transformers import BertTokenizer, BertModel
from transformers import pipeline, AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from .base import BaseFeatureExtractor
import torch

class BERTExtractor(BaseFeatureExtractor):

    def __init__(self):
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.model = BertModel.from_pretrained('bert-base-uncased')
        
        self.tokenizer = AutoTokenizer.from_pretrained('premsa/political-bias-prediction-allsides-BERT')
        self.model = AutoModel.from_pretrained('premsa/political-bias-prediction-allsides-BERT')

        super().__init__(vectorizer=None)
        

    def vectorize_speech(self, speech):
        
        inputs = self.tokenizer(speech, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Experiment with concatenated embeddings
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        max_embedding = outputs.last_hidden_state.max(dim=1).values.squeeze().numpy()
        mean_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        combined_embedding = np.concatenate([cls_embedding, max_embedding, mean_embedding], axis=-1)
        
        # test with mean pca 200, cls and no pca, + cls and pca 100, 200, 300
        # return cls_embedding
        return mean_embedding
    

    def extract(self, cleaned_speeches, labels):
        """Extract BERT features."""
        X_bert = np.array([self.vectorize_speech(speech) for speech in cleaned_speeches])

        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=100)
        X_bert = pca.fit_transform(X_bert)

        X_train_bert, X_test_bert, y_train, y_test = train_test_split(X_bert, labels, test_size=0.2, random_state=42)
        return X_train_bert, X_test_bert, y_train, y_test

