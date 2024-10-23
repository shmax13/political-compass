import numpy  as np
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from .base import BaseFeatureExtractor
import torch

class BERTExtractor(BaseFeatureExtractor):

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        super().__init__(vectorizer=None)

    def vectorize_speech(self, speech):
        """Vectorize a single speech using the BERT model."""
        inputs = self.tokenizer(speech, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            #use CLS token embeddings - might wanna try something else here
            cls_embedding = outputs.last_hidden_state[:,0,:].squeeze().numpy()
            return cls_embedding     

    def extract(self, cleaned_speeches, labels):
        """Extract BERT features."""
        X_bert = np.array([self.vectorize_speech(speech) for speech in cleaned_speeches])
        X_train_bert, X_test_bert, y_train, y_test = train_test_split(X_bert, labels, test_size=0.2, random_state=42)
        return X_train_bert, X_test_bert, y_train, y_test
