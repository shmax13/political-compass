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
        
        return mean_embedding
    

    def extract(self, cleaned_speeches, labels):
        """Extract BERT features."""
        X_bert = np.array([self.vectorize_speech(speech) for speech in cleaned_speeches])
        X_train_bert, X_test_bert, y_train, y_test = train_test_split(X_bert, labels, test_size=0.2, random_state=42)
        return X_train_bert, X_test_bert, y_train, y_test


    # def __init__(self):
    #     self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    #     self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
    #     super().__init__(vectorizer=None)

    # def vectorize_speech(self, speech):
    #     """Vectorize a single speech using the longformer model."""
    #     inputs = self.tokenizer(speech, return_tensors="pt", truncation=True, padding=True, max_length=512)
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)
    #         #use CLS token embeddings - might wanna try something else here
    #         # cls_embedding = outputs.last_hidden_state[:,0,:].squeeze().numpy()
    #         # return cls_embedding    

    #         # max_embedding = outputs.last_hidden_state.max(dim=1).values.squeeze().numpy()
    #         # return max_embedding 

    #         #mean pooling - gives better accuracy than other above pooling methods
    #         mean_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    #         return mean_embedding

    # def extract(self, cleaned_speeches, labels):
    #     """Extract BERT features."""
    #     X_bert = np.array([self.vectorize_speech(speech) for speech in cleaned_speeches])
    #     X_train_bert, X_test_bert, y_train, y_test = train_test_split(X_bert, labels, test_size=0.2, random_state=42)
    #     return X_train_bert, X_test_bert, y_train, y_test

    def train(self, train_data, train_labels, epochs=5, batch_size=32, learning_rate=1e-5):
        """
        Fine-tune the BERT model on the training data.

        Args:
            train_data (list): List of preprocessed speech texts.
            train_labels (list): List of corresponding labels.
            epochs (int, optional): Number of training epochs. Defaults to 5.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-5.
        """
        # Create a custom dataset class for our data
        class SpeechDataset(torch.utils.data.Dataset):
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels

            def __getitem__(self, idx):
                speech = self.data[idx]
                label = self.labels[idx]
                inputs = self.tokenizer(speech, return_tensors="pt", truncation=True, padding=True, max_length=512)
                return inputs, torch.tensor(label)

            def __len__(self):
                return len(self.data)

        # Create a dataset instance and data loader
        dataset = SpeechDataset(train_data, train_labels)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Set the model to training mode
        self.model.train()

        # Define the optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train the model
        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")

        # Save the fine-tuned model
        torch.save(self.model.state_dict(), "fine_tuned_bert.pth")