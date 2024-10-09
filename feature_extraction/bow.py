from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def extract_bow(cleaned_speeches, labels):
    bow_vectorizer = CountVectorizer(max_features=5000)
    X_bow = bow_vectorizer.fit_transform(cleaned_speeches)
    X_train_bow, X_test_bow, y_train, y_test = train_test_split(X_bow, labels, test_size=0.2, random_state=42)
    return X_train_bow, X_test_bow, y_train, y_test