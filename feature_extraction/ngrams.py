from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def extract_ngrams(cleaned_speeches, labels, ngram_range=(2, 2)):
    bigram_vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=5000)
    X_bigrams = bigram_vectorizer.fit_transform(cleaned_speeches)
    X_train_bigrams, X_test_bigrams, y_train, y_test = train_test_split(X_bigrams, labels, test_size=0.2, random_state=42)
    return X_train_bigrams, X_test_bigrams, y_train, y_test