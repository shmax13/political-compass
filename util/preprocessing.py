import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def assign_leaning(president):
    left_leaning = {
        'Andrew Jackson', 'Martin Van Buren', 'James K. Polk', 'Franklin Pierce', 'James Buchanan',
        'Grover Cleveland', 'Woodrow Wilson', 'Franklin D. Roosevelt', 'Harry S. Truman', 
        'John F. Kennedy', 'Lyndon B. Johnson', 'Jimmy Carter', 'Bill Clinton', 'Barack Obama', 'Joe Biden',
        'Thomas Jefferson', 'James Madison', 'James Monroe', 'Andrew Johnson'
    }

    right_leaning = {
        'Abraham Lincoln', 'Ulysses S. Grant', 'Rutherford B. Hayes', 'James A. Garfield', 'Chester A. Arthur',
        'Benjamin Harrison', 'William McKinley', 'Theodore Roosevelt', 'William Taft', 'Calvin Coolidge', 
        'Herbert Hoover', 'Dwight D. Eisenhower', 'Richard M. Nixon', 'Gerald Ford', 'Ronald Reagan', 
        'George H. W. Bush', 'George W. Bush', 'Donald Trump', 'John Adams', 'John Quincy Adams',
        'William Harrison', 'John Tyler', 'Zachary Taylor', 'Millard Fillmore', 'Warren G. Harding'
    }

    # lets not use this for now
    independent_presidents = {
        'George Washington'
    }

    if president in left_leaning:
        return 'Left-Leaning'
    elif president in right_leaning:
        return 'Right-Leaning'
    else:
        return 'Unknown'

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters using regex
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Join tokens back into a string
    return ' '.join(cleaned_tokens)

def preprocess():

    with open('./speeches/speeches.json', 'r') as file:
        data = json.load(file)
        
    # Extract speeches and labels
    speeches = []
    labels = []

    for entry in data:
        speech_text = entry['transcript']
        president = entry['president']
        party = assign_leaning(president)
        assign_leaning
        speeches.append(speech_text)
        labels.append(party)
    
    # preprocess speech text
    cleaned_speeches = [preprocess_text(speech) for speech in speeches]

    return cleaned_speeches, labels


def save_preprocessed_data(cleaned_speeches, labels, filename='speeches/preprocessed_speeches.json'):
    data = {'speeches': cleaned_speeches, 'labels': labels}
    with open(filename, 'w') as f:
        json.dump(data, f)

def main():
    cleaned_speeches, labels = preprocess()
    save_preprocessed_data(cleaned_speeches, labels)


if __name__ == '__main__':
    main()