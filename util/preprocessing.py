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

president_positions = {
    'Andrew Jackson': (6, 6),      # Right-leaning authoritarian, despite some populist rhetoric
    'Martin Van Buren': (-5, -3),  # Moderate left libertarian
    'James K. Polk': (-6, 3),      # Left-leaning authoritarian due to expansionist policies
    'Franklin Pierce': (-4, 1),    # Center-left with moderate authoritarian tendencies
    'James Buchanan': (-3, 2),     # Moderate left, slightly authoritarian
    'Grover Cleveland': (-4, -1),  # Center-left, moderate libertarian
    'Woodrow Wilson': (-6, 8),     # Left-leaning authoritarian (due to WWI policies)
    'Franklin D. Roosevelt': (-9, 6),  # Strong left, authoritarian (New Deal, WWII)
    'Harry S. Truman': (-7, 4),    # Strong left-leaning authoritarian (Cold War)
    'John F. Kennedy': (-4, -2),   # Center-left, moderate libertarian
    'Lyndon B. Johnson': (-5, 5),  # Left-leaning authoritarian (Great Society)
    'Jimmy Carter': (-4, -3),      # Moderate left, moderate libertarian
    'Bill Clinton': (-3, -3),      # Center-left, moderate libertarian
    'Barack Obama': (-4, -2),      # Center-left, moderate libertarian
    'Joe Biden': (-5, 2),          # Moderate left, moderate authoritarian
    'Thomas Jefferson': (-7, -8),  # Strong left, strongly libertarian (anti-centralization)
    'James Madison': (-6, -7),     # Left-leaning libertarian (focused on individual rights)
    'James Monroe': (-5, -5),      # Moderate left, libertarian (Monroe Doctrine)
    'Andrew Johnson': (5, 4),      # Slightly right-leaning, authoritarian
    'Abraham Lincoln': (-6, 7),    # Left-leaning authoritarian (due to wartime powers)
    'Ulysses S. Grant': (4, 4),    # Right-leaning authoritarian (Reconstruction policies)
    'Rutherford B. Hayes': (3, 2), # Slightly right-leaning, authoritarian
    'James A. Garfield': (2, 1),   # Moderate right, slightly authoritarian
    'Chester A. Arthur': (4, 1),   # Moderate right, moderately authoritarian
    'Benjamin Harrison': (5, 3),   # Right-leaning authoritarian
    'William McKinley': (6, 4),    # Strongly right-leaning, authoritarian
    'Theodore Roosevelt': (-5, 5), # Left-leaning authoritarian (progressive policies)
    'William Taft': (4, 3),        # Right-leaning authoritarian
    'Calvin Coolidge': (8, 4),     # Strongly right-leaning, authoritarian
    'Herbert Hoover': (7, 5),      # Right-leaning authoritarian
    'Dwight D. Eisenhower': (3, 2),# Moderate right, slightly authoritarian
    'Richard M. Nixon': (7, 8),    # Right-leaning authoritarian
    'Gerald Ford': (4, 3),         # Right-leaning authoritarian
    'Ronald Reagan': (9, 7),       # Strongly right-leaning, authoritarian
    'George H. W. Bush': (6, 5),   # Right-leaning authoritarian
    'George W. Bush': (6, 5),      # Right-leaning authoritarian
    'Donald Trump': (8, 7),        # Strongly right-leaning, authoritarian
    'John Adams': (3, 5),          # Moderate right, authoritarian (Federalist policies)
    'John Quincy Adams': (2, 2),   # Moderate right, slightly authoritarian
    'William Harrison': (5, 3),    # Right-leaning authoritarian
    'John Tyler': (6, 4),          # Right-leaning authoritarian
    'Zachary Taylor': (5, 4),      # Right-leaning authoritarian
    'Millard Fillmore': (4, 3),    # Right-leaning authoritarian
    'Warren G. Harding': (7, 6)    # Right-leaning authoritarian
}

def assign_leaning(president):
    x_value = president_positions.get(president, (0, 0))[0] # Default to (0, 0) if unknown
    if x_value < 0:
        return 'Left-Leaning'
    elif x_value > 0:
        return 'Right-Leaning'
    else:
        return 'Unknown'
    
def assign_coordinates(president):
    return president_positions.get(president, (0, 0))  # Default to (0, 0) if unknown

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
    coordinates = []

    for entry in data:
        speech_text = entry['transcript']
        president = entry['president']
        label = assign_leaning(president)
        labels.append(label)
        position = assign_coordinates(president)
        coordinates.append(position)
        speeches.append(speech_text)
    
    # preprocess speech text
    cleaned_speeches = [preprocess_text(speech) for speech in speeches]

    return cleaned_speeches, labels, coordinates


def save_preprocessed_data(cleaned_speeches, labels, coordinates, filename='speeches/preprocessed_speeches.json'):
    data = {'speeches': cleaned_speeches, 'labels': labels, 'coordinates': coordinates}
    with open(filename, 'w') as f:
        json.dump(data, f)

def main():
    cleaned_speeches, labels, coordinates = preprocess()
    save_preprocessed_data(cleaned_speeches, labels, coordinates)


if __name__ == '__main__':
    main()