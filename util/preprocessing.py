import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# NOTE: the following approximated positions were assigned by us, and are not the result of scientific study
president_positions = {
    'Andrew Jackson': (6, 6),      
    'Martin Van Buren': (-5, -3),  
    'James K. Polk': (-6, 3),      
    'Franklin Pierce': (-4, 1),    
    'James Buchanan': (-3, 2),     
    'Grover Cleveland': (-4, -1),  
    'Woodrow Wilson': (-6, 8),     
    'Franklin D. Roosevelt': (-9, 6),
    'Harry S. Truman': (-7, 4),    
    'John F. Kennedy': (-4, -2),   
    'Lyndon B. Johnson': (-5, 5),  
    'Jimmy Carter': (-4, -3),      
    'Bill Clinton': (-3, -3),      
    'Barack Obama': (-4, -2),      
    'Joe Biden': (-5, 2),          
    'Thomas Jefferson': (-7, -8),  
    'James Madison': (-6, -7),     
    'James Monroe': (-5, -5),      
    'Andrew Johnson': (5, 4),      
    'Abraham Lincoln': (-6, 7),    
    'Ulysses S. Grant': (4, 4),    
    'Rutherford B. Hayes': (3, 2), 
    'James A. Garfield': (2, 1),   
    'Chester A. Arthur': (4, 1),   
    'Benjamin Harrison': (5, 3),   
    'William McKinley': (6, 4),    
    'Theodore Roosevelt': (-5, 5), 
    'William Taft': (4, 3),        
    'Calvin Coolidge': (8, 4),     
    'Herbert Hoover': (7, 5),      
    'Dwight D. Eisenhower': (3, 2),
    'Richard M. Nixon': (7, 8),    
    'Gerald Ford': (4, 3),         
    'Ronald Reagan': (9, 7),       
    'George H. W. Bush': (6, 5),   
    'George W. Bush': (6, 5),      
    'Donald Trump': (8, 7),        
    'John Adams': (3, 5),          
    'John Quincy Adams': (2, 2),   
    'William Harrison': (5, 3),    
    'John Tyler': (6, 4),          
    'Zachary Taylor': (5, 4),      
    'Millard Fillmore': (4, 3),    
    'Warren G. Harding': (7, 6)    
}

def assign_leaning(president):
    x_value = president_positions.get(president, (0, 0))[0]  # Default to (0, 0) if unknown
    if x_value < 0:
        return 'Left-Leaning'
    elif x_value > 0:
        return 'Right-Leaning'
    else:
        return 'Unknown'
    
def assign_coordinates(president):
    global president_positions  # Allows modification of the original dictionary

    # Calculate mean of x and y coordinates
    x_mean = sum(pos[0] for pos in president_positions.values()) / len(president_positions)
    y_mean = sum(pos[1] for pos in president_positions.values()) / len(president_positions)

    # Adjust coordinates to center around the origin
    # NOTE: this affects the president's positions, and should only be used for demo purposes
    president_positions = {president: (pos[0] - x_mean, pos[1] - y_mean) for president, pos in president_positions.items()}

    # Return the centered coordinates for the specified president
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

# def extract_entities(text):
#     """Extract named entities from the text using spaCy."""
#     doc = nlp(text)
#     return [(ent.text, ent.label_) for ent in doc.ents]

def extract_entities_batch(speeches, batch_size=32):
    """Extract named entities from a batch of texts using spaCy with a specified batch size."""
    docs = list(nlp.pipe(speeches, batch_size=batch_size)) 
    return [[(ent.text, ent.label_) for ent in doc.ents] for doc in docs]

def preprocess():
    with open('./speeches/speeches.json', 'r') as file:
        data = json.load(file)
        
    # Extract speeches and labels
    speeches = []
    labels = []
    coordinates = []
    entities_list = []  # List to store entities

    for entry in data:
        speech_text = entry['transcript']
        president = entry['president']
        labels.append(assign_leaning(president))
        coordinates.append(assign_coordinates(president))
        speeches.append(speech_text)

    cleaned_speeches = [preprocess_text(speech) for speech in speeches]
    
    # Extract entities for all speeches in batch
    entities_list = extract_entities_batch(cleaned_speeches, batch_size=280) 

    # return speeches, labels, coordinates, and entities
    return cleaned_speeches, labels, coordinates, entities_list

def save_preprocessed_data(cleaned_speeches, labels, coordinates, entities_list, filename='./speeches/preprocessed_speeches.json'):
    data = {
        'speeches': cleaned_speeches,
        'labels': labels,
        'coordinates': coordinates,
        'entities': entities_list  
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

def main():
    cleaned_speeches, labels, coordinates, entities_list = preprocess()
    save_preprocessed_data(cleaned_speeches, labels, coordinates, entities_list)

if __name__ == '__main__':
    main()
