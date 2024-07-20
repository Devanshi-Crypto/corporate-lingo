import re
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def parse_data():
    with open("workplace-jargon-dictionary.txt", 'r', encoding='utf8') as file:
        content = file.read()
    
    # Split the content into sections
    sections = re.split(r'###\s*', content)[1:]  # Skip the intro part
    
    data = []
    for section in sections:
        lines = section.strip().split('\n')
        if not lines:  # Skip empty sections
            continue
        
        term = lines[0].strip('"')
        definition = ''
        example = ''
        
        for line in lines[1:]:
            if 'means' in line:
                definition = line.split('means', 1)[1].strip()
            elif 'Used in a sentence:' in line:
                example = line.replace('Used in a sentence:', '').strip()
        
        data.append({
            'Term': term,
            'Definition': definition,
            'Example': example
        })
    
    return data

def save_to_csv(data, file_path):
    with open(file_path, 'w', newline='', encoding='utf8') as file:
        writer = csv.DictWriter(file, fieldnames=['Term', 'Definition', 'Example'])
        writer.writeheader()
        writer.writerows(data)

# Main execution
data = parse_data()
save_to_csv(data, 'corporate_lingo.csv')

print(f"Processed {len(data)} terms and saved to corporate_lingo.csv")

# You can add your model training code here