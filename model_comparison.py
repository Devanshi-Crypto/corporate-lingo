import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the trained models
models = {
    'Multinomial Naive Bayes': joblib.load('models/multinomial_nb.pkl'),
    'Linear SVM': joblib.load('models/linear_svc.pkl'),
    'Random Forest': joblib.load('models/random_forest.pkl'),
    'Logistic Regression': joblib.load('models/logistic_regression.pkl')
}

# Load the TfidfVectorizer
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Load and preprocess the examples
with open('corporate-lingo-examples.txt', 'r') as file:
    content = file.read()

sections = content.split('# ')
examples = {}
for section in sections[1:]:  # Skip the first empty section
    lines = section.strip().split('\n')
    category = lines[0].strip()
    examples[category] = [line.split('.', 1)[1].strip() for line in lines[1:]]

# Function to predict and calculate accuracy
def predict_examples(model, examples):
    predictions = []
    for category, texts in examples.items():
        X = vectorizer.transform(texts)
        preds = model.predict(X)
        predictions.extend([(category, text, pred) for text, pred in zip(texts, preds)])
    return predictions

# Test all models and collect results
results = {}
for name, model in models.items():
    predictions = predict_examples(model, examples)
    results[name] = predictions

# Calculate accuracy for each category and model
accuracy_results = {}
for name, preds in results.items():
    accuracy_results[name] = {}
    for category in examples.keys():
        category_preds = [p for p in preds if p[0] == category]
        accuracy = sum(1 for p in category_preds if 'jargon' in p[2].lower()) / len(category_preds)
        accuracy_results[name][category] = accuracy

# Plot results
categories = list(examples.keys())
models_names = list(models.keys())

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(categories))
width = 0.2
multiplier = 0

for model, accuracies in accuracy_results.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, [accuracies[c] for c in categories], width, label=model)
    ax.bar_label(rects, fmt='{:.2f}', padding=3)
    multiplier += 1

ax.set_ylabel('Accuracy (Jargon Detection)')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width, categories)
ax.legend(loc='upper left', ncols=2)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('model_comparison_results.png')
plt.close()

# Print detailed results
for name, preds in results.items():
    print(f"\nResults for {name}:")
    for category, text, prediction in preds:
        print(f"Category: {category}")
        print(f"Text: {text}")
        print(f"Prediction: {prediction}")
        print("-" * 50)