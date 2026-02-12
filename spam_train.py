import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
# 1. Load data
data = pd.read_csv("spam.csv", encoding='latin-1')

# 2. Print columns once to confirm
print("Columns in dataset:", data.columns)

# Select only needed columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']


# 3. Convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 4. Split data
X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

print("Model Accuracy:", accuracy)

with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))

print("Model, Vectorizer, and Accuracy saved successfully!")
