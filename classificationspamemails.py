import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import kagglehub

# Download the dataset from kaggle
#path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
#print("Path to dataset files:", path)

# Load sample email data ('spam.csv' with columns ['label', 'message'])
df = pd.read_csv('C:\\Users\\danielchakraborty\\Desktop\\Projects\\dbs\\spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to binary values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
# Convert text data to numerical feature vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Predict email labels
y_pred = clf.predict(X_test_vec)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
