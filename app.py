from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

#Step 1: Data Cleaning
data = pd.read_csv("Fake_News.csv")
# Rename the unnamed column to 'id'
data = data.rename(columns={'Unnamed: 0': 'id'})
print(data.shape)
print(data.head(10))

# Check for null values
null_counts = data.isnull().sum()
print(null_counts)

# check for duplicates 
duplicates = data.duplicated()
print(duplicates)


#Step 2: Exploratory Data Analysis (EDA)
# Check distribution of labels
label_counts = data['label'].value_counts()
print("Distribution of labels:")
print(label_counts)


# Analyze text length
data['title_length'] = data['title'].apply(lambda x: len(x.split()))
data['text_length'] = data['text'].apply(lambda x: len(x.split()))


# Word frequency analysis (example)
from collections import Counter
words = ' '.join(data['text']).split()
word_freq = Counter(words)

print("Most common words:")
print(word_freq.most_common(10))


# Step 3: Data Preprocessing

# Combine title and text columns
data['combined_text'] = data['title'] + " " + data['text']

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization and lowercase
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stopwords and non-alphabetic tokens
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]  # Stemming
    return " ".join(stemmed_tokens)

data['processed_text'] = data['combined_text'].apply(preprocess_text)


#Step 4: Feature Extraction
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data['processed_text'])
y = data['label'].map({'FAKE': 0, 'REAL': 1})
print(X.shape)


# print(tfidf_vectorizer.get_feature_names_out()[:20])

# print(tfidf_vectorizer.get_params())

#Step 5: ML Models
# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Train a Logistic Regression model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)

# Predictions
y_pred = logistic_regression_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Initialize SVM classifier
svm_classifier = SVC(kernel='linear', probability=True)  # Linear kernel for linear classification

# Train the classifier
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

def predict(message):
    # Preprocess the input text
    processed_message = preprocess_text(message)
    # Transform the preprocessed text using the TF-IDF vectorizer
    tfidf_message = tfidf_vectorizer.transform([processed_message])
    # Make predictions using the trained Logistic Regression model
    prediction = svm_classifier.predict(tfidf_message)[0]
    # Map the predicted label to "fake" or "real"
    if prediction == 0:
        return "fake"
    else:
        return "real"

app = Flask(__name__)

@app.route("/")
def form():
    return render_template('form.html')

@app.route("/predict", methods=["POST"])
def submit():
    if request.method == "POST":
        message = request.form.get("news-text")
        # Get prediction for the input message
        result = predict(message)
        # Redirect to the response page with the prediction result
        return redirect(url_for('response', result=result))

@app.route("/response")
def response():
    result = request.args.get('result')
    return render_template('response.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
