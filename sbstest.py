import pandas as pd
import nltk
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords

def parse(new_article):
    # Load the datasets
    fake = pd.read_csv('C:/users/xanhug/Fake.csv', low_memory=False)
    true = pd.read_csv('C:/users/xanhug/True1.csv', low_memory=False)

    # Label the data
    fake['label'] = 1
    true['label'] = 0

    # Concatenate and shuffle
    data = pd.concat([fake, true], axis=0).sample(frac=1).reset_index(drop=True)

    # Extract text and label
    X = data['text']
    y = data['label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Define pipelines for both classifiers
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('clf', MultinomialNB())
    ])
    
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('clf', LogisticRegression())
    ])

    # Train both models
    nb_pipeline.fit(X_train, y_train)
    lr_pipeline.fit(X_train, y_train)

    # Predictions
    nb_predictions = nb_pipeline.predict(X_test)
    lr_predictions = lr_pipeline.predict(X_test)

    # Evaluate the models
    nb_result = {
        "title": "Naive Bayes Results:",
        "accuracy": accuracy_score(y_test, nb_predictions),
        "report": classification_report(y_test, nb_predictions),
        "prediction": "Real" if nb_pipeline.predict(new_article)[0] == 0 else "Misinformation"
    }
    
    lr_result = {
        "title": "Logistic Regression Results:",
        "accuracy": accuracy_score(y_test, lr_predictions),
        "report": classification_report(y_test, lr_predictions),
        "prediction": "Real" if lr_pipeline.predict(new_article)[0] == 0 else "Misinformation"
    }

    return y_test, nb_predictions, lr_predictions, nb_result, lr_result

def display_results(model_result, col, y_test):
    col.header(model_result["title"])
    col.write(f"Accuracy: {model_result['accuracy']:.2f}")
    col.write(f"Prediction on new article: {model_result['prediction']}")
    
    # Display classification report as a table
    report_df = pd.DataFrame.from_dict(
        classification_report(y_test, model_result['predictions'], output_dict=True)
    ).transpose()
    col.table(report_df)

def main():
    # Streamlit title
    st.title("Text Classification with Naive Bayes and Logistic Regression")

    # Textarea in Streamlit for user input
    user_input = st.text_area("Enter the article text or title here",)

    # When user provides input and clicks on classify button, make predictions and display them
    if st.button("Classify"):
        y_test, nb_predictions, lr_predictions, nb_res, lr_res = parse([user_input])

        nb_res['predictions'] = nb_predictions
        lr_res['predictions'] = lr_predictions

        col1, col2 = st.columns(2)
        display_results(nb_res, col1, y_test)
        display_results(lr_res, col2, y_test)

if __name__ == "__main__":
    main()
