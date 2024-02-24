
# imports

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pickle

# data set is loaded

df = pd.read_pickle('Final_dataset.pkl')
df.rename(columns={'Facts': 'facts'}, inplace=True)
df.drop(columns=['index'], inplace=True)
df.reset_index(inplace=True)

# Create a mirrored case for each case, where the parties are swapped to prevent favoring first_party
df_list = df.values.tolist()
result = []
for row in df_list:
    result.append(row[1:])
    mirrored_row = row.copy()
    #  first_party is at index=4, second=5, winner_index=7
    mirrored_row[4] = row[5]
    mirrored_row[5] = row[4]
    mirrored_row[7] = 1-mirrored_row[7]
    result.append(mirrored_row[1:])
df2 = pd.DataFrame(result)
df2.rename(columns={
    0: 'ID',
    1: 'name',
    2: 'href',
    3: 'first_party',
    4: 'second_party',
    5: 'winning_party',
    6: 'winner_index',
    7: 'facts',
}, inplace=True)
df = df2
df.reset_index(inplace=True)


# # Perform an 80-20 split for training and testing data
X_train_party1_text, X_test_party1_text, \
X_train_party2_text, X_test_party2_text, \
X_train_facts_text, X_test_facts_text, \
y_train, y_test = train_test_split(
    df['first_party'],
    df['second_party'],
    df['facts'],
    df['winner_index'],
    test_size=0.2,
    stratify=df['winner_index'],
    random_state=42
)

vectorizer = TfidfVectorizer()
X_train_facts = vectorizer.fit_transform(X_train_facts_text)
X_test_facts = vectorizer.transform(X_test_facts_text)
X_train_party1 = vectorizer.transform(X_train_party1_text)
X_test_party1 = vectorizer.transform(X_test_party1_text)
X_train_party2 = vectorizer.transform(X_train_party2_text)
X_test_party2 = vectorizer.transform(X_test_party2_text)

X_train = np.concatenate([X_train_party1.todense(), X_train_party2.todense(), X_train_facts.todense()], axis=1)
X_test = np.concatenate([X_test_party1.todense(), X_test_party2.todense(), X_test_facts.todense()], axis=1)


X_train = np.asarray(X_train)
X_test = np.asarray(X_test)


del X_train_facts, X_train_party1, X_train_party2
del X_test_facts, X_test_party1, X_test_party2


model = pickle.load(open('model.pkl', 'rb'))


st.title('Legal Case Prediction')

party1 = st.text_input('Enter the name of the first party:')
party2 = st.text_input('Enter the name of the second party:')
facts = st.text_area('Enter the case description:')

if st.button('Predict Winner'):
    X_party1 = vectorizer.transform([party1]).todense()
    X_party2 = vectorizer.transform([party2]).todense()
    X_facts = vectorizer.transform([facts]).todense()

    # Convert to NumPy arrays
    X_party1 = np.asarray(X_party1)
    X_party2 = np.asarray(X_party2)
    X_facts = np.asarray(X_facts)

    # Concatenate along the correct axis
    X = np.concatenate([X_party1, X_party2, X_facts], axis=1)
    prediction = model.predict(X)[0]
    winner = "First Party" if prediction == 0 else "Second Party"

    st.success('Winner: {}'.format(winner))
