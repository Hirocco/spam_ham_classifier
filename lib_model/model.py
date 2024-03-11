import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('mail_data.csv')

data = df.where((pd.notnull(df)), '')
"""
print(data.head())
print(data.info())
"""

data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1

X = data['Message']  # content wiadomosco
Y = data['Category']  # kategoria spam/ham 0/1

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#print(X_test_features)

Model = LogisticRegression()
Model.fit(X_train_features, Y_train)

predicition_on_training_data = Model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, predicition_on_training_data)
print('Precyzyjnosc w treningu: ', accuracy_on_training_data)


predicition_on_test_data = Model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, predicition_on_test_data)
print('Precyzyjnosc w treningu: ', accuracy_on_test_data)

input_your_mail = ["WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]
input_data_features = feature_extraction.transform(input_your_mail)
prediction = Model.predict(input_data_features)

if prediction[0] == 0:
    print('Spam')
else:
    print('Ham')