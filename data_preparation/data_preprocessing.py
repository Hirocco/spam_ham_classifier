import sys

from data_preparation.data_loading import data_classes, data_content
from data_preparation.stopwords import stopwords

import nltk

from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split

sys.path.append("../")

# zamienia slownik k: klasa - v: mail na liste maili

# bierze jeden mail i dzieli go na liste wyrazow
def tokenization(mail_content):
    two_dimensional_mail_array = []
    for mail in mail_content:
        splitted_mail = mail.split()
        two_dimensional_mail_array.append(splitted_mail)
    return two_dimensional_mail_array


def toLowerCase(tokenized_mails: list[list[str]]):
    lowered_words = []
    for mail in tokenized_mails:
        lowered_mail = []  # Create a new list for each mail's lowercase words
        for word in mail:
            lowered_mail.append(word.lower())  # Append the lowercase version of each word to the mail's list
        lowered_words.append(lowered_mail)  # Append the mail's list to the two-dimensional array
    return lowered_words


# usuwa znaki z listy chars_to_remove
def punctationRemoval(words: list[list[str]]):
    chars_to_remove = ['.', ',', '!', '?', ':', ';', "'", '"', '-', '(', ')', '[', ']', '{', '}',
                       '/']  # te znaki usuniemy by łatwiej szkolić model
    words_without_punctuation = []
    for sentence in words:
        word_arr = []
        for word in sentence:  # to sa slowa dopiero
            char_without_punctuation = ""
            for char in word:
                if char not in chars_to_remove:
                    char_without_punctuation += char
            word_arr.append(char_without_punctuation)
        words_without_punctuation.append(word_arr)
    return words_without_punctuation


def stopWordRemoval(mails: list[list[str]]):
    words_wuthout_stopwords = []
    for mail in mails:
        word_arr = []
        for word in mail:
            if word not in stopwords:
                word_arr.append(word)
        words_wuthout_stopwords.append(word_arr)
    return words_wuthout_stopwords


# zamienia np. running na run etc.
def lemmatization(mails: list[list[str]]):
    # Download the WordNet resource (needed for lemmatization)
    nltk.download('wordnet')

    # Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_mails = []
    for mail in mails:
        lemmatized_mail = []
        for word in mail:
            lemmatized_word = lemmatizer.lemmatize(word, pos="v")  # 'v' indicates that the word is a verb
            lemmatized_mail.append(lemmatized_word)
        lemmatized_mails.append(lemmatized_mail)
    return lemmatized_mails


preprocessed_mails = lemmatization(
    stopWordRemoval(punctationRemoval(toLowerCase(tokenization(data_content)))))  # dziala


def split_dataset(mail_arr):
    X_train, X_test = train_test_split(mail_arr, test_size=0.2, random_state=42)
    return X_train, X_test

Train_set, Test_set = split_dataset(preprocessed_mails)
"""
print("Train : " , Train_set)
print("Test : ", Test_set)
"""
