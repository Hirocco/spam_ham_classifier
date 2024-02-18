import sys

sys.path.append('C:/Users/Kamil/Desktop/spam_ham_ai/data_preparation')

from data_preparation.data_preprocessing import Train_set, Test_set
from data_preparation.data_loading import data_classes

def build_vocabulary(train_set: list[list[str]]):
    vocabulary = {}
    for mail in train_set:
        for word in mail:
            if word in vocabulary:
                vocabulary[word] += 1
            else:
                vocabulary[word] = 1
    return vocabulary


def vectorize_emails(vocabulary: dict, data_set: list[list[str]]):
    vectorized_emails = []
    for mail in data_set:
        feature_vector = []
        for word in mail:
            if word in vocabulary:
                feature_vector.append(mail.count(word))
            else:
                feature_vector.append(0)
        vectorized_emails.append(feature_vector)
    return vectorized_emails

def standaryzowanie_wektoru(vectorized_emails : list[list[int]]):
    max_len = 0
    for mail in vectorized_emails:
        if len(mail) > max_len:
            max_len = len(mail)

    for mail in vectorized_emails:
        while(len(mail) != max_len):
            mail.append(-1)

    return vectorized_emails

def set_label(data_classes):
    label_list = []
    for label in data_classes:
        if label == "spam":
            label_list.append(0)
        else:
            label_list.append(1)
    return label_list

vocabulary = build_vocabulary(Train_set)
vectorized_emails = standaryzowanie_wektoru(vectorize_emails(vocabulary, Train_set))
vectorized_test_emails = vectorize_emails(vocabulary, Test_set)
num_of_initial_neurons = len(vectorized_emails[0])
label_list = set_label(data_classes=data_classes)


"""dziala
print("vect:" , vectorized_emails[2])
print("mail: ", Train_set[0])
print("Vocabulary: ", vocabulary)
"""
