from data_preparation.data_preprocessing import *
from feature_extraction.Vocabulary import *
from perceptron_algorithm.Neural_network import *

"""
ham,"Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."
ham,Ok lar... Joking wif u oni...
spam,"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
ham,U dun say so early hor... U c already then say...
ham,"Nah I don't think he goes to usf, he lives around here though"
spam,"FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv"
ham,Even my brother is not like to speak with me. They treat me like aids patent.
ham,"As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"
spam,WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.
"""

emails = [
    "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
    "Ok lar... Joking wif u oni...",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
    ]

"""   
    "U dun say so early hor... U c already then say...",
    "Nah I don't think he goes to usf, he lives around here though",
    "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv",
    "Even my brother is not like to speak with me. They treat me like aids patent.",
    "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune",
    "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."
    ]
    [0, 0, 1, 0, 0, 1, 0, 0, 1]
"""

preprocessed_mails = lemmatization(
    stopWordRemoval(punctationRemoval(toLowerCase(tokenization(emails)))))

vocabulary = build_vocabulary(preprocessed_mails)
vectorized_emails = standaryzowanie_wektoru(vectorize_emails(vocabulary, preprocessed_mails))

network = NeuralNetwork(predictions=[0, 0, 1], vectorized_mails_input=vectorized_emails)
input_layer = network.create_input_layer()
print(network.check_results(input_layer))





"""
akuratne:
spam : 749
ham : 4825
"""
