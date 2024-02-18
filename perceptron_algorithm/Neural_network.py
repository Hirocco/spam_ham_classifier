from perceptron_algorithm.Neuron import Neuron
from feature_extraction.Vocabulary import num_of_initial_neurons, vectorized_emails, label_list
import random


# Todo:
#  - Każdy neuron dostaje jedną i tą samą liste na początek, potem następne warstwy dostają wyniki poprzedniej na zasadzie każdy do każdego
#  - zbieranie danych warstwy poprzedniej


class NeuralNetwork:
    def __init__(self, predictions: list[int], vectorized_mails_input: list[list[int]], layer_output=None,
                 num_of_neurons=num_of_initial_neurons):
        # podajemy cały set
        self.input = vectorized_mails_input
        self.layer_output = layer_output
        self.num_of_neurons = num_of_neurons
        self.input_layer = {}
        self.predictions = predictions

    # TODO - dodać kalkulacje straty na każdej warstwie - porównać z gotowymi klasami
    def hinge_loss(self, prediction: int, layer_output):
        return max(0, abs(prediction - layer_output / 10))  # dzielimy przez 10 by uzyskać wartości z przedziału [0;1]

    def create_input_layer(self) -> dict: # Zwraca { id neuronu : wartość neuronu}
        # Tworzenie warstwy neuronów
        layer_loss = {}
        input_layer = {}
        list_of_neurons = []
        # tworzenie warstwy (89 neuronów)
        for i in range(self.num_of_neurons):
            n = Neuron(id=i, weight=random.uniform(0.1, 0.3))
            list_of_neurons.append(n)
        # Każdy neuron genetuje x wyników, zatem do klucza (id neuronu) dolaczamy liste wyników dla każdego maila

        for neuron in list_of_neurons:
            list_of_outputs = []
            for i, input_data in enumerate(self.input):
                neuron.input = input_data
                neuron.bias = ((neuron.weight + sum(filter(lambda x: x != -1, input_data))) /
                               len(list(filter(lambda x: x != -1, input_data))))
                # zaokrąglenie do 1 liczby po przecinku w celu ułatwienia debugowania
                neuron_output = round(neuron.weighted_sum(), 1)
                list_of_outputs.append(neuron_output)
                loss = self.hinge_loss(prediction=self.predictions[i], layer_output=neuron_output)
                layer_loss[i] = loss  # Użyj indeksu i jako klucza dla layer_loss
            input_layer[neuron.id] = list_of_outputs

        self.input_layer = input_layer  # przypisanie słownika z wynikami pierwszej warstwy
        print("loss: ", layer_loss)
        return input_layer

    def create_hidden_layers(self) -> dict:
        # TODO: neurony które,miały więcej wartości <0 zostają wyłączone lub zmniejszamy ich wage
        #  kopiujemy obiekty i aktualizujemy ich wagi bez zmiany pierwotnych obiektów
        pass

    def create_output_layer(self):
        pass


vect = [
    [1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1],
    [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1]
]

network = NeuralNetwork(predictions=[1, 0], vectorized_mails_input=vect)
input_layer = network.create_input_layer()
print(input_layer)
