from perceptron_algorithm.Neuron import Neuron
from feature_extraction.Vocabulary import num_of_initial_neurons, vectorized_emails
import random

"""
Todo:
    - Każdy neuron dostaje jedną i tą samą liste na początek, potem następne warstwy dostają wyniki poprzedniej na zasadzie każdy do każdego
    - obliczanie błędu 
    - zbieranie danych warstwy poprzedniej 
"""


class NeuralNetwork:
    def __init__(self, vectorized_mails_input: list[list[int]], layer_output=None,
                 num_of_neurons=num_of_initial_neurons):
        # podajemy cały set
        self.input = vectorized_mails_input
        self.layer_output = layer_output
        self.num_of_neurons = num_of_neurons
        self.input_layer = {}

    # TODO - dodać kalkulacje straty na każdej warstwie - porównać z gotowymi klasami
    def hinge_loss(self, prediciton, layer_output):
        return max(0, 1 - prediciton * layer_output)

    def create_input_layer(self):
        # Tworzenie warstwy neuronów
        input_layer = {}
        list_of_neurons = []
        # tworzenie warstwy (89 neuronów)
        for i in range(self.num_of_neurons):
            n = Neuron(id=i, weight=random.uniform(0.1, 0.3))
            list_of_neurons.append(n)
        # Każdy neuron genetuje x wyników, zatem do klucza (id neuronu) dolaczamy liste wyników dla każdego maila
        list_of_outputs = []
        for neuron in list_of_neurons:
            for mail in self.input:
                neuron.input = mail
                neuron.bias = (neuron.weight + sum(filter(lambda x: x != -1, mail))) / len(list(filter(lambda x: x != -1, mail)))
                # print("Waga: ", neuron.weight, "Bias: ", neuron.bias)
                list_of_outputs.append(neuron.weighted_sum())
            input_layer[neuron.id] = list_of_outputs
            list_of_outputs = []
        self.input_layer = input_layer  # przypisanie słownika z wynikami pierwszej warstwy
        return input_layer

    def create_hidden_layers(self):
        # TODO: neurony które,miały więcej wartości <0 zostają wyłączone lub zmniejszamy ich wage
        #  kopiujemy obiekty i aktualizujemy ich wagi bez zmiany pierwotnych obiektów
        pass

    def create_output_layer(self):
        pass

    def propagation(self):
        pass

    def train(self):
        pass


vect = [
    [1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1]]

network = NeuralNetwork(vectorized_mails_input=vect)
input_layer = network.create_input_layer()
print(input_layer)
