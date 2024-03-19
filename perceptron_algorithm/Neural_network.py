from perceptron_algorithm.Neuron import Neuron
from feature_extraction.Vocabulary import num_of_initial_neurons, vectorized_emails, label_list
import random


# Todo:
#  - zbieranie danych warstwy poprzedniej
#  - poprawić działanie kalkulacji błędu
#  - przesiewanie sieci,
#  - trenowanie i aktualizacjia wag/biasów w oparciu o szukanie najmniejszego błędu (wyprowadzenie na kartce)


class NeuralNetwork:
    def __init__(self, predictions: list[int], vectorized_mails_input: list[list[int]], layer_output=None,
                 num_of_neurons=num_of_initial_neurons):
        # podajemy cały set
        self.input = vectorized_mails_input
        self.layer_output = layer_output
        self.num_of_neurons = num_of_neurons
        self.input_layer = {}
        self.hidden_layers = {}
        self.predictions = predictions
        self.network_loss = {}

    # output = 1 prediction = 0 suma_wazona = 2.130215165708253, waga = 0,23812004459646455, teta = 0.5
    # blad = -0.25362346512925716287024027846558
    # jak ten bład interpretować ?

    # ouput = 1 , pred = 0 , suma_wazona = 2.130215165708253, waga = 0,23812004459646455, teta = 0.5
    # blad = -0.25362346512925716287024027846558

    # bierzemy wartosc absolutna by ustandaryzowac kwestie bledu
    # pred = 0 output = 0 blad = 0.0


    def calculate_loss(self, prediction: int, layer_output: int,
                       neuron_weight: float, neuron_weighted_sum: float):
        teta = 0.5
        return teta * abs(prediction - layer_output) * neuron_weight * neuron_weighted_sum

    def create_input_layer(self) -> dict:  # Zwraca { id neuronu : wartość neuronu}
        # Tworzenie warstwy neuronów
        layer_loss = {}
        list_of_neurons = []
        # tworzenie warstwy (89 neuronów)
        for i in range(self.num_of_neurons):
            n = Neuron(id=i, weight=random.uniform(0.1, 0.3))
            list_of_neurons.append(n)
        # Każdy neuron genetuje x wyników, zatem do klucza (id neuronu) dolaczamy liste wyników dla każdego maila

        for neuron in list_of_neurons:
            list_of_outputs = []
            loss = []
            for i, input_data in enumerate(self.input):
                neuron.input = input_data
                neuron.bias = ((neuron.weight + sum(filter(lambda x: x != -1, input_data))) /
                               len(list(filter(lambda x: x != -1, input_data))))
                # zaokrąglenie do 1 liczby po przecinku w celu ułatwienia debugowania
                neuron_output = 1 if round(neuron.weighted_sum(), 1) > 0.51 else 0
                list_of_outputs.append(neuron_output)
                single_loss = self.calculate_loss(prediction=self.predictions[i], layer_output=neuron_output,
                                                  neuron_weight=neuron.weight,
                                                  neuron_weighted_sum=neuron.weighted_sum())
                # single_loss = self.hinge_loss(prediction=self.predictions[i], layer_output=neuron_output)
                loss.append(single_loss)  # Użyj indeksu i jako klucza dla layer_loss
            self.input_layer[neuron.id] = list_of_outputs
            layer_loss[neuron.id] = loss

        #print("loss: ", layer_loss.get(0))
        self.network_loss = layer_loss
        return self.input_layer

    def create_hidden_layers(self) -> dict:
        num_of_neurons = 0
        for neuron_id, neuron_output in self.input_layer.items():
            if set(neuron_output) == set(self.predictions):
                self.hidden_layers[neuron_id] = neuron_output
                num_of_neurons += 1
        """
        print(first_hidden_layer)
        print(num_of_neurons)
        """
        return self.hidden_layers

    def check_results(self):
        count_total_input = len(self.input)
        total = []
        count_good_guesses = 0
        # TODO: przepatrz to w debuggerze
        for neuron_output in self.hidden_layers.values():
            for otp, label in zip(neuron_output, label_list):
                if otp == label:
                    count_good_guesses += 1
            total.append(count_good_guesses / count_total_input)
            count_good_guesses = 0
        return sum(total) / len(total)

    def create_output_layer(self):
        pass


"""
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
pred = [1,0]
"""
network = NeuralNetwork(predictions=label_list, vectorized_mails_input=vectorized_emails)
input_layer = network.create_input_layer()
# print(input_layer)
network.create_hidden_layers()
result = round(network.check_results() * 100, 2)
print("Average accuracy: ", result, "%")
print("Average error: ", 100 - result, "%")

"""for value in network.network_loss.get(0):
    print(value)"""