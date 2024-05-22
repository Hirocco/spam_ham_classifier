import itertools

"""
Neuron:
---------
ID : Int
weight : float 
bias : float 
output : float
mail_input : list[int]
output : float 
is_activated : boolean
"""

class Neuron:
    def __init__(self, id, weight, bias=0, output=None, mail_input=None, is_activated=None):
        # Pierwsza wagi będą inicjowane losowo, jeżeli jesteśmy jużw innej warstwie niż pierwszej to będą zmieniane
        self.weight = weight
        self.input = mail_input
        self.id = id
        # suma elementów i wagi przez długość listy - pierwszy sposób obliczania biasu
        self.bias = bias
        self.output = output
        self.is_activated = is_activated

    def activation_func(self, output: float) -> bool:
        return output > 1

    def weighted_sum(self) -> float:
        input_values = itertools.takewhile(lambda x: x != -1, self.input)
        weighted_sum = self.weight + sum(x * self.weight for x in input_values)
        self.output = weighted_sum - self.bias
        return self.output if self.activation_func(self.output) else 0
